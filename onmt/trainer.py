"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from copy import deepcopy
import torch
import traceback

import onmt.utils
from onmt.utils.logging import logger
import time

import numpy as np
import sys


def build_trainer(opt, device_id, model, fields, optim, model_saver=None,tgt_vocab=None,vocab_sample_method="uniform",src_vocab=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (list :obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    pad_idx = tgt_field.vocab.stoi[tgt_field.pad_token]

    special_tokens = []

    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, special_tokens=special_tokens)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False, special_tokens=special_tokens)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           use_knowledge=opt.knowledge,
                           tgt_vocab=tgt_vocab,
                           vocab_sample_method=vocab_sample_method,
                           src_vocab=src_vocab,
                           pad_idx=pad_idx,
                           elmo=opt.encoder_type=='elmo',
                           pairenc=opt.pairenc)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0], use_knowledge=False, 
                 tgt_vocab=None,vocab_sample_method="uniform",
                 src_vocab=None, elmo=False, pad_idx=-1, pairenc=False):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.use_knowledge = use_knowledge
        self.tgt_vocab = tgt_vocab
        self.src_vocab=src_vocab
        self.pad_idx = pad_idx
        self.elmo=elmo
        self.pairenc=pairenc
        if self.tgt_vocab is not None:
            self._sampler_init(vocab_sample_method=vocab_sample_method)

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def _sampler_init(self, vocab_sample_method="uniform"):
        no_sample_list = ['<q>', '</q>', '<a>', '</a>', 'is', 'the', 'no', 'yes', 'are', 'a',
           'it', 'there', 'what', 'see', 'you', 'any', 'color', 'in', 'can',
           ',', '\'s', '\'m', 'lots']

        if vocab_sample_method == "freq":
            self.vocab_freq_id = dict([(self.tgt_vocab.stoi[key], 
                self.tgt_vocab.freqs[key]) for key in self.tgt_vocab.freqs if not key in no_sample_list])
        elif vocab_sample_method == "uniform":
            self.vocab_freq_id = dict([(self.tgt_vocab.stoi[key], 
                1) for key in self.tgt_vocab.freqs if not key in no_sample_list])
        elif vocab_sample_method == "log":
            self.vocab_freq_id = dict([(self.tgt_vocab.stoi[key], 
                int(np.log(self.tgt_vocab.freqs[key])+1) ) for key in self.tgt_vocab.freqs if not key in no_sample_list])

        self.vocab_ids = list(self.vocab_freq_id.keys())
        self.freq_cumsum = torch.cumsum(torch.LongTensor(list(self.vocab_freq_id.values())), dim=0)
        self.freq_sum = self.freq_cumsum[-1]

    def _neg_tgt_sampler(self, tgt, tgt_length,verbose=False):
        t_len, batch_size, _ = tgt.size()

        neg_tgt_indicators = torch.randint(0, self.freq_sum, (t_len, batch_size))

        neg_tgt_list = [[1 for i in range(batch_size)] for j in range(t_len)]

        for i in range(batch_size):
            for j in range(t_len):
                if j < tgt_length[i]:
                    neg_tgt_list[j][i] = self.vocab_ids[torch.where(neg_tgt_indicators[j, i] < self.freq_cumsum)[0][0]]
                    # temp = neg_tgt_list[j][i]
                    # print(temp)
                    # print(torch.cuda.LongTensor([temp]))
                    # print(tgt[j][i])
                    # print(neg_tgt_list[j][i])
                    # print(tgt[j][i], temp)
                    while torch.equal(tgt[j][i], torch.cuda.LongTensor([neg_tgt_list[j][i]])):
                        neg_tgt_list[j][i] = self.vocab_ids[torch.where(torch.randint(0, self.freq_sum, (1,)) < self.freq_cumsum)[0][0]]
        
        neg_tgt = torch.cuda.LongTensor(neg_tgt_list).unsqueeze(2)
        if verbose:
            print([(self.tgt_vocab.itos[item], self.vocab_freq_id[int(item)]) for item in neg_tgt.view([-1,])])
        return neg_tgt

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d"
                            % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim[0].training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim[0].training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim[0].training_step
            if step % 20 == 0:
                torch.cuda.empty_cache()
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))
            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)
            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim[0].learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim[0].learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None, normalization=1.0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if moving_average:
            valid_model = deepcopy(self.model)
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                param.data = avg.data.half() if self.optim[0]._fp16 == "legacy" \
                    else avg.data
        else:
            valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()
            i = 0
            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt, tgt_lengths = batch.tgt if isinstance(batch.tgt, tuple) \
                    else (batch.tgt, None)
                if self.pairenc:
                    q_lens_list = batch.q_lens.int()
                    a_lens_list = batch.a_lens.int()
                else:
                    q_lens_list = None
                    a_lens_list = None

                if self.use_knowledge:
                    knowledge, knowledge_length = batch.know
                    # F-prop through the model.
                    outputs, attns = valid_model(src, tgt, src_lengths, 
                                knowledge, knowledge_length,
                                q_lens_list=q_lens_list, a_lens_list=a_lens_list,
                            pairenc=self.pairenc)
                else:
                    outputs, attns = valid_model(src, tgt, src_lengths,
                        q_lens_list=q_lens_list, a_lens_list=a_lens_list,
                            pairenc=self.pairenc
                        )

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        if moving_average:
            del valid_model
        else:
            # Set model back to training mode.
            valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            for op in self.optim:
                op.zero_grad()

        for k, batch in enumerate(true_batches):
            if isinstance(batch.tgt, tuple):
                target_size = batch.tgt[0].size(0)
            else:
                target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            if isinstance(batch.src, tuple):
                if len(batch.src) == 3:
                    src, elmo_src, src_lengths = batch.src
                else:
                    src, src_lengths = batch.src
                    elmo_src = None
            else:
                src = batch.src
                src_lengths = None
                elmo_src = None
            # srcs, src_lengths = batch.src if isinstance(batch.src, tuple) \
            #     else (batch.src, None)
            # src, elmo_src = srcs if isinstance(srcs, tuple) \
            #     else (srcs, None)

            # s_len, batch_size, _ = src.size()


            # print('in trainer')
            # print(src)
            # print(len(src), [len(item) for item in src])
            # print(src_lengths)
            # sys.exit(1)

            # if self.elmo:
            #     # if elmo, prepare input src for elmo
            #     # there should be src_vocab
            #     assert self.src_vocab is not None

            #     # check whether src_vocab is ok

            #     elmo_src = [[self.src_vocab.itos[idx] for idx in src[:, b_s, 0] if idx != self.pad_idx] for b_s in range(batch_size)]
            #     # 只需要去掉pad就行了
            # else:
            #     elmo_src=None

            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer, tgt_lengths_outer = batch.tgt if isinstance(batch.tgt, tuple) \
                else (batch.tgt, None)
            # print('outer_size', tgt_outer.size(), tgt_lengths_outer.size())
            # print(tgt_outer, tgt_lengths_outer - 1)

            if self.use_knowledge:
                knowledge, knowledge_length = batch.know
            if self.pairenc:
                q_lens_list = batch.q_lens.int()
                a_lens_list = batch.a_lens.int()
            else:
                q_lens_list = None
                a_lens_list = None

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]
                # print(tgt)
                # print(tgt.size())
                # sys.exit(1)
                if tgt_lengths_outer is not None:
                    tgt_length = tgt_lengths_outer - 2 # [j: j + trunc_size] # minus start token and end token

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    for op in self.optim:
                        op.zero_grad()

                if self.use_knowledge:
                    outputs, attns = self.model(src, tgt, src_lengths, 
                            knowledge, knowledge_length, bptt=bptt,
                            q_lens_list=q_lens_list, a_lens_list=a_lens_list,
                            pairenc=self.pairenc) # Core. by TQ
                else:
                    outputs, attns = self.model(src, tgt, src_lengths, 
                        bptt=bptt, elmo_src=elmo_src,
                        q_lens_list=q_lens_list, a_lens_list=a_lens_list,
                        pairenc=self.pairenc)
                bptt = True
                # print('in trainer attns.get("context_vecs")', attns.get("context_vecs").size())
                
                # 3. Compute loss.
                try:
                    
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)

                    if loss is not None:
                        for op in self.optim:
                            op.backward(loss)
                    else:
                        # print('loss is None????')
                        pass
                        
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim[0].training_step, k)
                # 4. Update the parameters and statistics.
                start = time.time()
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    for op in self.optim:
                        op.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()
                
                end = time.time()
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for op in self.optim:
                op.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
