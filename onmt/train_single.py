#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim, build_optim_elmo, build_optim_dec, Optimizer_ori
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    opt.brnn = (opt.encoder_type == "brnn" or
                opt.encoder_type == "qa0" or
                opt.encoder_type == "qa1" or
                opt.encoder_type == "qa2" or
                opt.encoder_type == "qa3")
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    elif opt.know_train_from:
        logger.info('Knowledge train_from mode')
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.know_train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        
        # add knowledge opts: 

        setattr(model_opt, "know_train_from", opt.know_train_from)
        setattr(model_opt, "know_train_from_type", opt.know_train_from_type)
        setattr(model_opt, "knowledge", opt.knowledge)
        setattr(model_opt, "know_attn_type",opt.know_attn_type)
        setattr(model_opt, "input_attn", opt.input_attn)
        setattr(model_opt, "know_star_attn_share",opt.know_star_attn_share)
        setattr(model_opt, "know_loss_lambda",opt.know_loss_lambda)
        setattr(model_opt, "p_kgen_loss",opt.p_kgen_loss)
        setattr(model_opt, "p_kgen_func",opt.p_kgen_func)
        setattr(model_opt, "prob_logits_type",opt.prob_logits_type)
        setattr(model_opt, "trans_know_query_type",opt.trans_know_query_type)
        setattr(model_opt, "copy_attn",opt.copy_attn)
        setattr(model_opt, "reuse_copy_attn",opt.reuse_copy_attn)
        setattr(model_opt, "train_steps",opt.train_steps)
        
        # forbit copy_attn for now
        # model_opt.copy_attn = False
        # model_opt.reuse_copy_attn = False

        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        # vocab = checkpoint['vocab']
        vocab = torch.load(opt.data + '.vocab.pt')
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
                if sn == 'tgt':
                    tgt_vocab = sf.vocab
                if sn == 'src':
                    src_vocab = sf.vocab
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    if opt.encoder_type == 'elmo' and opt.optim_seperate:
        optim = [build_optim_elmo(model, opt, checkpoint),
                build_optim_dec(model, opt, checkpoint)]
    else:
        # optim = [build_optim(model, opt, checkpoint)]
        optim = [Optimizer_ori.from_opt(model, opt, checkpoint=checkpoint)]

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver, 
        tgt_vocab=tgt_vocab, vocab_sample_method=opt.vocab_sample_method,
        src_vocab=src_vocab)

    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            # logger.info('calculating all train_iter')
            train_iter = build_dataset_iter(shard_base, fields, opt)
            # logger.info('finished calculating all train_iter')


    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
