from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch


import onmt.model_builder
import onmt.inputters as inputters
from onmt.utils.misc import tile, set_random_seed

class TextRetTranslator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        ret_num (int) : Number of retrieved words
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            ret_num=10,
            out_file=None,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)
        self.tgt = torch.LongTensor(list(range(4, self._tgt_vocab_len)))
        self.tgt_length = torch.FloatTensor(1).fill_(self._tgt_vocab_len-4)

        self.ret_num = ret_num

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        if self._use_cuda:
            self.tgt = self.tgt.cuda(self._gpu)
            self.tgt_length = self.tgt_length.cuda(self._gpu)

        self.src_reader = src_reader
        self.tgt_reader = tgt_reader

        self.out_file = out_file
        self.logger = logger

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            out_file=None,
            logger=None):

        src_reader = inputters.str2reader["text"].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            ret_num=opt.ret_num,
            out_file=out_file,
            logger=logger,
            seed=opt.seed)

    def translate(
            self,
            src,
            tgt=None,
            src_dir=None,
            batch_size=None,
            batch_type="sents"):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")
        feed_reader = [self.src_reader]
        feed_data = [("src", src)]
        feed_dirs = [src_dir]
        if tgt:
            feed_reader.append(self.tgt_reader)
            feed_data.append(("tgt", tgt))
            feed_dirs.append(None)

        data = inputters.Dataset(
            self.fields,
            readers=feed_reader,
            data=feed_data,
            dirs=feed_dirs,
            sort_key=inputters.str2sortkey["text"],
            filter_pred=None,#self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        # xlation_builder = onmt.translate.TranslationBuilder(
        #     data, self.fields, self.n_best, self.replace_unk, tgt,
        #     self.phrase_table
        # )

        # Statistics
        # counter = count(1)
        # pred_score_total, pred_words_total = 0, 0
        # gold_score_total, gold_words_total = 0, 0

        # all_scores = []
        # all_predictions = []

        start_time = time.time()

        # TODO by tianqing. add tgt_vocab
        for batch in data_iter:
            batch_data = self.translate_batch(
                batch, data.src_vocabs
            )

            # print('batch_data', batch_data)

        end_time = time.time()

        return ;

    def translate_batch(self, batch, src_vocabs):
        """Translate a batch of sentences."""
        with torch.no_grad():
            return self._translate_batch(
                batch,
                src_vocabs)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _run_decoder(self, src, enc_states, memory_bank, src_lengths):

        # target is all the words in the tgt_vocabs
        _, batch_size, _ = memory_bank.size()
        tgt_size = self.tgt.size()[0]

        tgt = self.tgt.unsqueeze(0).expand(batch_size, tgt_size).transpose(0, 1).unsqueeze(2)
        tgt_length = self.tgt_length.expand(batch_size)

        # print(tgt.size())
        scores, _ = self.model.decoder(enc_states, memory_bank, tgt, None, src_lengths, tgt_length)

        return scores

    def _translate_batch(
            self,
            batch,
            src_vocabs,):

        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # get scores of all words in tgt_vocab

        scores_tgt_vocab = self._run_decoder(src, enc_states, memory_bank, src_lengths)
        # the returned stuff are words
        retrieved = torch.argsort(scores_tgt_vocab, dim=1, descending=True)[:, :self.ret_num]
        
        words = [[self._tgt_vocab.itos[i] for i in retrieved[b, :]] for b in range(batch_size)]
        print(words)
        
        return words