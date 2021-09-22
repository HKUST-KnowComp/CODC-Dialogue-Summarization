#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab,\
                                    _load_vocab

from functools import partial
from multiprocessing import Pool


def check_existing_pt_files(opt, corpus_type, ids, existing_fields):
    """ Check if there are existing .pt files to avoid overwriting them """
    existing_shards = []
    for maybe_id in ids:
        if maybe_id:
            shard_base = corpus_type + "_" + maybe_id
        else:
            shard_base = corpus_type
        pattern = opt.save_data + '.{}.*.pt'.format(shard_base)
        if glob.glob(pattern):
            if opt.overwrite:
                maybe_overwrite = ("will be overwritten because "
                                   "`-overwrite` option is set.")
            else:
                maybe_overwrite = ("won't be overwritten, pass the "
                                   "`-overwrite` option if you want to.")
            logger.warning("Shards for corpus {} already exist, {}"
                           .format(shard_base, maybe_overwrite))
            existing_shards += [maybe_id]
    return existing_shards


def process_one_shard(corpus_params, params):
    corpus_type, fields, src_reader, tgt_reader, know_reader, opt, existing_fields,\
        src_vocab, tgt_vocab, know_vocab = corpus_params
    i, (src_shard, tgt_shard, know_shard, maybe_id, filter_pred) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(tgt_shard) # and len(src_shard) == len(know_shard)
    if know_shard is not None:
        assert len(src_shard) == len(know_shard)
    logger.info("Building shard %d." % i)
    dataset = inputters.Dataset(
        fields,
        readers=([src_reader, tgt_reader, know_reader]
                 if tgt_reader else [src_reader]),
        data=([("src", src_shard), ("tgt", tgt_shard), ("know", know_shard)]
              if know_reader else [("src", src_shard), ("tgt", tgt_shard)]),
        dirs=([opt.src_dir, None, None]
              if tgt_reader else [opt.src_dir]),
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred,
        pairenc=opt.pairenc
    )
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            for name, field in fields.items():
                if ((opt.data_type == "audio") and (name == "src")):
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and
                                 src_vocab is not None) or \
                                (sub_n == 'tgt' and
                                 tgt_vocab is not None) or \
                                (sub_n == 'know' and
                                 know_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".\
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def maybe_load_vocab(corpus_type, counters, opt):
    """
        Check if there're existing vocabs, if it does, load the prev vocab.
        if not, do nothing
    """
    src_vocab = None
    tgt_vocab = None
    know_vocab = None
    existing_fields = None
    if corpus_type == "train":
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = _load_vocab(
                opt.tgt_vocab, "tgt", counters,
                opt.tgt_words_min_frequency)
        if opt.know_vocab != "":
            know_vocab, know_vocab_size = _load_vocab(
                opt.know_vocab, "know", counters,
                opt.know_words_min_frequency)
    return src_vocab, tgt_vocab, know_vocab, existing_fields


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, know_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        tgts = opt.train_tgt
        if know_reader is not None:
            knows = opt.train_know
        else:
            knows = [None]
        ids = opt.train_ids
    elif corpus_type == 'valid':
        counters = None
        srcs = [opt.valid_src]
        tgts = [opt.valid_tgt]
        if know_reader is not None:
            knows = [opt.valid_know]
        else:
            knows = [None]
        ids = [None]

    # load prev-processed vocab
    src_vocab, tgt_vocab, know_vocab, existing_fields = maybe_load_vocab(
        corpus_type, counters, opt)

    # load prev-processed shards
    existing_shards = check_existing_pt_files(
        opt, corpus_type, ids, existing_fields)

    # every corpus has shards, no new one
    if existing_shards == ids and not opt.overwrite:
        return

    def shard_iterator(srcs, tgts, knows, ids, existing_shards,
                       existing_fields, corpus_type, opt):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, tgt, know, maybe_id in zip(srcs, tgts, knows, ids):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}"
                                   .format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None,\
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if ((corpus_type == "train" or opt.filter_valid)
                    and tgt is not None):
                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length,
                    max_know_len=opt.know_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)
            if know is not None:
                know_shards = split_corpus(know, opt.shard_size)
                for i, (ss, ts, ks) in enumerate(zip(src_shards, tgt_shards, know_shards)):
                    yield (i, (ss, ts, ks, maybe_id, filter_pred))
            else:
                know_shards = None
                for i, (ss, ts) in enumerate(zip(src_shards, tgt_shards)):
                    yield (i, (ss, ts, None, maybe_id, filter_pred))
            
            # What does filter_pred do? by ftq

    shard_iter = shard_iterator(srcs, tgts, knows, ids, existing_shards,
                                existing_fields, corpus_type, opt)

    with Pool(opt.num_threads) as p:
        dataset_params = (corpus_type, fields, src_reader, tgt_reader, know_reader,
                          opt, existing_fields, src_vocab, tgt_vocab, know_vocab)
        func = partial(process_one_shard, dataset_params)
        # process_one_shard(dataset_params, shard_iter)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.tgt_vocab_size, opt.tgt_words_min_frequency,
                opt.know_vocab_size, opt.know_words_min_frequency,
                opt.knowledge)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)

    init_logger(opt.log_file)

    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0
    for src, tgt in zip(opt.train_src, opt.train_tgt):
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
        tgt_nfeats += count_features(tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    if opt.knowledge:
        know_nfeats = 0
        for know in opt.train_know:
            know_nfeats += count_features(know)
        logger.info(" * number of knowledge features: %d." % know_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        n_know_feats=know_nfeats if opt.knowledge else 0,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc,
        know_truncate=opt.know_seq_length_trunc,
        knowledge=opt.knowledge,
        elmo=opt.elmo,
        pairenc=opt.pairenc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    if opt.knowledge: # for now knowledge are texts
        know_reader = inputters.str2reader["text"].from_opt(opt)
    else:
        know_reader = None

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, know_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, know_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
