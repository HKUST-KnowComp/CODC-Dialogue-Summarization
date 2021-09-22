""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from torch.autograd import Variable


class TextRetModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(TextRetModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.labels = torch.cuda.FloatTensor(1).fill_(1)

    def forward(self, src, tgt, neg_tgt, src_lengths, tgt_length, elmo_src=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
     
        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            know (LongTensor as tgt): A knowledge sequence (words) of size 
                ``(know_len, batch)``.
            know_lengths

            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns: 
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs. Why? by ftq (end token)
        neg_tgt = neg_tgt[:-1]

        if elmo_src is not None:
            enc_state, memory_bank, src_lengths = self.encoder(elmo_src, src_lengths)
        else:
            enc_state, memory_bank, src_lengths = self.encoder(src, src_lengths)

        pos_score, neg_score = self.decoder(enc_state, memory_bank, tgt, neg_tgt, src_lengths, tgt_length)

        labels = Variable(self.labels.expand_as(pos_score)) # labels

        return pos_score, neg_score, labels

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
