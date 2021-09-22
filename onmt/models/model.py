""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, input_attention=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_attention = input_attention

    def forward(self, src, tgt, lengths, know=None, know_length=None, bptt=False, 
                elmo_src=None,q_lens_list=None, a_lens_list=None,pairenc=False):
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
            know_length

            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs. Why? by ftq

        if elmo_src is not None:
            enc_state, memory_bank, lengths = self.encoder(elmo_src, lengths)
        elif pairenc:
            enc_state, memory_bank, _ = self.encoder(src, q_lens_list, a_lens_list, lengths)
        else:
            enc_state, memory_bank, lengths = self.encoder(src, lengths)

        input_know_vecs, input_know_attns = None, None
        if self.input_attention is not None:
            input_know_vecs, input_know_attns = self.input_attention(memory_bank, lengths, know, know_length)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        dec_out, attns = self.decoder(tgt, memory_bank, 
                                  know=know, know_lengths=know_length, memory_lengths=lengths, 
                                  input_know_vecs=input_know_vecs, input_know_attns=input_know_attns,)

        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
