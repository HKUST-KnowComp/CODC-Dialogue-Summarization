""" Onmt NMT Model base class definition """
import torch.nn as nn


class BowGenModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(BowGenModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths):
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
        # tgt = tgt[:-1]  # exclude last target from inputs. Why? by ftq
        # exclude the first and the last

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        self.decoder.init_state(src, memory_bank, enc_state)
        # print('in model, tgt[0,:].size():', tgt[0, :, :])
        dec_out, _ = self.decoder(tgt[0, :, :].unsqueeze(0), memory_bank, memory_lengths=lengths)
        
        return dec_out

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
