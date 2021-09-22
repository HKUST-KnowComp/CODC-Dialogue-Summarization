"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F


from onmt.encoders.encoder import EncoderBase
from allennlp.modules.elmo import Elmo, batch_to_ids


class ELMoEncoder(EncoderBase):
    """ 
    """

    def __init__(self, dropout=0.0, finetune=False):
        super(ELMoEncoder, self).__init__()

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0, requires_grad=finetune)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.elmo_dropout,
            opt.finetune)

    def forward(self, src, lengths=None, dev=None):
        """See :func:`EncoderBase.forward()`"""
        # src should be tokens of words (list of strings)

        # b = batch_to_ids(src).cuda() if dev is None else batch_to_ids(src).cuda(dev)

        # print('in elmo_encoder, batch', b)

        # elmo_input = torch.cuda.LongTensor(b)

        # print('in elmo_encoder, cuda batch', elmo_input)

        dev = 0 if dev is None else dev
        # if dev is None:
            # embeddings = self.elmo(src.cuda())
        # else:
        embeddings = self.elmo(src.cuda(dev))

        memory_bank = embeddings['elmo_representations'][0]

        encoder_final = None

        return encoder_final, memory_bank.contiguous().transpose(0, 1), lengths
