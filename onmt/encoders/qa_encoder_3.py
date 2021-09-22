"""Define RNN-based encoders."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.modules.global_attention import GlobalAttention
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules.embeddings import PositionalEncoding
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from torch.nn import LayerNorm
from onmt.encoders.transformer import TransformerEncoderLayer

# from onmt.myconstants import Q_MAX_LEN, A_MAX_LEN

Q_MAX_LEN = 20
A_MAX_LEN = 20

TOTAL_MAX_LEN = Q_MAX_LEN + A_MAX_LEN


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x))
            transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine
            transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
         """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class QAEncoder3(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, attention_dropout=0.0,embeddings=None,
                 use_bridge=False,max_relative_positions=0):
        super(QAEncoder3, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size = hidden_size
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.context_rnn, _ = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size + hidden_size*8,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.dialog_rnn, _ = \
            rnn_factory(rnn_type,
                        input_size=hidden_size*2*4,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.highway = Highway(hidden_size*2, num_layers=2, f=nn.ReLU())
        self.attn = GlobalAttention(hidden_size*2)
        self_linear_hidden_size = 64
        self.self_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, self_linear_hidden_size),
            # nn.BatchNorm1d(self_linear_hidden_size),
            nn.Tanh(),
            nn.Linear(self_linear_hidden_size, 1)
        )
        self.dropout = nn.Dropout(0.1)
        self.pe = PositionalEncoding(0.1, hidden_size*2)
        self.transformer = TransformerEncoderLayer(hidden_size * 2, 8, 2048, 
            0.2,attention_dropout, max_relative_positions)
        #d_model, heads, d_ff, dropout
        #d_model, heads, d_ff, dropout, attention_dropout,
                 # max_relative_positions=0

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type, 
            True, 
            opt.enc_layers,
            opt.rnn_size, 
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.bridge,
            opt.max_relative_positions
            )

    @staticmethod
    def sorted_rnn_encode(inputs, inputs_lens, encoder, last_hidden=None):
        _, inp_len, _ = inputs.size()
        sorted_lens, sorted_idx = inputs_lens.sort(descending=True)
        _, original_idx = sorted_idx.sort(descending=False)
        sorted_inputs = inputs[sorted_idx]
        sorted_packed_embeds = pack(
            sorted_inputs, sorted_lens.cpu().numpy().astype(int), batch_first=True)
        packed_encoder_output, (sorted_hidden, sorted_cell) = encoder(
            sorted_packed_embeds, last_hidden)
        sorted_encoder_output, _ = unpack(packed_encoder_output)
        sorted_encoder_output = sorted_encoder_output.transpose(0, 1).contiguous()
        encoder_outputs = sorted_encoder_output[original_idx, :, :]
        bsize, outp_len, outp_dim = encoder_outputs.size()
        if inp_len != outp_len:
            app_shape = bsize, inp_len - outp_len, outp_dim
            app_tensor = encoder_outputs.data.new(*app_shape).zero_().float()
            app_tensor.data.add_(1e-7)
            encoder_outputs = torch.cat((encoder_outputs, app_tensor), dim=1)
        hidden = sorted_hidden[:, original_idx, :]
        cell = sorted_cell[:, original_idx, :]

        return encoder_outputs, (hidden, cell)

    def local_inference(self, q_encoder_output, a_encoder_output,
                        q_lens, a_lens):
        q_align, q_attns = self.attn(q_encoder_output, a_encoder_output, a_lens)
        q_align = q_align.transpose(0, 1).contiguous()
        a_align, a_attns = self.attn(a_encoder_output, q_encoder_output, q_lens)
        a_align = a_align.transpose(0, 1).contiguous()
        q_align = self.highway(q_align)
        a_align = self.highway(a_align)
        q_combine = torch.cat(
            (q_encoder_output, q_align,
             torch.abs(q_encoder_output - q_align)+ 1e-7,
             q_encoder_output * q_align + 1e-7), dim=2)
        # batch_size x len1 x (hidden_size x 8)
        a_combine = torch.cat(
            (a_encoder_output, a_align,
             torch.abs(a_encoder_output - a_align)+ 1e-7,
             a_encoder_output * a_align + 1e-7), dim=2)
        # batch_size x len2 x (hidden_size x 8)
        return q_combine, a_combine, (q_attns, a_attns)

    def self_fc_encode(self, encoder_output):
        bsize, seq_len, _ = encoder_output.size()
        flatten_encoder_output = encoder_output.view(
            -1, self.hidden_size * 2)
        out = self.self_fc(flatten_encoder_output)
        out = out.view(bsize, seq_len, -1)
        self_attn_weights = F.softmax(out, dim=1).transpose(1, 2)
        encoded = self_attn_weights.bmm(encoder_output)
        return encoded

    def encode_qa_pair(self, q, a, q_lens, a_lens):
        # q_emb = self.embeddings(q.transpose(0, 1).contiguous().unsqueeze(2))
        # a_emb = self.embeddings(a.transpose(0, 1).contiguous().unsqueeze(2))
        q_emb = self.embeddings(q.transpose(0, 1).contiguous())
        a_emb = self.embeddings(a.transpose(0, 1).contiguous())
        q_encoder_output, (q_last_hidden, q_last_cell) = self.sorted_rnn_encode(
            q_emb, q_lens, self.rnn)
        a_encoder_output, (a_last_hidden, a_last_cell) = self.sorted_rnn_encode(
            a_emb, a_lens, self.rnn)
        q_combine_1, a_combine_1, corr_attns = self.local_inference(
            q_encoder_output, a_encoder_output,
            q_lens, a_lens)
        q_encoder_output, (q_last_hidden, q_last_cell) = self.sorted_rnn_encode(
            torch.cat((q_emb, q_combine_1), dim=2), q_lens, self.context_rnn)
        a_encoder_output, (a_last_hidden, a_last_cell) = self.sorted_rnn_encode(
            torch.cat((a_emb, a_combine_1), dim=2), a_lens, self.context_rnn)

        return torch.cat((q_encoder_output, a_encoder_output), dim=1), corr_attns

    def forward(self, src, q_lens_list, a_lens_list, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        # src = batch.src[0] # batch: (src, src_lengths)
        q_list = [src[i*TOTAL_MAX_LEN: (i*TOTAL_MAX_LEN + Q_MAX_LEN)] for i in range(10)]
        a_list = [src[(i*TOTAL_MAX_LEN + Q_MAX_LEN): (i*TOTAL_MAX_LEN + TOTAL_MAX_LEN)] for i in range(10)]
        # q_lens_list = batch.q_lens.int()
        # a_lens_list = batch.a_lens.int()
        mem_list = []
        corr_attns_list = []
        for i in range(10):
            mem, corr_attns = self.encode_qa_pair(
                q_list[i], a_list[i],
                q_lens_list[:, i], a_lens_list[:, i])
            mem_list.append(mem)
            corr_attns_list.append(corr_attns)
        memory_bank = torch.cat(mem_list, dim=1)
        memory_bank = self.pe(memory_bank)
        memory_bank = self.transformer(memory_bank, mask=None)
        memory_bank = memory_bank.transpose(0, 1).contiguous()

        return None, memory_bank, None

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
