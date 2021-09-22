import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention, KnowledgeAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq, sequence_mask

class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings, know_emb=None):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError


class BinClassDecoder(DecoderBase):
    """
        Used to decode text retrieval problem
    """

    def __init__(self, 
                attn_type,
                enc_dim,
                word_dim,
                embeddings,
                bidirectional_encoder,
                text_attn_type,
                score_output_function,
                elmo
                ):
        super(BinClassDecoder, self).__init__()
        self.input_dim = enc_dim
        self.enc_dim = enc_dim
        self.word_dim = word_dim # dimension of 
        self.context_dim = enc_dim # dimension of encoder context vector

        self.attn_type = attn_type
        self.embeddings = embeddings

        self.bidirectional_encoder = bidirectional_encoder

        self.text_attn_type = text_attn_type
        self.score_output_function = score_output_function
        self.elmo = elmo
        
        # self.attend_vec = Variable(torch.FloatTensor(self.input_dim).fill_(0.0)).cuda()
        if self.text_attn_type == 'one':
            self.attend_vec = torch.nn.Parameter(torch.FloatTensor(self.word_dim), requires_grad=True)

        # for attention
        if self.attn_type == "general":
            self.linear_in = nn.Linear(self.input_dim, self.input_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(self.input_dim, self.input_dim, bias=False)
            self.linear_query = nn.Linear(self.word_dim, self.input_dim, bias=True)
            self.v = nn.Linear(self.input_dim, 1, bias=False)

        # for ranking 
        if not elmo:
            if self.bidirectional_encoder:
                self.linear_enc_out = nn.Linear(self.enc_dim * 2, self.word_dim, bias=True)
            else:
                self.linear_enc_out = nn.Linear(self.enc_dim, self.word_dim, bias=True)
        self.linear_enc_context = nn.Linear(self.context_dim, self.word_dim, bias=True)
        self.linear_candi_word = nn.Linear(self.word_dim, self.word_dim, bias=True)
        self.v_rank = nn.Linear(self.word_dim, 1, bias=True)

    @classmethod
    def from_opt(cls, opt, embeddings, know_emb=None):
        """Alternate constructor."""
        return cls(
                attn_type=opt.global_attention,
                enc_dim=opt.enc_rnn_size,
                word_dim=opt.word_vec_size,
                embeddings=embeddings,
                bidirectional_encoder=opt.brnn,
                text_attn_type=opt.text_ret_decoder_attn,
                score_output_function=opt.score_output_function,
                elmo=opt.encoder_type=='elmo',
            )

    # def init_state(self, src, memory_bank, encoder_final):
    #     """Initialize decoder state with last state of the encoder."""
    #     def _fix_enc_hidden(hidden):
    #         # The encoder hidden is  (layers*directions) x batch x dim.
    #         # We need to convert it to layers x batch x (directions*dim).
    #         if self.bidirectional_encoder:
    #             hidden = torch.cat([hidden[0:hidden.size(0):2],
    #                                 hidden[1:hidden.size(0):2]], 2)
    #         return hidden

    #     if isinstance(encoder_final, tuple):  # LSTM
    #         self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
    #                                      for enc_hid in encoder_final)
    #     else:  # GRU
    #         self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

    #     # Init the input feed. 
    #     batch_size = self.state["hidden"][0].size(1)
    #     h_size = (batch_size, self.hidden_size)
    #     self.state["input_feed"] = \
    #         self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
    #     self.state["coverage"] = None

    def attend_hidden_states(self, attend_vec, memory_bank, memory_lengths):
        """
            attend_vec: [input_dim]
            memory_bank: [batch, src_len, hidden_dim]
            memory_lengths: [batch, ]
        """
        # print('memory_bank size', memory_bank.size())
        attend_dim = attend_vec.size()[0]
        src_len, batch_size, hidden_dim = memory_bank.size()

        aeq(attend_dim, hidden_dim) 

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = attend_vec.view(1, attend_dim)
                h_t_ = self.linear_in(h_t_) # [1, attend_dim,]
                h_t = h_t_.view(1, 1, attend_dim).expand(batch_size, 1, attend_dim)
            h_s_ = memory_bank.transpose(1, 2)
            # (batch, 1, d) x (batch, d, s_len) --> (batch, 1, s_len)
            align = torch.bmm(h_t, h_s_)
        else:
            dim = self.input_dim
            wq = self.linear_query(attend_vec.view(1, attend_dim)) # [1, input_dim(==hidden_dim)]
            wq = wq.view(1, 1, 1, attend_dim) 
            wq = wq.expand(batch_size, 1, src_len, attend_dim) # [batch_size, 1, src_len, hidden_dim]

            uh = self.linear_context(memory_bank.contiguous().view(-1, hidden_dim)) # [src_len * batch_size, input_dim(==hidden_dim)]
            uh = uh.view(batch_size, 1, src_len, hidden_dim)
            # uh = uh.expand(batch_size, 1, src_len, hidden_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            align = self.v(wquh.view(-1, hidden_dim)).view(batch_size, 1, src_len)

        # compute attention scores, as in Luong et al.
        # align = self.score(source, memory_bank)

        mask = sequence_mask(memory_lengths, max_len=align.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        align.masked_fill_(~mask, -float('inf'))

        align_vectors = F.softmax(align.view(batch_size, src_len), -1)
        align_vectors = align_vectors.view(batch_size, 1, src_len)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank.contiguous().transpose(0, 1))

        c = c.view(batch_size, hidden_dim)

        return c # [batch_size, hidden_dim]

    def attention(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim)``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        # aeq(src_dim, tgt_dim)
        dim = src_dim

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            wq = self.linear_query(h_t.contiguous().view(-1, tgt_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, src_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, src_dim)

            uh = self.linear_context(h_s.contiguous().view(-1, src_dim))
            uh = uh.view(src_batch, 1, src_len, src_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, src_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def rank_score(self, enc_hidden, memory_bank, memory_lengths, tgt, tgt_length):

        """
            enc_hidden, memory_bank, memory_lengths, tgt_emb, tgt_length
            enc_hidden: encoder output state, [1, batch_size, enc_hidden_dim]
            h_context: context vector of encoder, [batch_size, hidden_dim]
            tgt: candidate tgt, [batch_size, t_len, emb_dim]
            tge_length: [batch_size, ]
        """
        s_len, _, hidden_dim = memory_bank.size()
        
        t_len, batch_size, word_dim = tgt.size()
        # print('rank_score::', tgt.size())
    

        if enc_hidden is not None:
            enc_hid = self.linear_enc_out(
                enc_hidden.contiguous().squeeze(0)
                ).view(batch_size, 1, word_dim).expand([batch_size, t_len, word_dim])

        word_hid = self.linear_candi_word(tgt.contiguous().transpose(0, 1)) # .view(batch_size, t_len, word_dim)


        if self.text_attn_type == 'None':
            assert not self.elmo 
            out_vec = torch.tanh(enc_hid.contiguous().view(-1, word_dim) 
                           + word_hid.contiguous().view(-1, word_dim))
        elif self.text_attn_type == 'one':
            align = self.attention(
                self.attend_vec.view(1, 1, self.attend_vec.size()[0]).expand(batch_size, 1, self.attend_vec.size()[0]),
                memory_bank.transpose(0, 1))
            if memory_lengths is not None:
                mask = sequence_mask(memory_lengths, max_len=align.size(-1))
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                align.masked_fill_(~mask, -float('inf'))
            align_vectors = F.softmax(align.view(batch_size*1, s_len), -1)  
            align_vectors = align_vectors.view(batch_size, 1, s_len)
            c = torch.bmm(align_vectors, memory_bank.transpose(0, 1))
            c = c.view(batch_size, 1, hidden_dim)  
            cont_hid = self.linear_enc_context(
                    c.view(batch_size * 1, hidden_dim)
                ).contiguous().view(batch_size, 1, word_dim).expand([batch_size, t_len, word_dim])

            if enc_hidden is not True:
                out_vec = torch.tanh(enc_hid.contiguous().view(-1, word_dim) 
                               + cont_hid.contiguous().view(-1, word_dim) 
                               + word_hid.contiguous().view(-1, word_dim))
            else:
                out_vec = torch.tanh(cont_hid.contiguous().view(-1, word_dim) 
                               + word_hid.contiguous().view(-1, word_dim))
        elif self.text_attn_type == 'global':
            # use every word to attend to encoder memories
            align = self.attention( tgt.transpose(0, 1), memory_bank.transpose(0, 1),) # (tgt_batch, tgt_len, src_len)
            if memory_lengths is not None:
                mask = sequence_mask(memory_lengths, max_len=align.size(-1))
                # print('attention::mask', mask.size())
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                align.masked_fill_(~mask, -float('inf'))
            align_vectors = F.softmax(align.view(batch_size*t_len, s_len), -1)
            align_vectors = align_vectors.view(batch_size, t_len, s_len)
            c = torch.bmm(align_vectors, memory_bank.transpose(0, 1))
            c = c.view(batch_size, t_len, hidden_dim)
            cont_hid = self.linear_enc_context(
                     c.view(batch_size * t_len, hidden_dim)
                ).contiguous().view(batch_size, t_len, word_dim)
            if enc_hidden is not None:
                out_vec = torch.tanh(enc_hid.contiguous().view(-1, word_dim) 
                               + cont_hid.contiguous().view(-1, word_dim) 
                               + word_hid.contiguous().view(-1, word_dim))
            else:
                out_vec = torch.tanh(cont_hid.contiguous().view(-1, word_dim) 
                               + word_hid.contiguous().view(-1, word_dim))

        scores = self.v_rank(out_vec).view(batch_size, t_len, 1)
        # batch_size, t_len, 1

        if self.score_output_function == 'sigmoid':
            scores = torch.sigmoid(scores)
        elif self.score_output_function == 'tanh':
            scores = torch.tanh(scores)

        # mask
        # mask = sequence_mask(tgt_length, max_len=t_len)
        # print('rank_score::mask', mask.size())
        # mask = mask.unsqueeze(2)  # Make it broadcastable.
        # scores.masked_fill_(~mask, 0.0) # for neg and pos the scores are both 0.

        return scores # [batch_size, t_len, 1]



    def forward(self, enc_state, memory_bank, tgt, memory_lengths, tgt_length):
        """
        Args:
            enc_state ()
            tgt (LongTensor): sequences of padded tokens ``(tgt_len, batch, emb_dim)``
            tgt_length (LongTensor) : padded length
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden_dim)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.

        """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if enc_state is not None:

            if isinstance(enc_state, tuple):  # LSTM
                enc_hidden = torch.cat([_fix_enc_hidden(enc_hid)
                                             for enc_hid in enc_state], 2)
            else:  # GRU
                enc_hidden = _fix_enc_hidden(enc_state)
        else:
            # elmo should be true when there's no encoder output
            assert self.elmo 
            enc_hidden = None

        # the label of the first element of tgt is 1, the second is 0

        tgt_emb = self.embeddings(tgt)
        score = self.rank_score(enc_hidden, memory_bank, memory_lengths, tgt_emb, tgt_length)

        return score

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)
