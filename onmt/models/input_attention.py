"""Define Input Attention Layer"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import aeq, sequence_mask

class InputAttention(nn.Module):
    def __init__(self, hid_dim, know_dim, know_embeddings):
        super(InputAttention, self).__init__()
        # init\
        # 这里直接用mlp做attention了
        self.hid_dim = hid_dim
        self.know_dim = know_dim
        self.know_embeddings= know_embeddings


        self.linear_hidden = nn.Linear(hid_dim, hid_dim, bias=True) # query (memory banks as queries)
        self.linear_know = nn.Linear(know_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    @classmethod
    def from_opt(cls, opt, know_embeddings):
        """Alternate constructor."""
        return cls(
                opt.rnn_size,
                opt.word_vec_size,
                know_embeddings
            )

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, tgt_dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, src_dim)``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)


        wq = self.linear_hidden(h_t.view(-1, tgt_dim))
        wq = wq.view(tgt_batch, tgt_len, 1, tgt_dim)
        wq = wq.expand(tgt_batch, tgt_len, src_len, tgt_dim)

        uh = self.linear_know(h_s.contiguous().view(-1, src_dim))
        uh = uh.view(src_batch, 1, src_len, tgt_dim)
        uh = uh.expand(src_batch, tgt_len, src_len, tgt_dim)

        # (batch, t_len, s_len, d)
        wquh = torch.tanh(wq + uh)

        return self.v(wquh.view(-1, self.hid_dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, memory_bank, src_lengths, know, know_lengths):
        """
            memory_bank: encoder hidden states, (src_len, batch, hidden_dim)
            src_lengths: (LongTensor): the padded source lengths
                ``(batch,)``.

            know: (know_len, batch, 1)
            know_lengths: (batch, )
        """

        source_l, batch, dim = memory_bank.size()
        know_embs = self.know_embeddings(know)
        know_l, batch_, dim_ = know_embs.size()

        # compute attention scores, as in Luong et al.
        align = self.score(h_t=memory_bank.transpose(0, 1).contiguous(), 
                        h_s=know_embs.transpose(0, 1).contiguous())
        # (batch, s_len, k_len)

        mask = sequence_mask(know_lengths, max_len=align.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        align_vectors = F.softmax(align.view(batch * source_l, know_l), -1)

        align_vectors = align_vectors.view(batch, source_l, know_l)

        # (batch, source_l, know_l) * (batch, know_l, emb_size)
        k_star = torch.bmm(align_vectors, know_embs.transpose(0, 1))

        k_star = k_star.view(batch, source_l, dim_)
        # [batch, s_len, k_dim]

        return k_star, align_vectors



        
        