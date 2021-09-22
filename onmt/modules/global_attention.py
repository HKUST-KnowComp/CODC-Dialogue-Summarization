"""Global attention modules (Luong / Bahdanau)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        if isinstance(dim, list):
            self.src_dim = dim[0]
            self.tgt_dim = dim[1]
        else:
            self.src_dim = dim
            self.tgt_dim = dim
            self.dim = dim


        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(self.tgt_dim, self.tgt_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(self.src_dim, self.tgt_dim, bias=False)
            self.linear_query = nn.Linear(self.tgt_dim, self.tgt_dim, bias=True)
            self.v = nn.Linear(self.tgt_dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(self.tgt_dim + self.src_dim, self.tgt_dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, self.tgt_dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        # aeq(src_dim, tgt_dim)
        # aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            # dim = self.dim
            wq = self.linear_query(h_t.view(-1, self.tgt_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, self.tgt_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, self.tgt_dim)

            uh = self.linear_context(h_s.contiguous().view(-1, self.src_dim))
            uh = uh.view(src_batch, 1, src_len, self.tgt_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, self.tgt_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, self.tgt_dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None, know=False):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_lengths (LongTensor): the source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        # aeq(dim, dim_)
        # aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        c = c.view(batch, target_l, self.src_dim)

        if know:
            if one_step:
                c = c.squeeze(1)
                align_vectors = align_vectors.squeeze(1)
            else:
                c = c.transpose(0, 1).contiguous()
                align_vectors = align_vectors.transpose(0, 1).contiguous()
            

            return c, align_vectors

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, self.tgt_dim + self.src_dim)
        attn_h = self.linear_out(concat_c).view(batch, target_l, self.tgt_dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            # aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            # aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors

class KnowledgeAttention(nn.Module):
    r"""
    Knowledge attention takes a matrix (know matrix) and
    a query vector. It then computes a parameterized convex
    combination of the matrix based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dims (list): dimensionality of [context, query, knowledge(key)]
       know_attn_type (str): type of attention to use, options [dot,general,mlp]
       know_attn_func (str): attention function to use, options [softmax,sparsemax]

    """
    def __init__(self, dims, attn_type="dot",
                 attn_func="softmax"):
        super(KnowledgeAttention, self).__init__()

        self.src_dim, self.tgt_dim, self.know_dim = dims

        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(self.tgt_dim, self.know_dim, bias=False)
        elif self.attn_type == "mlp":
            # hidden dim of MLP equals self.tgt_dim (Should be able to tune. by ftq)
            self.linear_know = nn.Linear(self.know_dim, self.tgt_dim, bias=False)
            self.linear_query = nn.Linear(self.tgt_dim, self.tgt_dim, bias=True)
            self.v = nn.Linear(self.tgt_dim, 1, bias=False)

        # know_mlp
        # self.know_f1 = nn.Linear(self.know_dim, 2*self.know_dim, bias=True)
        # self.know_f2 = nn.Linear(2*self.know_dim, self.know_dim, bias=True)

        # mlp wants it with bias

        # equations:
        # beta_is = MLP(k_i, h_s)
        # attended know_vec = \sum softmax(exp(beta_is))*k_i
        # $P_{vocab} = softmax(V'(V[s_t, h_t^*, o_t]+b)+b')$

        out_bias = self.attn_type == "mlp" 
        self.linear_out = nn.Linear(self.src_dim + self.tgt_dim + self.know_dim, self.tgt_dim, bias=out_bias)

    def score(self, h_t, k_i):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          k_i (FloatTensor): sequence of sources ``(batch, know_len, dim)``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, know_len)``
        """

        # Check input sizes
        know_batch, know_len, know_dim = k_i.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(know_batch, tgt_batch)
        aeq(know_dim, self.know_dim)
        aeq(tgt_dim, self.tgt_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, self.know_dim)
                # [tgt_batch, tgt_len, self.know_dim]
            k_i_ = k_i.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, k_i_)
        else:
            # 这里用的是一层MLP，得唔得？by ftq
            # dim = self.dim
            wq = self.linear_query(h_t.view(-1, self.tgt_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, self.tgt_dim)
            wq = wq.expand(tgt_batch, tgt_len, know_len, self.tgt_dim)

            uh = self.linear_know(k_i.contiguous().view(-1, self.know_dim))
            uh = uh.view(know_batch, 1, know_len, self.tgt_dim)
            uh = uh.expand(know_batch, tgt_len, know_len, self.tgt_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, self.tgt_dim)).view(tgt_batch, tgt_len, know_len)

    def forward(self, target, know_bank, context_vecs=None, know_lengths=None):
        """

        Args:
          target (FloatTensor): query vectors ``(batch, tgt_len, tgt_dim)`` {x_t}
          know_bank (FloatTensor): know vectors ``(batch, know_len, know_dim)`` {k_i}
          context_vecs : context vector of the input (batch, tgt_len, src_dim). {h_t^*}

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, know_dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, know_len)``
        """

        # one step input
        if target.dim() == 2:
            one_step = True
            target = target.unsqueeze(1)
        else:
            one_step = False

        batch, know_l, know_dim = know_bank.size()
        batch_, target_l, t_dim = target.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.

        # out1 = torch.relu(self.know_f1(
        #             know_bank.contiguous().view(batch*know_l, know_dim)
        #             ))

        # know_transformed = torch.tanh(
        #     self.know_f2(
        #             out1
        #         )
        #     ).view(batch, know_l, know_dim)

        align = self.score(target, know_bank) # (batch, tgt_len, know_len)
        # align = self.score(target, know_transformed) # (batch, tgt_len, know_len)

        if know_lengths is not None:
            mask = sequence_mask(know_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, know_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, know_l), -1)
        align_vectors = align_vectors.view(batch, target_l, know_l) # weights
        align_vectors = torch.where(
            torch.isnan(align_vectors), torch.full_like(align_vectors, 0), align_vectors
        )
        # check if a row is nan，if nan, set all to zeros
        # print('alignvectors', align_vectors)
        # each context vector c_t is the weighted average
        # over all the target hidden states

        

        c = torch.bmm(align_vectors, know_bank) # (batch, target_l, know_dim) # vector of attended knowledge
        # c = torch.bmm(align_vectors, know_transformed)
        # print('c', c.sum())
        # target : (batch, tgt_len, tgt_dim )
        # context_vecs : (batch, tgt_len, src_dim)

        if context_vecs is None:
            # return c directly. in transformer mode
            if one_step:
                c = c.squeeze(1)
                align_vectors = align_vectors.squeeze(1)
            else:
                c = c.transpose(0, 1).contiguous()
                align_vectors = align_vectors.transpose(0, 1).contiguous()
            return None, align_vectors, c

        if one_step:
            context_vecs = context_vecs.unsqueeze(1)
        else:
            context_vecs = context_vecs.transpose(0, 1).contiguous()
        # concatenate
        concat_c = torch.cat([c, target, context_vecs], 2).view(batch*target_l, 
            self.src_dim + self.tgt_dim + self.know_dim)
        # print('concat_c', concat_c.sum())
        attn_k = self.linear_out(concat_c).view(batch, target_l, self.tgt_dim)

        if self.attn_type in ["general", "dot"]:
            attn_k = torch.tanh(attn_k)

        # Why doesn't MLP need a tanh? and why is this called attn_k? I think it should be called P_vocab

        if one_step:
            attn_k = attn_k.squeeze(1)
            c = c.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, t_dim = attn_k.size()
            aeq(batch, batch_)
            batch_, know_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(know_l, know_l_)

        else:
            c = c.transpose(0, 1).contiguous()
            attn_k = attn_k.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, t_dim = attn_k.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            target_l_, batch_, know_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
        # print('global_attention::align_vectors', align_vectors.size())
        return attn_k, align_vectors, c

