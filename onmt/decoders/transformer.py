"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from onmt.modules import KnowledgeAttention


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, know_query_type=None, 
                 know_dim=0, know_attn_type=None, know_attn_func=None):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.know_query_type = know_query_type
        if self.know_query_type == "iter_query":
            self.feed_forward_linear = nn.Linear(d_model + know_dim, d_model, bias=True)
            self.k_attn = KnowledgeAttention(
                [0, d_model, know_dim], 
                attn_type=know_attn_type, attn_func=know_attn_func,
            )
            


    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, know_emb, know_lengths,
                layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``
            know_emb (FloatTensor): ``(k_len, batch_size, know_dim)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        # ftq
        # 这里用query attend to knowledge就好了！看看需不需要加add&norm？ 用最后一层的query来query 
        ori_query = self.layer_norm_2(self.drop(query) + inputs)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")

        attns = {}
        attns["std"] = attn
        
        # knowledge attention:
        if self.know_query_type == "iter_query":
            _, k_attns, k_star_contexts = self.k_attn(
                    ori_query.contiguous(),
                    know_emb.transpose(0, 1),
                    know_lengths=know_lengths,)

            
            attns["knowledge"] = k_attns

            k_star_contexts = k_star_contexts.transpose(0, 1).contiguous()

            output_logits = self.feed_forward_linear(torch.cat([k_star_contexts, mid], 2 ))

            output = self.feed_forward( self.drop(output_logits) + query)

            attns["context_vecs"] = torch.cat([k_star_contexts, output], 2)
            
        else:
            output = self.feed_forward(self.drop(mid) + query)

        return output, attns, ori_query

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,encoder_hidden_dim,
                 know_attn_type, know_attn_func, logits_type, 
                 trans_know_query_type, know_emb,):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings
        self.d_model = d_model
        self.encoder_hidden_dim = encoder_hidden_dim
        if self.d_model != self.encoder_hidden_dim:
          self.encoder_linear=nn.Linear(self.encoder_hidden_dim, self.d_model, bias=True)
        # Decoder State
        self.state = {}

        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        know_dim = 0
        self._know_attn = False
        self.trans_know_query_type = trans_know_query_type

        if know_emb is not None:
            self.know_emb = know_emb
            know_dim = know_emb.word_vec_size #TODO
            self._know_attn = True
            self.logits_type = logits_type
            if trans_know_query_type == "query_once":
                self.k_attn = KnowledgeAttention(
                            [0, d_model, know_dim], 
                            attn_type=know_attn_type, attn_func=know_attn_func,
                        )
                if self.logits_type == "include_k_context":
                    self.linear_out = nn.Linear(d_model+know_dim, d_model, bias=True)

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             know_query_type=trans_know_query_type, know_dim=know_dim,
             know_attn_type=know_attn_type, know_attn_func=know_attn_func)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        
        

    @classmethod
    def from_opt(cls, opt, embeddings, know_emb=None):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.enc_rnn_size,
            opt.know_attn_type,
            opt.know_attention_function,
            opt.prob_logits_type,
            opt.trans_know_query_type,
            know_emb)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        batch_size, src_len, _ = memory_bank.size()
        if self.d_model != self.encoder_hidden_dim:
          memory_bank = self.encoder_linear(memory_bank.contiguous().view(-1, self.encoder_hidden_dim)).contiguous().view(batch_size, src_len, self.d_model)

        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        know_emb = None
        know_lengths = None
        if self._know_attn:
            know = kwargs["know"]
            know_emb = self.know_emb(know)
            know_lengths = kwargs["know_lengths"]

        assert emb.dim() == 3  # len x batch x embedding_dim 

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attns, ori_query = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                know_emb, # remember to transpose(0, 1)
                know_lengths,
                layer_cache=layer_cache,
                step=step)

        output = self.layer_norm(output)
        trans_dec_outs = output.transpose(0, 1).contiguous()

        attns["std"] = attns["std"].transpose(0, 1).contiguous()

        if self._copy:
            attns["copy"] = attns["std"]
        if "context_vecs" in attns:
            attns["context_vecs"] = attns["context_vecs"].transpose(0, 1).contiguous()

        dec_outs = trans_dec_outs

        if self._know_attn and self.trans_know_query_type == "query_once":
            # use the query of the upper layer of transformer to attend to knowledge
            # 计算knowledge attention
            _, k_attns, k_star_contexts = self.k_attn(
                    ori_query.contiguous(),
                    know_emb.transpose(0, 1),
                    know_lengths=know_lengths,) 
            attns["knowledge"] = k_attns
            attns["context_vecs"] = torch.cat([k_star_contexts, trans_dec_outs], 2)
            if self.logits_type == 'include_k_context':
                dec_outs = self.linear_out(torch.cat([k_star_contexts, trans_dec_outs], 2))

        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
