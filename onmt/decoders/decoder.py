import torch
import torch.nn as nn

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention, KnowledgeAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq


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


class RNNDecoderBase(DecoderBase):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~onmt.models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       dec_rnn_size (int) : hidden size of each layer
       enc_rnn_size (int) : hidden size of enc layer
       attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
       attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
       coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
       context_gate (str): see :class:`~onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
       reuse_copy_attn (bool): reuse the attention for copying
       copy_attn_type (str): The copy attention style. See
        :class:`~onmt.modules.GlobalAttention`.
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 dec_rnn_size, enc_rnn_size, word_vec_size,
                 attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None, 
                 know_emb=None, know_attn=False, know_attn_type="mlp",
                 know_attn_func="softmax",
                 reuse_copy_attn=False, copy_attn_type="general",
                 use_input_know_attn=False, know_star_attn_share=False):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        self.src_dim = enc_rnn_size
        self.tgt_dim = dec_rnn_size
        self.know_hidden_dim = word_vec_size
        self.attn_type = attn_type
        self.know_star_attn_share = know_star_attn_share
        if use_input_know_attn:
            self.linear_out = nn.Linear(self.src_dim + self.tgt_dim + self.know_hidden_dim, 
                self.tgt_dim, bias=self.attn_type == "mlp" )

        

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = dec_rnn_size # self.hidden_size denotes dec_rnn_size
        self.embeddings = embeddings
        self.know_emb=know_emb
        self.dropout = nn.Dropout(dropout)
        
        know_dim = 0
        
        if know_emb is not None:
            know_dim = know_emb.word_vec_size #TODO

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=dec_rnn_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                dec_rnn_size, dec_rnn_size, dec_rnn_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self._know_attn = know_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            if self._know_attn:
                raise ValueError("Cannot use knowledge attention term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                dec_rnn_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )
            if self._know_attn:
                if use_input_know_attn: 
                    # use inputs to attend to knowledge
                    if not self.know_star_attn_share:
                        self.k_star_attn = GlobalAttention(
                            [self.know_hidden_dim, dec_rnn_size], coverage=coverage_attn,
                            attn_type=attn_type, attn_func=attn_func
                        )
                else:
                    self.k_attn = KnowledgeAttention(
                        [enc_rnn_size, dec_rnn_size, know_dim], 
                        attn_type=know_attn_type, attn_func=know_attn_func,
                    )

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = GlobalAttention(
                dec_rnn_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

    @classmethod
    def from_opt(cls, opt, embeddings, know_emb=None):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.enc_rnn_size,
            opt.word_vec_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            know_emb,
            opt.knowledge,
            opt.know_attn_type,
            opt.know_attention_function,
            opt.reuse_copy_attn,
            opt.copy_attn_type,
            opt.input_attn,
            opt.know_star_attn_share)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None, 
                know=None, know_lengths=None, 
                input_know_vecs=None, input_know_attns=None, 
                step=None):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden_dim)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths, know=know, know_lengths=know_lengths, 
            input_know_vecs=input_know_vecs, input_know_attns=input_know_attns)
        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if self._know_attn:
            self.state["knowledge"] = attns["knowledge"]
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class StdRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.

    Faster implementation, uses CuDNN for implementation.
    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, know=None, know_lengths=None,
                input_know_vecs=None, input_know_attns=None, ):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                ``(len, batch, nfeats)``.
            memory_bank (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(src_len, batch, hidden_size)``.
            memory_lengths (LongTensor): the source memory_bank lengths.

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """

        assert self.copy_attn is None  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        attns = {}
        emb = self.embeddings(tgt)

        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"])

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_output
        else:
            dec_outs, p_attn = self.attn(
                rnn_output.transpose(0, 1).contiguous(),
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths
            )
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                dec_outs.view(-1, dec_outs.size(2))
            )
            dec_outs = dec_outs.view(tgt_len, tgt_batch, self.hidden_size)

        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`w`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, know=None, know_lengths=None,
        input_know_vecs=None, input_know_attns=None,):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                ``(len, batch, nfeats)``.
            memory_bank (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(src_len, batch, hidden_size)``.
            memory_lengths (LongTensor): the source memory_bank lengths.

        input_know_vecs: knowledge vectors attended by input hidden states:
            [batch, source_l, know_emb_dim]
        input_know_attns: corresponding attentions:
            [batch, source_l, know_l]

        if input_know_vecs is not None, then use s_t to attend to the input_know_vecs -> k_star_context

        Then k_star_context is added to P_vocab equation (??????????????????????????????????????????h_*???k*_context share attention??????????????????attention) ?????????????????????

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        batch = tgt_batch
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        if self._know_attn:
            attns["knowledge"] = []
            attns["context_vecs"] = []

        emb = self.embeddings(tgt)
        if self.know_emb is not None:
            know_emb = self.know_emb(know) # TODO, by ftq, add judgement about whether to share embeddings
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            if self.attentional:
                # decoder_output: attended vectors (tgt_len, batch, dim)
                # p_attn: attention weights: (tgt_len, batch, src_len)
                
                if self._know_attn:
                    # target, know_bank, context_vecs, know_lengths=None
                    # know_attn_output: (tgt_len, batch, know_dim) knowledge vectors
                    # k_attn: (tgt_len, batch, know_len) knowledge attentions
                    input_context, p_attn = self.attn(
                        rnn_output,
                        memory_bank.transpose(0, 1),
                        memory_lengths=memory_lengths,
                        know=True)
                    attns["std"].append(p_attn)

                    if input_know_vecs is not None:
                        if self.know_star_attn_share: # ????????????attention parameter???
                            # (batch, tgt_len, src_len) * [batch, src_len, know_emb_dim] 
                            # print('p_attn', p_attn.size(), 'input_know_vecs', input_know_vecs.size())

                            if len(p_attn.size()) == 2:
                                one_step = True
                                p_attn = p_attn.unsqueeze(1) # tgt_len == 1
                            else:
                                one_step = False

                            k_star_context = torch.bmm(p_attn, input_know_vecs)
                            k_star_context = k_star_context.contiguous()
                            # [batch, tgt_len, hidden_dim]
                            

                            if one_step:
                                k_star_context = k_star_context.squeeze(1)
                                p_attn = p_attn.squeeze(1)
                                k_star_attn = p_attn
                            # print("k_star_context", k_star_context.shape)
                        else: # ??????????????????s_t attend to [{h_i}, {k_star_i}], ????????????Attention[a_i_t, b_i_t]
                            k_star_context, k_star_attn = self.k_star_attn(
                                rnn_output,
                                input_know_vecs,
                                memory_lengths=memory_lengths,
                                know=True)

                        if len(k_star_attn.size()) == 2:
                            one_step = True
                            k_star_context = k_star_context.unsqueeze(1)
                            k_star_attn = k_star_attn.unsqueeze(1)
                            rnn_output = rnn_output.unsqueeze(1)
                            input_context = input_context.unsqueeze(1)
                        else:
                            one_step = False

                        tgt_len = 1 if one_step else k_star_attn.size()[0]

                        # knowledge attention equals the bmm of p_attn and input_know_attns
                        # (batch, tgt_len, src_len) * [batch, source_l, know_l]
                        k_attn = torch.bmm(k_star_attn, input_know_attns)
                        
                        concat_c = torch.cat([k_star_context, 
                            rnn_output, 
                            input_context], 2).view(batch * tgt_len, 
                            self.src_dim + self.tgt_dim + self.know_hidden_dim)
                        
                        decoder_output = self.linear_out(concat_c).view(batch, tgt_len, self.tgt_dim)

                        if self.attn_type in ["general", "dot"]:
                            decoder_output = torch.tanh(decoder_output)
                        decoder_output = decoder_output.contiguous()

                        if one_step:
                            k_attn = k_attn.squeeze(1)
                            decoder_output = decoder_output.squeeze(1)
                            rnn_output = rnn_output.squeeze(1)
                            input_context = input_context.squeeze(1)
                            k_star_context = k_star_context.squeeze(1)

                        # print("k_attn", k_attn.size())
                        # print("decoder_output", decoder_output.size())
                    else:
                        # calculate knowledge attention directly by attending s_t to {k_i}
                        decoder_output, k_attn, k_star_context = self.k_attn(
                            rnn_output,
                            know_emb.transpose(0, 1),
                            input_context,
                            know_lengths=know_lengths,) 

                    # k_star_context: [b_s, hidden_dim]
                    attns["knowledge"].append(k_attn)

                    # by ftq
                    if rnn_output.dim() == 2:
                        one_step = True
                        rnn_output = rnn_output.unsqueeze(1)
                    else:
                        one_step = False
                    if one_step:
                        input_context = input_context.unsqueeze(1)
                        k_star_context = k_star_context.unsqueeze(1)
                    else:
                        input_context = input_context.transpose(0, 1).contiguous()
                        k_star_context = k_star_context.transpose(0, 1).contiguous()
                    cont_vec = torch.cat([input_context, k_star_context, rnn_output], 2)
                    if one_step:
                        cont_vec = cont_vec.squeeze(1)
                        rnn_output = rnn_output.squeeze(1)
                    attns["context_vecs"].append(cont_vec)
                else:
                    # seq2seq mode without knowledge
                    decoder_output, p_attn = self.attn(
                        rnn_output,
                        memory_bank.transpose(0, 1),
                        memory_lengths=memory_lengths,
                        know=False)
                    attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None: # this is determined at init stage
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)
