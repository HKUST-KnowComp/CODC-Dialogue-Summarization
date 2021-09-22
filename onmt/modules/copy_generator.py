import torch
import torch.nn as nn
import sys

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores

def collapse_copy_scores_know(scores, batch, tgt_vocab, know_vocabs=None,
                         batch_dim=1, batch_offset=None, decode_know_without_copy=False):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if know_vocabs is None:
            know_vocab = batch.know_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            know_vocab = know_vocabs[index]

        for i in range(1, len(know_vocab)):
            sw = know_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            if not decode_know_without_copy: # if not, do as normal
                score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores

class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           input_size = 
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)`` 
               extra_words means the number of words in the extended vocab
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden) # (batch x tlen, output_size)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1) # (batch x tlen, output_size)

        # Probability of copying p(z=1) batch.
        # Note: 这里p_copy = sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t +b_{ptr} (少了个+x)
        # But in the original paper it is sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t + w_x^Tx_t+b_{ptr}

        p_copy = torch.sigmoid(self.linear_copy(hidden)) # (batch x tlen, 1)
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy) # (batch x tlen, output_size)
        mul_attn = torch.mul(attn, p_copy) # (batch x tlen, input_size)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),  # (batch, tlen, slen)
            src_map.transpose(0, 1) # (batch, src_len, extra_words)
        ).transpose(0, 1) # (batch, tlen, extra_words) -> (tlen, batch, extra_words)
        copy_prob = copy_prob.contiguous().view(-1, cvocab) # (tlen x batch, extra_words)
        return torch.cat([out_prob, copy_prob], 1) # (tlen x batch, extra_words + output_size)

class KnowGenerator(nn.Module):
    """
    These networks consider copying words from the knowledge words

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{kgen}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{kgen}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx, know_size, p_kgen_func='linear', 
            logits_type='include_k_context', decoder_type='brnn'):
        super(KnowGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.linear_copy = nn.Linear(input_size, 1)
        self.p_kgen_func = p_kgen_func
        if self.p_kgen_func == 'linear':
            self.linear_copy = nn.Linear(input_size * 2 + know_size, 1, bias=True)
        elif self.p_kgen_func == 'mlp':
            if decoder_type == 'transformer':
                input_dim = input_size + know_size
            else:
                input_dim = input_size * 2 + know_size
            self.linear_hidden = nn.Linear(input_dim, input_dim, bias=False)
            self.linear_copy = nn.Linear(input_dim, 1, bias=True)
        self.pad_idx = pad_idx
        self.output_size = output_size

    def forward(self, hidden, know_attn, know_map,context_vecs):
        # """
        # Compute a distribution over the target dictionary
        # extended by the dynamic dictionary implied by copying
        # source words.

        # Args:
        #    hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
        #    know_attn (FloatTensor): know_attn for each ``(batch x tlen, input_size)``
        #    input_size = input emb_size
        # """

        # # CHECKS
        # batch_by_tlen, _ = hidden.size()
        # batch_by_tlen_, klen = know_attn.size()
        # klen_, batch, cvocab = know_map.size()
        # aeq(batch_by_tlen, batch_by_tlen_)
        # aeq(klen, klen_)

        # # Original probabilities.
        # logits = self.linear(hidden) # (batch x tlen, output_size)
        # logits[:, self.pad_idx] = -float('inf')
        # prob = torch.softmax(logits, 1) # (batch x tlen, output_size)

        # # Probability of copying p(z=1) batch.
        # # Note: 这里p_copy = sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t +b_{ptr} (少了个+x)
        # # But in the original paper it is sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t + w_x^Tx_t+b_{ptr}

        # p_copy = torch.sigmoid(self.linear_copy(hidden)) # (batch x tlen, 1)
        # # Probability of not copying: p_{word}(w) * (1 - p(z))
        # out_prob = torch.mul(prob, 1 - p_copy) # (batch x tlen, output_size)
        # mul_attn = torch.mul(know_attn, p_copy) # (batch x tlen, input_size)
        # copy_prob = torch.bmm(
        #     mul_attn.view(-1, batch, klen).transpose(0, 1),  # (batch, tlen, klen)
        #     know_map.to_dense().cuda().transpose(0, 1) # (batch, src_len, extra_words)
        # ).transpose(0, 1) # (batch, tlen, extra_words) -> (tlen, batch, extra_words)
        # copy_prob = copy_prob.contiguous().view(-1, cvocab) # (tlen x batch, extra_words)

        # return torch.cat([out_prob, torch.zeros(batch_by_tlen, cvocab - self.output_size)], 1) + copy_prob
        # # (tlen x batch, extra_words + output_size)

        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           input_size = 
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)`` 
               extra_words means the number of words in the extended vocab
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = know_attn.size()
        slen_, batch, cvocab = know_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden) # (batch x tlen, output_size)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1) # (batch x tlen, output_size)

        # Probability of copying p(z=1) batch.
        # Note: 这里p_copy = sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t +b_{ptr} (少了个+x)
        # But in the original paper it is sigma({w_{h^{*}}^Th_t^{*} + w_s^Ts_t + w_x^Tx_t+b_{ptr}
        # print('in knowgenerator context_vecs', context_vecs.size())
        if self.p_kgen_func == 'linear':
            p_copy = torch.sigmoid(self.linear_copy(context_vecs)) # (batch x tlen, 1)
        elif self.p_kgen_func == 'mlp':
            p_copy = torch.sigmoid(self.linear_copy(torch.tanh(self.linear_hidden(context_vecs)))) # (batch x tlen, 1)

        # print('pcopy:', p_copy.size())
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy) # (batch x tlen, output_size)
        mul_attn = torch.mul(know_attn, p_copy) # (batch x tlen, input_size)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),  # (batch, tlen, slen)
            know_map.transpose(0, 1) # (batch, src_len, extra_words)
        ).transpose(0, 1) # (batch, tlen, extra_words) -> (tlen, batch, extra_words)
        copy_prob = copy_prob.contiguous().view(-1, cvocab) # (tlen x batch, extra_words)

        return torch.cat([out_prob, copy_prob], 1), p_copy # (tlen x batch, extra_words + output_size)

class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        # 这里写的真💩
        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs # 这个表达式真的醉了，如果是unk_token（|target不是unk）则用vocab_probs + copy_token, 不是Unk则使用copy_tok_probs
        )
        # if not force copy, 那么如果tgt词match到copy candidates中，那么概率是copy_tok_probs, which is attention. 如果不在，那么update from p_vocab
        # if force copy, 如果target不是unk，那么不论怎样都update from copy_tok_probs + vocab_probs，当且仅当target==unk，只update attention。反正傻逼就完了

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss

class KnowGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(KnowGeneratorLoss, self).__init__()
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index
        self.force_copy=force_copy

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            target (LongTensor): ``(batch_size x tgt_len)``
            align (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        # I think the target should be ids of words
        # vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # # # probability of tokens copied from source
        # # copy_ix = align.unsqueeze(1) + self.vocab_size
        # # copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # # # Set scores for unk to 0 and add eps
        # # copy_tok_probs[align == self.unk_index] = 0
        # # copy_tok_probs += self.eps  # to avoid -inf logs

        # # find the indices in which you do not use the copy mechanism
        # # non_copy = align == self.unk_index
        # # if not self.force_copy:
        # #     non_copy = non_copy | (target != self.unk_index)

        # # probs = torch.where(
        # #     non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        # # )
        # # probs = torch.where(
        # #     non_copy, vocab_probs, copy_tok_probs
        # # )
        # probs = vocab_probs

        # loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # # Drop padding.
        # loss[target == self.ignore_index] = 0
        # return loss
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size

        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        # temporarily commented as I dk what this is . by ftq
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        })
        return shard_state

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats

class KnowGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0, know_loss_lambda=-1, p_kgen_loss=False):
        super(KnowGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.know_loss_lambda = know_loss_lambda
        self.ignore_index = criterion.ignore_index
        self.p_kgen_loss = p_kgen_loss

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "know_alignment", None) is None:
            raise AssertionError("using -know_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(KnowGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        # print('in KnowGeneratorLossCompute _make_shard_state attns["context_vecs"]', attns.get("context_vecs").size())

        # make know the same form of context_vecs and others
        # [tlen, b_s, dim]

        know = batch.know[0]

        t_len = attns.get("context_vecs").size()[0]
        k_len, batch_size, _ = know.size()
        know = know.squeeze(2).transpose(0, 1).unsqueeze(0).expand(t_len, batch_size, k_len)

        shard_state.update({
            "context_vecs":attns.get("context_vecs"),
            "know_attn": attns.get("knowledge"),
            "know_alignment": batch.know_alignment[range_[0] + 1: range_[1]],
            "know": know
        })
        
        return shard_state
    def _compute_know_loss(self, scores, know, know_attn, t_len, p_kgen):

        """
            scores: (batch_size x tgt_len, dimanic_vocab_size)
            know_attn: (batch_size x tlen, klen)
            know:``(klen, batch_size)``
            p_kgen: (tlen * batch_size, 1)

        """

        k_len, batch_size, _ = know.size()

        # know = know.squeeze(2).transpose(0, 1).unsqueeze(0).expand(t_len, batch_size, k_len)
        know = know.contiguous().view(t_len * batch_size, k_len)
        # (tlen, b_s, k_len)

        # know.expand(batch_size * tlen, know_len)

        know_vocab_probs = scores.gather(1, know).squeeze(1).contiguous() 
        # (b * tlen, k_len)
        # print('know', know.size())
        # print('know_vocab_probs', know_vocab_probs.size())

        # print('know_vocab_probs[know == self.ignore_index]', know_vocab_probs[know == self.ignore_index].size())
        # print('know==xx', (know == self.ignore_index).size())
        know_vocab_probs[know == self.ignore_index] = 0
        # a = fucker
        # know_vocab_probs = torch.where(
        #     know == self.ignore_index, torch.cuda.FloatTensor(know_vocab_probs.size(), device=know_vocab_probs.device).fill_(0.0), know_vocab_probs)
        # print('know_vocab_probs after', know_vocab_probs.size())
        know_vocab_probs = torch.bmm(know_vocab_probs.view(-1, 1, k_len),
                        know_attn.view(-1, k_len, 1)).squeeze(2) # (t_len * b_s, 1)
        if p_kgen is not None:
            know_vocab_probs = torch.mul(know_vocab_probs, p_kgen).squeeze(1)
        else:
            know_vocab_probs = know_vocab_probs.squeeze(1)

        loss = -know_vocab_probs.log()
        # Drop padding.
        
        return self.know_loss_lambda * loss

    def _compute_loss(self, batch, output, target, know_attn, know_alignment,context_vecs,
                      std_attn=None, coverage_attn=None, know=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            know_attn: the knowledge  attention value.
            align: the align info.
        """
        align = know_alignment
        t_len, b_s = target.size()
        target = target.view(-1)
        align = align.view(-1)
        # print(know_attn.size(), self._bottle(know_attn).size(), context_vecs.size(), self._bottle(context_vecs).size())
        # print('in KnowGeneratorLossCompute', context_vecs.size(), self._bottle(context_vecs).size())
        if self.p_kgen_loss:
            scores, p_kgen = self.generator(
                self._bottle(output), self._bottle(know_attn), batch.know_map, self._bottle(context_vecs)
            ) # (tlen x batch, extra_words + output_size)
        else:
            scores, _ = self.generator(
                self._bottle(output), self._bottle(know_attn), batch.know_map, self._bottle(context_vecs)
            ) # (tlen x batch, extra_words + output_size)
            p_kgen = None
        loss = self.criterion(scores, align, target)

        # t_len, b_s, _ = target.size()
        # print('target_size', target.size())
        # bs_by_tlen = target.size()[0]

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss
        if self.know_loss_lambda > 0:
            know_loss = self._compute_know_loss(scores, know, self._bottle(know_attn), t_len, p_kgen)
            # print("copy_generator", know_loss)
            loss += know_loss

        # add knowledge generalization

        # if self.know_gen_lambda > 0:
        #     reg_loss = 0
        #     for name, param in self.generator.named_parameters():
        #         if param.requires_grad and name.startswith('linear_copy'):
        #             reg_loss += torch.norm(param, p=2)
        #             # print('in copy_generator', reg_loss)
        #     loss += self.know_gen_lambda * reg_loss



        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores_know(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
