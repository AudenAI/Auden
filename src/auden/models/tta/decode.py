import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union

try:
    import k2
except Exception as e:
    raise ImportError(
        "k2 is required for RNNT decoding utilities. "
        "Install via conda: `conda install -c k2-fsa k2` (ensure PyTorch/CUDA match), "
        "or see https://k2-fsa.github.io/k2/."
    ) from e
import torch
from torch import nn


def compute_bleu(hyp: List[str], ref: List[List[str]]):
    """Compute BLEU score
    args:
        hyp: list of hypothesis;
        ref: list of references, enclosed by a list;
            (support multiple sets of refrences)
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU(lowercase=True, tokenize="13a")
    score = bleu.corpus_score(hyp, ref)
    bleu_score = float(score.score)
    return bleu_score


@dataclass
class DecodingResults:
    # hyps[i] is the recognition results, i.e., word IDs or token IDs
    # for the i-th utterance with fast_beam_search_nbest_LG.
    hyps: Union[List[List[int]], k2.RaggedTensor]

    # scores[i][k] contains the log-prob of tokens[i][k]
    # or the total log-prob of tokens[i]
    scores: Optional[List[Union[List[float], float]]] = None

    # prefixs[i][k] contains the k-th prefix output token of i-th sample
    prefixs: Optional[List[List[int]]] = None

    # timestamps[i][k] contains the frame number on which tokens[i][k]
    # is decoded
    timestamps: Optional[List[List[int]]] = None
    source_language: Optional[List[int]] = None
    target_language: Optional[List[int]] = None


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    ac_probs: Optional[List[float]] = None

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # N-gram LM state
    # state_cost: Optional[NgramLmStateCost] = None

    # Context graph state
    # context_state: Optional[ContextState] = None

    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)
    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape
        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    if not return_timestamps:
        return DecodingResults(
            hyps=ans,
        )
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )


def ctc_greedy_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """CTC greedy search.

    Args:
        ctc_output: (batch, seq_len, vocab_size)
        encoder_out_lens: (batch,)
    Returns:
        List[List[int]]: greedy search result
    """
    batch = ctc_output.shape[0]
    scores, index = ctc_output.max(dim=-1)
    hyps = [
        torch.unique_consecutive(index[i, : encoder_out_lens[i]]) for i in range(batch)
    ]
    scores = scores.sum(-1)
    hyps = [h[h != blank_id].tolist() for h in hyps]
    return DecodingResults(hyps=hyps, timestamps=None, scores=scores, prefixs=None)


def attention_beam_search(
    model,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam_size: int = 10,
    length_penalty: float = 0.0,
    source_language: Optional[List[int]] = None,
    target_language: Optional[List[int]] = None,
) -> DecodingResults:
    """Decode with auto-regressive attention decoder.
    Args:
        model: model with an attention `decoder`.
        encoder_out:
        encoder_out_lens:
        beam_size:
        length_penalty:
        source_language: Optional list of language token ids per utterance (len == batch size).
            If not provided, source language is predicted by the decoder.
        target_language: Optional list of target language token ids per utterance (len == batch size).
            If not provided, it defaults to source_language (ASR mode).
    """
    device = encoder_out.device
    batch_size = encoder_out.shape[0]

    sos = model.attention_decoder.sos_id
    eos = model.attention_decoder.eos_id

    maxlen = encoder_out.size(1)
    running_size = batch_size * beam_size
    # init scores like this to make sure the first beam is not trivial
    scores = torch.tensor([0.0] + [-float("inf")] * (beam_size - 1), dtype=torch.float)
    scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)

    sos_seq = (
        torch.ones((running_size, 1), dtype=torch.long, device=encoder_out.device) * sos
    )
    if source_language is not None:
        if len(source_language) != batch_size:
            raise ValueError(
                f"source_language length {len(source_language)} != batch size {batch_size}"
            )
        src_per_batch = torch.tensor(
            source_language, device=device, dtype=torch.long
        ).unsqueeze(
            1
        )  # (B,1)
        language_token = src_per_batch.repeat(1, beam_size).view(-1, 1)
    else:
        # predict topk languages
        language_index, language_logp = model.forward_attention_decoder_language_token(
            encoder_out, encoder_out_lens, topk=beam_size
        )
        language_token = language_index[:, :1].repeat(1, beam_size)
        language_token[:, :beam_size] = language_index
        language_token = language_token.view(-1, 1)
        # modify initial scores for LID-informed beam search
        scores = scores.view(batch_size, beam_size)
        scores[:, :beam_size] = torch.log_softmax(language_logp, dim=-1)
        scores = scores.view(-1, 1)
    prefix_tokens = torch.cat([sos_seq, language_token], dim=1)

    if target_language is not None:
        if len(target_language) != batch_size:
            raise ValueError(
                f"target_language length {len(target_language)} != batch size {batch_size}"
            )
        tgt_per_batch = torch.tensor(
            target_language, device=device, dtype=torch.long
        ).unsqueeze(
            1
        )  # (B,1)
        tgt_token = tgt_per_batch.repeat(1, beam_size).view(-1, 1)
        prefix_tokens = torch.cat([prefix_tokens, tgt_token], dim=1)
    else:
        # If target_language is not provided, set it equal to source_language token(s)
        tgt_token = language_token
        prefix_tokens = torch.cat([prefix_tokens, tgt_token], dim=1)

    prefix_len = prefix_tokens.size(1)
    hyps = prefix_tokens
    encoder_out = encoder_out.repeat((1, beam_size, 1)).reshape(
        (batch_size * beam_size, maxlen, -1)
    )
    encoder_out_lens = encoder_out_lens.unsqueeze(1).repeat((1, beam_size)).view(-1)

    scores = torch.tensor([0.0] + [-float("inf")] * (beam_size - 1), dtype=torch.float)
    scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    cache = {"self_attn_cache": {}, "src_attn_cache": {}}

    for i in range(prefix_len, maxlen + 1):
        if end_flag.sum() == running_size:
            break
        logp = model.attention_decoder.forward_one_step(
            encoder_out, encoder_out_lens, hyps, cache
        )
        logp = torch.log_softmax(logp[:, -1, :], dim=-1)  # (B*N, V)
        top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)

        zero_mask = torch.zeros_like(end_flag, dtype=torch.bool)
        head_mask = torch.cat([end_flag, zero_mask.repeat([1, beam_size - 1])], dim=1)
        other_mask = torch.cat([zero_mask, end_flag.repeat([1, beam_size - 1])], dim=1)
        top_k_logp.masked_fill_(other_mask, -float("inf"))
        top_k_logp.masked_fill_(head_mask, 0.0)
        top_k_index.masked_fill_(end_flag.repeat([1, beam_size]), eos)

        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)

        cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
        base_cache_index = (
            torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])
            * beam_size
        ).view(
            -1
        )  # (B*N)
        cache_index = base_cache_index + cache_index
        cache["self_attn_cache"] = {
            i_layer: torch.index_select(value, dim=1, index=cache_index)
            for (i_layer, value) in cache["self_attn_cache"].items()
        }
        scores = scores.view(-1, 1)  # (B*N, 1)

        base_k_index = (
            torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])
        )  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

        best_k_pred = torch.index_select(
            top_k_index.view(-1), dim=-1, index=best_k_index
        )  # (B*N)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(
            hyps, dim=0, index=best_hyps_index
        )  # (B*N, i)
        hyps = torch.cat(
            (last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1
        )  # (B*N, i+1)

        end_flag = torch.eq(hyps[:, -1], eos).view(-1, 1)

    scores = scores.view(batch_size, beam_size)
    lengths = hyps.ne(eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = (
        best_index
        + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
    )
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    # extract predicted source/target language ids from prefix
    pred_src_lang = best_hyps[:, 1].tolist()
    pred_tgt_lang = best_hyps[:, 2].tolist()
    # strip prefix tokens
    best_hyps = best_hyps[:, prefix_len:]

    return DecodingResults(
        hyps=best_hyps.tolist(),
        timestamps=None,
        scores=best_scores,
        prefixs=None,
        source_language=pred_src_lang,
        target_language=pred_tgt_lang,
    )


def store_transcripts(
    filename: Path, texts: Iterable[Tuple[str, str, str]], char_level: bool = False
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
        If it is a multi-talker ASR system, the ref and hyp may also be lists of
        strings.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf8") as f:
        for cut_id, ref, hyp in texts:
            if char_level:
                ref = list("".join(ref))
                hyp = list("".join(hyp))
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def save_bleu_results(
    res_dir,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
    suffix: str = None,
):
    test_set_bleus = dict()
    for key, results in results_dict.items():
        recog_path = res_dir / f"translations-{test_set_name}-{key}-{suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The translations are stored in {recog_path}")

        bleu_score = compute_bleu(
            [" ".join(i[2]) for i in results],
            [[" ".join(i[1]) for i in results]],
        )
        test_set_bleus[key] = float("{:.2f}".format(bleu_score))

    test_set_bleus = sorted(test_set_bleus.items(), key=lambda x: x[1])
    errs_info = res_dir / f"bleu-summary-{test_set_name}-{key}-{suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tBLEU", file=f)
        for key, val in test_set_bleus:
            print("{}\t{}".format(key, val), file=f)

    bleu_info = res_dir / f"bleu-summary-all-{key}-{suffix}.txt"
    if not os.path.exists(bleu_info):
        with open(bleu_info, "w") as f:
            print("dataset\tsettings\tBLEU", file=f)
    with open(bleu_info, "a+") as f:
        for key, val in test_set_bleus:
            print("{}\t{}\t{}".format(test_set_name, key, val), file=f)

    s = "\nFor {}, BLEU of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_bleus:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)
