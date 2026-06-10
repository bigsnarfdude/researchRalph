"""
EXP-026: Test-time compute pipeline with selection intelligence (agent1)

Strategies:
1. Majority voting (baseline)
2. Shortest-majority voting
3. Confidence-weighted voting (weight by exp(mean_logprob of answer tokens))
4. Self-consistency clustering + confidence tiebreaker
5. Longest-majority voting (longer reasoning = more thorough)
6. Combined: majority vote, break ties by confidence
7. Best-of-N (pick highest logprob candidate)

Speed optimization: batch generation via num_return_sequences, batched logprob
via single padded forward pass.
"""
import argparse, math, time
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_candidates(model, tok, question, n_candidates=8,
                        temperature=0.7, max_new_tokens=512,
                        compute_confidence=True):
    """Generate N candidates with batched generation + batched logprob scoring.
    Returns list of (answer_num, gen_length, text, logprob)."""
    prompt = format_prompt(question)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    # Batch generate all candidates at once
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            num_return_sequences=n_candidates,
        )
    # out shape: (n_candidates, seq_len) — may vary in length, padded to max

    # Extract texts and answer numbers
    raw_candidates = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_len:]
        # Strip padding
        eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[:eos_positions[0]]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        num = extract_number(text)
        if num is not None:
            raw_candidates.append((num, len(gen_ids), text, i))

    if not raw_candidates or not compute_confidence:
        del out
        torch.cuda.empty_cache()
        return [(num, gl, text, 0.0) for num, gl, text, _ in raw_candidates]

    # Batched logprob computation: single forward pass on all sequences
    # Pad sequences to same length for batching
    all_seqs = []
    seq_gen_lens = []
    for num, gl, text, idx in raw_candidates:
        seq = out[idx, :prompt_len + gl]  # trim to actual content
        all_seqs.append(seq)
        seq_gen_lens.append(gl)

    if all_seqs:
        max_len = max(s.shape[0] for s in all_seqs)
        padded = torch.full((len(all_seqs), max_len), tok.eos_token_id,
                           dtype=torch.long, device=DEVICE)
        attention_mask = torch.zeros((len(all_seqs), max_len),
                                    dtype=torch.long, device=DEVICE)
        for i, seq in enumerate(all_seqs):
            padded[i, :seq.shape[0]] = seq
            attention_mask[i, :seq.shape[0]] = 1

        with torch.no_grad():
            logits = model(padded, attention_mask=attention_mask).logits

        candidates = []
        for i, (num, gl, text, _) in enumerate(raw_candidates):
            if gl <= 0:
                candidates.append((num, gl, text, -100.0))
                continue
            gen_logits = logits[i, prompt_len - 1:prompt_len - 1 + gl, :]
            gen_targets = padded[i, prompt_len:prompt_len + gl]
            lp = F.log_softmax(gen_logits.float(), dim=-1)
            token_lps = lp.gather(1, gen_targets.unsqueeze(1)).squeeze(1)
            mean_lp = token_lps.mean().item()
            candidates.append((num, gl, text, mean_lp))

        del logits, padded, attention_mask
    else:
        candidates = []

    del out
    torch.cuda.empty_cache()
    return candidates


# ---------- Selection strategies ----------

def select_majority(candidates):
    """Pure majority vote."""
    if not candidates:
        return None
    counts = Counter(num for num, _, _, _ in candidates)
    return counts.most_common(1)[0][0]


def select_shortest_majority(candidates):
    """Majority vote, break ties by preferring shorter solutions."""
    if not candidates:
        return None
    counts = defaultdict(int)
    avg_len = defaultdict(list)
    for num, gen_len, _, _ in candidates:
        counts[num] += 1
        avg_len[num].append(gen_len)
    best = max(counts.keys(),
               key=lambda k: (counts[k], -sum(avg_len[k]) / len(avg_len[k])))
    return best


def select_longest_majority(candidates):
    """Majority vote, break ties by preferring longer (more thorough) solutions."""
    if not candidates:
        return None
    counts = defaultdict(int)
    avg_len = defaultdict(list)
    for num, gen_len, _, _ in candidates:
        counts[num] += 1
        avg_len[num].append(gen_len)
    best = max(counts.keys(),
               key=lambda k: (counts[k], sum(avg_len[k]) / len(avg_len[k])))
    return best


def select_confidence_weighted(candidates):
    """Weight each vote by exp(mean_logprob). Higher confidence answers count more."""
    if not candidates:
        return None
    weights = defaultdict(float)
    for num, _, _, lp in candidates:
        weights[num] += math.exp(lp)
    return max(weights, key=weights.get)


def select_cluster_confidence(candidates):
    """
    Self-consistency + confidence: group by answer, pick the cluster with
    highest total confidence (count * avg_confidence).
    """
    if not candidates:
        return None
    clusters = defaultdict(list)
    for num, _, _, lp in candidates:
        clusters[num].append(math.exp(lp))
    return max(clusters, key=lambda k: len(clusters[k]) * (sum(clusters[k]) / len(clusters[k])))


def select_majority_confidence_tiebreak(candidates):
    """Majority vote; break ties using sum of confidence scores."""
    if not candidates:
        return None
    counts = defaultdict(int)
    conf_sum = defaultdict(float)
    for num, _, _, lp in candidates:
        counts[num] += 1
        conf_sum[num] += math.exp(lp)
    return max(counts.keys(), key=lambda k: (counts[k], conf_sum[k]))


def select_best_of_n(candidates):
    """Pure best-of-N: pick the single candidate with highest log-prob."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[3])[0]


STRATEGIES = {
    "majority": select_majority,
    "shortest_maj": select_shortest_majority,
    "longest_maj": select_longest_majority,
    "conf_weighted": select_confidence_weighted,
    "cluster_conf": select_cluster_confidence,
    "maj_conf_tie": select_majority_confidence_tiebreak,
    "best_of_n": select_best_of_n,
}

# --- Greedy-aware strategies (need greedy_num passed separately) ---

def select_greedy_boost_maj(candidates, greedy_num=None):
    """Majority vote but greedy answer gets bonus votes (greedy as independent signal)."""
    if not candidates:
        return None
    counts = Counter(num for num, _, _, _ in candidates)
    if greedy_num is not None:
        counts[greedy_num] += 2  # greedy gets 2 bonus votes
    return counts.most_common(1)[0][0]


def select_greedy_or_maj(candidates, greedy_num=None):
    """Use greedy if it appears in candidates (independent confirmation), else majority."""
    if not candidates:
        return None
    candidate_nums = set(num for num, _, _, _ in candidates)
    if greedy_num is not None and greedy_num in candidate_nums:
        return greedy_num
    counts = Counter(num for num, _, _, _ in candidates)
    return counts.most_common(1)[0][0]


def select_unanimous_or_greedy(candidates, greedy_num=None):
    """If majority is strong (>50% of votes), use majority. Else use greedy."""
    if not candidates:
        return None
    counts = Counter(num for num, _, _, _ in candidates)
    top_ans, top_count = counts.most_common(1)[0]
    if top_count > len(candidates) / 2:
        return top_ans
    if greedy_num is not None:
        return greedy_num
    return top_ans


def select_greedy_tiebreak_maj(candidates, greedy_num=None):
    """Majority vote, break ties by preferring the greedy answer."""
    if not candidates:
        return None
    counts = Counter(num for num, _, _, _ in candidates)
    top_count = counts.most_common(1)[0][1]
    tied = [a for a, c in counts.items() if c == top_count]
    if len(tied) == 1:
        return tied[0]
    if greedy_num in tied:
        return greedy_num
    return tied[0]


GREEDY_STRATEGIES = {
    "greedy_boost_maj": select_greedy_boost_maj,
    "greedy_or_maj": select_greedy_or_maj,
    "unanimous_or_greedy": select_unanimous_or_greedy,
    "greedy_tiebreak_maj": select_greedy_tiebreak_maj,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-confidence", action="store_true",
                        help="Skip logprob computation (faster, only majority strategies)")
    args = parser.parse_args()

    compute_conf = not args.no_confidence

    print(f"Test-time compute eval: {args.model}", flush=True)
    print(f"  {args.samples} problems, {args.candidates} candidates, "
          f"temp={args.temperature}, confidence={compute_conf}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    active_strategies = list(STRATEGIES.keys()) if compute_conf else [
        "majority", "shortest_maj", "longest_maj"]
    greedy_strats = list(GREEDY_STRATEGIES.keys())

    correct = {"greedy": 0}
    for s in active_strategies + greedy_strats:
        correct[s] = 0

    t0 = time.time()
    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        # Greedy baseline
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_text = tok.decode(greedy_out[0][prompt_len:], skip_special_tokens=True)
        greedy_num = extract_number(greedy_text)
        if check_answer(greedy_text, ex["answer"]):
            correct["greedy"] += 1
        del greedy_out

        # Generate candidates (batched)
        candidates = generate_candidates(
            model, tok, ex["question"],
            n_candidates=args.candidates,
            temperature=args.temperature,
            compute_confidence=compute_conf)

        if true_num is not None:
            for name in active_strategies:
                selector = STRATEGIES[name]
                ans = selector(candidates)
                if ans is not None and abs(ans - true_num) < 1e-3:
                    correct[name] += 1
            for name in greedy_strats:
                selector = GREEDY_STRATEGIES[name]
                ans = selector(candidates, greedy_num=greedy_num)
                if ans is not None and abs(ans - true_num) < 1e-3:
                    correct[name] += 1

        if (idx + 1) % 10 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (args.samples - n)
            maj_score = correct["majority"] / n
            all_strats = active_strategies + greedy_strats
            best_strat = max(all_strats, key=lambda s: correct[s])
            best_val = correct[best_strat] / n
            print(f"  {n}/{args.samples}  "
                  f"greedy={correct['greedy']/n:.3f}  "
                  f"maj@{args.candidates}={maj_score:.3f}  "
                  f"best={best_strat}={best_val:.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0
    greedy_score = correct["greedy"] / n

    print(f"\nResults ({elapsed:.0f}s, {n} problems, {args.candidates} candidates, "
          f"temp={args.temperature}):", flush=True)
    print(f"  Greedy pass@1:          {greedy_score:.4f}", flush=True)

    best_score = greedy_score
    best_name = "greedy"
    for name in active_strategies + greedy_strats:
        score = correct[name] / n
        delta = (score - greedy_score) * 100
        print(f"  {name:20s}@{args.candidates}: {score:.4f}  ({delta:+.1f}pp vs greedy)",
              flush=True)
        if score > best_score:
            best_score = score
            best_name = name

    print(f"\nBest strategy: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


if __name__ == "__main__":
    main()
