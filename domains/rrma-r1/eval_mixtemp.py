"""
EXP-033: Mixed-temperature majority voting (agent0)

Idea: Generate candidates at multiple temperatures to get both reliable (low-temp)
and diverse (high-temp) samples. Majority vote over the mixed pool.

Configs to test:
1. 4@temp=0.5 + 4@temp=0.9 (balanced)
2. 6@temp=0.7 + 2@temp=1.0 (mostly reliable + some wild)
3. 4@temp=0.7 + 4@temp=0.5 (all reliable, more diversity via more candidates)
"""
import argparse, time
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_at_temp(model, tok, prompt_ids, prompt_len, n, temperature, max_new_tokens=512):
    """Generate n candidates at given temperature."""
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            num_return_sequences=n,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_len:]
        eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[:eos_positions[0]]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        num = extract_number(text)
        if num is not None:
            results.append(num)
    del out
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    print(f"Mixed-temperature majority voting: {args.model}", flush=True)
    print(f"  {args.samples} problems", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    # Define configs: list of (n_candidates, temperature)
    configs = {
        "uniform_8@0.7": [(8, 0.7)],
        "mix_4@0.5+4@0.9": [(4, 0.5), (4, 0.9)],
        "mix_6@0.7+2@1.0": [(6, 0.7), (2, 1.0)],
        "mix_4@0.7+4@0.5": [(4, 0.7), (4, 0.5)],
        "mix_4@0.6+4@0.8": [(4, 0.6), (4, 0.8)],
        "mix_2@0.5+4@0.7+2@0.9": [(2, 0.5), (4, 0.7), (2, 0.9)],
    }

    correct = {"greedy": 0}
    for name in configs:
        correct[name] = 0

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
        if check_answer(greedy_text, ex["answer"]):
            correct["greedy"] += 1
        del greedy_out

        if true_num is None:
            continue

        # Generate candidates for each config
        for name, temp_specs in configs.items():
            all_nums = []
            for n, temp in temp_specs:
                nums = generate_at_temp(model, tok, inputs["input_ids"],
                                       prompt_len, n, temp)
                all_nums.extend(nums)

            if all_nums:
                counts = Counter(all_nums)
                maj_ans = counts.most_common(1)[0][0]
                if abs(maj_ans - true_num) < 1e-3:
                    correct[name] += 1

        if (idx + 1) % 10 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            eta = elapsed / n * (args.samples - n)
            parts = []
            for name in configs:
                parts.append(f"{name}={correct[name]/n:.3f}")
            print(f"  {n}/{args.samples}  greedy={correct['greedy']/n:.3f}  "
                  + "  ".join(parts[:3])
                  + f"  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.0f}s, {n} problems):", flush=True)
    print(f"  Greedy pass@1:          {correct['greedy']/n:.4f}", flush=True)

    best_score = correct["greedy"] / n
    best_name = "greedy"
    for name in configs:
        score = correct[name] / n
        delta = (score - correct["greedy"] / n) * 100
        print(f"  {name:30s}: {score:.4f}  ({delta:+.1f}pp vs greedy)", flush=True)
        if score > best_score:
            best_score = score
            best_name = name

    print(f"\nBest: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


if __name__ == "__main__":
    main()
