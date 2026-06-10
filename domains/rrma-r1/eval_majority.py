"""
Majority voting evaluation — sample N completions per problem,
extract answers from each, take the most common answer.

This is a test-time compute technique, not training.
Can significantly boost scores without any model changes.
"""
import argparse, time
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def majority_vote_eval(model_path, n_samples=100, n_votes=8, temperature=0.7):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:n_samples]
    correct_greedy = 0
    correct_majority = 0

    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)

        # Greedy (single sample)
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_pred = tok.decode(greedy_out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        if check_answer(greedy_pred, ex["answer"]) == 1.0:
            correct_greedy += 1

        # Majority voting (N samples)
        answers = []
        with torch.no_grad():
            for _ in range(n_votes):
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=temperature, top_p=0.95,
                    pad_token_id=tok.eos_token_id)
                pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                num = extract_number(pred)
                if num is not None:
                    answers.append(num)

        if answers:
            # Most common answer
            counter = Counter(answers)
            majority_answer = counter.most_common(1)[0][0]
            true_num = extract_number(ex["answer"])
            if true_num is not None and abs(majority_answer - true_num) < 1e-3:
                correct_majority += 1

        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{n_samples}  greedy={correct_greedy/(idx+1):.3f}  "
                  f"majority@{n_votes}={correct_majority/(idx+1):.3f}", flush=True)

    greedy_score = correct_greedy / n_samples
    majority_score = correct_majority / n_samples
    return greedy_score, majority_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--votes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Majority voting eval: {args.model}, {args.samples} problems, "
          f"{args.votes} votes, temp={args.temperature}", flush=True)

    t0 = time.time()
    greedy, majority = majority_vote_eval(
        args.model, args.samples, args.votes, args.temperature)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.0f}s):", flush=True)
    print(f"  Greedy pass@1:     {greedy:.4f}", flush=True)
    print(f"  Majority@{args.votes} pass@1: {majority:.4f}", flush=True)
    print(f"  Improvement:       +{(majority-greedy)*100:.1f}pp", flush=True)
    print(f"SCORE: {majority:.4f}")


if __name__ == "__main__":
    main()
