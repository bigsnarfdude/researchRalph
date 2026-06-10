"""
EXP-022: Profiled Curriculum GRPO + Binary Reward + AdamW (agent1)

Different approach from agent0's EXP-021a:
- Agent0: online curriculum + shaped reward (0.3 for wrong-but-formatted) + manual GD
- Agent1: offline profiling + BINARY reward (pure correctness) + AdamW

Hypothesis: Shaped reward reinforces wrong reasoning chains that happen to use
#### format. Pure binary reward on carefully-selected problems is cleaner.
The key is to ONLY train on problems where the model has 25-75% accuracy
(maximum entropy → maximum gradient signal per step).

Also: 2 prompts per optimizer step for more diverse gradient signal.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from gsm8k_data import load_gsm8k, format_prompt
from reward import check_answer
import random, time, gc, os

CHECKPOINT  = "./best"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS   = 200
GROUP_SIZE  = 8
GEN_BATCH   = 2
LR          = 3e-6
LR_MIN      = 3e-7
MAX_GEN_LEN = 384
MAX_PROMPT_LEN = 200
WEIGHT_DECAY = 0.02
TEMPERATURE = 0.9
SEED        = 522
CLIP_GRAD   = 0.5
BUDGET_MIN  = 32
PROMPTS_PER_STEP = 2  # Multiple prompts per optimizer step

# Profiling config
PROFILE_K   = 4   # samples per problem during profiling
PROFILE_N   = 300 # problems to profile (covers ~1/3 of training set)
MIN_ACC     = 0.25
MAX_ACC     = 0.75

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
random.seed(SEED)
torch.manual_seed(SEED)


def profile_problems(model, tok, problems, k=4, max_n=300):
    """Quick profiling: sample K completions per problem, record accuracy."""
    model.eval()
    results = []

    for idx, ex in enumerate(problems[:max_n]):
        prompt_text = format_prompt(ex["question"])
        prompt_enc = tok(prompt_text, return_tensors="pt",
                        truncation=True, max_length=MAX_PROMPT_LEN).to(DEVICE)
        prompt_len = prompt_enc["input_ids"].shape[1]

        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(0, k, GEN_BATCH):
                batch_sz = min(GEN_BATCH, k - total)
                if batch_sz <= 0:
                    break
                try:
                    outs = model.generate(
                        **prompt_enc,
                        max_new_tokens=MAX_GEN_LEN,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=0.95,
                        num_return_sequences=batch_sz,
                        pad_token_id=tok.eos_token_id,
                    )
                    for i in range(outs.shape[0]):
                        resp = tok.decode(outs[i, prompt_len:], skip_special_tokens=True)
                        r = check_answer(resp, ex["answer"])
                        correct += r
                        total += 1
                    del outs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    break

        torch.cuda.empty_cache()
        if total > 0:
            acc = correct / total
            results.append((idx, ex, acc))

        if idx % 50 == 0:
            elapsed_problems = idx + 1
            sweet = sum(1 for _, _, a in results if MIN_ACC <= a <= MAX_ACC)
            print(f"  Profiled {elapsed_problems}/{max_n}... "
                  f"sweet_spot={sweet}/{len(results)}", flush=True)

    return results


def main():
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading policy from {CHECKPOINT}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16).to(DEVICE)
    model.gradient_checkpointing_enable()

    train_data = load_gsm8k(split="train")
    random.shuffle(train_data)

    # Phase 1: Profile
    print(f"=== PHASE 1: Profiling {PROFILE_N} problems (K={PROFILE_K}) ===", flush=True)
    profiled = profile_problems(model, tok, train_data, k=PROFILE_K, max_n=PROFILE_N)

    curriculum = [(idx, ex) for idx, ex, acc in profiled if MIN_ACC <= acc <= MAX_ACC]
    easy = [(idx, ex) for idx, ex, acc in profiled if acc > MAX_ACC]
    hard = [(idx, ex) for idx, ex, acc in profiled if acc < MIN_ACC]

    profile_time = time.time() - t0
    print(f"Profile done in {profile_time:.0f}s", flush=True)
    print(f"  Curriculum (sweet spot): {len(curriculum)} problems", flush=True)
    print(f"  Easy (>{MAX_ACC}): {len(easy)}, Hard (<{MIN_ACC}): {len(hard)}", flush=True)

    if len(curriculum) < 15:
        print("WARNING: Too few curriculum problems, widening range", flush=True)
        curriculum = [(idx, ex) for idx, ex, acc in profiled if 0.10 <= acc <= 0.90]
        print(f"  Widened curriculum: {len(curriculum)} problems", flush=True)

    remaining_budget = BUDGET_MIN * 60 - (time.time() - t0)
    if remaining_budget < 120:
        print(f"ERROR: Only {remaining_budget:.0f}s left after profiling, not enough for training", flush=True)
        # Still save for eval
        for save_dir in ["./checkpoints/exp022", "./checkpoints/latest"]:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
        return

    # Phase 2: GRPO on curriculum problems with AdamW
    print(f"=== PHASE 2: AdamW GRPO on {len(curriculum)} curriculum problems ===", flush=True)
    print(f"  G={GROUP_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, temp={TEMPERATURE}", flush=True)
    print(f"  prompts_per_step={PROMPTS_PER_STEP}, remaining_budget={remaining_budget:.0f}s", flush=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                      foreach=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=LR_MIN)

    random.shuffle(curriculum)

    total_correct = 0
    total_samples = 0
    signal_steps = 0
    data_idx = 0

    for step in range(MAX_STEPS):
        elapsed = time.time() - t0
        if elapsed > BUDGET_MIN * 60:
            print(f"Budget limit at step {step} ({elapsed:.0f}s)", flush=True)
            break

        optimizer.zero_grad()
        step_loss = 0.0
        step_signal = False
        n_backward_total = 0

        # Process multiple prompts per optimizer step
        for prompt_i in range(PROMPTS_PER_STEP):
            _, ex = curriculum[data_idx % len(curriculum)]
            data_idx += 1

            prompt_text = format_prompt(ex["question"])
            prompt_enc = tok(prompt_text, return_tensors="pt",
                            truncation=True, max_length=MAX_PROMPT_LEN).to(DEVICE)
            prompt_len = prompt_enc["input_ids"].shape[1]

            model.eval()
            all_outputs = []
            rewards = []

            with torch.no_grad():
                for batch_start in range(0, GROUP_SIZE, GEN_BATCH):
                    batch_size = min(GEN_BATCH, GROUP_SIZE - batch_start)
                    try:
                        outs = model.generate(
                            **prompt_enc,
                            max_new_tokens=MAX_GEN_LEN,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=0.95,
                            num_return_sequences=batch_size,
                            pad_token_id=tok.eos_token_id,
                        )
                        for i in range(outs.shape[0]):
                            resp = tok.decode(outs[i, prompt_len:], skip_special_tokens=True)
                            r = check_answer(resp, ex["answer"])
                            all_outputs.append(outs[i].clone())
                            rewards.append(r)
                        del outs
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue

            torch.cuda.empty_cache()

            if len(rewards) < 2:
                continue

            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_t.mean().item()
            std_r = rewards_t.std().item()
            total_correct += sum(rewards)
            total_samples += len(rewards)

            if std_r < 1e-6:
                del all_outputs
                continue

            step_signal = True
            advantages = (rewards_t - mean_r) / (std_r + 1e-8)

            model.train()
            n_backward = 0

            for i in range(len(all_outputs)):
                adv = advantages[i].item()
                if abs(adv) < 1e-6:
                    continue

                seq = all_outputs[i].unsqueeze(0)
                if seq.shape[1] - prompt_len <= 1:
                    continue

                try:
                    logits = model(input_ids=seq, use_cache=False).logits
                    shift_logits = logits[:, prompt_len-1:-1, :]
                    shift_labels = seq[:, prompt_len:]
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    policy_lp = log_probs.gather(
                        2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)

                    # Scale by number of prompts per step
                    pg_loss = -(adv * policy_lp.mean()) / (len(all_outputs) * PROMPTS_PER_STEP)
                    pg_loss.backward()

                    step_loss += pg_loss.item()
                    n_backward += 1

                    del logits, log_probs, policy_lp
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                torch.cuda.empty_cache()

            n_backward_total += n_backward
            del all_outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Optimizer step after all prompts
        if n_backward_total > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        if step_signal:
            signal_steps += 1

        scheduler.step()

        if step % 25 == 0:
            acc = total_correct / max(total_samples, 1)
            cur_lr = scheduler.get_last_lr()[0]
            elapsed_now = time.time() - t0
            print(f"  step {step}/{MAX_STEPS}  loss={step_loss:.4f}  "
                  f"acc={acc:.3f}  lr={cur_lr:.1e}  "
                  f"sig={signal_steps}  {elapsed_now:.0f}s", flush=True)

    # Save — only to exp022, don't overwrite best
    for save_dir in ["./checkpoints/exp022", "./checkpoints/latest"]:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    elapsed = time.time() - t0
    acc = total_correct / max(total_samples, 1)
    print(f"Done. {elapsed:.0f}s, acc={acc:.3f}, signal={signal_steps}/{step+1}",
          flush=True)
    print(f"Curriculum: {len(curriculum)} sweet-spot problems "
          f"(easy={len(easy)}, hard={len(hard)})", flush=True)


if __name__ == "__main__":
    main()
