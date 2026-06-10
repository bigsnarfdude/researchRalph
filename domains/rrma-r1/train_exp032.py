"""
EXP-032: Offline Difficulty Curriculum + GRPO with Manual GD (agent0)

Manual gradient descent = zero optimizer memory, can coexist with other GPU processes.
Profile 300 problems with K=4 (faster), train on sweet-spot (20-80% accuracy).
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import check_answer
import random, time, gc, os, json

CHECKPOINT  = "./best"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS   = 300
GROUP_SIZE  = 8
GEN_BATCH   = 2
LR          = 5e-6
LR_MIN      = 5e-7
MAX_GEN_LEN = 384
MAX_PROMPT_LEN = 200
WEIGHT_DECAY = 0.02
TEMPERATURE = 0.9
SEED        = 1337
CLIP_GRAD   = 0.5
BUDGET_MIN  = 30

# Profiling config
PROFILE_K       = 4     # 4 samples (faster, still decent signal)
PROFILE_N       = 300   # profile 300 problems
PROFILE_GEN_LEN = 384
SWEET_LO        = 0.20
SWEET_HI        = 0.80
PROFILE_BUDGET  = 480   # 8 min for profiling
PROFILE_FILE    = "problem_difficulty_v2.json"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
random.seed(SEED)
torch.manual_seed(SEED)


def fast_profile(model, tok, problems, k=4, max_n=200, gen_len=256, budget_s=480):
    """Profile problems by sampling K completions and checking accuracy."""
    model.eval()
    results = []
    t_start = time.time()

    for idx, ex in enumerate(problems[:max_n]):
        if time.time() - t_start > budget_s:
            print(f"  Profiling budget hit at {idx}/{max_n}", flush=True)
            break

        prompt_text = format_prompt(ex["question"])
        prompt_enc = tok(prompt_text, return_tensors="pt",
                        truncation=True, max_length=MAX_PROMPT_LEN).to(DEVICE)
        prompt_len = prompt_enc["input_ids"].shape[1]

        correct = 0
        total = 0
        with torch.no_grad():
            # Generate K samples in batches of 2
            for batch_start in range(0, k, 2):
                bs = min(2, k - batch_start)
                try:
                    outs = model.generate(
                        **prompt_enc,
                        max_new_tokens=gen_len,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=0.95,
                        num_return_sequences=bs,
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
                    continue

        torch.cuda.empty_cache()
        if total > 0:
            acc = correct / total
            results.append((idx, ex, acc))

        if idx % 50 == 0:
            t_elapsed = time.time() - t_start
            sweet = sum(1 for _, _, a in results if SWEET_LO <= a <= SWEET_HI)
            print(f"  Profiled {idx+1}/{max_n}... "
                  f"sweet={sweet}/{len(results)} ({t_elapsed:.0f}s)", flush=True)

    return results


def main():
    t0 = time.time()

    checkpoint = CHECKPOINT
    if not os.path.isdir(checkpoint):
        print(f"WARNING: {checkpoint} not found, falling back to ./best", flush=True)
        checkpoint = "./best"

    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading policy from {checkpoint}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16).to(DEVICE)
    model.gradient_checkpointing_enable()

    train_data = load_gsm8k(split="train")
    random.shuffle(train_data)

    # Phase 1: Profile problems (load cached or compute fresh)
    if os.path.exists(PROFILE_FILE):
        print(f"Loading cached profile from {PROFILE_FILE}...", flush=True)
        with open(PROFILE_FILE) as f:
            profile_data = json.load(f)
        # Reconstruct profiled list using question text as key
        q_to_acc = {d["question"]: d["accuracy"] for d in profile_data}
        profiled = []
        for idx, ex in enumerate(train_data):
            if ex["question"] in q_to_acc:
                profiled.append((idx, ex, q_to_acc[ex["question"]]))
        print(f"  Loaded {len(profiled)} cached profiles", flush=True)
    else:
        print(f"=== PHASE 1: Profiling {PROFILE_N} problems (K={PROFILE_K}) ===", flush=True)
        profiled = fast_profile(model, tok, train_data,
                               k=PROFILE_K, max_n=PROFILE_N,
                               gen_len=PROFILE_GEN_LEN, budget_s=PROFILE_BUDGET)
        # Save profile
        profile_data = [{"question": ex["question"], "accuracy": acc}
                        for _, ex, acc in profiled]
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile_data, f)
        print(f"  Saved profile to {PROFILE_FILE}", flush=True)

    sweet = [(idx, ex) for idx, ex, acc in profiled if SWEET_LO <= acc <= SWEET_HI]
    easy = [(idx, ex) for idx, ex, acc in profiled if acc > SWEET_HI]
    hard = [(idx, ex) for idx, ex, acc in profiled if acc < SWEET_LO]

    profile_time = time.time() - t0
    print(f"Profile done in {profile_time:.0f}s", flush=True)
    print(f"  Sweet spot ({SWEET_LO}-{SWEET_HI}): {len(sweet)}", flush=True)
    print(f"  Easy (>{SWEET_HI}): {len(easy)}, Hard (<{SWEET_LO}): {len(hard)}", flush=True)

    if len(sweet) < 15:
        print("WARNING: Too few sweet-spot problems, expanding range", flush=True)
        sweet = [(idx, ex) for idx, ex, acc in profiled if 0.05 <= acc <= 0.95]
        print(f"  Expanded sweet spot: {len(sweet)}", flush=True)

    remaining_budget = BUDGET_MIN * 60 - (time.time() - t0)
    print(f"  Remaining budget: {remaining_budget:.0f}s", flush=True)

    if remaining_budget < 120:
        print(f"ERROR: Not enough time for training ({remaining_budget:.0f}s)", flush=True)
        for save_dir in ["./checkpoints/exp032", "./checkpoints/latest"]:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
        return

    # Phase 2: GRPO training ONLY on sweet-spot problems (Manual GD — zero optimizer memory)
    print(f"=== PHASE 2: GRPO (manual GD) on {len(sweet)} sweet-spot problems ===", flush=True)

    random.shuffle(sweet)

    total_correct = 0
    total_samples = 0
    signal_steps = 0
    data_idx = 0

    for step in range(MAX_STEPS):
        elapsed = time.time() - t0
        if elapsed > BUDGET_MIN * 60:
            print(f"Budget limit at step {step} ({elapsed:.0f}s)", flush=True)
            break

        # Cosine LR schedule (manual)
        progress = step / max(MAX_STEPS, 1)
        import math as _math
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + _math.cos(_math.pi * progress))

        model.zero_grad()

        _, ex = sweet[data_idx % len(sweet)]
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
            if step % 25 == 0:
                acc = total_correct / max(total_samples, 1)
                print(f"  step {step}/{MAX_STEPS}  NO_SIG  r={mean_r:.1f}  "
                      f"acc={acc:.3f}  sig={signal_steps}  {elapsed:.0f}s", flush=True)
            continue

        advantages = (rewards_t - mean_r) / (std_r + 1e-8)
        signal_steps += 1

        model.train()
        step_loss = 0.0
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

                pg_loss = -(adv * policy_lp.mean()) / len(all_outputs)
                pg_loss.backward()

                step_loss += pg_loss.item()
                n_backward += 1

                del logits, log_probs, policy_lp
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                continue

            torch.cuda.empty_cache()

        del all_outputs
        torch.cuda.empty_cache()
        gc.collect()

        if n_backward > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Manual GD — zero optimizer memory
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.mul_(1.0 - lr * WEIGHT_DECAY)  # decoupled weight decay
                        p.data.add_(p.grad, alpha=-lr)         # gradient step

        if step % 25 == 0:
            acc = total_correct / max(total_samples, 1)
            elapsed_now = time.time() - t0
            sig_rate = signal_steps / max(step + 1, 1) * 100
            print(f"  step {step}/{MAX_STEPS}  loss={step_loss:.4f}  "
                  f"r={mean_r:.3f}  acc={acc:.3f}  "
                  f"lr={lr:.1e}  sig={signal_steps}({sig_rate:.0f}%)  "
                  f"{elapsed_now:.0f}s", flush=True)

    # Save
    for save_dir in ["./checkpoints/exp032", "./checkpoints/latest"]:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    elapsed = time.time() - t0
    acc = total_correct / max(total_samples, 1)
    sig_rate = signal_steps / max(step + 1, 1)
    print(f"Done. {elapsed:.0f}s, acc={acc:.3f}, "
          f"signal={signal_steps}/{step+1} ({sig_rate:.0%})",
          flush=True)


if __name__ == "__main__":
    main()
