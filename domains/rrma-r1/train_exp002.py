"""
EXP-023: Curriculum GRPO + Proximity Reward (agent0)

Key innovation: PROXIMITY-BASED REWARD creates within-group variance where
format-based shaping (EXP-021a) failed.

Reward function:
- 1.0: correct answer
- 0.1 + 0.4 * proximity: wrong answer, where proximity = 1/(1 + |log(pred/true)|)
  This means answers close to the right number get ~0.4, far answers get ~0.1
- 0.0: no parseable answer

Why this works when format-based doesn't: Within a group of 8 wrong answers,
different samples produce DIFFERENT wrong numbers. Their proximity to the correct
answer varies, creating reward variance → gradient signal even in all-wrong groups.

Also uses curriculum filtering (online, from EXP-021a).
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import check_answer, extract_number
import random, time, gc, os, math
from collections import defaultdict

CHECKPOINT  = "./best"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS   = 400
GROUP_SIZE  = 8
GEN_BATCH   = 2
LR          = 5e-6
LR_MIN      = 5e-7
MAX_GEN_LEN = 384
MAX_PROMPT_LEN = 200
WEIGHT_DECAY = 0.02
TEMPERATURE = 0.9
SEED        = 523
CLIP_GRAD   = 0.5
BUDGET_MIN  = 32

# Curriculum
BURN_IN_STEPS = 40
MIN_DIFFICULTY = 0.20
MAX_DIFFICULTY = 0.85
MIN_OBS = 4

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
random.seed(SEED)
torch.manual_seed(SEED)


def proximity_reward(prediction: str, ground_truth: str) -> float:
    """
    Proximity-based reward: varies with HOW CLOSE the wrong answer is.
    Creates within-group variance even when all answers are wrong.
    """
    if check_answer(prediction, ground_truth) == 1.0:
        return 1.0

    pred_num = extract_number(prediction)
    true_num = extract_number(ground_truth)

    if pred_num is None or true_num is None:
        return 0.0

    if true_num == 0:
        proximity = 1.0 / (1.0 + abs(pred_num))
    elif pred_num == 0:
        proximity = 1.0 / (1.0 + abs(true_num))
    else:
        try:
            ratio = abs(pred_num / true_num)
            if ratio <= 0:
                proximity = 0.0
            else:
                proximity = 1.0 / (1.0 + abs(math.log(ratio)))
        except (ValueError, OverflowError):
            proximity = 0.0

    return 0.1 + 0.4 * proximity


class OnlineCurriculum:
    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.order = list(range(self.n))
        random.shuffle(self.order)
        self.ptr = 0
        self.stats = defaultdict(lambda: [0, 0])

    def update(self, idx, n_correct, n_total):
        self.stats[idx][0] += n_correct
        self.stats[idx][1] += n_total

    def accuracy(self, idx):
        s = self.stats[idx]
        if s[1] < MIN_OBS:
            return None
        return s[0] / s[1]

    def in_sweet_spot(self, idx):
        acc = self.accuracy(idx)
        if acc is None:
            return True
        return MIN_DIFFICULTY <= acc <= MAX_DIFFICULTY

    def sample(self, step):
        if step < BURN_IN_STEPS:
            idx = self.order[self.ptr % self.n]
            self.ptr += 1
            return idx
        for _ in range(30):
            idx = self.order[self.ptr % self.n]
            self.ptr += 1
            if self.ptr >= self.n:
                random.shuffle(self.order)
                self.ptr = 0
            if self.in_sweet_spot(idx):
                return idx
        return self.order[self.ptr % self.n]

    def summary(self):
        known = {k: v for k, v in self.stats.items() if v[1] >= MIN_OBS}
        if not known:
            return "no tracked"
        sweet = sum(1 for v in known.values() if MIN_DIFFICULTY <= v[0]/v[1] <= MAX_DIFFICULTY)
        easy = sum(1 for v in known.values() if v[0]/v[1] > MAX_DIFFICULTY)
        hard = sum(1 for v in known.values() if v[0]/v[1] < MIN_DIFFICULTY)
        return f"t={len(known)} s={sweet} e={easy} h={hard}"


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
    curriculum = OnlineCurriculum(train_data)

    print(f"EXP-023 Curriculum + Proximity Reward GRPO (agent0)", flush=True)
    print(f"  G={GROUP_SIZE}, steps={MAX_STEPS}, lr={LR}, wd={WEIGHT_DECAY}, "
          f"temp={TEMPERATURE}", flush=True)
    print(f"  Reward: proximity (1.0 / 0.1-0.5 / 0.0)", flush=True)
    print(f"  Curriculum: burn_in={BURN_IN_STEPS}, sweet=[{MIN_DIFFICULTY},{MAX_DIFFICULTY}]",
          flush=True)

    total_correct = 0
    total_samples = 0
    signal_steps = 0
    prox_signal_steps = 0

    for step in range(MAX_STEPS):
        elapsed = time.time() - t0
        if elapsed > BUDGET_MIN * 60:
            print(f"Budget limit at step {step} ({elapsed:.0f}s)", flush=True)
            break

        model.zero_grad()
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * step / MAX_STEPS))

        prob_idx = curriculum.sample(step)
        ex = train_data[prob_idx]

        prompt_text = format_prompt(ex["question"])
        prompt_enc = tok(prompt_text, return_tensors="pt",
                        truncation=True, max_length=MAX_PROMPT_LEN).to(DEVICE)
        prompt_len = prompt_enc["input_ids"].shape[1]

        model.eval()
        all_outputs = []
        rewards = []
        binary_rewards = []

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
                        r_prox = proximity_reward(resp, ex["answer"])
                        r_binary = check_answer(resp, ex["answer"])
                        all_outputs.append(outs[i].clone())
                        rewards.append(r_prox)
                        binary_rewards.append(r_binary)
                    del outs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

        torch.cuda.empty_cache()

        if len(rewards) < 2:
            continue

        n_binary_correct = sum(int(r) for r in binary_rewards)
        curriculum.update(prob_idx, n_binary_correct, len(binary_rewards))

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        binary_t = torch.tensor(binary_rewards, dtype=torch.float32)
        mean_r = rewards_t.mean().item()
        std_r = rewards_t.std().item()
        total_correct += n_binary_correct
        total_samples += len(rewards)

        binary_std = binary_t.std().item()
        has_prox_only_signal = (std_r > 1e-6 and binary_std < 1e-6)

        if std_r < 1e-6:
            del all_outputs
            if step % 25 == 0:
                acc = total_correct / max(total_samples, 1)
                print(f"  step {step}/{MAX_STEPS}  NO_SIG  r={mean_r:.2f}  "
                      f"acc={acc:.3f}  sig={signal_steps}(+{prox_signal_steps}prox)  "
                      f"[{curriculum.summary()}]  {elapsed:.0f}s", flush=True)
            continue

        if has_prox_only_signal:
            prox_signal_steps += 1

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
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.mul_(1.0 - lr * WEIGHT_DECAY)
                        p.data.add_(p.grad, alpha=-lr)

        if step % 25 == 0:
            acc = total_correct / max(total_samples, 1)
            elapsed_now = time.time() - t0
            print(f"  step {step}/{MAX_STEPS}  loss={step_loss:.4f}  "
                  f"r={mean_r:.3f}  acc={acc:.3f}  "
                  f"lr={lr:.1e}  sig={signal_steps}(+{prox_signal_steps}prox)  "
                  f"[{curriculum.summary()}]  {elapsed_now:.0f}s", flush=True)

    # Save
    for save_dir in ["./checkpoints/exp023", "./checkpoints/latest"]:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    elapsed = time.time() - t0
    acc = total_correct / max(total_samples, 1)
    print(f"Done. {elapsed:.0f}s, acc={acc:.3f}, signal={signal_steps}/{step+1} "
          f"(+{prox_signal_steps} prox-only)", flush=True)
    print(f"Curriculum: {curriculum.summary()}", flush=True)


if __name__ == "__main__":
    main()
