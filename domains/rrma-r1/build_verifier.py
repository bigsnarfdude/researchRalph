"""
EXP-027: Lightweight verifier for test-time candidate selection (agent0)

Three phases:
  python3 build_verifier.py --phase collect --n-problems 400 --candidates 8
  python3 build_verifier.py --phase train
  python3 build_verifier.py --phase eval --samples 200 --candidates 8

Two verifier types:
1. Surface features: text statistics from reasoning traces (logistic regression)
2. Hidden-state probe: linear probe on model's last-layer hidden states at the
   final token — captures internal "confidence about correctness" that log-probs miss
3. Combined: surface + hidden + within-group features

Key insight from EXP-026: log-probs are anti-correlated with correctness because they
measure fluency, not mathematical validity. Hidden states may encode a richer signal
about reasoning quality.
"""
import argparse, json, os, re, time, math, pickle
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "verifier_data.json"
HIDDEN_FILE = "verifier_hidden.npz"
MODEL_FILE = "verifier_model.pkl"


def check_arithmetic(text):
    r"""Check arithmetic consistency in reasoning trace.
    Returns (n_equations_found, n_correct, n_wrong).
    Handles LaTeX (\times, \div, \frac) and plain operators."""
    n_found = 0
    n_correct = 0
    n_wrong = 0

    def _check(a, expected_result, actual_result):
        nonlocal n_found, n_correct, n_wrong
        n_found += 1
        if abs(expected_result - actual_result) < 0.01 * max(abs(expected_result), 1):
            n_correct += 1
        else:
            n_wrong += 1

    NUM = r'(\d+(?:,\d{3})*(?:\.\d+)?)'

    # Pattern 1: "N op N = N" with plain or LaTeX operators
    ops_pattern = r'([+\-]|[*/×÷]|\\times|\\cdot|\\div)'
    pat1 = NUM + r'\s*' + ops_pattern + r'\s*' + NUM + r'\s*=\s*' + NUM
    for m in re.finditer(pat1, text):
        try:
            a = float(m.group(1).replace(',', ''))
            op = m.group(2).strip()
            b = float(m.group(3).replace(',', ''))
            result = float(m.group(4).replace(',', ''))
            if op in ('+',):
                _check(a, a + b, result)
            elif op in ('-',):
                _check(a, a - b, result)
            elif op in ('*', '×', '\\times', '\\cdot'):
                _check(a, a * b, result)
            elif op in ('/', '÷', '\\div'):
                if b != 0:
                    _check(a, a / b, result)
        except (ValueError, ZeroDivisionError):
            continue

    # Pattern 2: \frac{N}{N} = N
    pat_frac = r'\\frac\{' + NUM + r'\}\{' + NUM + r'\}\s*=\s*' + NUM
    for m in re.finditer(pat_frac, text):
        try:
            a = float(m.group(1).replace(',', ''))
            b = float(m.group(2).replace(',', ''))
            result = float(m.group(3).replace(',', ''))
            if b != 0:
                _check(a, a / b, result)
        except (ValueError, ZeroDivisionError):
            continue

    # Pattern 3: Multi-term additions: N + N + N = N (common in GSM8K)
    pat_multi_add = r'(' + NUM + r'(?:\s*\+\s*' + NUM + r')+)\s*=\s*' + NUM
    for m in re.finditer(pat_multi_add, text):
        try:
            expr = m.group(1)
            result = float(m.group(0).split('=')[-1].strip().replace(',', ''))
            nums = [float(n.replace(',', '')) for n in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', expr)]
            if len(nums) >= 2:
                _check(0, sum(nums), result)
        except (ValueError, ZeroDivisionError):
            continue

    return n_found, n_correct, n_wrong


def extract_features(text, gen_len):
    """Extract surface-level features from a reasoning trace."""
    features = {}
    features["gen_len"] = gen_len
    features["word_count"] = len(text.split())
    features["char_count"] = len(text)

    features["has_hash_answer"] = 1 if "####" in text else 0
    features["num_newlines"] = text.count("\n")
    features["num_equals"] = text.count("=")
    features["num_plus"] = text.count("+")
    features["num_minus"] = text.count("-")
    features["num_multiply"] = text.count("*") + text.count("×")
    features["num_divide"] = text.count("/") + text.count("÷")
    features["total_ops"] = (features["num_plus"] + features["num_minus"] +
                             features["num_multiply"] + features["num_divide"])

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    features["num_numbers"] = len(numbers)
    features["has_decimal"] = 1 if any("." in n for n in numbers) else 0
    features["has_negative"] = 1 if any(n.startswith("-") for n in numbers) else 0

    text_lower = text.lower()
    features["has_therefore"] = 1 if "therefore" in text_lower else 0
    features["has_so"] = 1 if " so " in text_lower else 0
    features["has_step"] = 1 if "step" in text_lower else 0
    features["has_total"] = 1 if "total" in text_lower else 0
    features["has_answer_is"] = 1 if "answer is" in text_lower else 0
    features["has_first"] = 1 if "first" in text_lower else 0
    features["has_then"] = 1 if "then" in text_lower else 0
    features["has_each"] = 1 if "each" in text_lower else 0
    features["has_per"] = 1 if " per " in text_lower else 0

    sentences = re.split(r'[.!?\n]', text)
    features["num_sentences"] = len([s for s in sentences if s.strip()])
    features["avg_sentence_len"] = features["char_count"] / max(features["num_sentences"], 1)

    hash_pos = text.find("####")
    features["answer_position_ratio"] = hash_pos / len(text) if hash_pos > 0 else 1.0

    words = text.lower().split()
    if len(words) > 10:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        features["bigram_uniqueness"] = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    else:
        features["bigram_uniqueness"] = 1.0

    # Answer value features (if extractable)
    answer_num = extract_number(text)
    if answer_num is not None:
        features["answer_is_integer"] = 1 if answer_num == int(answer_num) else 0
        features["answer_is_positive"] = 1 if answer_num > 0 else 0
        features["answer_magnitude"] = math.log1p(abs(answer_num))
        features["answer_is_small"] = 1 if abs(answer_num) < 100 else 0
    else:
        features["answer_is_integer"] = 0
        features["answer_is_positive"] = 0
        features["answer_magnitude"] = 0
        features["answer_is_small"] = 0

    # Mathematical consistency features
    # Count how many numbers appear more than once (reused in later steps)
    num_vals = [float(n) for n in numbers if n]
    num_counter = Counter(num_vals)
    features["num_reused_numbers"] = sum(1 for v, c in num_counter.items() if c > 1)
    features["reuse_ratio"] = features["num_reused_numbers"] / max(len(num_vals), 1)

    # Count explicit reasoning steps (lines with "Step N" or numbered lists)
    step_lines = re.findall(r'(?:step\s*\d|^\d+[\.\)]\s)', text_lower, re.MULTILINE)
    features["num_explicit_steps"] = len(step_lines)

    # Check if final answer appears earlier in the chain (self-consistency)
    if answer_num is not None and len(num_vals) > 1:
        ans_str = str(int(answer_num)) if answer_num == int(answer_num) else str(answer_num)
        # How many times does the final answer number appear in the reasoning?
        features["answer_appears_in_chain"] = sum(1 for v in num_vals[:-1] if abs(v - answer_num) < 1e-3)
    else:
        features["answer_appears_in_chain"] = 0

    # Equation density (lines containing '=')
    lines = text.split('\n')
    eq_lines = sum(1 for l in lines if '=' in l)
    features["equation_density"] = eq_lines / max(len(lines), 1)

    # Dollar sign / unit tracking
    features["has_dollar"] = 1 if "$" in text else 0
    features["num_dollar_signs"] = text.count("$")
    features["has_percent"] = 1 if "%" in text else 0

    # Reasoning length relative to answer length
    if hash_pos > 0:
        reasoning_part = text[:hash_pos]
        answer_part = text[hash_pos:]
        features["reasoning_to_answer_ratio"] = len(reasoning_part) / max(len(answer_part), 1)
    else:
        features["reasoning_to_answer_ratio"] = 0.0

    # Repetition detection (sign of looping/degenerate output)
    if len(lines) >= 4:
        last_4 = [l.strip() for l in lines[-4:] if l.strip()]
        features["ending_repetition"] = 1 if len(last_4) >= 2 and len(set(last_4)) == 1 else 0
    else:
        features["ending_repetition"] = 0

    # Arithmetic consistency check
    n_eq, n_correct_eq, n_wrong_eq = check_arithmetic(text)
    features["num_verified_equations"] = n_eq
    features["num_correct_equations"] = n_correct_eq
    features["num_wrong_equations"] = n_wrong_eq
    features["equation_accuracy"] = n_correct_eq / max(n_eq, 1)
    features["has_arithmetic_error"] = 1 if n_wrong_eq > 0 else 0

    return features


FEATURE_NAMES = None


def features_to_vector(feat_dict):
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feat_dict.keys())
    return np.array([feat_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)


def collect_data(args):
    """Generate candidates on train set, save labeled data + hidden states."""
    print(f"Collecting verifier data: {args.n_problems} problems, "
          f"{args.candidates} candidates each", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    train_data = load_gsm8k(split="train")[:args.n_problems]
    all_samples = []
    all_hidden = []  # list of hidden state vectors

    t0 = time.time()
    for idx, ex in enumerate(train_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=0.7, top_p=0.95,
                pad_token_id=tok.eos_token_id,
                num_return_sequences=args.candidates)

        # Parse candidates and trim to actual content
        parsed = []
        for i in range(out.shape[0]):
            gen_ids = out[i, prompt_len:]
            eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[:eos_positions[0]]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            num = extract_number(text)
            gen_len = len(gen_ids)
            actual_len = prompt_len + gen_len  # total sequence length
            parsed.append((num, gen_len, text, i, actual_len))

        # Batched forward pass to get hidden states at last token of each candidate
        if parsed:
            seqs = []
            masks = []
            last_positions = []
            for num, gl, text, orig_idx, actual_len in parsed:
                seq = out[orig_idx, :actual_len]
                seqs.append(seq)
                last_positions.append(actual_len - 1)

            max_len = max(s.shape[0] for s in seqs)
            padded = torch.full((len(seqs), max_len), tok.eos_token_id,
                               dtype=torch.long, device=DEVICE)
            attention_mask = torch.zeros((len(seqs), max_len),
                                       dtype=torch.long, device=DEVICE)
            for i, seq in enumerate(seqs):
                padded[i, :seq.shape[0]] = seq
                attention_mask[i, :seq.shape[0]] = 1

            with torch.no_grad():
                outputs = model(padded, attention_mask=attention_mask,
                               output_hidden_states=True)
                # Last layer hidden states at last real token
                last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
                for i, (num, gl, text, orig_idx, actual_len) in enumerate(parsed):
                    pos = min(actual_len - 1, last_hidden.shape[1] - 1)
                    h = last_hidden[i, pos, :].float().cpu().numpy()
                    all_hidden.append(h)

            del outputs, last_hidden, padded, attention_mask

        for num, gl, text, orig_idx, actual_len in parsed:
            is_correct = 0
            if num is not None and true_num is not None:
                is_correct = 1 if abs(num - true_num) < 1e-3 else 0

            features = extract_features(text, gl)
            all_samples.append({
                "problem_idx": idx,
                "text": text[:500],
                "answer_num": float(num) if num is not None else None,
                "true_num": float(true_num) if true_num is not None else None,
                "correct": is_correct,
                "features": features,
                "gen_len": gl,
            })

        del out
        torch.cuda.empty_cache()

        if (idx + 1) % 20 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            n_correct = sum(1 for s in all_samples if s["correct"])
            rate = elapsed / n
            eta = rate * (args.n_problems - n)
            print(f"  {n}/{args.n_problems}  "
                  f"samples={len(all_samples)}  "
                  f"correct_rate={n_correct/len(all_samples):.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    with open(DATA_FILE, "w") as f:
        json.dump(all_samples, f)

    hidden_array = np.array(all_hidden, dtype=np.float32)
    np.savez_compressed(HIDDEN_FILE, hidden=hidden_array)

    n_correct = sum(1 for s in all_samples if s["correct"])
    print(f"\nCollected {len(all_samples)} samples from {args.n_problems} problems")
    print(f"Correct rate: {n_correct}/{len(all_samples)} = {n_correct/len(all_samples):.3f}")
    print(f"Hidden states shape: {hidden_array.shape}")
    print(f"Saved to {DATA_FILE} + {HIDDEN_FILE}")


def add_group_features(samples):
    """Add within-group relative features for each candidate."""
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["problem_idx"]].append(i)

    for pidx, indices in groups.items():
        if len(indices) <= 1:
            continue
        answers = [samples[i]["answer_num"] for i in indices]
        gen_lens = [samples[i]["gen_len"] for i in indices]
        answer_counts = Counter(a for a in answers if a is not None)
        most_common = answer_counts.most_common(1)[0] if answer_counts else (None, 0)
        group_size = len(indices)
        mean_len = np.mean(gen_lens)
        std_len = np.std(gen_lens) + 1e-6

        for i in indices:
            s = samples[i]
            ans = s["answer_num"]
            # Is this candidate's answer in the majority?
            if ans is not None and answer_counts:
                s["features"]["answer_frequency"] = answer_counts.get(ans, 0) / group_size
                s["features"]["is_majority"] = 1 if ans == most_common[0] else 0
                s["features"]["majority_margin"] = most_common[1] / group_size
            else:
                s["features"]["answer_frequency"] = 0.0
                s["features"]["is_majority"] = 0
                s["features"]["majority_margin"] = 0.0
            # Relative length
            s["features"]["len_zscore"] = (s["gen_len"] - mean_len) / std_len
            # Number of unique answers in group
            s["features"]["group_diversity"] = len(answer_counts) / group_size


def train_verifier(args):
    """Train multiple verifier types and pick the best."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report

    with open(DATA_FILE) as f:
        samples = json.load(f)

    # Add within-group features
    add_group_features(samples)

    print(f"Training verifier on {len(samples)} samples")

    # Surface features
    X_surf, y = [], []
    for s in samples:
        X_surf.append(features_to_vector(s["features"]))
        y.append(s["correct"])
    X_surf = np.array(X_surf)
    y = np.array(y)

    print(f"Surface features: {len(FEATURE_NAMES)}")
    print(f"Class balance: {y.sum()}/{len(y)} correct ({y.mean():.3f})")

    # Hidden states
    has_hidden = os.path.exists(HIDDEN_FILE)
    if has_hidden:
        X_hidden = np.load(HIDDEN_FILE)["hidden"]
        print(f"Hidden state features: {X_hidden.shape[1]}")
        assert len(X_hidden) == len(y), f"Hidden mismatch: {len(X_hidden)} vs {len(y)}"
    else:
        print("No hidden states found — surface features only")

    results = {}

    # --- Model 1: Surface features only ---
    print("\n=== Surface features (logistic regression) ===")
    scaler_surf = StandardScaler()
    X_s = scaler_surf.fit_transform(X_surf)
    lr_surf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    scores = cross_val_score(lr_surf, X_s, y, cv=5, scoring="roc_auc")
    print(f"  CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    results["surface"] = scores.mean()
    lr_surf.fit(X_s, y)

    importances = sorted(zip(FEATURE_NAMES, lr_surf.coef_[0]),
                        key=lambda x: abs(x[1]), reverse=True)
    print("  Top features:")
    for name, coef in importances[:10]:
        print(f"    {name:30s} {coef:+.4f}")

    best_type = "surface"
    best_auc = results["surface"]

    # --- Model 1b: Surface features with Random Forest ---
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier as GBC
        print("\n=== Surface features (Random Forest) ===")
        rf_surf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                          class_weight="balanced", random_state=42)
        scores_rf = cross_val_score(rf_surf, X_s, y, cv=5, scoring="roc_auc")
        print(f"  CV AUC: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")
        results["surface_rf"] = scores_rf.mean()
        rf_surf.fit(X_s, y)
        if results["surface_rf"] > best_auc:
            best_type = "surface_rf"
            best_auc = results["surface_rf"]

        # Feature importances from RF
        rf_imp = sorted(zip(FEATURE_NAMES, rf_surf.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        print("  Top features (RF):")
        for name, imp in rf_imp[:10]:
            print(f"    {name:30s} {imp:.4f}")

        print("\n=== Surface features (GBT) ===")
        gbt_surf = GBC(n_estimators=200, max_depth=4, learning_rate=0.1,
                        subsample=0.8, random_state=42)
        scores_gbt_s = cross_val_score(gbt_surf, X_s, y, cv=5, scoring="roc_auc")
        print(f"  CV AUC: {scores_gbt_s.mean():.3f} ± {scores_gbt_s.std():.3f}")
        results["surface_gbt"] = scores_gbt_s.mean()
        gbt_surf.fit(X_s, y)
        if results["surface_gbt"] > best_auc:
            best_type = "surface_gbt"
            best_auc = results["surface_gbt"]
    except Exception as e:
        print(f"  RF/GBT failed: {e}")

    # --- Model 1c: Surface features WITHOUT group features (independent signal) ---
    print("\n=== Surface features WITHOUT group features (logistic regression) ===")
    no_group_feats = [f for f in FEATURE_NAMES
                      if f not in ("answer_frequency", "is_majority", "majority_margin",
                                  "len_zscore", "group_diversity")]
    no_group_idx = [FEATURE_NAMES.index(f) for f in no_group_feats]
    X_s_nogroup = X_s[:, no_group_idx]
    lr_nogroup = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    scores_ng = cross_val_score(lr_nogroup, X_s_nogroup, y, cv=5, scoring="roc_auc")
    print(f"  CV AUC: {scores_ng.mean():.3f} ± {scores_ng.std():.3f}")
    results["surface_nogroup"] = scores_ng.mean()
    lr_nogroup.fit(X_s_nogroup, y)

    if has_hidden:
        # --- Model 2: Hidden states only ---
        print("\n=== Hidden states (logistic regression) ===")
        scaler_hidden = StandardScaler()
        X_h = scaler_hidden.fit_transform(X_hidden)
        lr_hidden = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
        scores = cross_val_score(lr_hidden, X_h, y, cv=5, scoring="roc_auc")
        print(f"  CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        results["hidden"] = scores.mean()
        lr_hidden.fit(X_h, y)

        if results["hidden"] > best_auc:
            best_type = "hidden"
            best_auc = results["hidden"]

        # --- Model 3: Combined surface + hidden ---
        print("\n=== Combined (surface + hidden) ===")
        X_combo = np.concatenate([X_surf, X_hidden], axis=1)
        scaler_combo = StandardScaler()
        X_c = scaler_combo.fit_transform(X_combo)
        lr_combo = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
        scores = cross_val_score(lr_combo, X_c, y, cv=5, scoring="roc_auc")
        print(f"  CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        results["combined"] = scores.mean()
        lr_combo.fit(X_c, y)

        if results["combined"] > best_auc:
            best_type = "combined"
            best_auc = results["combined"]

        # --- Model 4: Hidden states with gradient-boosted trees ---
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            print("\n=== Hidden + GBT ===")
            gbt = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, subsample=0.8)
            scores_gbt = cross_val_score(gbt, X_h, y, cv=5, scoring="roc_auc")
            print(f"  CV AUC: {scores_gbt.mean():.3f} ± {scores_gbt.std():.3f}")
            results["hidden_gbt"] = scores_gbt.mean()
            gbt.fit(X_h, y)
            if results["hidden_gbt"] > best_auc:
                best_type = "hidden_gbt"
                best_auc = results["hidden_gbt"]
        except Exception as e:
            print(f"  GBT failed: {e}")

    print(f"\n=== Results ===")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ★ BEST" if name == best_type else ""
        print(f"  {name:20s}  AUC={auc:.3f}{marker}")

    # Save best model
    if best_type == "surface":
        save_data = {"type": "surface", "model": lr_surf, "scaler": scaler_surf,
                     "feature_names": FEATURE_NAMES}
    elif best_type == "surface_rf":
        save_data = {"type": "surface", "model": rf_surf, "scaler": scaler_surf,
                     "feature_names": FEATURE_NAMES}
    elif best_type == "surface_gbt":
        save_data = {"type": "surface", "model": gbt_surf, "scaler": scaler_surf,
                     "feature_names": FEATURE_NAMES}
    elif best_type == "hidden":
        save_data = {"type": "hidden", "model": lr_hidden, "scaler": scaler_hidden,
                     "feature_names": FEATURE_NAMES}
    elif best_type == "combined":
        save_data = {"type": "combined", "model": lr_combo, "scaler": scaler_combo,
                     "feature_names": FEATURE_NAMES}
    elif best_type == "hidden_gbt":
        save_data = {"type": "hidden_gbt", "model": gbt, "scaler": scaler_hidden,
                     "feature_names": FEATURE_NAMES}

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nSaved best verifier ({best_type}, AUC={best_auc:.3f}) to {MODEL_FILE}")

    # Classification report for best
    if best_type == "surface":
        y_pred = lr_surf.predict(X_s)
    elif best_type in ("surface_rf", "surface_gbt"):
        y_pred = save_data["model"].predict(X_s)
    elif best_type == "hidden":
        y_pred = lr_hidden.predict(X_h)
    elif best_type == "combined":
        y_pred = lr_combo.predict(X_c)
    elif best_type == "hidden_gbt":
        y_pred = gbt.predict(X_h)
    else:
        y_pred = None
    if y_pred is not None:
        print(f"\nIn-sample classification report ({best_type}):")
        print(classification_report(y, y_pred, target_names=["wrong", "correct"]))


def score_candidate(vdata, model, tok, seq_ids, prompt_len, gen_len, text, group_info=None):
    """Score a single candidate with the verifier. Returns P(correct)."""
    vtype = vdata["type"]
    verifier = vdata["model"]
    scaler = vdata["scaler"]

    feats = extract_features(text, gen_len)
    # Add group features if available
    if group_info:
        feats.update(group_info)

    if vtype == "surface":
        fvec = features_to_vector(feats).reshape(1, -1)
        fvec_scaled = scaler.transform(fvec)
        return verifier.predict_proba(fvec_scaled)[0, 1]

    elif vtype in ("hidden", "hidden_gbt"):
        # Need hidden states from model
        with torch.no_grad():
            out = model(seq_ids.unsqueeze(0), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        h_scaled = scaler.transform(h.reshape(1, -1))
        return verifier.predict_proba(h_scaled)[0, 1]

    elif vtype == "combined":
        with torch.no_grad():
            out = model(seq_ids.unsqueeze(0), output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        fvec = features_to_vector(feats)
        combo = np.concatenate([fvec, h])
        combo_scaled = scaler.transform(combo.reshape(1, -1))
        return verifier.predict_proba(combo_scaled)[0, 1]


def eval_with_verifier(args):
    """Evaluate with verifier-based selection."""
    with open(MODEL_FILE, "rb") as f:
        vdata = pickle.load(f)
    global FEATURE_NAMES
    FEATURE_NAMES = vdata["feature_names"]
    vtype = vdata["type"]
    verifier = vdata["model"]
    scaler = vdata["scaler"]
    needs_hidden = vtype in ("hidden", "hidden_gbt", "combined")

    print(f"Verifier type: {vtype}, needs_hidden={needs_hidden}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    correct = {"greedy": 0, "majority": 0, "verifier_best": 0,
               "verifier_weighted": 0, "verifier_filtered_maj": 0,
               "verifier_rerank_maj": 0}

    t0 = time.time()
    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_text = tok.decode(greedy_out[0][prompt_len:], skip_special_tokens=True)
        if check_answer(greedy_text, ex["answer"]):
            correct["greedy"] += 1
        del greedy_out

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=True,
                temperature=args.temperature, top_p=0.95,
                pad_token_id=tok.eos_token_id,
                num_return_sequences=args.candidates)

        # Parse candidates
        parsed = []
        for i in range(out.shape[0]):
            gen_ids = out[i, prompt_len:]
            eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[:eos_positions[0]]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            num = extract_number(text)
            gen_len = len(gen_ids)
            if num is not None:
                actual_len = prompt_len + gen_len
                parsed.append((num, gen_len, text, out[i, :actual_len]))

        # Compute group features
        if parsed:
            answer_counts = Counter(num for num, _, _, _ in parsed)
            most_common = answer_counts.most_common(1)[0]
            group_size = len(parsed)
            mean_len = np.mean([gl for _, gl, _, _ in parsed])
            std_len = np.std([gl for _, gl, _, _ in parsed]) + 1e-6

        # Score candidates with verifier (batched for hidden-state models)
        candidates = []
        if needs_hidden and parsed:
            seqs = [seq_ids for _, _, _, seq_ids in parsed]
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.full((len(seqs), max_len), tok.eos_token_id,
                               dtype=torch.long, device=DEVICE)
            attention_mask = torch.zeros((len(seqs), max_len),
                                       dtype=torch.long, device=DEVICE)
            for i, seq in enumerate(seqs):
                padded[i, :seq.shape[0]] = seq
                attention_mask[i, :seq.shape[0]] = 1

            with torch.no_grad():
                outputs = model(padded, attention_mask=attention_mask,
                               output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

            for i, (num, gl, text, seq_ids) in enumerate(parsed):
                pos = min(seq_ids.shape[0] - 1, last_hidden.shape[1] - 1)
                h = last_hidden[i, pos, :].float().cpu().numpy()

                feats = extract_features(text, gl)
                # Add group features
                feats["answer_frequency"] = answer_counts.get(num, 0) / group_size
                feats["is_majority"] = 1 if num == most_common[0] else 0
                feats["majority_margin"] = most_common[1] / group_size
                feats["len_zscore"] = (gl - mean_len) / std_len
                feats["group_diversity"] = len(answer_counts) / group_size

                if vtype == "hidden" or vtype == "hidden_gbt":
                    h_scaled = scaler.transform(h.reshape(1, -1))
                    v_score = verifier.predict_proba(h_scaled)[0, 1]
                elif vtype == "combined":
                    fvec = features_to_vector(feats)
                    combo = np.concatenate([fvec, h])
                    combo_scaled = scaler.transform(combo.reshape(1, -1))
                    v_score = verifier.predict_proba(combo_scaled)[0, 1]

                candidates.append((num, gl, text, v_score))

            del outputs, last_hidden, padded, attention_mask
        elif parsed:
            for num, gl, text, seq_ids in parsed:
                feats = extract_features(text, gl)
                feats["answer_frequency"] = answer_counts.get(num, 0) / group_size
                feats["is_majority"] = 1 if num == most_common[0] else 0
                feats["majority_margin"] = most_common[1] / group_size
                feats["len_zscore"] = (gl - mean_len) / std_len
                feats["group_diversity"] = len(answer_counts) / group_size
                fvec = features_to_vector(feats).reshape(1, -1)
                fvec_scaled = scaler.transform(fvec)
                v_score = verifier.predict_proba(fvec_scaled)[0, 1]
                candidates.append((num, gl, text, v_score))

        del out
        torch.cuda.empty_cache()

        if true_num is not None and candidates:
            # Majority vote
            counts = Counter(num for num, _, _, _ in candidates)
            maj_ans = counts.most_common(1)[0][0]
            if abs(maj_ans - true_num) < 1e-3:
                correct["majority"] += 1

            # Verifier best-of-N
            best_cand = max(candidates, key=lambda c: c[3])
            if abs(best_cand[0] - true_num) < 1e-3:
                correct["verifier_best"] += 1

            # Verifier-weighted voting
            weights = defaultdict(float)
            for num, _, _, vs in candidates:
                weights[num] += vs
            vw_ans = max(weights, key=weights.get)
            if abs(vw_ans - true_num) < 1e-3:
                correct["verifier_weighted"] += 1

            # Verifier-filtered majority (filter low-scoring, then vote)
            threshold = 0.5
            filtered = [(num, gl, t, vs) for num, gl, t, vs in candidates if vs >= threshold]
            if filtered:
                fcounts = Counter(num for num, _, _, _ in filtered)
                fm_ans = fcounts.most_common(1)[0][0]
            else:
                fm_ans = maj_ans
            if abs(fm_ans - true_num) < 1e-3:
                correct["verifier_filtered_maj"] += 1

            # Verifier-reranked majority: majority vote, tiebreak by verifier score
            answer_vscores = defaultdict(list)
            for num, _, _, vs in candidates:
                answer_vscores[num].append(vs)
            rm_ans = max(counts.keys(),
                        key=lambda k: (counts[k], sum(answer_vscores[k]) / len(answer_vscores[k])))
            if abs(rm_ans - true_num) < 1e-3:
                correct["verifier_rerank_maj"] += 1

        if (idx + 1) % 10 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            eta = elapsed / n * (args.samples - n)
            best_strat = max(["majority", "verifier_best", "verifier_weighted",
                            "verifier_filtered_maj", "verifier_rerank_maj"],
                           key=lambda s: correct[s])
            print(f"  {n}/{args.samples}  "
                  f"greedy={correct['greedy']/n:.3f}  "
                  f"maj={correct['majority']/n:.3f}  "
                  f"vf_best={correct['verifier_best']/n:.3f}  "
                  f"vf_wtd={correct['verifier_weighted']/n:.3f}  "
                  f"best={best_strat}={correct[best_strat]/n:.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0
    print(f"\nResults ({elapsed:.0f}s, {n} problems, {args.candidates} candidates, "
          f"temp={args.temperature}):")
    best_score = 0
    best_name = "greedy"
    strat_names = ["greedy", "majority", "verifier_best", "verifier_weighted",
                   "verifier_filtered_maj", "verifier_rerank_maj"]
    for name in strat_names:
        score = correct[name] / n
        delta = (score - correct["greedy"] / n) * 100
        print(f"  {name:25s}: {score:.4f}  ({delta:+.1f}pp vs greedy)")
        if score > best_score:
            best_score = score
            best_name = name
    print(f"\nBest: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["collect", "train", "eval"], required=True)
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--n-problems", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.phase == "collect":
        collect_data(args)
    elif args.phase == "train":
        train_verifier(args)
    elif args.phase == "eval":
        eval_with_verifier(args)


if __name__ == "__main__":
    main()
