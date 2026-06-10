#!/usr/bin/env python3
"""
gen_proofs_v8.py — Compiler-in-the-loop proof generator (agent1).

Key innovation: For each unsolved problem, try tactics INDIVIDUALLY against
the Lean compiler. This avoids cascade timeout where a 60-tactic cascade
times out even though tactic #47 would have worked.

Strategy:
1. Start from exp014 baseline (preserves handcrafted proofs)
2. Check which problems compile
3. For failures: generate goal-conditioned tactics and test each one
4. Two tiers: fast tactics (30s) then slow tactics (90s)
"""
import os, re, sys, subprocess, shutil

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ELAN_PATH = os.path.expanduser("~/.elan/bin")

method = sys.argv[1] if len(sys.argv) > 1 else "exp_v8"
out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)

FAST_TIMEOUT = 30
SLOW_TIMEOUT = 90

# ── Helpers ──

def extract_hypotheses(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', content)

def extract_subst_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', content)

def extract_forall_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', content)

def get_goal(content):
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    return m.group(1).strip() if m else ""

def get_var_names(content):
    all_vars = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', content)
    return [v for v, t in all_vars if not v.startswith('h')]

def check_lean(filepath, timeout=FAST_TIMEOUT):
    try:
        env = os.environ.copy()
        env["PATH"] = ELAN_PATH + ":" + env.get("PATH", "")
        result = subprocess.run(
            ["lake", "env", "lean", filepath],
            capture_output=True, text=True, timeout=timeout,
            cwd=MINIF2F_DIR, env=env
        )
        if result.returncode == 0:
            return True, ""
        err_lines = [l for l in result.stderr.split('\n') if 'error' in l.lower()]
        return False, (err_lines[0] if err_lines else result.stderr[:300])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

def make_lean_file(orig_content, proof_body):
    new = orig_content.replace("by sorry", f"by\n{proof_body}")
    new = new.replace("set_option maxHeartbeats 0", "set_option maxHeartbeats 8000000")
    return new

def try_tactic(orig_content, tactic, filepath, timeout=FAST_TIMEOUT):
    body = f"  {tactic}"
    content = make_lean_file(orig_content, body)
    with open(filepath, 'w') as f:
        f.write(content)
    return check_lean(filepath, timeout=timeout)


# ── Tactic generation ──

def generate_tactics(content, problem):
    """Generate list of (tactic_string, is_slow) tuples, goal-conditioned."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)
    goal = get_goal(content)
    var_names = get_var_names(content)
    h_list = ", ".join(hyps) if hyps else ""

    has_and = "∧" in goal
    has_or = "∨" in goal
    has_exists = "∃" in goal
    has_iff = "↔" in goal
    has_le = any(op in goal for op in ["≤", "≥", "<", ">"])
    has_eq = "=" in goal and "≠" not in goal
    has_complex = "ℂ" in content or "Complex" in content
    has_nat = ": ℕ)" in content or ": ℕ " in content
    has_real = ": ℝ" in content
    has_rat = ": ℚ" in content
    has_div = "/" in goal
    has_mod = "%" in content
    has_pow = "^" in content
    has_finset = "Finset" in content or "∑" in content or "∏" in content
    has_dvd = "∣" in goal
    has_sqrt = "sqrt" in content.lower()
    has_prime = "Prime" in content
    has_gcd = "gcd" in content or "lcm" in content
    has_abs = "|" in goal or "abs" in content
    has_induction = bool(re.search(r'\(n\s*:\s*ℕ\)', content))
    has_neg = "¬" in goal
    no_hyps = len(hyps) == 0

    fast = []
    slow = []

    # ── Universal ──
    fast += ["omega", "norm_num", "ring", "linarith", "nlinarith",
             "simp", "simp_all", "decide", "positivity"]

    # ── Complex ──
    if has_complex:
        fast += ["ring_nf; norm_num", "norm_num [Complex.ext_iff, Complex.I_sq]"]
        if subst_h:
            fast += ["subst_vars; ring",
                     "subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]"]
            if has_and:
                fast += ["constructor <;> (subst_vars; ring)"]
        if hyps:
            fast.append(f"simp only [{h_list}]; ring")
            for c in [1, -1, 2, -2, 4, -4, 9, -9, 16, -16, 25, 49]:
                fast.append(f"ring_nf; linear_combination {c} * Complex.I_sq")

    # ── Subst ──
    if subst_h and not has_complex:
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith", "simp_all"]:
            fast.append(f"subst_vars; {closer}")
        if has_and:
            for closer in ["norm_num", "omega", "ring", "linarith"]:
                fast.append(f"constructor <;> (subst_vars; {closer})")

    # ── Forall / function def ──
    if forall_h:
        all_simp = ", ".join(forall_h + subst_h)
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            fast.append(f"simp only [{all_simp}] at *; {closer}")
        if has_div:
            fast += [f"simp only [{all_simp}] at *; field_simp; ring",
                     f"simp only [{all_simp}] at *; field_simp; linarith"]
        if has_and:
            fast.append(f"simp only [{all_simp}] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)")
        if has_finset:
            fast += [f"simp only [{all_simp}]; norm_num",
                     f"simp only [{all_simp}]; native_decide"]
        # Specialize at values
        for h in forall_h:
            for val in ["0", "1", "2", "-1", "3"]:
                fast.append(f"have := {h} {val}; linarith")
                fast.append(f"have := {h} {val}; omega")
                fast.append(f"have := {h} {val}; norm_num at this ⊢; linarith")

    # ── Conjunction ──
    if has_and:
        closers = ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp"]
        if hyps:
            for c in closers:
                fast.append(f"constructor <;> {c}")
            fast += [f"constructor <;> linarith [{h_list}]",
                     f"constructor <;> nlinarith [{h_list}]"]
            if has_le and has_pow:
                fast.append(f"constructor <;> nlinarith [sq_nonneg _, {h_list}]")
        else:
            for c in closers:
                fast.append(f"constructor <;> {c}")

    # ── Disjunction ──
    if has_or:
        for side in ["left", "right"]:
            for c in ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp"]:
                fast.append(f"{side}; {c}")
            if hyps:
                fast += [f"{side}; linarith [{h_list}]",
                         f"{side}; nlinarith [{h_list}]"]

    # ── Existential ──
    if has_exists:
        witnesses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 25,
                     27, 30, 32, 36, 48, 50, 60, 64, 72, 81, 96, 100, -1, -2, -3, -4, -5]
        for w in witnesses:
            for c in ["omega", "norm_num", "ring", "simp"]:
                fast.append(f"exact ⟨{w}, by {c}⟩")
                fast.append(f"refine ⟨{w}, ?_⟩; {c}")

    # ── Iff ──
    if has_iff:
        fast += ["constructor <;> intro <;> omega",
                 "constructor <;> intro <;> linarith",
                 "constructor <;> intro <;> simp_all",
                 "constructor <;> (intro; simp_all)"]

    # ── Negation ──
    if has_neg:
        fast += ["intro h; omega", "intro h; linarith", "intro h; simp_all",
                 "intro h; norm_num at h"]
        if hyps:
            fast += [f"intro h; linarith [{h_list}, h]",
                     f"intro h; nlinarith [{h_list}, h]"]

    # ── Field division ──
    if has_div:
        if hyps:
            fast += [f"field_simp; linarith [{h_list}]",
                     f"field_simp; nlinarith [{h_list}]",
                     f"field_simp; ring_nf; linarith [{h_list}]",
                     f"field_simp; ring_nf; nlinarith [{h_list}]"]
        fast += ["field_simp; ring", "field_simp; linarith",
                 "field_simp; norm_num", "field_simp; nlinarith"]

    # ── Inequality with SOS ──
    if has_le:
        if hyps:
            fast += [f"linarith [{h_list}]", f"nlinarith [{h_list}]"]
            for v in var_names[:4]:
                fast.append(f"nlinarith [sq_nonneg {v}, {h_list}]")
                fast.append(f"nlinarith [sq_nonneg ({v} - 1), {h_list}]")
                for v2 in var_names[:4]:
                    if v < v2:
                        fast.append(f"nlinarith [sq_nonneg ({v} - {v2}), {h_list}]")
                        fast.append(f"nlinarith [sq_nonneg ({v} + {v2}), {h_list}]")
        fast += ["positivity", "linarith", "nlinarith"]
        if has_nat: fast.append("omega")

    # ── Divisibility ──
    if has_dvd:
        if hyps:
            fast.append(f"simp only [{h_list}]; omega")
        fast += ["omega", "norm_num", "simp; omega"]

    # ── Hyp-based simp ──
    if hyps and not forall_h and not subst_h:
        for c in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            fast.append(f"simp only [{h_list}]; {c}")
            fast.append(f"simp only [{h_list}] at *; {c}")
        fast += [f"linarith [{h_list}]", f"nlinarith [{h_list}]"]

    # ── linear_combination ──
    if hyps and len(hyps) <= 6 and (has_eq or has_and) and not has_finset:
        for h in hyps:
            fast.append(f"linear_combination {h}")
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            for a, b in [(1,1),(1,-1),(-1,1),(2,-1),(-1,2),(3,-2),(-2,3),
                         (3,-1),(-1,3),(2,1),(1,2),(4,-3),(-3,4),(5,-4)]:
                parts = []
                if a == 1: parts.append(h0)
                elif a == -1: parts.append(f"- {h0}")
                elif a != 0: parts.append(f"{a} * {h0}")
                if b == 1: parts.append(h1)
                elif b == -1: parts.append(f"- {h1}")
                elif b != 0: parts.append(f"{b} * {h1}")
                if parts:
                    fast.append(f"linear_combination {' + '.join(parts)}")
            if has_and:
                fast.append(f"constructor\n  · linear_combination {h0}\n  · linear_combination {h1}")

    # ── Decidable ──
    if no_hyps or has_gcd or has_prime:
        fast.insert(0, "norm_num")
        if not has_real and not has_complex:
            fast.append("native_decide")

    if has_finset:
        fast.insert(0, "native_decide")
        fast += ["simp; norm_num", "simp; native_decide"]

    if has_mod:
        fast += ["omega", "native_decide"]

    # ── Cast ──
    if has_nat and (has_real or has_rat):
        fast += ["push_cast; ring", "push_cast; linarith", "push_cast; norm_num",
                 "push_cast; omega", "push_cast; nlinarith",
                 "exact_mod_cast (by omega)", "exact_mod_cast (by norm_num)",
                 "norm_cast; omega", "norm_cast; ring"]

    # ── Sqrt ──
    if has_sqrt:
        fast += ["norm_num [Real.sqrt_lt', Real.lt_sqrt]"]
        slow += ["nlinarith [Real.sq_sqrt (by positivity : (0:ℝ) ≤ _)]"]

    # ── Abs ──
    if has_abs:
        fast += ["simp [abs_le]; constructor <;> linarith",
                 "simp [abs_lt]; constructor <;> linarith"]
        if hyps:
            fast += [f"rw [abs_le]; constructor <;> linarith [{h_list}]",
                     f"simp only [abs_le]; constructor <;> linarith [{h_list}]"]

    # ── Induction ──
    if has_induction and (has_dvd or has_eq or has_le):
        ind_vars = [v for v in var_names if v in ['n', 'k', 'm']]
        for v in ind_vars[:2]:
            for closer in ["omega", "ring", "simp", "norm_num", "linarith", "ring_nf; omega"]:
                slow.append(f"induction {v} with\n  | zero => {closer}\n  | succ k ih => {closer}")
                slow.append(f"induction {v} with\n  | zero => {closer}\n  | succ k ih => simp_all; {closer}")
                slow.append(f"induction {v} with\n  | zero => {closer}\n  | succ k ih => ring_nf; {closer}")

    # ── Simp fallbacks ──
    for c in ["", "; ring", "; omega", "; linarith", "; nlinarith", "; norm_num"]:
        fast.append(f"simp_all{c}")

    fast += ["simp; ring", "simp; omega", "simp; norm_num",
             "push_cast; ring", "ring_nf; norm_num", "ring_nf; omega",
             "norm_num; omega"]

    slow += ["aesop"]

    # Deduplicate
    seen = set()
    result = []
    for t in fast:
        if t not in seen:
            seen.add(t)
            result.append((t, False))
    for t in slow:
        if t not in seen:
            seen.add(t)
            result.append((t, True))
    return result


# ── Main ──

def main():
    problems = sorted([f[:-5] for f in os.listdir(VALID_DIR) if f.endswith(".lean")])
    print(f"[v8] {len(problems)} problems, method={method}")

    # Phase 1: Copy exp014 baseline
    exp014_dir = os.path.join(DOMAIN_DIR, "attempts", "exp014")
    for p in problems:
        src = os.path.join(exp014_dir, f"{p}.lean")
        dst = os.path.join(out_dir, f"{p}.lean")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            shutil.copy2(os.path.join(VALID_DIR, f"{p}.lean"), dst)

    # Phase 2: Check baseline
    print("\n[Phase 2] Checking baseline...")
    passing = set()
    failing = {}
    for i, p in enumerate(problems):
        fp = os.path.join(out_dir, f"{p}.lean")
        ok, err = check_lean(fp, timeout=SLOW_TIMEOUT)
        if ok:
            passing.add(p)
        else:
            failing[p] = err
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(problems)}: {len(passing)} pass, {len(failing)} fail")
    print(f"\n  Baseline: {len(passing)}/{len(problems)}")

    # Phase 3: Individual tactic testing for unsolved
    print(f"\n[Phase 3] Testing tactics on {len(failing)} unsolved...")
    newly_solved = 0
    for pi, p in enumerate(sorted(failing.keys())):
        orig_path = os.path.join(VALID_DIR, f"{p}.lean")
        if not os.path.exists(orig_path):
            continue
        with open(orig_path) as f:
            orig_content = f.read()

        fp = os.path.join(out_dir, f"{p}.lean")
        tactics = generate_tactics(orig_content, p)

        solved = False
        for ti, (tactic, is_slow) in enumerate(tactics):
            timeout = SLOW_TIMEOUT if is_slow else FAST_TIMEOUT
            ok, err = try_tactic(orig_content, tactic, fp, timeout=timeout)
            if ok:
                print(f"  SOLVED [{newly_solved+1}]: {p} ← {tactic[:70]}")
                passing.add(p)
                newly_solved += 1
                solved = True
                break

        if not solved:
            # Restore exp014 version
            src = os.path.join(exp014_dir, f"{p}.lean")
            if os.path.exists(src):
                shutil.copy2(src, fp)

        if (pi + 1) % 10 == 0:
            print(f"  Progress: {pi+1}/{len(failing)}, +{newly_solved} new")

    total = len(problems)
    score = len(passing) / total
    print(f"\n{'='*60}")
    print(f"RESULT: {len(passing)}/{total} = {score:.4f}")
    print(f"  Baseline: {len(passing) - newly_solved}")
    print(f"  New solves: {newly_solved}")


if __name__ == "__main__":
    main()
