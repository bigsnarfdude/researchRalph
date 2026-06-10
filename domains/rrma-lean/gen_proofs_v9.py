#!/usr/bin/env python3
"""
gen_proofs_v9.py — Targeted compiler-in-the-loop proof generator (agent1).

Strategy: Start from exp014 baseline, then for each unsolved problem:
1. Classify the problem type
2. Generate a SHORT list of high-probability tactics for that type
3. Test each tactic individually against the Lean compiler
4. Keep first that works

Key improvements:
- rw [div_eq_iff] pattern for division goals
- native_decide with high heartbeats for Finset counting
- Targeted tactic sets per problem type (max ~30 per problem)
- positivity for nonnegativity goals
"""
import os, re, sys, subprocess, shutil

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ELAN_PATH = os.path.expanduser("~/.elan/bin")

method = sys.argv[1] if len(sys.argv) > 1 else "exp_v9"
out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)

TIMEOUT = 45  # per-tactic timeout
SLOW_TIMEOUT = 120  # for expensive tactics like native_decide on large problems

def extract_hypotheses(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', content)

def extract_subst_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', content)

def extract_forall_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', content)

def extract_neq_hyps(content):
    """Find hypotheses asserting ≠."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:[^)]*≠[^)]*\)', content)

def get_goal(content):
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    return m.group(1).strip() if m else ""

def get_var_names(content):
    all_vars = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', content)
    return [v for v, t in all_vars if not v.startswith('h')]

def check_lean(filepath, timeout=TIMEOUT):
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

def try_tactic(orig_content, tactic, filepath, timeout=TIMEOUT):
    content = make_lean_file(orig_content, f"  {tactic}")
    with open(filepath, 'w') as f:
        f.write(content)
    return check_lean(filepath, timeout=timeout)


def generate_tactics(content, problem):
    """Generate targeted tactic list as [(tactic, timeout)] pairs."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)
    neq_h = extract_neq_hyps(content)
    goal = get_goal(content)
    var_names = get_var_names(content)
    h_list = ", ".join(hyps) if hyps else ""

    # Features
    has_and = "∧" in goal
    has_or = "∨" in goal
    has_exists = "∃" in goal
    has_iff = "↔" in goal
    has_le = any(op in goal for op in ["≤", "≥", "<", ">"])
    has_eq = "=" in goal and "≠" not in goal
    has_complex = "ℂ" in content or "Complex" in content
    has_nat = ": ℕ)" in content or ": ℕ " in content
    has_int = ": ℤ)" in content
    has_real = ": ℝ" in content
    has_rat = ": ℚ" in content
    has_div = "/" in goal
    has_mod = "%" in content or "MOD" in content
    has_pow = "^" in content
    has_finset = "Finset" in content or "∑" in content or "∏" in content
    has_dvd = "∣" in goal
    has_sqrt = "sqrt" in content.lower()
    has_prime = "Prime" in content
    has_gcd = "gcd" in content or "lcm" in content
    has_abs = "abs" in content or "|" in goal
    has_card = ".card" in goal
    has_equiv = "Equiv" in content
    has_induction = bool(re.search(r'\(n\s*:\s*ℕ\)', content))
    has_neg = "¬" in goal
    has_isleast = "IsLeast" in content or "IsGreatest" in content
    no_hyps = len(hyps) == 0

    tactics = []  # (tactic_string, timeout)

    # ═══ UNIVERSAL BASICS ═══
    basics = ["omega", "norm_num", "ring", "linarith", "nlinarith",
              "simp", "simp_all", "positivity"]
    for t in basics:
        tactics.append((t, TIMEOUT))

    # ═══ NATIVE_DECIDE — for Finset, card, concrete computation ═══
    if has_finset or has_card or has_prime or has_gcd or no_hyps:
        tactics.append(("native_decide", SLOW_TIMEOUT))
        tactics.append(("decide", SLOW_TIMEOUT))

    # ═══ COMPLEX ═══
    if has_complex:
        if subst_h:
            tactics.append(("subst_vars; ring", TIMEOUT))
            tactics.append(("subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]", TIMEOUT))
            if has_and:
                tactics.append(("constructor <;> (subst_vars; ring)", TIMEOUT))
        if hyps:
            tactics.append((f"simp only [{h_list}]; ring", TIMEOUT))
            for c in [1, -1, 4, -4, 9, 16, 25, 49]:
                tactics.append((f"ring_nf; linear_combination {c} * Complex.I_sq", TIMEOUT))

    # ═══ SUBST ═══
    if subst_h and not has_complex:
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith", "simp_all"]:
            tactics.append((f"subst_vars; {closer}", TIMEOUT))
        if has_and:
            for c in ["norm_num", "omega", "ring", "linarith"]:
                tactics.append((f"constructor <;> (subst_vars; {c})", TIMEOUT))

    # ═══ FORALL / FUNCTION DEF ═══
    if forall_h:
        all_simp = ", ".join(forall_h + subst_h)
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            tactics.append((f"simp only [{all_simp}] at *; {closer}", TIMEOUT))
        if has_div:
            tactics.append((f"simp only [{all_simp}] at *; field_simp; ring", TIMEOUT))
            tactics.append((f"simp only [{all_simp}] at *; field_simp; linarith", TIMEOUT))
        if has_finset:
            tactics.append((f"simp only [{all_simp}]; norm_num", TIMEOUT))
            tactics.append((f"simp only [{all_simp}]; native_decide", SLOW_TIMEOUT))

    # ═══ CONJUNCTION ═══
    if has_and:
        for c in ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp"]:
            tactics.append((f"constructor <;> {c}", TIMEOUT))
        if hyps:
            tactics.append((f"constructor <;> linarith [{h_list}]", TIMEOUT))
            tactics.append((f"constructor <;> nlinarith [{h_list}]", TIMEOUT))

    # ═══ DISJUNCTION ═══
    if has_or:
        for side in ["left", "right"]:
            for c in ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp"]:
                tactics.append((f"{side}; {c}", TIMEOUT))
            if hyps:
                tactics.append((f"{side}; linarith [{h_list}]", TIMEOUT))
                tactics.append((f"{side}; nlinarith [{h_list}]", TIMEOUT))
                tactics.append((f"{side}; field_simp; nlinarith [{h_list}]", TIMEOUT))

    # ═══ EXISTENTIAL ═══
    if has_exists:
        for w in [0,1,2,3,4,5,6,7,8,9,10,12,16,20,25,32,50,64,100,-1,-2,-3,-4,-5]:
            for c in ["omega", "norm_num"]:
                tactics.append((f"exact ⟨{w}, by {c}⟩", TIMEOUT))
                tactics.append((f"refine ⟨{w}, ?_⟩; {c}", TIMEOUT))

    # ═══ IFF ═══
    if has_iff:
        tactics.append(("constructor <;> intro <;> omega", TIMEOUT))
        tactics.append(("constructor <;> intro <;> simp_all", TIMEOUT))
        tactics.append(("constructor <;> (intro; simp_all)", TIMEOUT))

    # ═══ DIVISION — the key new pattern ═══
    if has_div and hyps:
        # rw [div_eq_iff] pattern: clear denominators using ≠0 hyps
        neq_names = [h for h in neq_h]  # h : x ≠ 0 style
        if neq_names:
            for nh in neq_names:
                tactics.append((f"rw [div_eq_iff {nh}]; linarith", TIMEOUT))
                tactics.append((f"rw [div_eq_iff {nh}]; nlinarith [{h_list}]", TIMEOUT))
                tactics.append((f"rw [div_eq_iff {nh}]; ring", TIMEOUT))
        # Derive denominator ≠ 0 then clear
        # Pattern: have denom_ne : expr ≠ 0 := by <intro/linarith>; rw [div_eq_iff]; linarith
        for v in var_names[:4]:
            for expr in [f"{v}", f"{v} - 1", f"2 * {v}"]:
                for proof in ["intro h; omega", "intro h; linarith", "by positivity"]:
                    tactics.append((f"have hne : {expr} ≠ 0 := by {proof}\n  rw [div_eq_iff hne]\n  linarith", TIMEOUT))

        # field_simp patterns
        tactics.append(("field_simp; ring", TIMEOUT))
        tactics.append(("field_simp; linarith", TIMEOUT))
        tactics.append(("field_simp; nlinarith", TIMEOUT))
        if hyps:
            tactics.append((f"field_simp; linarith [{h_list}]", TIMEOUT))
            tactics.append((f"field_simp; nlinarith [{h_list}]", TIMEOUT))
            tactics.append((f"field_simp; ring_nf; linarith [{h_list}]", TIMEOUT))
            tactics.append((f"field_simp; ring_nf; nlinarith [{h_list}]", TIMEOUT))

        # For h₂ : expr/denom = val pattern: rewrite h₂ then use
        for h in hyps:
            tactics.append((f"rw [div_eq_iff (by positivity)] at {h}; rw [div_eq_iff (by positivity)]; nlinarith", TIMEOUT))
            tactics.append((f"field_simp at {h} ⊢; nlinarith", TIMEOUT))
            tactics.append((f"field_simp at {h} ⊢; linarith", TIMEOUT))

    # Also for division without hyps
    if has_div and not hyps:
        tactics.append(("field_simp; ring", TIMEOUT))
        tactics.append(("field_simp; norm_num", TIMEOUT))
        tactics.append(("norm_num; ring", TIMEOUT))

    # ═══ INEQUALITY with SOS ═══
    if has_le and hyps:
        tactics.append((f"linarith [{h_list}]", TIMEOUT))
        tactics.append((f"nlinarith [{h_list}]", TIMEOUT))
        for v in var_names[:4]:
            tactics.append((f"nlinarith [sq_nonneg {v}, {h_list}]", TIMEOUT))
            for v2 in var_names[:4]:
                if v < v2:
                    tactics.append((f"nlinarith [sq_nonneg ({v} - {v2}), {h_list}]", TIMEOUT))

    # ═══ DIVISIBILITY ═══
    if has_dvd:
        tactics.append(("omega", TIMEOUT))
        if hyps:
            tactics.append((f"simp only [{h_list}]; omega", TIMEOUT))

    # ═══ HYP-BASED SIMP ═══
    if hyps and not forall_h and not subst_h:
        for c in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            tactics.append((f"simp only [{h_list}]; {c}", TIMEOUT))
            tactics.append((f"simp only [{h_list}] at *; {c}", TIMEOUT))

    # ═══ LINEAR_COMBINATION ═══
    if hyps and len(hyps) <= 5 and (has_eq or has_and) and not has_finset:
        for h in hyps:
            tactics.append((f"linear_combination {h}", TIMEOUT))
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            for a, b in [(1,1),(1,-1),(-1,1),(2,-1),(-1,2),(3,-1),(-1,3),(2,1),(1,2)]:
                parts = []
                if a == 1: parts.append(h0)
                elif a == -1: parts.append(f"- {h0}")
                elif a != 0: parts.append(f"{a} * {h0}")
                if b == 1: parts.append(h1)
                elif b == -1: parts.append(f"- {h1}")
                elif b != 0: parts.append(f"{b} * {h1}")
                if parts:
                    tactics.append((f"linear_combination {' + '.join(parts)}", TIMEOUT))
            if has_and:
                tactics.append((f"constructor\n  · linear_combination {h0}\n  · linear_combination {h1}", TIMEOUT))

    # ═══ CAST ═══
    if has_nat and (has_real or has_rat):
        for c in ["ring", "linarith", "norm_num", "omega", "nlinarith"]:
            tactics.append((f"push_cast; {c}", TIMEOUT))
        tactics.append(("exact_mod_cast (by omega)", TIMEOUT))
        tactics.append(("exact_mod_cast (by norm_num)", TIMEOUT))
        tactics.append(("norm_cast; omega", TIMEOUT))

    # ═══ MOD ═══
    if has_mod:
        tactics.append(("omega", TIMEOUT))
        tactics.append(("native_decide", SLOW_TIMEOUT))

    # ═══ NEGATION ═══
    if has_neg:
        tactics.append(("intro h; omega", TIMEOUT))
        tactics.append(("intro h; linarith", TIMEOUT))
        tactics.append(("intro h; simp_all", TIMEOUT))
        if hyps:
            tactics.append((f"intro h; nlinarith [{h_list}, h]", TIMEOUT))

    # ═══ EQUIV ═══
    if has_equiv and hyps:
        # Try rewriting with the Equiv
        for h in hyps:
            tactics.append((f"simp [{h}]", TIMEOUT))
            tactics.append((f"simp only [{h}]; ring", TIMEOUT))
            tactics.append((f"simp only [{h}]; norm_num", TIMEOUT))
            tactics.append((f"simp only [{h}]; omega", TIMEOUT))

    # ═══ SQRT ═══
    if has_sqrt:
        tactics.append(("norm_num [Real.sqrt_lt', Real.lt_sqrt]", TIMEOUT))
        tactics.append(("nlinarith [Real.sq_sqrt (by positivity : (0:ℝ) ≤ _)]", SLOW_TIMEOUT))

    # ═══ ABSOLUTE VALUE ═══
    if has_abs:
        tactics.append(("simp [abs_le]; constructor <;> linarith", TIMEOUT))
        if hyps:
            tactics.append((f"rw [abs_le]; constructor <;> linarith [{h_list}]", TIMEOUT))

    # ═══ INDUCTION ═══
    if has_induction and (has_dvd or has_eq or has_le):
        ind_vars = [v for v in var_names if v in ['n', 'k', 'm']]
        for v in ind_vars[:2]:
            for closer in ["omega", "ring", "simp_all; omega", "ring_nf; omega",
                            "simp_all; ring_nf; omega"]:
                tactics.append((f"induction {v} with\n  | zero => {closer}\n  | succ k ih => {closer}", SLOW_TIMEOUT))

    # ═══ SIMP_ALL FALLBACKS ═══
    for c in ["; ring", "; omega", "; linarith", "; nlinarith", "; norm_num"]:
        tactics.append((f"simp_all{c}", TIMEOUT))

    # ═══ MISC FALLBACKS ═══
    misc = ["simp; ring", "simp; omega", "simp; norm_num",
            "push_cast; ring", "ring_nf; norm_num", "ring_nf; omega",
            "aesop"]
    for t in misc:
        tactics.append((t, SLOW_TIMEOUT if t == "aesop" else TIMEOUT))

    # Deduplicate preserving order
    seen = set()
    result = []
    for t, to in tactics:
        if t not in seen:
            seen.add(t)
            result.append((t, to))
    return result


def main():
    problems = sorted([f[:-5] for f in os.listdir(VALID_DIR) if f.endswith(".lean")])
    print(f"[v9] {len(problems)} problems, method={method}")

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
        ok, err = check_lean(fp, timeout=90)
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
        for ti, (tactic, timeout) in enumerate(tactics):
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

        if (pi + 1) % 5 == 0:
            print(f"  [{pi+1}/{len(failing)}] +{newly_solved} new")

    total = len(problems)
    score = len(passing) / total
    print(f"\n{'='*60}")
    print(f"RESULT: {len(passing)}/{total} = {score:.4f}")
    print(f"  Baseline: {len(passing) - newly_solved}")
    print(f"  New: {newly_solved}")

    # Write summary for blackboard
    print(f"\nFailing ({total - len(passing)}):")
    still_failing = [p for p in problems if p not in passing]
    for p in still_failing:
        print(f"  {p}")


if __name__ == "__main__":
    main()
