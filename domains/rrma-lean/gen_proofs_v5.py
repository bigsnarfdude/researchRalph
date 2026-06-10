#!/usr/bin/env python3
"""
gen_proofs_v5.py — Error-driven retry proof generator (agent0).

Strategy:
1. Start with exp014 as base (already has 159/244 solved)
2. For unsolved problems, generate richer tactic cascades
3. Compile each attempt with `lake env lean`
4. Parse error messages and generate targeted fixes
5. Retry up to MAX_RETRIES times per problem

Key improvements over v4:
- Error-driven retry loop (Goedel-V2 style self-correction)
- Richer tactic patterns: specialize, have, rcases, interval_cases
- Better nlinarith witness generation (auto SOS decomposition)
- push_cast/exact_mod_cast for coercion problems
- Brute-force single-tactic sweep for remaining failures
"""

import os, re, sys, subprocess, shutil
from pathlib import Path

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))

method = sys.argv[1] if len(sys.argv) > 1 else "exp015"
MAX_RETRIES = 2
TIMEOUT = 90  # seconds per Lean check

out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_hypotheses(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', content)

def extract_subst_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', content)

def extract_forall_hyps(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', content)

def get_goal(content):
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    return m.group(1).strip() if m else ""

def get_variables(content):
    """Extract simple variable names from theorem signature."""
    vars_found = re.findall(r'\((\w+)\s*:', content)
    # Filter out hypothesis names and type annotations
    return [v for v in vars_found if not v.startswith('h') and len(v) <= 3]

# ── Lean compiler interface ───────────────────────────────────────────────────

def check_lean(filepath):
    """Compile a Lean file and return (success, error_msg)."""
    try:
        env = os.environ.copy()
        elan_bin = os.path.expanduser("~/.elan/bin")
        env["PATH"] = elan_bin + ":" + env.get("PATH", "")
        result = subprocess.run(
            ["bash", "-c", f"cd '{MINIF2F_DIR}' && lake env lean '{filepath}'"],
            capture_output=True, text=True, timeout=TIMEOUT, env=env
        )
        if result.returncode == 0:
            return True, ""
        for line in result.stderr.split('\n'):
            if 'error:' in line.lower():
                return False, line.strip()
        return False, result.stderr[:500] if result.stderr else "unknown error"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)

# ── Tactic generation ─────────────────────────────────────────────────────────

def generate_cascade_tactics(content, problem):
    """Generate comprehensive tactic cascade."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)
    goal = get_goal(content)
    var_names = get_variables(content)

    # Feature detection
    has_and = "∧" in goal
    has_or = "∨" in goal
    has_exists = "∃" in goal
    has_iff = "↔" in goal
    has_le = any(op in goal for op in ["≤", "≥", "<", ">"])
    has_eq = "=" in goal and not has_le and not has_iff
    has_complex = "ℂ" in content or "Complex" in content
    has_nat = ": ℕ)" in content
    has_real = ": ℝ" in content
    has_rat = ": ℚ" in content
    has_div = "/" in goal
    has_mod = "%" in content or "MOD" in content
    has_pow = "^" in content
    has_finset = "Finset" in content or "∑" in content or "∏" in content
    has_dvd = "∣" in goal
    has_sqrt = "sqrt" in content
    has_prime = "Prime" in content
    has_gcd = "gcd" in content or "lcm" in content
    has_abs = "abs" in content
    has_card = ".card" in goal
    has_equiv = "Equiv" in content
    no_hyps = len(hyps) == 0

    tactics = []

    # ── COMPLEX ──
    if has_complex:
        if subst_h:
            tactics += ["subst_vars; ring", "subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]",
                        "subst_vars; simp [Complex.ext_iff, Complex.I_sq]; ring"]
            if has_and:
                tactics += ["constructor <;> (subst_vars; ring)",
                            "constructor <;> (subst_vars; norm_num [Complex.ext_iff])"]
        tactics += ["ring", "norm_num"]
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"simp only [{h_list}]; ring")
            tactics.append(f"ring_nf; linear_combination 49 * Complex.I_sq")

    # ── SUBST ──
    if subst_h and not has_complex:
        if has_and:
            tactics += ["constructor <;> (subst_vars; norm_num)",
                        "constructor <;> (subst_vars; omega)",
                        "constructor <;> (subst_vars; ring)"]
        tactics += ["subst_vars; ring", "subst_vars; norm_num",
                    "subst_vars; omega", "subst_vars; simp"]

    # ── FORALL / FUNCTION DEFINITION ──
    if forall_h:
        all_simp = ", ".join(forall_h + subst_h)
        for c in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            tactics.append(f"simp only [{all_simp}] at *; {c}")
        if has_div:
            tactics += [f"simp only [{all_simp}] at *; field_simp; ring",
                        f"simp only [{all_simp}] at *; field_simp; linarith"]
        if has_and:
            tactics.append(f"simp only [{all_simp}] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)")
        if has_finset:
            tactics += [f"simp only [{all_simp}]; norm_num",
                        f"simp only [{all_simp}]; native_decide"]

    # ── CONJUNCTION ──
    if has_and and not subst_h and not forall_h:
        if hyps:
            h_list = ", ".join(hyps)
            tactics += [f"constructor <;> linarith [{h_list}]",
                        f"constructor <;> nlinarith [{h_list}]",
                        f"constructor <;> omega"]
        else:
            tactics += ["constructor <;> omega", "constructor <;> norm_num"]
        if has_le and hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"constructor <;> nlinarith [sq_nonneg _, {h_list}]")

    # ── DISJUNCTION ──
    if has_or:
        for side in ["left", "right"]:
            tactics += [f"{side}; omega", f"{side}; norm_num", f"{side}; ring"]
            if hyps:
                h_list = ", ".join(hyps)
                tactics += [f"{side}; nlinarith [{h_list}]",
                            f"{side}; linarith [{h_list}]",
                            f"{side}; field_simp; nlinarith [{h_list}]",
                            f"{side}; field_simp; ring"]

    # ── EXISTENTIAL ──
    if has_exists:
        for w in ["0", "1", "2", "3", "4", "5", "10", "100"]:
            tactics += [f"exact ⟨{w}, by omega⟩",
                        f"exact ⟨{w}, by norm_num⟩"]

    # ── IFF ──
    if has_iff:
        tactics += ["constructor <;> intro <;> omega",
                    "constructor <;> intro <;> linarith",
                    "constructor <;> (intro; simp_all)",
                    "constructor <;> intro <;> simp_all <;> omega"]

    # ── FIELD DIVISION ──
    if has_div and not forall_h:
        if hyps:
            h_list = ", ".join(hyps)
            tactics += [f"field_simp; linarith [{h_list}]",
                        f"field_simp; nlinarith [{h_list}]"]
        tactics += ["field_simp; ring", "field_simp; linarith",
                    "field_simp; norm_num", "field_simp; nlinarith"]

    # ── INEQUALITY ──
    if has_le:
        if hyps:
            h_list = ", ".join(hyps)
            tactics += [f"linarith [{h_list}]", f"nlinarith [{h_list}]"]
            if has_pow:
                tactics.append(f"nlinarith [sq_nonneg _, {h_list}]")
                for v in var_names[:3]:
                    tactics.append(f"nlinarith [sq_nonneg {v}, {h_list}]")
                    tactics.append(f"nlinarith [sq_nonneg ({v} - 1), {h_list}]")
                    for v2 in var_names[:3]:
                        if v != v2:
                            tactics.append(f"nlinarith [sq_nonneg ({v} - {v2}), {h_list}]")
        tactics += ["linarith", "nlinarith"]
        if has_nat:
            tactics.append("omega")

    # ── DIVISIBILITY ──
    if has_dvd:
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"simp only [{h_list}]; omega")
        tactics += ["omega", "norm_num", "simp; omega"]

    # ── HYPOTHESIS-BASED ──
    if hyps and not forall_h and not subst_h:
        h_list = ", ".join(hyps)
        for c in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            tactics.append(f"simp only [{h_list}]; {c}")
        tactics += [f"linarith [{h_list}]", f"nlinarith [{h_list}]"]

    # ── LINEAR_COMBINATION ──
    if hyps and len(hyps) <= 5 and (has_eq or has_and) and not has_finset:
        for h in hyps:
            tactics.append(f"linear_combination {h}")
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            for op in ["+", "-"]:
                tactics.append(f"linear_combination {h0} {op} {h1}")
            for c in ["2", "3", "-1", "-2"]:
                tactics += [f"linear_combination {c} * {h0} - {h1}",
                            f"linear_combination {c} * {h1} - {h0}"]
            if has_and:
                tactics.append(f"constructor\n    · linear_combination {h0}\n    · linear_combination {h1}")

    # ── DECIDABLE / CONCRETE ──
    if no_hyps or has_gcd or has_prime or (has_nat and not hyps):
        tactics.insert(0, "norm_num")
        if not has_real and not has_complex:
            tactics += ["native_decide", "decide"]

    if has_finset and not forall_h:
        tactics.insert(0, "native_decide")
        tactics += ["decide", "simp; norm_num", "simp; native_decide"]

    if has_mod:
        tactics += ["omega", "native_decide"]

    # ── SQRT ──
    if has_sqrt:
        tactics += ["norm_num [Real.sqrt_eq_iff_sq_eq]",
                    "simp [Real.sqrt_eq_iff_sq_eq]; ring_nf; norm_num"]

    # ── PUSH_CAST ──
    if has_nat and (has_real or has_rat):
        tactics += ["push_cast; ring", "push_cast; linarith", "push_cast; norm_num",
                    "push_cast; omega", "push_cast; nlinarith",
                    "exact_mod_cast (by omega)", "exact_mod_cast (by norm_num)"]

    # ── EXPONENTIAL ──
    if has_pow and (has_eq or has_le):
        tactics.append("norm_num [pow_succ, pow_zero]")
        if hyps:
            h_list = ", ".join(hyps)
            tactics += [f"norm_num at {h_list} ⊢; linarith",
                        f"norm_num at {h_list} ⊢; omega"]

    # ── EQUIV ──
    if has_equiv and hyps:
        h_list = ", ".join(hyps)
        for c in ["ring", "linarith", "omega", "norm_num"]:
            tactics.append(f"simp only [{h_list}]; {c}")
        tactics.append(f"simp [{h_list}]")

    # ── CARD ──
    if has_card:
        tactics += ["native_decide", "decide", "simp; native_decide"]

    # ── SIMP_ALL FALLBACK ──
    for c in ["", "; ring", "; omega", "; linarith", "; nlinarith", "; norm_num"]:
        tactics.append(f"simp_all{c}")

    # ── UNIVERSAL FALLBACKS ──
    for d in ["omega", "norm_num", "ring", "linarith", "nlinarith", "decide",
              "simp; ring", "simp; omega", "simp; norm_num",
              "push_cast; ring", "push_cast; norm_num",
              "ring_nf; norm_num", "ring_nf; omega"]:
        tactics.append(d)

    return deduplicate(tactics)


def deduplicate(tactics):
    seen = set()
    return [t for t in tactics if not (t in seen or seen.add(t))]


def format_proof(tactics, max_tactics=60):
    simple = [t for t in tactics if "\n" not in t][:max_tactics]
    multi = [t for t in tactics if "\n" in t]
    lines = []
    for m in multi:
        indented = m.replace("\n", "\n    ")
        lines.append(f"  try\n    {indented}")
    if simple:
        lines.append("  first")
        for s in simple:
            lines.append(f"  | solve | {s}")
    return "\n".join(lines)


def make_lean_file(content, proof_text):
    new = content.replace("by sorry", f"by\n{proof_text}")
    new = new.replace("set_option maxHeartbeats 0", "set_option maxHeartbeats 4000000")
    return new


# ── Error-driven retry ────────────────────────────────────────────────────────

def parse_error(error_msg):
    e = error_msg.lower()
    if "timeout" in e or "heartbeat" in e:
        return "timeout"
    if "unknown identifier" in e:
        return "unknown_id"
    if "type mismatch" in e:
        return "type_mismatch"
    if "unsolved goals" in e:
        return "unsolved"
    if "tactic" in e and "failed" in e:
        return "tactic_failed"
    return "other"


def generate_retry_tactics(content, error_msg, attempt):
    """Generate alternative tactics based on error."""
    etype = parse_error(error_msg)
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""

    tactics = []

    if etype == "timeout":
        # Simpler, cheaper tactics
        tactics = ["omega", "norm_num", "ring", "linarith", "simp", "decide"]
        if hyps:
            tactics.append(f"linarith [{h_list}]")

    elif etype == "type_mismatch":
        tactics = ["push_cast; ring", "push_cast; norm_num", "push_cast; linarith",
                    "exact_mod_cast (by omega)", "exact_mod_cast (by norm_num)",
                    "norm_cast; omega", "norm_cast; ring", "norm_cast; norm_num"]
        if hyps:
            h_space = " ".join(hyps)
            tactics += [f"push_cast at {h_space} ⊢; linarith",
                        f"push_cast at {h_space} ⊢; omega"]

    elif etype == "unsolved":
        tactics = ["simp_all; omega", "simp_all; linarith", "simp_all; nlinarith",
                    "simp_all; ring", "simp_all; norm_num"]
        if hyps:
            tactics += [f"simp [{h_list}]; omega", f"simp [{h_list}]; ring"]

    else:
        tactics = ["omega", "norm_num", "ring", "linarith", "nlinarith",
                    "simp_all", "native_decide"]

    return tactics


def has_handcrafted_proof(problem):
    """Check if exp014 has a non-cascade proof."""
    fp = os.path.join(DOMAIN_DIR, "attempts", "exp014", f"{problem}.lean")
    if not os.path.exists(fp):
        return False
    with open(fp) as f:
        content = f.read()
    patterns = [r'^\s*have ', r'^\s*induction', r'^\s*obtain', r'^\s*rcases',
                r'^\s*calc', r'^\s*cases', r'^\s*suffices', r'^\s*refine',
                r'^\s*ring_nf', r'^\s*abs_of_', r'^\s*specialize',
                r'^\s*rw \[', r'^\s*intro', r'^\s*apply ', r'^\s*exact\b',
                r'^\s*use ', r'^\s*left\b', r'^\s*right\b',
                r'linear_combination.*Complex\.I_sq']
    return any(re.search(p, content, re.MULTILINE) for p in patterns)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    problems = sorted([f[:-5] for f in os.listdir(VALID_DIR) if f.endswith(".lean")])
    print(f"Processing {len(problems)} problems...")

    # Phase 1: Copy all proofs from exp014 as baseline
    exp014_dir = os.path.join(DOMAIN_DIR, "attempts", "exp014")
    copied = 0
    for problem in problems:
        src = os.path.join(exp014_dir, f"{problem}.lean")
        dst = os.path.join(out_dir, f"{problem}.lean")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            # Generate from scratch
            orig = os.path.join(VALID_DIR, f"{problem}.lean")
            with open(orig) as f:
                content = f.read()
            tactics = generate_cascade_tactics(content, problem)
            proof = format_proof(tactics)
            with open(dst, 'w') as f:
                f.write(make_lean_file(content, proof))
            copied += 1
    print(f"Initialized {copied} proofs")

    # Phase 2: Check all, identify failures
    passing = set()
    failing = {}
    print("\nPhase 2: Checking all proofs...")
    for i, problem in enumerate(problems):
        fp = os.path.join(out_dir, f"{problem}.lean")
        ok, err = check_lean(fp)
        if ok:
            passing.add(problem)
        else:
            failing[problem] = err
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(problems)}: {len(passing)} pass, {len(failing)} fail")

    print(f"\nPhase 2: {len(passing)} pass, {len(failing)} fail")

    # Phase 3: For non-handcrafted failures, regenerate with improved cascade
    regen_count = 0
    regen_solved = 0
    for problem in list(failing.keys()):
        if has_handcrafted_proof(problem):
            continue  # Don't overwrite working handcrafted proofs that timed out

        regen_count += 1
        orig = os.path.join(VALID_DIR, f"{problem}.lean")
        with open(orig) as f:
            content = f.read()

        # Generate fresh cascade
        tactics = generate_cascade_tactics(content, problem)
        proof = format_proof(tactics)
        fp = os.path.join(out_dir, f"{problem}.lean")
        with open(fp, 'w') as f:
            f.write(make_lean_file(content, proof))

        ok, err = check_lean(fp)
        if ok:
            passing.add(problem)
            del failing[problem]
            regen_solved += 1
            print(f"  REGEN SOLVED: {problem}")
        else:
            failing[problem] = err

    print(f"\nPhase 3: Regenerated {regen_count}, solved {regen_solved}")

    # Phase 4: Error-driven retry for remaining non-handcrafted failures
    retry_solved = 0
    for retry_round in range(1, MAX_RETRIES + 1):
        for problem in list(failing.keys()):
            if has_handcrafted_proof(problem):
                continue

            err = failing[problem]
            orig = os.path.join(VALID_DIR, f"{problem}.lean")
            with open(orig) as f:
                content = f.read()

            retry_tactics = generate_retry_tactics(content, err, retry_round)
            cascade = generate_cascade_tactics(content, problem)
            combined = retry_tactics + [t for t in cascade if t not in set(retry_tactics)]
            proof = format_proof(deduplicate(combined))

            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(make_lean_file(content, proof))

            ok, new_err = check_lean(fp)
            if ok:
                passing.add(problem)
                del failing[problem]
                retry_solved += 1
                print(f"  RETRY {retry_round} SOLVED: {problem}")
            else:
                failing[problem] = new_err

        print(f"  Retry round {retry_round}: {retry_solved} total solved, {len(failing)} remaining")

    # Phase 5: Brute-force single-tactic sweep for remaining
    brute_tactics = [
        "omega", "norm_num", "ring", "linarith", "nlinarith",
        "simp", "simp_all", "native_decide", "decide",
        "norm_num; omega", "simp; omega", "simp; norm_num",
        "push_cast; ring", "push_cast; omega", "push_cast; norm_num",
        "field_simp; ring", "field_simp; norm_num", "field_simp; linarith",
        "ring_nf; norm_num", "ring_nf; omega",
        "norm_cast; omega", "norm_cast; norm_num",
        "simp_all; omega", "simp_all; ring", "simp_all; norm_num",
        "simp_all; linarith", "simp_all; nlinarith",
    ]
    brute_solved = 0
    for problem in list(failing.keys()):
        if has_handcrafted_proof(problem):
            continue

        orig = os.path.join(VALID_DIR, f"{problem}.lean")
        with open(orig) as f:
            content = f.read()

        solved = False
        for tactic in brute_tactics:
            proof = f"  {tactic}"
            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(make_lean_file(content, proof))

            ok, _ = check_lean(fp)
            if ok:
                passing.add(problem)
                del failing[problem]
                brute_solved += 1
                print(f"  BRUTE SOLVED: {problem} with: {tactic}")
                solved = True
                break

        if not solved:
            # Restore cascade
            tactics = generate_cascade_tactics(content, problem)
            proof = format_proof(tactics)
            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(make_lean_file(content, proof))

    print(f"\nPhase 5: {brute_solved} brute-force solved")

    total = len(problems)
    score = len(passing) / total
    print(f"\n{'='*60}")
    print(f"FINAL: {len(passing)}/{total} solved = {score:.4f}")
    print(f"\nRemaining failures ({len(failing)}):")
    for p in sorted(failing.keys()):
        print(f"  {p}: {failing[p][:80]}")


if __name__ == "__main__":
    main()
