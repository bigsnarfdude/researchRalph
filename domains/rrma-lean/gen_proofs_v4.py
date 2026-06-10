#!/usr/bin/env python3
"""
EXP-006 generator (agent1): Hybrid tactic cascade.
Combines exp003 (solve|, linear_combination) + exp005 (subst_vars, native_decide).
Key improvements:
- subst_vars for equality hypotheses (solves complex number problems)
- native_decide for decidable/finite problems
- simp only [...] at * for function definition unfolding
- Better linear_combination coefficient generation
- Higher heartbeats for expensive tactics
"""
import os, re, sys

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))

method = sys.argv[1] if len(sys.argv) > 1 else "exp006"
out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)


def extract_hypotheses(content):
    """Extract hypothesis names starting with h."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', content)

def extract_subst_hyps(content):
    """Find hypotheses of the form (h : var = expr)."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', content)

def extract_forall_hyps(content):
    """Find hypotheses of form (h : ∀ ...)"""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', content)

def extract_neq_hyps(content):
    """Find hypotheses that assert ≠."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:[^)]*≠[^)]*\)', content)

def get_goal(content):
    """Extract goal from theorem statement."""
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    return m.group(1).strip() if m else ""


def generate_proof(content, problem):
    """Generate hybrid tactic cascade proof."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)
    neq_h = extract_neq_hyps(content)
    goal = get_goal(content)

    # Feature detection
    has_and = "∧" in goal
    has_or = "∨" in goal
    has_exists = "∃" in goal
    has_iff = "↔" in goal
    has_neg = "¬" in goal
    has_le = any(op in goal for op in ["≤", "≥", "<", ">"])
    has_eq = "=" in goal and not has_le and not has_iff
    has_complex = "ℂ" in content or "Complex" in content
    has_nat = ": ℕ)" in content
    has_int = ": ℤ)" in content
    has_real = ": ℝ" in content
    has_div = "/" in goal
    has_mod = "%" in content
    has_pow = "^" in content
    has_finset = "Finset" in content or "∑" in content
    has_dvd = "∣" in goal
    has_sqrt = "sqrt" in content
    has_prime = "Prime" in content
    has_gcd = "gcd" in content or "lcm" in content
    no_hyps = len(hyps) == 0

    tactics = []

    # ========== PHASE 1: SUBST + COMPLEX ==========
    if has_complex:
        if subst_h:
            tactics.append("subst_vars; ring")
            tactics.append("subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]")
            tactics.append("subst_vars; simp [Complex.ext_iff, Complex.I_sq]; ring")
            if has_and:
                tactics.append("constructor <;> (subst_vars; ring)")
                tactics.append("constructor <;> (subst_vars; norm_num [Complex.ext_iff])")
        tactics.append("ring")
        tactics.append("norm_num")
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"simp only [{h_list}]; ring")

    # ========== PHASE 2: SUBST FOR EQUALITY HYPS ==========
    if subst_h and not has_complex:
        if has_and:
            tactics.append("constructor <;> (subst_vars; norm_num)")
            tactics.append("constructor <;> (subst_vars; omega)")
            tactics.append("constructor <;> (subst_vars; ring)")
        tactics.append("subst_vars; ring")
        tactics.append("subst_vars; norm_num")
        tactics.append("subst_vars; omega")
        tactics.append("subst_vars; simp")

    # ========== PHASE 3: FUNCTION DEFINITION UNFOLDING ==========
    if forall_h:
        all_simp = ", ".join(forall_h + subst_h)
        closers = ["ring", "norm_num", "omega", "linarith", "nlinarith"]
        for c in closers:
            tactics.append(f"simp only [{all_simp}] at *; {c}")
        if has_div:
            tactics.append(f"simp only [{all_simp}] at *; field_simp; ring")
            tactics.append(f"simp only [{all_simp}] at *; field_simp; linarith")
        if has_and:
            tactics.append(f"simp only [{all_simp}] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)")
        if has_finset:
            tactics.append(f"simp only [{all_simp}]; norm_num")
            tactics.append(f"simp only [{all_simp}]; omega")

    # ========== PHASE 4: CONJUNCTION GOALS ==========
    if has_and and not subst_h and not forall_h:
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"constructor <;> linarith [{h_list}]")
            tactics.append(f"constructor <;> nlinarith [{h_list}]")
            tactics.append(f"constructor <;> omega")
        else:
            tactics.append("constructor <;> omega")
            tactics.append("constructor <;> norm_num")
        # For inequality conjunction (a ≤ x ∧ x ≤ b from quadratic)
        if has_le or "≤" in goal or "≥" in goal:
            if hyps:
                h_list = ", ".join(hyps)
                tactics.append(f"constructor <;> nlinarith [sq_nonneg _, {h_list}]")

    # ========== PHASE 5: DISJUNCTION GOALS ==========
    if has_or:
        for side in ["left", "right"]:
            tactics.append(f"{side}; omega")
            tactics.append(f"{side}; norm_num")
            if hyps:
                h_list = ", ".join(hyps)
                tactics.append(f"{side}; nlinarith [{h_list}]")

    # ========== PHASE 6: EXISTENTIAL GOALS ==========
    if has_exists:
        for w in ["0", "1", "2", "3", "4", "5", "10", "100"]:
            tactics.append(f"exact ⟨{w}, by omega⟩")
            tactics.append(f"exact ⟨{w}, by norm_num⟩")

    # ========== PHASE 7: IFF GOALS ==========
    if has_iff:
        tactics.append("constructor <;> intro <;> omega")
        tactics.append("constructor <;> intro <;> linarith")
        tactics.append("constructor <;> (intro; simp_all)")

    # ========== PHASE 8: FIELD DIVISION ==========
    if has_div and not forall_h:
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"field_simp; linarith [{h_list}]")
            tactics.append(f"field_simp; nlinarith [{h_list}]")
        tactics.append("field_simp; ring")
        tactics.append("field_simp; linarith")
        tactics.append("field_simp; norm_num")

    # ========== PHASE 9: INEQUALITY GOALS ==========
    if has_le:
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"linarith [{h_list}]")
            tactics.append(f"nlinarith [{h_list}]")
            if has_pow:
                tactics.append(f"nlinarith [sq_nonneg _, {h_list}]")
        tactics.append("linarith")
        tactics.append("nlinarith")
        if has_nat:
            tactics.append("omega")

    # ========== PHASE 10: DIVISIBILITY ==========
    if has_dvd:
        if hyps:
            h_list = ", ".join(hyps)
            tactics.append(f"simp only [{h_list}]; omega")
        tactics.append("omega")
        tactics.append("norm_num")
        tactics.append("simp; omega")

    # ========== PHASE 11: HYPOTHESIS-BASED TACTICS ==========
    if hyps and not forall_h and not subst_h:
        h_list = ", ".join(hyps)
        tactics.append(f"simp only [{h_list}]; ring")
        tactics.append(f"simp only [{h_list}]; norm_num")
        tactics.append(f"simp only [{h_list}]; omega")
        tactics.append(f"simp only [{h_list}]; linarith")
        tactics.append(f"simp only [{h_list}]; nlinarith")
        tactics.append(f"linarith [{h_list}]")
        tactics.append(f"nlinarith [{h_list}]")

    # ========== PHASE 12: LINEAR_COMBINATION ==========
    if hyps and len(hyps) <= 5 and (has_eq or has_and) and not has_finset:
        for h in hyps:
            tactics.append(f"linear_combination {h}")
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            tactics.append(f"linear_combination {h0} + {h1}")
            tactics.append(f"linear_combination {h0} - {h1}")
            tactics.append(f"linear_combination 2 * {h0} - {h1}")
            tactics.append(f"linear_combination {h1} - {h0}")
            tactics.append(f"linear_combination 2 * {h1} - {h0}")
            if has_and:
                tactics.append(f"constructor\n    · linear_combination {h0}\n    · linear_combination {h1}")
                tactics.append(f"constructor\n    · linear_combination 2 * {h0} - {h1}\n    · linear_combination {h1} - {h0}")

    # ========== PHASE 13: DECIDABLE/CONCRETE ==========
    if no_hyps or has_gcd or has_prime or (has_nat and not hyps):
        tactics.insert(0, "norm_num")
        if not has_real and not has_complex:
            tactics.append("native_decide")
            tactics.append("decide")

    if has_finset and not forall_h:
        tactics.insert(0, "native_decide")
        tactics.append("decide")
        tactics.append("simp; norm_num")

    if has_mod:
        tactics.append("omega")
        tactics.append("native_decide")

    # ========== PHASE 14: SIMP_ALL FALLBACK ==========
    tactics.append("simp_all")
    tactics.append("simp_all; ring")
    tactics.append("simp_all; omega")
    tactics.append("simp_all; linarith")
    tactics.append("simp_all; nlinarith")
    tactics.append("simp_all; norm_num")

    # ========== PHASE 15: UNIVERSAL FALLBACKS ==========
    defaults = ["omega", "norm_num", "ring", "linarith", "nlinarith", "decide",
                "simp; ring", "simp; omega", "simp; norm_num", "push_cast; ring",
                "push_cast; norm_num"]
    for d in defaults:
        tactics.append(d)

    # Deduplicate, preserving order
    seen = set()
    unique = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Separate simple (single-line) and multi-line tactics
    simple = [t for t in unique if "\n" not in t]
    multi = [t for t in unique if "\n" in t]

    lines = []

    # Multi-line tactics go in try blocks
    for m in multi:
        indented = m.replace("\n", "\n    ")
        lines.append(f"  try\n    {indented}")

    # Simple tactics in first | solve | cascade
    if simple:
        lines.append("  first")
        for s in simple[:50]:  # limit to 50 tactics max
            lines.append(f"  | solve | {s}")

    return "\n".join(lines)


count = 0
for fname in sorted(os.listdir(VALID_DIR)):
    if not fname.endswith(".lean"):
        continue
    problem = fname[:-5]

    with open(os.path.join(VALID_DIR, fname)) as f:
        content = f.read()

    proof = generate_proof(content, problem)
    new_content = content.replace("by sorry", f"by\n{proof}")
    # Use 4M heartbeats for more expensive tactics
    new_content = new_content.replace("set_option maxHeartbeats 0", "set_option maxHeartbeats 4000000")

    with open(os.path.join(out_dir, fname), 'w') as f:
        f.write(new_content)
    count += 1

print(f"Generated {count} proof attempts in {out_dir}")
