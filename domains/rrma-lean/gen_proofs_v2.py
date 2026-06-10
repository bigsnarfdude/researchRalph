#!/usr/bin/env python3
"""
Generate smarter tactic-cascade proof attempts for MiniF2F valid problems.
EXP-003: Pattern-matched tactic selection per problem type.
"""
import os, re, sys

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))

method = sys.argv[1] if len(sys.argv) > 1 else "exp003"
out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)

def extract_goal_type(content):
    """Extract the return type (goal) of the theorem."""
    # Find the part after the last ':' before ':= by sorry'
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""

def extract_hypotheses(content):
    """Extract hypothesis names from (h₀ : ...) patterns."""
    return re.findall(r'\((\w+)\s*:', content)

def generate_proof(content, problem):
    """Generate a proof attempt based on problem analysis."""
    goal = extract_goal_type(content)
    hyps = [h for h in extract_hypotheses(content) if h.startswith('h')]

    has_and = "∧" in goal
    has_or = "∨" in goal
    has_exists = "∃" in goal
    has_complex = "ℂ" in content
    has_nat = "ℕ" in content and "ℝ" not in content
    has_real = "ℝ" in content
    has_int = "ℤ" in content
    has_sum = "∑" in content
    has_div = "/" in goal
    has_abs = "abs" in content or "|" in goal
    has_sqrt = "sqrt" in content
    has_pow = "^" in content
    has_mod = "%" in content or "dvd" in content
    has_iff = "↔" in goal
    has_neg = "¬" in goal
    has_le = "≤" in goal or "≥" in goal or "<" in goal or ">" in goal
    has_eq = "=" in goal and not has_le and not has_iff

    # Hypothesis usage strings
    hyp_star = "[" + ", ".join(hyps) + "]" if hyps else "[*]"
    hyp_list = ", ".join(hyps)

    tactics = []

    # === CONJUNCTION GOALS ===
    if has_and:
        # For ∧ goals with hypotheses, constructor + solve subgoals
        sub_tactics = "omega) <;> (try linarith) <;> (try ring) <;> (try norm_num) <;> (try nlinarith [sq_nonneg _]) <;> (try simp [*])"
        tactics.append(f"constructor <;> (try {sub_tactics}")

        if has_complex:
            if hyps:
                # Try linear_combination for each branch
                tactics.append(f"constructor <;> linarith {hyp_star}")
                # Specific pattern: solve from hypotheses
                for h in hyps:
                    tactics.append(f"constructor <;> linear_combination {h}")
        elif has_nat:
            tactics.append(f"constructor <;> omega")
        elif has_real or has_int:
            tactics.append(f"constructor <;> linarith")
            tactics.append(f"constructor <;> nlinarith [sq_nonneg _]")

        tactics.append(f"refine ⟨by omega, by omega⟩")
        tactics.append(f"refine ⟨by linarith, by linarith⟩")
        tactics.append(f"refine ⟨by norm_num, by norm_num⟩")
        tactics.append(f"refine ⟨by ring, by ring⟩")
        tactics.append(f"simp only [*]; constructor <;> ring")

    # === EXISTENTIAL GOALS ===
    if has_exists:
        # Try common witnesses
        for witness in ["0", "1", "2", "3", "4", "5", "10"]:
            tactics.append(f"exact ⟨{witness}, by omega⟩")
            tactics.append(f"exact ⟨{witness}, by norm_num⟩")
            tactics.append(f"exact ⟨{witness}, by decide⟩")

    # === IFF GOALS ===
    if has_iff:
        tactics.append("constructor <;> intro h <;> (try omega) <;> (try linarith) <;> (try simp [*])")

    # === NEGATION GOALS ===
    if has_neg:
        tactics.append("intro h; omega")
        tactics.append("intro h; linarith")
        tactics.append("intro h; norm_num at h")
        tactics.append("simp; omega")

    # === INEQUALITY GOALS ===
    if has_le:
        if has_pow:
            tactics.append("nlinarith [sq_nonneg _, sq_nonneg _, sq_abs _]")
            # (a-b)^2 >= 0 trick
            for var in ['a', 'b', 'x', 'y', 'n', 'm']:
                tactics.append(f"nlinarith [sq_nonneg ({var})]")
            tactics.append("nlinarith [sq_nonneg (a - b), sq_nonneg (a + b)]")
        if has_nat:
            tactics.insert(0, "omega")
        tactics.append("linarith")
        tactics.append("nlinarith")

    # === EQUALITY GOALS ===
    if has_eq and not has_and:
        if has_sum:
            tactics.append("simp [Finset.sum_range_succ, Nat.properDivisors]")
            tactics.append("native_decide")
            tactics.append("decide")
            tactics.append("norm_num [Finset.sum_range_succ]")
        if has_abs:
            tactics.append("simp [abs_of_pos, abs_of_neg, abs_of_nonneg]")
            tactics.append("norm_num")
        if has_div and has_real:
            tactics.append("field_simp; ring")
            tactics.append("field_simp; linarith")
            tactics.append("field_simp; nlinarith")
        if has_complex:
            tactics.append("ring")
            tactics.append(f"linear_combination {hyp_list}" if len(hyps) >= 1 else "ring")
            # Complex specific
            tactics.append("ext <;> simp [*] <;> ring")
            tactics.append("simp [*]; ring")
        if has_sqrt:
            tactics.append("nlinarith [Real.sq_sqrt (by linarith : (0:ℝ) ≤ _)]")
        if has_nat:
            tactics.insert(0, "omega")
        if has_mod:
            tactics.append("omega")
            tactics.append("decide")

    # === UNIVERSAL FALLBACKS ===
    # Try substituting hypotheses then solving
    if hyps:
        hyp_simp = "simp only [" + ", ".join(hyps) + "]"
        tactics.append(f"{hyp_simp}; ring")
        tactics.append(f"{hyp_simp}; norm_num")
        tactics.append(f"{hyp_simp}; omega")
        tactics.append(f"{hyp_simp}; linarith")
        # subst for equality hypotheses
        tactics.append("subst_vars; ring")
        tactics.append("subst_vars; norm_num")
        tactics.append("subst_vars; omega")
        tactics.append("subst_vars; simp")

    # Always try these
    tactics.extend([
        "omega",
        "norm_num",
        "ring",
        "linarith",
        "simp [*]",
        "nlinarith [sq_nonneg _]",
        "decide",
        "norm_num [*]",
        "simp [*]; ring",
        "simp [*]; omega",
        "simp [*]; linarith",
    ])

    # Remove duplicates while preserving order
    seen = set()
    unique_tactics = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique_tactics.append(t)

    # Build proof
    lines = ["  first"]
    for t in unique_tactics:
        lines.append(f"  | {t}")
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
    new_content = new_content.replace("set_option maxHeartbeats 0", "set_option maxHeartbeats 800000")

    with open(os.path.join(out_dir, fname), 'w') as f:
        f.write(new_content)
    count += 1

print(f"Generated {count} proof attempts in {out_dir}")
