#!/usr/bin/env python3
"""
Generate tactic-cascade proof attempts for MiniF2F valid problems.
EXP-003: Fixed Lean 4 syntax, pattern-matched tactic selection.
Key fix: avoid [*] in first combinator, use <;> for subgoal dispatch.
"""
import os, re, sys

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))

method = sys.argv[1] if len(sys.argv) > 1 else "exp003"
out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)

def extract_hypotheses(content):
    """Extract hypothesis names."""
    return re.findall(r'\((\w+)\s*:', content)

def extract_variables(content):
    """Extract variable names from theorem signature (before the colon)."""
    # Match single-letter variables bound before hypotheses
    # E.g. (a b : ℝ) → ['a', 'b']
    vars_found = []
    for m in re.finditer(r'\(([a-z_][\w]*(?:\s+[a-z_][\w]*)*)\s*:', content):
        names = m.group(1).split()
        for n in names:
            if not n.startswith('h') and len(n) <= 3:
                vars_found.append(n)
    return vars_found

def generate_proof(content, problem):
    """Generate a multi-tactic proof using Lean 4 `try` blocks."""
    hyps = [h for h in extract_hypotheses(content) if h.startswith('h')]
    goal_match = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    goal = goal_match.group(1).strip() if goal_match else ""

    has_and = "∧" in goal
    has_exists = "∃" in goal
    has_complex = "ℂ" in content
    has_nat = "ℕ" in content and "ℝ" not in content
    has_real = "ℝ" in content
    has_sum = "∑" in content
    has_div = "/" in goal
    has_pow = "^" in content
    has_iff = "↔" in goal
    has_neg = "¬" in goal
    has_le = "≤" in goal or "≥" in goal or "<" in goal or ">" in goal
    has_eq = "=" in goal and not has_le and not has_iff
    has_mod = "%" in content or "dvd" in content
    has_finset = "Finset" in content
    has_nat_proper_div = "properDivisors" in content or "divisors" in content

    # Build tactic blocks - each is a standalone proof attempt
    # Using `try` chains since `first |` has issues with complex tactics
    blocks = []

    # === CONJUNCTION GOALS ===
    if has_and:
        if has_nat:
            blocks.append("constructor <;> omega")
        if has_real or has_complex:
            blocks.append("constructor <;> linarith")
            blocks.append("constructor <;> nlinarith")
        blocks.append("constructor <;> norm_num")
        blocks.append("constructor <;> ring")
        # Try with hypothesis-specific linear combinations
        if has_complex and len(hyps) >= 2:
            combos = []
            for i, h in enumerate(hyps):
                for j, h2 in enumerate(hyps):
                    if i != j:
                        combos.append(f"constructor\n    · linear_combination {h}\n    · linear_combination {h2}")
                        combos.append(f"constructor\n    · linear_combination 2 * {h} - {h2}\n    · linear_combination {h2} - {h}")
            blocks.extend(combos[:4])  # limit

    # === EXISTENTIAL GOALS ===
    if has_exists:
        for w in ["0", "1", "2", "3", "4", "5"]:
            blocks.append(f"exact ⟨{w}, by omega⟩")
            blocks.append(f"exact ⟨{w}, by norm_num⟩")

    # === IFF GOALS ===
    if has_iff:
        blocks.append("constructor <;> intro h <;> omega")
        blocks.append("constructor <;> intro h <;> linarith")

    # === INEQUALITY GOALS ===
    if has_le:
        if has_pow:
            blocks.append("nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]")
            # Generic sq_nonneg patterns
            blocks.append("nlinarith [sq_nonneg (_ - _)]")
        blocks.append("linarith")
        blocks.append("nlinarith")
        if has_nat:
            blocks.append("omega")

    # === EQUALITY GOALS ===
    if has_eq and not has_and:
        blocks.append("ring")
        blocks.append("norm_num")
        if has_nat:
            blocks.append("omega")
        if has_div:
            blocks.append("field_simp; ring")
            blocks.append("field_simp; linarith")
            blocks.append("field_simp; nlinarith")
        if has_sum or has_finset:
            blocks.append("simp [Finset.sum_range_succ]")
            blocks.append("simp [Finset.sum_range_succ]; ring")
            blocks.append("simp [Finset.sum_range_succ]; norm_num")
            blocks.append("native_decide")
        if has_nat_proper_div:
            blocks.append("native_decide")
            blocks.append("decide")
        if has_complex:
            blocks.append("push_cast; ring")

    # === HYPOTHESIS-BASED TACTICS ===
    if hyps:
        h_list = ", ".join(hyps)
        blocks.append(f"simp only [{h_list}]")
        blocks.append(f"simp only [{h_list}]; ring")
        blocks.append(f"simp only [{h_list}]; norm_num")
        blocks.append(f"simp only [{h_list}]; linarith")
        blocks.append(f"simp only [{h_list}]; omega")
        blocks.append("subst_vars; ring")
        blocks.append("subst_vars; norm_num")
        blocks.append("subst_vars; omega")
        blocks.append("subst_vars; simp")
        if len(hyps) == 1:
            blocks.append(f"linarith [{hyps[0]}]")
        elif len(hyps) >= 2:
            blocks.append(f"linarith [{', '.join(hyps)}]")
            blocks.append(f"nlinarith [{', '.join(hyps)}]")

    # === UNIVERSAL FALLBACKS ===
    blocks.extend([
        "omega",
        "norm_num",
        "ring",
        "linarith",
        "nlinarith",
        "decide",
        "simp",
        "simp; ring",
        "simp; omega",
        "simp; norm_num",
        "simp; linarith",
        "norm_num; omega",
        "push_cast; ring",
        "push_cast; norm_num",
    ])

    # Deduplicate
    seen = set()
    unique = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    # Build proof using first | (solve | t) syntax
    # solve | t ensures the tactic fully closes all goals
    simple = [b for b in unique if "\n" not in b]
    multi = [b for b in unique if "\n" in b]

    lines = []
    # Try multi-line tactics first (these usually have <;> for subgoals)
    for m in multi:
        # Indent multi-line blocks
        indented = m.replace("\n", "\n    ")
        lines.append(f"  try\n    {indented}")

    # Then simple tactics via first + solve
    if simple:
        lines.append("  first")
        for s in simple:
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
    new_content = new_content.replace("set_option maxHeartbeats 0", "set_option maxHeartbeats 800000")

    with open(os.path.join(out_dir, fname), 'w') as f:
        f.write(new_content)
    count += 1

print(f"Generated {count} proof attempts in {out_dir}")
