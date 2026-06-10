#!/usr/bin/env python3
"""Generate tactic-cascade proof attempts for all MiniF2F valid problems.

v4: Better category detection, complex numbers, field_simp, function subst.
"""

import os
import re
import sys

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "exp005"
OUT_DIR = f"/home/vincent/researchRalph/domains/rrma-lean/attempts/{METHOD}"

os.makedirs(OUT_DIR, exist_ok=True)


def extract_subst_hyps(text):
    """Find hypotheses of the form (h : var = expr) where var is a simple name."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', text)


def extract_forall_hyps(text):
    """Find hypotheses of form (h : ∀ ...)"""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', text)


def extract_ineq_hyps(text):
    """Find all hypothesis names."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:', text)


def extract_neq_hyps(text):
    """Find hypotheses that assert ≠."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:[^)]*≠[^)]*\)', text)


def get_goal_area(text):
    """Get rough goal area after the last ) :"""
    parts = text.split(') :')
    return parts[-1] if len(parts) > 1 else text


def gen_proof(text, fname):
    """Generate SHORT, priority-ordered cascade."""
    subst = extract_subst_hyps(text)
    forall_h = extract_forall_hyps(text)
    all_h = extract_ineq_hyps(text)
    neq_h = extract_neq_hyps(text)
    goal = get_goal_area(text)

    has_conj = '∧' in goal
    has_iff = '↔' in goal
    has_exists = '∃' in goal
    has_finset = 'Finset' in text
    has_complex = 'Complex' in text or ': ℂ' in text
    has_nat = ': ℕ)' in text
    has_int = ': ℤ)' in text
    has_real = ': ℝ' in text
    has_div = '/' in goal
    has_mod = '%' in text
    has_gcd = 'gcd' in text or 'lcm' in text
    has_prime = 'Prime' in text
    no_hyps = len(all_h) == 0
    has_induction = fname.startswith('induction')
    has_sqrt = 'sqrt' in text
    has_abs = 'abs ' in text or 'abs(' in text

    tactics = []

    # === COMPLEX NUMBER PROBLEMS ===
    if has_complex:
        if subst:
            tactics.append("subst_vars; norm_num [Complex.ext_iff]")
            tactics.append("subst_vars; ring")
            tactics.append("subst_vars; apply Complex.ext <;> simp <;> ring")
        if has_conj:
            if subst:
                tactics.append("constructor <;> (subst_vars; norm_num [Complex.ext_iff])")
                tactics.append("constructor <;> (subst_vars; ring)")
        tactics.append("ring")
        tactics.append("norm_num")
        tactics.append("simp_all")
        return _build_cascade(tactics)

    # === SUBSTITUTABLE HYPOTHESES (not ∀) ===
    if subst and not forall_h:
        sa = ', '.join(subst)
        if has_conj:
            tactics.append(f"constructor <;> (first | (simp only [{sa}]; linarith) | (simp only [{sa}]; omega) | (simp only [{sa}]; ring) | (simp only [{sa}]; norm_num))")
        else:
            tactics.append(f"simp only [{sa}]; ring")
            tactics.append(f"simp only [{sa}]; norm_num")
            tactics.append(f"simp only [{sa}]; omega")
            tactics.append(f"simp only [{sa}]; linarith")
            if has_div:
                tactics.append(f"simp only [{sa}]; field_simp; ring")
                tactics.append(f"simp only [{sa}]; field_simp; linarith")

    # === FUNCTION DEFINITION HYPOTHESES ===
    if forall_h:
        # simp with all forall + subst hyps, then close
        all_simp = ', '.join(forall_h + subst)
        # Use at * to also simplify other hypotheses that reference the function
        tactics.append(f"simp only [{all_simp}] at *; nlinarith")
        tactics.append(f"simp only [{all_simp}] at *; linarith")
        tactics.append(f"simp only [{all_simp}] at *; omega")
        tactics.append(f"simp only [{all_simp}] at *; norm_num")
        if has_div:
            tactics.append(f"simp only [{all_simp}] at *; field_simp; ring")
            tactics.append(f"simp only [{all_simp}] at *; field_simp; linarith")
        tactics.append(f"simp only [{all_simp}]; ring")
        tactics.append(f"simp only [{all_simp}]; norm_num")
        if has_conj:
            tactics.append(f"simp only [{all_simp}] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)")

    # === FIELD DIVISION ===
    if has_div and not forall_h:
        if all_h:
            h_list = ', '.join(all_h)
            tactics.append(f"field_simp; linarith [{h_list}]")
            tactics.append(f"field_simp; nlinarith [{h_list}]")
        tactics.append("field_simp; ring")
        tactics.append("field_simp; linarith")
        tactics.append("field_simp; norm_num")

    # === CONJUNCTION ===
    if has_conj and not has_exists and not subst and not forall_h:
        if all_h:
            h_list = ', '.join(all_h)
            tactics.append(f"constructor <;> linarith [{h_list}]")
            tactics.append(f"constructor <;> omega")
        tactics.append("constructor <;> norm_num")
        tactics.append("constructor <;> simp_all")

    # === IFF ===
    if has_iff:
        tactics.append("constructor <;> intro <;> omega")
        tactics.append("constructor <;> (intro; simp_all)")

    # === CONCRETE / CLOSED GOALS ===
    if no_hyps or has_gcd or has_prime:
        tactics.insert(0, "norm_num")
        if not has_real and not has_complex:
            tactics.insert(1, "native_decide")
            tactics.insert(2, "decide")

    # === FINSET ===
    if has_finset:
        tactics.insert(0, "native_decide")
        tactics.insert(1, "decide")
        tactics.insert(2, "simp [Finset.sum]; norm_num")

    # === NATURAL / INTEGER ===
    if has_nat or has_int or has_mod:
        if 0 not in [i for i, t in enumerate(tactics) if t == "omega"]:
            tactics.insert(0, "omega")

    # === ABSOLUTE VALUE ===
    if has_abs:
        tactics.append("simp [abs_of_nonneg, abs_of_nonpos]; norm_num")
        tactics.append("norm_num")

    # === DEFAULT FALLBACKS ===
    defaults = ["ring", "norm_num", "omega", "linarith", "simp_all", "decide"]
    for d in defaults:
        if d not in tactics:
            tactics.append(d)

    return _build_cascade(tactics)


def _build_cascade(tactics):
    """Build first | ... cascade, deduped, max 12."""
    seen = set()
    unique = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    unique = unique[:12]
    cascade = "\n    | ".join(unique)
    return f"""first
    | {cascade}"""


def make_attempt(filepath):
    fname = os.path.basename(filepath).replace('.lean', '')
    with open(filepath) as f:
        text = f.read()
    proof = gen_proof(text, fname)
    result = text.replace("by sorry", f"by\n  {proof}")
    result = result.replace("maxHeartbeats 0", "maxHeartbeats 4000000")
    return result


count = 0
for fname in sorted(os.listdir(VALID_DIR)):
    if not fname.endswith('.lean'):
        continue
    attempt = make_attempt(os.path.join(VALID_DIR, fname))
    with open(os.path.join(OUT_DIR, fname), 'w') as f:
        f.write(attempt)
    count += 1

print(f"Generated {count} attempts in {OUT_DIR}")
