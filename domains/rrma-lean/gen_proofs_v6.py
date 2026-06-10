#!/usr/bin/env python3
"""
gen_proofs_v6.py — Problem-type classifier + specialized tactic templates (agent1).

Key innovation: instead of a one-size-fits-all cascade, classify each problem
into a TYPE based on goal structure, then apply a specialized tactic template
for that type. Also includes error-driven retry.

Problem types:
  1. NUMERIC_EVAL  — No hypotheses, concrete computation → norm_num / native_decide
  2. LINEAR_SYSTEM — Linear equations in hypotheses → linear_combination / linarith
  3. POLYNOMIAL_ID — Ring identity goal → ring / ring_nf
  4. INEQUALITY    — ≤/≥/</>  goal → nlinarith with SOS witnesses
  5. INDUCTION     — ∀ n, ... with Nat → induction
  6. DIVISIBILITY  — ∣ goal → omega / mod case split
  7. FINSET_SUM    — ∑/∏ over Finset → simp + norm_num / native_decide
  8. FUNC_DEF      — ∀ x, f x = ... hypotheses → unfold + closer
  9. CONJUNCTION   — ∧ goal → split + recurse
  10. COMPLEX_ALG  — ℂ type → ring_nf + Complex.I_sq
  11. SET_CARD     — .card goal with Finset → native_decide / ext
  12. EQUIV_FUNC   — Equiv in hypotheses → apply/simp Equiv
  13. GENERIC      — fallback cascade
"""
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ELAN_PATH = os.path.expanduser("~/.elan/bin")

method = sys.argv[1] if len(sys.argv) > 1 else "exp015_v6"
MAX_RETRIES = 2
TIMEOUT = 120

out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

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
    """Extract variable names (not hypothesis names) from theorem signature."""
    # Match (varname : Type) but exclude hypotheses (h₀ etc)
    all_vars = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', content)
    return [(v, t.strip()) for v, t in all_vars
            if not v.startswith('h') and not v.startswith('S')
            and v not in ('f', 'σ', 'u', 'v', 't')]

def get_all_var_names(content):
    """Get all variable and hypothesis names."""
    return re.findall(r'\((\w+)\s*:', content)


# ══════════════════════════════════════════════════════════════
# Lean compiler
# ══════════════════════════════════════════════════════════════

def check_lean(filepath):
    try:
        env = os.environ.copy()
        env["PATH"] = ELAN_PATH + ":" + env.get("PATH", "")
        result = subprocess.run(
            ["lake", "env", "lean", filepath],
            capture_output=True, text=True, timeout=TIMEOUT,
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


# ══════════════════════════════════════════════════════════════
# Problem type classifier
# ══════════════════════════════════════════════════════════════

def classify_problem(content, goal):
    """Classify problem into a type for specialized tactic selection."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)

    # Feature detection
    feat = {
        "has_and": "∧" in goal,
        "has_or": "∨" in goal,
        "has_exists": "∃" in goal,
        "has_iff": "↔" in goal,
        "has_le": any(op in goal for op in ["≤", "≥", "<", ">"]),
        "has_eq": "=" in goal,
        "has_complex": "ℂ" in content or "Complex" in content,
        "has_nat": ": ℕ)" in content,
        "has_int": ": ℤ)" in content,
        "has_real": ": ℝ" in content,
        "has_rat": ": ℚ" in content,
        "has_div": "/" in goal,
        "has_mod": "%" in content,
        "has_pow": "^" in content,
        "has_finset": "Finset" in content or "∑" in content or "∏" in content,
        "has_dvd": "∣" in goal,
        "has_sqrt": "sqrt" in content.lower(),
        "has_prime": "Prime" in content or "Nat.Prime" in content,
        "has_gcd": "gcd" in content,
        "has_abs": "abs" in content,
        "has_card": ".card" in goal,
        "has_equiv": "Equiv" in content,
        "has_isleast": "IsLeast" in content,
        "has_induction_type": bool(re.search(r'\(n\s*:\s*ℕ\)', content)) and "∀" in content,
        "no_hyps": len(hyps) == 0,
        "n_hyps": len(hyps),
        "has_sum_prod": "∑" in content or "∏" in content,
    }

    # Classification logic (order matters — more specific first)
    if feat["has_equiv"]:
        return "EQUIV_FUNC", feat
    if feat["has_card"] and feat["has_finset"]:
        return "SET_CARD", feat
    if feat["has_sum_prod"]:
        return "FINSET_SUM", feat
    if feat["has_complex"]:
        return "COMPLEX_ALG", feat
    if feat["has_dvd"] and not feat["has_le"]:
        return "DIVISIBILITY", feat
    if feat["has_le"] and not feat["has_and"]:
        return "INEQUALITY", feat
    if feat["has_and"] and feat["has_le"]:
        return "INEQUALITY", feat  # conjunction of inequalities
    if forall_h and not feat["has_finset"]:
        return "FUNC_DEF", feat
    if subst_h and feat["has_eq"]:
        return "LINEAR_SYSTEM", feat
    if feat["no_hyps"] and feat["has_eq"]:
        return "NUMERIC_EVAL", feat
    if feat["has_eq"] and hyps and not forall_h:
        if feat["has_pow"] or feat["has_div"]:
            return "POLYNOMIAL_ID", feat
        return "LINEAR_SYSTEM", feat
    if feat["has_and"]:
        return "CONJUNCTION", feat
    if feat["has_exists"]:
        return "EXISTENTIAL", feat
    if feat["has_iff"]:
        return "IFF", feat
    return "GENERIC", feat


# ══════════════════════════════════════════════════════════════
# Specialized tactic generators per type
# ══════════════════════════════════════════════════════════════

def tactics_numeric_eval(content, goal, feat):
    """No hypotheses, concrete computation."""
    t = ["norm_num", "native_decide", "decide", "simp; norm_num",
         "simp; native_decide", "ring", "omega"]
    if feat["has_sqrt"]:
        t.extend([
            "norm_num [Real.sqrt_lt', Real.lt_sqrt]",
            "simp [Real.sqrt_eq_iff_sq_eq]; norm_num",
        ])
    if feat["has_prime"]:
        t.extend(["norm_num [Nat.Prime]", "decide"])
    if feat["has_gcd"]:
        t.extend(["norm_num [Nat.gcd, Nat.lcm]", "native_decide"])
    return t


def tactics_linear_system(content, goal, feat):
    """Linear equations → linear_combination / linarith."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    h_list = ", ".join(hyps)
    t = []

    # Try subst first
    if subst_h:
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            t.append(f"subst_vars; {closer}")
        if feat["has_and"]:
            t.append("constructor <;> (subst_vars; norm_num)")
            t.append("constructor <;> (subst_vars; linarith)")

    # linear_combination with various coefficient combos
    if hyps:
        for h in hyps:
            t.append(f"linear_combination {h}")
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            coeffs = [(1, 1), (1, -1), (-1, 1), (2, -1), (-1, 2), (3, -2), (-2, 3),
                       (1, 0), (0, 1), (3, -1), (-1, 3), (4, -3), (-3, 4), (5, -4)]
            for a, b in coeffs:
                parts = []
                if a == 1: parts.append(h0)
                elif a == -1: parts.append(f"-{h0}")
                elif a != 0: parts.append(f"{a} * {h0}")
                if b == 1: parts.append(h1)
                elif b == -1: parts.append(f"-{h1}")
                elif b != 0: parts.append(f"{b} * {h1}")
                if parts:
                    t.append(f"linear_combination {' + '.join(parts)}")
            if len(hyps) >= 3:
                h2 = hyps[2]
                t.append(f"linear_combination {h0} + {h1} + {h2}")
                t.append(f"linear_combination {h0} - {h1} + {h2}")

        # linarith with hints
        t.extend([
            f"linarith [{h_list}]",
            f"nlinarith [{h_list}]",
        ])

    # Field division
    if feat["has_div"]:
        t.extend([
            f"field_simp; linear_combination {hyps[0]}" if hyps else "field_simp; ring",
            "field_simp; ring",
            f"field_simp; linarith [{h_list}]" if hyps else "field_simp; linarith",
            f"field_simp; nlinarith [{h_list}]" if hyps else "field_simp; nlinarith",
        ])

    # Conjunction
    if feat["has_and"] and hyps:
        t.append(f"constructor <;> linarith [{h_list}]")
        t.append(f"constructor <;> nlinarith [{h_list}]")
        if len(hyps) >= 2:
            t.append(f"constructor\n    · linear_combination {hyps[0]}\n    · linear_combination {hyps[1]}")

    # Hypothesis-based simp
    if hyps:
        for closer in ["ring", "norm_num", "omega", "linarith"]:
            t.append(f"simp only [{h_list}]; {closer}")

    return t


def tactics_polynomial_id(content, goal, feat):
    """Polynomial/ring identities."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps)
    t = ["ring", "ring_nf; norm_num", "ring_nf; omega"]
    if feat["has_div"]:
        t.extend(["field_simp; ring", "field_simp; ring_nf; norm_num"])
    if hyps:
        t.extend([
            f"simp only [{h_list}]; ring",
            f"simp only [{h_list}]; ring_nf; norm_num",
        ])
        for h in hyps:
            t.append(f"linear_combination {h}")
        if feat["has_pow"]:
            t.extend([
                f"nlinarith [{h_list}]",
                f"nlinarith [sq_nonneg _, {h_list}]",
            ])
    return t


def tactics_inequality(content, goal, feat):
    """Inequality goals."""
    hyps = extract_hypotheses(content)
    variables = get_variables(content)
    h_list = ", ".join(hyps) if hyps else ""
    var_names = [v for v, _ in variables]
    t = []

    if hyps:
        t.extend([
            f"linarith [{h_list}]",
            f"nlinarith [{h_list}]",
        ])
        # SOS witnesses from variables
        if feat["has_pow"] or feat["has_le"]:
            for v in var_names[:4]:
                t.append(f"nlinarith [sq_nonneg {v}, {h_list}]")
                t.append(f"nlinarith [sq_nonneg ({v} - 1), {h_list}]")
            if len(var_names) >= 2:
                a, b = var_names[0], var_names[1]
                t.extend([
                    f"nlinarith [sq_nonneg ({a} - {b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a} + {b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a} - {b}), sq_nonneg ({a} + {b}), {h_list}]",
                    f"nlinarith [sq_nonneg (2*{a} - {b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a} - 2*{b}), {h_list}]",
                    f"nlinarith [mul_self_nonneg ({a} - {b}), mul_self_nonneg ({a} + {b}), {h_list}]",
                ])
    else:
        t.extend(["linarith", "nlinarith", "omega"])

    # Conjunction of inequalities
    if feat["has_and"] and hyps:
        t.extend([
            f"constructor <;> linarith [{h_list}]",
            f"constructor <;> nlinarith [{h_list}]",
            f"constructor <;> omega",
        ])

    if feat["has_nat"]:
        t.append("omega")
    if feat["has_abs"]:
        t.extend([
            f"simp only [abs_le]; constructor <;> linarith [{h_list}]" if hyps else "simp only [abs_le]; constructor <;> linarith",
        ])
    if feat["has_div"]:
        t.extend([
            f"field_simp; nlinarith [{h_list}]" if hyps else "field_simp; nlinarith",
            f"field_simp; linarith [{h_list}]" if hyps else "field_simp; linarith",
        ])
    return t


def tactics_func_def(content, goal, feat):
    """Function-definition hypotheses (∀ x, f x = ...)."""
    forall_h = extract_forall_hyps(content)
    subst_h = extract_subst_hyps(content)
    hyps = extract_hypotheses(content)
    all_simp = ", ".join(forall_h + subst_h)
    h_list = ", ".join(hyps)
    t = []

    # Unfold + close
    closers = ["ring", "norm_num", "omega", "linarith", "nlinarith",
               "simp", "native_decide", "decide"]
    for c in closers:
        t.append(f"simp only [{all_simp}] at *; {c}")

    # With field_simp
    if feat["has_div"]:
        t.extend([
            f"simp only [{all_simp}] at *; field_simp; ring",
            f"simp only [{all_simp}] at *; field_simp; linarith",
            f"simp only [{all_simp}] at *; field_simp; nlinarith",
        ])

    # With conjunction
    if feat["has_and"]:
        t.append(f"simp only [{all_simp}] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)")

    # For Finset sums with function definitions
    if feat["has_sum_prod"] or feat["has_finset"]:
        t.extend([
            f"simp only [{all_simp}]; ring",
            f"simp only [{all_simp}]; norm_num",
            f"simp [{all_simp}]; ring",
            f"simp [{all_simp}]; norm_num",
            f"simp only [{all_simp}]; native_decide",
        ])
        # Rewrite + Finset.sum_congr
        t.extend([
            f"conv_lhs => arg 2; ext; rw [show ∀ x, _ from {forall_h[0]}]" if forall_h else "",
            f"simp only [{all_simp}]; simp [Finset.sum_sub_distrib]; ring",
        ])

    # Extensional simp
    t.extend([
        f"simp [{all_simp}]",
        f"simp [{h_list}]" if hyps else "",
    ])

    return [x for x in t if x]


def tactics_finset_sum(content, goal, feat):
    """Finset sum/product problems."""
    hyps = extract_hypotheses(content)
    forall_h = extract_forall_hyps(content)
    h_list = ", ".join(hyps) if hyps else ""
    all_simp = ", ".join(forall_h) if forall_h else ""
    t = []

    t.extend(["native_decide", "decide", "norm_num"])

    if forall_h:
        t.extend([
            f"simp only [{all_simp}]",
            f"simp only [{all_simp}]; ring",
            f"simp only [{all_simp}]; norm_num",
            f"simp [{all_simp}]; ring",
            f"simp [{all_simp}]; norm_num",
        ])
        # For sum of linear functions
        t.extend([
            f"simp only [{all_simp}]; simp [Finset.sum_add_adjacent]; ring",
            f"simp only [{all_simp}]; rw [Finset.sum_sub_distrib]; simp; ring",
        ])

    if hyps:
        t.extend([
            f"simp [{h_list}]; ring",
            f"simp [{h_list}]; norm_num",
            f"simp [{h_list}]; native_decide",
        ])

    # Product-specific
    if "∏" in content:
        t.extend([
            "simp [Finset.prod_Icc_succ]",
            f"simp only [{all_simp}]; simp [Finset.prod_div_distrib]; norm_num" if all_simp else "",
        ])

    t.extend(["simp; native_decide", "simp; norm_num", "simp_all; native_decide"])
    return [x for x in t if x]


def tactics_set_card(content, goal, feat):
    """Finset cardinality problems."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = ["native_decide", "decide"]

    if hyps:
        t.extend([
            f"simp only [{h_list}]; native_decide",
            f"simp only [{h_list}]; decide",
            # Convert membership condition and decide
            f"ext x; simp only [{h_list}]; omega",
            f"have : s = _ := by ext x; simp [{h_list}]; omega",
        ])
        # For problems with interval-like conditions
        if feat["has_le"] or feat["has_abs"]:
            t.extend([
                f"convert_to (Finset.Icc _ _).card = _; · ext; simp [{h_list}]; omega; · simp",
            ])
    return t


def tactics_complex_alg(content, goal, feat):
    """Complex number algebra."""
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = []

    if subst_h:
        t.extend([
            "subst_vars; ring",
            "subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]",
            "subst_vars; simp [Complex.ext_iff, Complex.I_sq]; ring",
        ])
        if feat["has_and"]:
            t.extend([
                "constructor <;> (subst_vars; ring)",
                "constructor <;> (subst_vars; norm_num [Complex.ext_iff])",
            ])
    t.extend([
        "ring",
        "ring_nf; norm_num [Complex.I_sq]",
        "ring_nf; linear_combination 49 * Complex.I_sq",
        "simp [Complex.ext_iff, Complex.I_sq]; constructor <;> ring",
    ])
    if hyps:
        t.extend([
            f"simp only [{h_list}]; ring",
            f"linear_combination {hyps[0]}" if hyps else "",
        ])
    return [x for x in t if x]


def tactics_divisibility(content, goal, feat):
    """Divisibility goals."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = ["omega", "norm_num", "simp; omega", "native_decide", "decide"]
    if hyps:
        t.extend([
            f"simp only [{h_list}]; omega",
            f"simp [{h_list}]; omega",
        ])
    if feat["has_mod"]:
        t.extend(["omega", "native_decide"])
    return t


def tactics_equiv_func(content, goal, feat):
    """Equiv-based problems — need to unfold Equiv."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    forall_h = extract_forall_hyps(content)
    t = []

    # Key insight: σ.2 is the inverse of σ.1
    # If h₀ : σ.2 a = b, then σ.1 b = a
    # If h₀ : ∀ x, σ.1 x = expr, then σ.2 y solves expr = y

    if forall_h:
        all_simp = ", ".join(forall_h)
        t.extend([
            f"simp only [{all_simp}] at *; linarith",
            f"simp only [{all_simp}] at *; ring",
            f"simp only [{all_simp}] at *; nlinarith",
            f"simp only [{all_simp}] at *; field_simp; linarith",
            f"simp only [{all_simp}] at *; omega",
            f"simp only [{all_simp}] at *; norm_num",
        ])

    if hyps:
        # Try using Equiv.apply_symm_apply and Equiv.symm_apply_apply
        t.extend([
            f"simp only [Equiv.apply_eq_iff_eq] at *; linarith [{h_list}]",
            f"have := Equiv.apply_symm_apply σ _; simp only [{h_list}] at *; linarith",
        ])
        # For problems with σ.2 (inverse) values
        for h in hyps:
            t.extend([
                f"have := σ.apply_symm_apply _; rw [{h}] at this; linarith",
                f"have := σ.symm_apply_apply _; rw [{h}] at this; linarith",
            ])

    # For mathd_algebra_422 pattern: σ.1 (x+1) = σ.2 x means σ(x+1) = σ⁻¹(x)
    # So σ(σ(x+1)) = x, i.e., 5*(5*(x+1)-12)-12 = x → solve for x
    if forall_h:
        t.extend([
            f"have h_inv := σ.right_inv; simp only [{', '.join(forall_h)}] at *; linarith",
            f"have h_inv := σ.left_inv; simp only [{', '.join(forall_h)}] at *; linarith",
        ])

    # For mathd_algebra_451 pattern: chain of σ.2 values
    if not forall_h and hyps:
        # Try rewriting with each hypothesis
        rw_chain = " ".join([f"[{h}]" for h in reversed(hyps)])
        t.extend([
            f"simp only [{h_list}]",
            f"simp [{h_list}]",
        ])

    return t


def tactics_existential(content, goal, feat):
    """Existential goals."""
    t = []
    witnesses = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "11", "12", "16", "20", "25", "32", "50", "64", "100"]
    for w in witnesses:
        t.extend([
            f"exact ⟨{w}, by omega⟩",
            f"exact ⟨{w}, by norm_num⟩",
        ])
    # Negative
    for w in ["1", "2", "3", "4", "5"]:
        t.extend([
            f"exact ⟨-{w}, by omega⟩",
            f"exact ⟨-{w}, by norm_num⟩",
        ])
    return t


def tactics_iff(content, goal, feat):
    """Iff goals."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = [
        "constructor <;> intro <;> omega",
        "constructor <;> intro <;> linarith",
        "constructor <;> (intro; simp_all)",
        "constructor <;> intro <;> norm_num",
        "constructor <;> intro <;> simp_all <;> omega",
    ]
    if hyps:
        t.append(f"constructor <;> intro <;> linarith [{h_list}]")
    return t


def tactics_conjunction(content, goal, feat):
    """Conjunction goals."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = []
    if hyps:
        t.extend([
            f"constructor <;> linarith [{h_list}]",
            f"constructor <;> nlinarith [{h_list}]",
            f"constructor <;> omega",
            f"constructor <;> norm_num",
        ])
    else:
        t.extend([
            "constructor <;> omega",
            "constructor <;> norm_num",
            "constructor <;> ring",
        ])
    return t


def tactics_generic(content, goal, feat):
    """Generic fallback cascade."""
    hyps = extract_hypotheses(content)
    h_list = ", ".join(hyps) if hyps else ""
    t = []

    # Universal basics
    t.extend(["omega", "norm_num", "ring", "linarith", "nlinarith",
              "simp", "simp_all", "decide", "native_decide"])

    if hyps:
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            t.append(f"simp only [{h_list}]; {closer}")
        t.extend([
            f"linarith [{h_list}]",
            f"nlinarith [{h_list}]",
        ])

    t.extend([
        "push_cast; ring", "push_cast; norm_num", "push_cast; omega",
        "field_simp; ring", "field_simp; norm_num",
        "ring_nf; norm_num", "ring_nf; omega",
        "simp_all; ring", "simp_all; omega", "simp_all; norm_num",
        "simp_all; linarith", "simp_all; nlinarith",
    ])
    return t


TYPE_HANDLERS = {
    "NUMERIC_EVAL": tactics_numeric_eval,
    "LINEAR_SYSTEM": tactics_linear_system,
    "POLYNOMIAL_ID": tactics_polynomial_id,
    "INEQUALITY": tactics_inequality,
    "FUNC_DEF": tactics_func_def,
    "FINSET_SUM": tactics_finset_sum,
    "SET_CARD": tactics_set_card,
    "COMPLEX_ALG": tactics_complex_alg,
    "DIVISIBILITY": tactics_divisibility,
    "EQUIV_FUNC": tactics_equiv_func,
    "EXISTENTIAL": tactics_existential,
    "IFF": tactics_iff,
    "CONJUNCTION": tactics_conjunction,
    "GENERIC": tactics_generic,
}


# ══════════════════════════════════════════════════════════════
# Proof formatting
# ══════════════════════════════════════════════════════════════

def deduplicate(tactics):
    seen = set()
    return [t for t in tactics if not (t in seen or seen.add(t))]


def format_proof(tactics, max_tactics=70):
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


def make_lean_file(content, proof_text, heartbeats=4000000):
    new = content.replace("by sorry", f"by\n{proof_text}")
    new = new.replace("set_option maxHeartbeats 0", f"set_option maxHeartbeats {heartbeats}")
    return new


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

def generate_proof_for_problem(content, problem):
    """Generate classified proof for a single problem."""
    goal = get_goal(content)
    ptype, feat = classify_problem(content, goal)

    # Get type-specific tactics
    handler = TYPE_HANDLERS.get(ptype, tactics_generic)
    specific_tactics = handler(content, goal, feat)

    # Always add generic fallbacks
    generic = tactics_generic(content, goal, feat)
    all_tactics = specific_tactics + [t for t in generic if t not in set(specific_tactics)]

    return deduplicate(all_tactics), ptype


def main():
    problems = sorted([f[:-5] for f in os.listdir(VALID_DIR) if f.endswith(".lean")])
    print(f"Processing {len(problems)} problems with type-classified tactics...")

    # Phase 1: Copy handcrafted proofs from exp011 (current best)
    exp011_dir = os.path.join(DOMAIN_DIR, "attempts", "exp011")
    handcrafted = set()
    for problem in problems:
        src = os.path.join(exp011_dir, f"{problem}.lean")
        if os.path.exists(src):
            # Check if it's a handcrafted (non-cascade) proof
            with open(src) as f:
                src_content = f.read()
            # Handcrafted proofs contain specific patterns
            is_handcrafted = any(p in src_content for p in [
                "have ", "induction", "obtain", "rcases", "calc", "cases ",
                "suffices", "refine", "ring_nf", "abs_of_", "specialize",
                "intro ", "apply ", "rw [", "left\n", "right\n",
                "linear_combination", "interval_cases",
                "dvd_", "Nat.choose",
            ])
            if is_handcrafted:
                shutil.copy2(src, os.path.join(out_dir, f"{problem}.lean"))
                handcrafted.add(problem)

    print(f"Copied {len(handcrafted)} handcrafted proofs from exp011")

    # Phase 2: Generate type-classified proofs for all non-handcrafted
    type_counts = {}
    for problem in problems:
        if problem in handcrafted:
            continue

        orig_path = os.path.join(VALID_DIR, f"{problem}.lean")
        with open(orig_path) as f:
            content = f.read()

        tactics, ptype = generate_proof_for_problem(content, problem)
        type_counts[ptype] = type_counts.get(ptype, 0) + 1

        proof = format_proof(tactics)
        lean_text = make_lean_file(content, proof)

        with open(os.path.join(out_dir, f"{problem}.lean"), 'w') as f:
            f.write(lean_text)

    print(f"\nType distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # Phase 3: Verify all and identify failures
    print(f"\nPhase 3: Verifying all {len(problems)} proofs...")
    passing = set()
    failing = {}
    for i, problem in enumerate(problems):
        fp = os.path.join(out_dir, f"{problem}.lean")
        ok, err = check_lean(fp)
        if ok:
            passing.add(problem)
        else:
            failing[problem] = err
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(problems)}] {len(passing)} pass, {len(failing)} fail")

    print(f"\nAfter Phase 3: {len(passing)}/{len(problems)} pass")

    # Phase 4: Error-driven retry for non-handcrafted failures
    print("\nPhase 4: Error-driven retry...")
    for retry in range(1, MAX_RETRIES + 1):
        retry_solved = 0
        for problem in list(failing.keys()):
            if problem in handcrafted:
                continue

            err = failing[problem]
            orig_path = os.path.join(VALID_DIR, f"{problem}.lean")
            with open(orig_path) as f:
                content = f.read()

            # Parse error type
            err_lower = err.lower()
            extra_tactics = []

            if "timeout" in err_lower or "heartbeat" in err_lower:
                # Use simpler tactics only
                extra_tactics = ["omega", "norm_num", "ring", "linarith", "decide", "simp"]
                heartbeats = 2000000
            elif "type mismatch" in err_lower:
                extra_tactics = [
                    "push_cast; ring", "push_cast; norm_num", "push_cast; omega",
                    "exact_mod_cast (by omega)", "exact_mod_cast (by norm_num)",
                    "norm_cast; omega", "norm_cast; ring",
                ]
                heartbeats = 4000000
            elif "unsolved" in err_lower:
                extra_tactics = [
                    "simp_all; omega", "simp_all; linarith", "simp_all; nlinarith",
                    "simp_all; ring", "simp_all; norm_num",
                ]
                heartbeats = 4000000
            else:
                heartbeats = 4000000
                extra_tactics = []

            if extra_tactics:
                tactics, _ = generate_proof_for_problem(content, problem)
                combined = extra_tactics + tactics
                proof = format_proof(deduplicate(combined))
                lean_text = make_lean_file(content, proof, heartbeats)

                fp = os.path.join(out_dir, f"{problem}.lean")
                with open(fp, 'w') as f:
                    f.write(lean_text)

                ok, new_err = check_lean(fp)
                if ok:
                    passing.add(problem)
                    del failing[problem]
                    retry_solved += 1
                    print(f"  RETRY {retry} SOLVED: {problem}")
                else:
                    failing[problem] = new_err

        print(f"  Retry {retry}: solved {retry_solved}, remaining {len(failing)}")

    # Phase 5: Brute-force single-tactic sweep for stubborn failures
    print("\nPhase 5: Brute-force sweep...")
    brute_tactics = [
        "omega", "norm_num", "ring", "linarith", "nlinarith",
        "simp", "simp_all", "native_decide", "decide",
        "field_simp; ring", "field_simp; norm_num", "field_simp; linarith",
        "push_cast; ring", "push_cast; omega", "push_cast; norm_num",
        "ring_nf; norm_num", "ring_nf; omega",
        "norm_cast; omega", "norm_cast; norm_num",
        "simp_all; omega", "simp_all; ring", "simp_all; norm_num",
        "simp_all; linarith", "simp_all; nlinarith",
        "norm_num; omega", "simp; omega", "simp; norm_num",
    ]
    brute_solved = 0
    for problem in list(failing.keys()):
        if problem in handcrafted:
            continue

        orig_path = os.path.join(VALID_DIR, f"{problem}.lean")
        with open(orig_path) as f:
            content = f.read()

        solved = False
        for tactic in brute_tactics:
            proof = f"  {tactic}"
            lean_text = make_lean_file(content, proof)
            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(lean_text)

            ok, _ = check_lean(fp)
            if ok:
                passing.add(problem)
                del failing[problem]
                brute_solved += 1
                print(f"  BRUTE SOLVED: {problem} with: {tactic}")
                solved = True
                break

        if not solved:
            # Restore classified proof
            tactics, _ = generate_proof_for_problem(content, problem)
            proof = format_proof(tactics)
            lean_text = make_lean_file(content, proof)
            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(lean_text)

    print(f"\nPhase 5: brute-force solved {brute_solved}")

    # Final summary
    total = len(problems)
    score = len(passing) / total
    print(f"\n{'='*60}")
    print(f"FINAL: {len(passing)}/{total} solved = {score:.4f}")
    print(f"  Handcrafted: {len(handcrafted & passing)}/{len(handcrafted)}")
    print(f"  Cascade-solved: {len(passing - handcrafted)}")
    print(f"\nUnsolved ({len(failing)}):")
    for p in sorted(failing.keys())[:30]:
        print(f"  {p}: {failing[p][:60]}")
    if len(failing) > 30:
        print(f"  ... and {len(failing) - 30} more")


if __name__ == "__main__":
    main()
