#!/usr/bin/env python3
"""
gen_proofs_v7.py — Algorithmic proof generation with hypothesis specialization,
intermediate derivations, and parallel compilation.

Key innovations over v6:
1. Hypothesis specialization: for ∀-quantified hyps, specialize at specific values
2. Intermediate derivations: combine hypotheses to derive new facts before closing
3. Parallel Lean checking via multiprocessing
4. Multi-round error-driven retry with error-specific tactic selection
5. Better Nat equation handling: omega after derived bounds
6. Systematic two-step tactic composition
"""
import os
import re
import sys
import subprocess
import shutil
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ELAN_PATH = os.path.expanduser("~/.elan/bin")

method = sys.argv[1] if len(sys.argv) > 1 else "exp015_v7"
MAX_RETRIES = 3
TIMEOUT = 90
N_WORKERS = min(4, cpu_count())  # parallel Lean checks

out_dir = os.path.join(DOMAIN_DIR, "attempts", method)
os.makedirs(out_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def extract_hypotheses(content):
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', content)

def extract_subst_hyps(content):
    """Find hypotheses of the form (h : var = expr)."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*\w+\s*=\s*[^)]+\)', content)

def extract_forall_hyps(content):
    """Find hypotheses that are universally quantified."""
    return re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀', content)

def extract_forall_hyp_details(content):
    """Extract ∀ hypotheses with their variable names for specialization."""
    results = []
    # Find each hypothesis block — use balanced parentheses
    # Pattern: (h₀ : ∀ x, ...) or (h₀ : ∀ x : Type, ...)
    hyp_pattern = re.compile(r'\((h[₀₁₂₃₄₅₆₇₈₉\d]*)\s*:\s*∀\s+(\w+)(?:\s*:\s*(\w+))?,')
    for m in hyp_pattern.finditer(content):
        name = m.group(1)
        var = m.group(2)
        vtype = m.group(3) or ''
        # Determine if body contains → (precondition)
        # Look ahead from the match to find the body
        rest = content[m.end():]
        has_arrow = '→' in rest.split(')')[0] if ')' in rest else '→' in rest[:200]
        has_neq = '≠' in rest.split(')')[0] if ')' in rest else '≠' in rest[:200]
        results.append({
            'name': name,
            'var': var,
            'type': vtype.strip(),
            'body': '→' if has_arrow else '',
            'has_precond': has_arrow or has_neq,
        })
    return results

def get_goal(content):
    m = re.search(r':\s*(.+?)\s*:=\s*by\s*sorry', content, re.DOTALL)
    return m.group(1).strip() if m else ""

def get_variables(content):
    """Extract variable names and types from theorem signature."""
    all_vars = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', content)
    return [(v, t.strip()) for v, t in all_vars
            if not v.startswith('h') and not v.startswith('S')
            and v not in ('f', 'σ', 'u', 'v', 't')]

def get_hyp_types(content):
    """Extract hypothesis types for smarter tactic selection."""
    results = {}
    for m in re.finditer(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:\s*([^)]+)\)', content):
        results[m.group(1)] = m.group(2).strip()
    return results


# ══════════════════════════════════════════════════════════════
# Lean compiler (parallelizable)
# ══════════════════════════════════════════════════════════════

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
        return False, (err_lines[0] if err_lines else result.stderr[:500])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def check_lean_worker(args):
    """Worker function for parallel checking."""
    problem, filepath = args
    ok, err = check_lean(filepath)
    return problem, ok, err


# ══════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════

def extract_features(content, goal):
    hyps = extract_hypotheses(content)
    subst_h = extract_subst_hyps(content)
    forall_h = extract_forall_hyps(content)
    forall_details = extract_forall_hyp_details(content)

    return {
        "has_and": "∧" in goal,
        "has_or": "∨" in goal,
        "has_exists": "∃" in goal,
        "has_iff": "↔" in goal,
        "has_le": any(op in goal for op in ["≤", "≥", "<", ">"]),
        "has_eq": "=" in goal and "≠" not in goal,
        "has_neq": "≠" in goal,
        "has_complex": "ℂ" in content or "Complex" in content,
        "has_nat": ": ℕ)" in content or ": ℕ\n" in content,
        "has_int": ": ℤ)" in content,
        "has_real": ": ℝ" in content,
        "has_rat": ": ℚ" in content,
        "has_div": "/" in goal,
        "has_mod": "%" in content or "Mod" in content,
        "has_pow": "^" in content,
        "has_finset": "Finset" in content or "∑" in content or "∏" in content,
        "has_dvd": "∣" in goal,
        "has_sqrt": "sqrt" in content.lower() or "Real.sqrt" in content,
        "has_prime": "Prime" in content or "Nat.Prime" in content,
        "has_gcd": "gcd" in content or "lcm" in content,
        "has_abs": "abs" in content or "|" in goal,
        "has_card": ".card" in goal,
        "has_equiv": "Equiv" in content,
        "has_induction": bool(re.search(r'\(n\s*:\s*ℕ\)', content)) and "∀" in content,
        "has_forall_hyps": len(forall_h) > 0,
        "no_hyps": len(hyps) == 0,
        "n_hyps": len(hyps),
        "hyps": hyps,
        "subst_hyps": subst_h,
        "forall_hyps": forall_h,
        "forall_details": forall_details,
        "has_sum_prod": "∑" in content or "∏" in content,
        "has_coe": "↑" in content or "Nat.cast" in content,
    }


# ══════════════════════════════════════════════════════════════
# Proof strategy generation
# ══════════════════════════════════════════════════════════════

def gen_specialization_block(content, feat):
    """Generate `have` statements that specialize ∀-quantified hypotheses."""
    lines = []
    forall_details = feat["forall_details"]

    for fd in forall_details:
        h = fd['name']
        var = fd['var']
        vtype = fd['type']

        # Choose specialization values based on type
        if vtype in ('ℝ', 'ℚ', 'ℂ', '') or 'ℝ' in vtype:
            values = ['0', '1', '-1', '2', '-2', '3', '5', '10']
        elif vtype == 'ℕ' or 'ℕ' in vtype:
            values = ['0', '1', '2', '3', '4', '5']
        elif vtype == 'ℤ':
            values = ['0', '1', '-1', '2', '-2']
        else:
            values = ['0', '1', '2']

        for val in values:
            # For hypotheses that have preconditions (∀ x, P x → Q x)
            if '→' in fd['body'] or '≠' in fd['body']:
                # Need to provide precondition proof
                lines.append(f"  have h_{h}_{val} := {h} {val} (by norm_num)")
                lines.append(f"  have h_{h}_{val} := {h} ({val}) (by norm_num)")
                lines.append(f"  have h_{h}_{val} := {h} ({val}) ⟨by norm_num, by norm_num⟩")
            else:
                lines.append(f"  have h_{h}_{val} := {h} {val}")
                lines.append(f"  have h_{h}_{val} := {h} ({val})")

    return lines


def gen_derivation_block(content, feat):
    """Generate intermediate `have` derivations from combining hypotheses."""
    lines = []
    hyps = feat["hyps"]

    if len(hyps) >= 2:
        h0, h1 = hyps[0], hyps[1]
        # Linear combinations of hypotheses
        lines.extend([
            f"  have hd1 : _ := by linarith [{h0}, {h1}]",
            f"  have hd2 : _ := by nlinarith [{h0}, {h1}]",
        ])

        if feat["has_nat"]:
            # For Nat: try omega after deriving facts
            lines.append(f"  have hd3 : _ := by omega")

    # For problems with products and sums in Nat
    if feat["has_nat"] and feat["has_pow"]:
        for h in hyps:
            lines.append(f"  have hsq_{h} := sq_nonneg ({h})")

    return lines


def gen_closing_tactics(content, goal, feat):
    """Generate closing tactic sequences."""
    hyps = feat["hyps"]
    h_list = ", ".join(hyps) if hyps else ""
    variables = get_variables(content)
    var_names = [v for v, _ in variables]
    tactics = []

    # ── Basic closers ──
    basics = ["omega", "norm_num", "ring", "linarith", "nlinarith",
              "simp", "simp_all", "native_decide", "decide"]

    # ── With hypothesis hints ──
    if hyps:
        for base in ["linarith", "nlinarith"]:
            tactics.append(f"{base} [{h_list}]")
        tactics.extend(basics)

        # SOS witnesses
        if feat["has_le"] or feat["has_pow"]:
            for v in var_names[:3]:
                tactics.append(f"nlinarith [sq_nonneg {v}, {h_list}]")
                tactics.append(f"nlinarith [sq_nonneg ({v} - 1), {h_list}]")
            if len(var_names) >= 2:
                a, b = var_names[0], var_names[1]
                tactics.extend([
                    f"nlinarith [sq_nonneg ({a} - {b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a} + {b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a} - {b}), sq_nonneg ({a} + {b}), {h_list}]",
                    f"nlinarith [sq_nonneg (2*{a} - {b}), sq_nonneg ({a} - 2*{b}), {h_list}]",
                    f"nlinarith [sq_nonneg ({a}*{b} - 1), {h_list}]",
                ])
                if len(var_names) >= 3:
                    c = var_names[2]
                    tactics.extend([
                        f"nlinarith [sq_nonneg ({a} - {b}), sq_nonneg ({b} - {c}), sq_nonneg ({a} - {c}), {h_list}]",
                    ])

        # Hypothesis-driven simp
        for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
            tactics.append(f"simp only [{h_list}]; {closer}")
            tactics.append(f"simp [{h_list}]; {closer}")

        # With subst
        if feat["subst_hyps"]:
            for closer in ["ring", "norm_num", "omega", "linarith", "nlinarith"]:
                tactics.append(f"subst_vars; {closer}")

        # linear_combination
        for h in hyps[:3]:
            tactics.append(f"linear_combination {h}")
        if len(hyps) >= 2:
            h0, h1 = hyps[0], hyps[1]
            for a, b in [(1,1), (1,-1), (-1,1), (2,-1), (-1,2), (3,-1), (-3,2)]:
                parts = []
                if a == 1: parts.append(h0)
                elif a == -1: parts.append(f"-{h0}")
                elif a != 0: parts.append(f"{a} * {h0}")
                if b == 1: parts.append(h1)
                elif b == -1: parts.append(f"-{h1}")
                elif b != 0: parts.append(f"{b} * {h1}")
                if parts:
                    tactics.append(f"linear_combination {' + '.join(parts)}")
    else:
        tactics.extend(basics)

    # ── Field division ──
    if feat["has_div"]:
        base = f"[{h_list}]" if hyps else ""
        tactics.extend([
            "field_simp; ring",
            "field_simp; norm_num",
            f"field_simp; linarith {base}" if base else "field_simp; linarith",
            f"field_simp; nlinarith {base}" if base else "field_simp; nlinarith",
        ])

    # ── Conjunction ──
    if feat["has_and"]:
        base = f"[{h_list}]" if hyps else ""
        for closer in ["linarith", "nlinarith", "omega", "norm_num", "ring", "simp"]:
            arg = f" {base}" if base and closer in ("linarith", "nlinarith") else ""
            tactics.append(f"constructor <;> {closer}{arg}")
        if feat["subst_hyps"]:
            tactics.append("constructor <;> (subst_vars; norm_num)")
            tactics.append("constructor <;> (subst_vars; linarith)")

    # ── Complex ──
    if feat["has_complex"]:
        tactics.extend([
            "ring_nf; norm_num [Complex.I_sq]",
            "simp [Complex.ext_iff, Complex.I_sq]; constructor <;> ring",
        ])
        if feat["subst_hyps"]:
            tactics.extend([
                "subst_vars; ring",
                "subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]",
            ])

    # ── Existential ──
    if feat["has_exists"]:
        for w in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "12", "16", "20", "25", "32", "50", "64", "100"]:
            tactics.append(f"exact ⟨{w}, by omega⟩")
            tactics.append(f"exact ⟨{w}, by norm_num⟩")
        for w in ["1", "2", "3", "4", "5"]:
            tactics.append(f"exact ⟨-{w}, by omega⟩")
            tactics.append(f"exact ⟨-{w}, by norm_num⟩")

    # ── Iff ──
    if feat["has_iff"]:
        tactics.extend([
            "constructor <;> intro <;> omega",
            "constructor <;> intro <;> linarith",
            "constructor <;> (intro; simp_all)",
        ])

    # ── Cast ──
    if feat["has_coe"] or feat["has_nat"]:
        for closer in ["ring", "omega", "norm_num", "linarith"]:
            tactics.append(f"push_cast; {closer}")
            tactics.append(f"norm_cast; {closer}")

    # ── Two-step compositions ──
    step1 = ["ring_nf", "simp_all", "push_cast", "norm_cast", "field_simp"]
    step2 = ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp"]
    for s1 in step1:
        for s2 in step2:
            if s1 != s2:
                tactics.append(f"{s1}; {s2}")

    return tactics


def gen_specialization_proof(content, goal, feat):
    """Generate a proof that specializes ∀-quantified hypotheses then closes."""
    if not feat["forall_details"]:
        return []

    proofs = []
    forall_details = feat["forall_details"]

    # For each ∀-hypothesis, try specializing at various values
    for fd in forall_details:
        h = fd['name']
        has_precond = fd.get('has_precond', False)

        if has_precond:
            # Hypothesis like: ∀ x, x - 3 ≠ 0 ∧ x - 5 ≠ 0 → P x
            # Try both tuple and simple precondition forms
            precond_forms = ["⟨by norm_num, by norm_num⟩", "by norm_num"]
            values = ["0", "1", "2", "(-1)", "4", "6", "10"]
            closers = ["linarith", "nlinarith", "omega", "norm_num",
                        "field_simp; ring", "field_simp; linarith"]

            # Single specialization + close
            for val in values:
                for pf in precond_forms:
                    for closer in closers:
                        block = f"  have := {h} {val} ({pf})\n  {closer}"
                        proofs.append(block)

            # Two specializations then close (only best value pairs)
            value_pairs = [("0", "1"), ("0", "2"), ("1", "2"), ("0", "(-1)"),
                           ("1", "(-1)"), ("4", "6"), ("0", "10")]
            for v1, v2 in value_pairs:
                for pf in precond_forms[:1]:  # just tuple form
                    for closer in ["linarith", "nlinarith", "field_simp; linarith"]:
                        block = (f"  have h_s1 := {h} {v1} ({pf})\n"
                                 f"  have h_s2 := {h} {v2} ({pf})\n"
                                 f"  {closer} [h_s1, h_s2]")
                        proofs.append(block)
                    if feat["has_and"]:
                        block = (f"  have h_s1 := {h} {v1} ({pf})\n"
                                 f"  have h_s2 := {h} {v2} ({pf})\n"
                                 f"  constructor <;> linarith [h_s1, h_s2]")
                        proofs.append(block)
                        block = (f"  have h_s1 := {h} {v1} ({pf})\n"
                                 f"  have h_s2 := {h} {v2} ({pf})\n"
                                 f"  constructor <;> nlinarith [h_s1, h_s2]")
                        proofs.append(block)

        else:
            # Simple ∀ x, ... hypothesis — specialize directly
            values = ['0', '1', '2', '3', '(-1)', '(-2)', '5', '10']
            for val in values:
                for closer in ["linarith", "nlinarith", "omega", "norm_num", "ring",
                                "simp", "field_simp; ring"]:
                    block = f"  have := {h} {val}\n  {closer}"
                    proofs.append(block)

            # Two specializations
            for i, v1 in enumerate(values[:5]):
                for v2 in values[i+1:6]:
                    for closer in ["linarith", "nlinarith", "omega"]:
                        block = (f"  have h_s1 := {h} {v1}\n"
                                 f"  have h_s2 := {h} {v2}\n"
                                 f"  {closer} [h_s1, h_s2]")
                        proofs.append(block)

    return proofs


def gen_induction_proof(content, goal, feat):
    """Generate induction proof templates for ∀ n : ℕ, ... problems."""
    if not feat["has_induction"]:
        return []

    proofs = []
    # Find the induction variable
    m = re.search(r'\((\w+)\s*:\s*ℕ\)', content)
    if not m:
        return []
    ind_var = m.group(1)

    closers = ["omega", "ring", "norm_num", "linarith", "nlinarith",
               "simp", "ring_nf; omega", "push_cast; ring", "push_cast; linarith"]

    for closer in closers:
        proofs.append(f"  induction {ind_var} with\n  | zero => {closer}\n  | succ k ih => {closer}")
        proofs.append(f"  induction {ind_var} with\n  | zero => simp; {closer}\n  | succ k ih => simp_all; {closer}")
        proofs.append(f"  induction {ind_var} with\n  | zero => {closer}\n  | succ k ih => simp_all [ih]; {closer}")
        proofs.append(f"  induction {ind_var} with\n  | zero => norm_num\n  | succ k ih => push_cast [Nat.succ_eq_add_one] at *; {closer}")

    return proofs


def gen_nat_equation_proof(content, goal, feat):
    """For Nat equation problems, try interval_cases after bounding."""
    if not feat["has_nat"]:
        return []

    proofs = []
    hyps = feat["hyps"]
    variables = get_variables(content)
    nat_vars = [v for v, t in variables if 'ℕ' in t]

    if not nat_vars or not hyps:
        return []

    # For small Nat problems, try omega with various intermediate facts
    h_list = ", ".join(hyps)

    # Try interval_cases on Nat variables after establishing bounds
    for v in nat_vars[:2]:
        for bound in [10, 20, 50, 100]:
            block = (f"  have hb : {v} ≤ {bound} := by nlinarith [{h_list}]\n"
                     f"  interval_cases {v} <;> omega")
            proofs.append(block)
            block = (f"  have hb : {v} ≤ {bound} := by nlinarith [{h_list}]\n"
                     f"  interval_cases {v} <;> simp_all <;> omega")
            proofs.append(block)

    # For two Nat vars with equations, try bounding both
    if len(nat_vars) >= 2:
        v1, v2 = nat_vars[0], nat_vars[1]
        for b in [10, 20, 50]:
            block = (f"  have hb1 : {v1} ≤ {b} := by nlinarith [{h_list}]\n"
                     f"  have hb2 : {v2} ≤ {b} := by nlinarith [{h_list}]\n"
                     f"  interval_cases {v1} <;> interval_cases {v2} <;> omega")
            proofs.append(block)
            block = (f"  have hb1 : {v1} ≤ {b} := by nlinarith [{h_list}]\n"
                     f"  have hb2 : {v2} ≤ {b} := by nlinarith [{h_list}]\n"
                     f"  interval_cases {v1} <;> interval_cases {v2} <;> simp_all <;> omega")
            proofs.append(block)

    return proofs


def gen_abs_proof(content, goal, feat):
    """Handle absolute value problems."""
    if not feat["has_abs"]:
        return []

    hyps = feat["hyps"]
    h_list = ", ".join(hyps) if hyps else ""
    proofs = []

    for unfold in ["abs_of_nonneg", "abs_of_neg", "abs_of_pos"]:
        for closer in ["linarith", "nlinarith", "omega", "norm_num"]:
            if hyps:
                proofs.append(f"  rw [{unfold} (by linarith [{h_list}])]; {closer} [{h_list}]")
                proofs.append(f"  simp only [abs_le]; constructor <;> linarith [{h_list}]")
            proofs.append(f"  rw [{unfold} (by linarith)]; {closer}")

    return proofs


def gen_sqrt_proof(content, goal, feat):
    """Handle Real.sqrt problems."""
    if not feat["has_sqrt"]:
        return []

    hyps = feat["hyps"]
    h_list = ", ".join(hyps) if hyps else ""
    proofs = []

    sqrt_lemmas = [
        "Real.sqrt_sq", "Real.sq_sqrt", "Real.sqrt_eq_iff_sq_eq",
        "Real.sqrt_lt'", "Real.lt_sqrt", "Real.sqrt_le_sqrt",
        "Real.sqrt_one", "Real.sqrt_zero",
    ]

    for lem in sqrt_lemmas:
        for closer in ["linarith", "nlinarith", "ring", "norm_num"]:
            proofs.append(f"  rw [{lem}] <;> [{closer}, linarith]")
            proofs.append(f"  simp [{lem}]; {closer}")
            if hyps:
                proofs.append(f"  have := {lem}; {closer} [{h_list}]")

    # Square both sides strategy
    if hyps:
        proofs.extend([
            f"  nlinarith [Real.sq_sqrt (by linarith [{h_list}] : (0:ℝ) ≤ _), {h_list}]",
            f"  nlinarith [Real.sq_sqrt (by positivity), {h_list}]",
        ])

    return proofs


# ══════════════════════════════════════════════════════════════
# Proof assembly
# ══════════════════════════════════════════════════════════════

def format_proof_cascade(tactics, max_tactics=80):
    """Format simple tactics into first | solve | ... cascade."""
    simple = [t for t in tactics if "\n" not in t][:max_tactics]
    lines = ["  first"]
    for s in simple:
        lines.append(f"  | solve | {s}")
    return "\n".join(lines)


def format_multiline_attempts(proofs, max_attempts=25):
    """Format multi-line proof attempts as try blocks."""
    lines = []
    for p in proofs[:max_attempts]:
        if "\n" in p:
            lines.append(f"  try\n{p}")
        else:
            lines.append(f"  try {p}")
    return "\n".join(lines)


def make_lean_file(content, proof_text, heartbeats=4000000):
    new = content.replace("by sorry", f"by\n{proof_text}")
    new = re.sub(r'set_option maxHeartbeats \d+', f'set_option maxHeartbeats {heartbeats}', new)
    if 'set_option maxHeartbeats' not in new:
        new = new.replace('import Mathlib', f'import Mathlib\nset_option maxHeartbeats {heartbeats}')
    return new


# ══════════════════════════════════════════════════════════════
# Error-driven retry
# ══════════════════════════════════════════════════════════════

def error_specific_tactics(err, content, goal, feat):
    """Generate tactics based on specific error message."""
    hyps = feat["hyps"]
    h_list = ", ".join(hyps) if hyps else ""
    extra = []

    err_lower = err.lower()

    if "timeout" in err_lower or "heartbeat" in err_lower:
        # Simpler tactics only
        return ["omega", "norm_num", "ring", "linarith", "decide",
                "simp", "native_decide"], 2000000

    if "type mismatch" in err_lower:
        extra = [
            "push_cast; ring", "push_cast; norm_num", "push_cast; omega",
            "exact_mod_cast (by omega)", "exact_mod_cast (by norm_num)",
            "norm_cast; omega", "norm_cast; ring", "norm_cast; norm_num",
        ]
        if hyps:
            extra.extend([
                f"push_cast; linarith [{h_list}]",
                f"norm_cast; linarith [{h_list}]",
            ])

    if "unsolved" in err_lower:
        extra = [
            "simp_all; omega", "simp_all; linarith", "simp_all; nlinarith",
            "simp_all; ring", "simp_all; norm_num",
            "aesop", "tauto",
        ]

    if "unknown identifier" in err_lower:
        # Likely wrong lemma name — try without specific lemmas
        extra = ["omega", "norm_num", "ring", "simp", "linarith", "nlinarith",
                 "simp_all", "native_decide"]

    if "failed" in err_lower and "tactic" in err_lower:
        extra = [
            "simp_all", "simp_all; omega", "aesop",
            "norm_num; omega", "ring_nf; norm_num",
        ]

    return extra, 4000000


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

def generate_all_proofs(problem, content):
    """Generate proof text for a problem."""
    goal = get_goal(content)
    feat = extract_features(content, goal)

    # 1. Closing tactics (cascade)
    closing = gen_closing_tactics(content, goal, feat)

    # 2. Multi-line proof strategies
    multi_proofs = []
    multi_proofs.extend(gen_specialization_proof(content, goal, feat))
    multi_proofs.extend(gen_induction_proof(content, goal, feat))
    multi_proofs.extend(gen_nat_equation_proof(content, goal, feat))
    multi_proofs.extend(gen_abs_proof(content, goal, feat))
    multi_proofs.extend(gen_sqrt_proof(content, goal, feat))

    return closing, multi_proofs


def main():
    problems = sorted([f[:-5] for f in os.listdir(VALID_DIR) if f.endswith(".lean")])
    print(f"Processing {len(problems)} problems with gen_proofs_v7...")

    # Phase 1: Copy handcrafted proofs from best experiment (exp011)
    best_dir = os.path.join(DOMAIN_DIR, "attempts", "exp011")
    handcrafted = set()

    # First identify which exp011 proofs are handcrafted (contain multi-step reasoning)
    handcraft_markers = [
        "have ", "induction", "obtain", "rcases", "calc", "cases ",
        "suffices", "refine", "ring_nf", "abs_of_", "specialize",
        "intro ", "apply ", "rw [", "left\n", "right\n",
        "linear_combination", "interval_cases",
        "dvd_", "Nat.choose", "conv ", "show ",
    ]

    if os.path.isdir(best_dir):
        for problem in problems:
            src = os.path.join(best_dir, f"{problem}.lean")
            if not os.path.exists(src):
                continue
            with open(src) as f:
                src_content = f.read()
            is_handcrafted = any(p in src_content for p in handcraft_markers)
            if is_handcrafted:
                shutil.copy2(src, os.path.join(out_dir, f"{problem}.lean"))
                handcrafted.add(problem)

    print(f"Copied {len(handcrafted)} handcrafted proofs from exp011")

    # Phase 2: Generate proofs for all non-handcrafted problems
    for problem in problems:
        if problem in handcrafted:
            continue

        orig_path = os.path.join(VALID_DIR, f"{problem}.lean")
        with open(orig_path) as f:
            content = f.read()

        closing, multi_proofs = generate_all_proofs(problem, content)

        # Build proof: try multi-line strategies first, then cascade
        parts = []
        if multi_proofs:
            parts.append(format_multiline_attempts(multi_proofs))
        parts.append(format_cascade_block(closing))

        proof_text = "\n".join(parts)
        lean_text = make_lean_file(content, proof_text)

        with open(os.path.join(out_dir, f"{problem}.lean"), 'w') as f:
            f.write(lean_text)

    # Phase 3: Parallel verification
    print(f"\nPhase 3: Parallel verification ({N_WORKERS} workers)...")
    tasks = []
    for problem in problems:
        fp = os.path.join(out_dir, f"{problem}.lean")
        tasks.append((problem, fp))

    passing = set()
    failing = {}

    with Pool(N_WORKERS) as pool:
        results = pool.map(check_lean_worker, tasks)

    for problem, ok, err in results:
        if ok:
            passing.add(problem)
        else:
            failing[problem] = err

    print(f"After Phase 3: {len(passing)}/{len(problems)} pass ({len(passing)/len(problems):.4f})")

    # Phase 4: Error-driven retry (serial, only for failures)
    print(f"\nPhase 4: Error-driven retry ({len(failing)} failures)...")
    for retry_round in range(1, MAX_RETRIES + 1):
        retry_solved = 0
        for problem in list(failing.keys()):
            if problem in handcrafted:
                continue

            err = failing[problem]
            orig_path = os.path.join(VALID_DIR, f"{problem}.lean")
            with open(orig_path) as f:
                content = f.read()

            goal = get_goal(content)
            feat = extract_features(content, goal)

            extra_tactics, heartbeats = error_specific_tactics(err, content, goal, feat)
            if not extra_tactics:
                continue

            # Combine error-specific with original
            closing, _ = generate_all_proofs(problem, content)
            combined = extra_tactics + closing
            # Remove duplicates preserving order
            seen = set()
            deduped = []
            for t in combined:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)

            proof = format_cascade_block(deduped)
            lean_text = make_lean_file(content, proof, heartbeats)

            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(lean_text)

            ok, new_err = check_lean(fp)
            if ok:
                passing.add(problem)
                del failing[problem]
                retry_solved += 1
                print(f"  RETRY {retry_round} SOLVED: {problem}")
            else:
                failing[problem] = new_err

        print(f"  Retry {retry_round}: solved {retry_solved}, remaining {len(failing)}")
        if retry_solved == 0:
            break

    # Phase 5: Brute-force single-tactic sweep
    print(f"\nPhase 5: Brute-force sweep on {len([p for p in failing if p not in handcrafted])} non-handcrafted failures...")
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
        "aesop",
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

            ok, _ = check_lean(fp, timeout=30)  # shorter timeout for brute
            if ok:
                passing.add(problem)
                del failing[problem]
                brute_solved += 1
                print(f"  BRUTE SOLVED: {problem} with: {tactic}")
                solved = True
                break

        if not solved:
            # Restore original generated proof
            closing, multi_proofs = generate_all_proofs(problem, content)
            parts = []
            if multi_proofs:
                parts.append(format_multiline_attempts(multi_proofs[:20]))
            parts.append(format_cascade_block(closing))
            proof_text = "\n".join(parts)
            lean_text = make_lean_file(content, proof_text)
            fp = os.path.join(out_dir, f"{problem}.lean")
            with open(fp, 'w') as f:
                f.write(lean_text)

    print(f"Phase 5: brute-force solved {brute_solved}")

    # Final summary
    total = len(problems)
    score = len(passing) / total
    print(f"\n{'='*60}")
    print(f"FINAL: {len(passing)}/{total} solved = {score:.4f}")
    print(f"  Handcrafted: {len(handcrafted & passing)}/{len(handcrafted)}")
    print(f"  Cascade/algo solved: {len(passing - handcrafted)}")
    print(f"\nUnsolved ({len(failing)}):")
    for p in sorted(failing.keys())[:40]:
        print(f"  {p}: {failing[p][:80]}")
    if len(failing) > 40:
        print(f"  ... and {len(failing) - 40} more")

    # Save detailed results
    results_file = os.path.join(out_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "score": score,
            "passing": sorted(passing),
            "failing": {k: v[:200] for k, v in failing.items()},
            "handcrafted": sorted(handcrafted),
        }, f, indent=2)


def format_cascade_block(tactics):
    """Format tactics into a first | solve | cascade."""
    seen = set()
    deduped = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    simple = [t for t in deduped if "\n" not in t][:80]
    lines = ["  first"]
    for s in simple:
        lines.append(f"  | solve | {s}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
