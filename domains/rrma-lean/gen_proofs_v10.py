#!/usr/bin/env python3
"""
gen_proofs_v10.py — Automated proof search for MiniF2F-valid problems.
Generates candidate proofs using systematic tactic combinations and tests with lake env lean.
"""

import subprocess
import os
import sys
import time
from pathlib import Path

DOMAIN_DIR = Path(__file__).parent
MINIF2F_DIR = Path("/home/vincent/miniF2F-lean4")
ATTEMPTS_DIR = DOMAIN_DIR / "attempts"
VALID_DIR = MINIF2F_DIR / "MiniF2F" / "Valid"

IMPOSSIBLE = {
    "mathd_algebra_433", "mathd_algebra_437", "aime_1984_p5",
    "aime_1988_p3", "mathd_numbertheory_126", "amc12a_2020_p13",
    "amc12a_2002_p21", "mathd_algebra_282",
}

UNSOLVED = [
    "aime_1987_p8", "aime_1991_p6", "aime_1994_p4", "aime_1997_p11",
    "amc12a_2009_p25", "amc12a_2010_p22",
    "amc12a_2019_p21", "amc12a_2020_p21", "amc12b_2021_p21",
    "imo_1962_p4", "imo_1967_p3", "imo_1978_p5", "imo_1979_p1",
    "imo_1987_p6", "imo_1988_p6", "imo_1990_p3", "imo_1993_p5",
    "imo_2006_p3",
]


def read_theorem_header(problem):
    path = VALID_DIR / f"{problem}.lean"
    if not path.exists():
        return None
    text = path.read_text()
    idx = text.find("by sorry")
    if idx == -1:
        return None
    return text[:idx + 2]  # ends with "by"


def generate_candidates(problem, header):
    base = header.rstrip()
    candidates = []

    # ─── Tier 1: Single tactics ───
    singles = [
        "omega", "norm_num", "simp", "ring", "linarith", "nlinarith",
        "decide", "native_decide", "positivity", "aesop",
        "simp_all", "norm_num [Nat.factorial]",
    ]
    for t in singles:
        candidates.append((f"single_{t[:20]}", f"{base}\n  {t}"))

    # ─── Tier 2: Semicolon chains ───
    chains = [
        "simp; omega", "simp; ring", "simp; linarith", "simp; nlinarith",
        "simp; norm_num", "norm_num; omega", "field_simp; ring",
        "field_simp; linarith", "field_simp; nlinarith",
        "push_cast; omega", "push_cast; ring", "push_cast; norm_num",
        "zify; omega", "zify; ring", "simp_all; omega", "simp_all; ring",
        "simp_all; linarith", "simp_all; nlinarith",
        "norm_num; simp", "norm_num; native_decide",
        "simp; native_decide", "push_cast; simp; omega",
        "push_cast; simp; ring", "push_cast; simp; norm_num",
        "simp [Finset.sum]; norm_num", "simp [Finset.sum]; ring",
        "push_cast; norm_num; omega",
    ]
    for c in chains:
        candidates.append((f"chain_{c[:30]}", f"{base}\n  {c}"))

    # ─── Tier 3: Constructor splits ───
    splits = [
        "constructor <;> linarith", "constructor <;> omega",
        "constructor <;> norm_num", "constructor <;> simp",
        "constructor <;> ring", "constructor <;> nlinarith",
        "constructor <;> native_decide", "constructor <;> decide",
        "refine ⟨?_, ?_⟩ <;> norm_num", "refine ⟨?_, ?_⟩ <;> omega",
        "refine ⟨?_, ?_⟩ <;> simp", "refine ⟨?_, ?_⟩ <;> native_decide",
    ]
    for s in splits:
        candidates.append((f"split_{s[:30]}", f"{base}\n  {s}"))

    # ─── Tier 4: Multi-line patterns ───
    multiline = [
        "  constructor\n  · norm_num\n  · intro x hx\n    omega",
        "  constructor\n  · norm_num\n    native_decide\n  · intro x hx\n    omega",
        "  constructor\n  · constructor\n    · norm_num\n    · native_decide\n  · intro x hx\n    omega",
        "  simp only [Set.mem_setOf_eq]\n  constructor\n  · norm_num\n  · intro x hx\n    omega",
        "  ext x\n  simp\n  omega",
        "  ext x\n  simp\n  constructor <;> omega",
        "  simp only [Finset.mem_Icc]\n  omega",
    ]
    for i, m in enumerate(multiline):
        candidates.append((f"multi_{i}", f"{base}\n{m}"))

    # ─── Problem-specific strategies ───

    # For imo_1990_p3 (n²|2^n+1 → n=3)
    if problem == "imo_1990_p3":
        # Bounded search: n² | 2^n+1 with n≥2, so n ≤ some bound
        for bound in [10, 20, 50, 100]:
            candidates.append((f"imo1990_ic_{bound}", f"""{base}
  have hle : n ≤ {bound} := by
    by_contra h
    push_neg at h
    have : n^2 > 2^n + 1 := by nlinarith [Nat.lt_pow_self (show 1 < n by omega)]
    omega
  interval_cases n <;> omega"""))

    # For imo_2006_p3 (Schur-like inequality)
    if problem == "imo_2006_p3":
        candidates.append(("imo2006_nlinarith", f"""{base}
  nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (a - c),
    sq_nonneg (a*b - b*c), sq_nonneg (b*c - a*c), sq_nonneg (a*c - a*b),
    sq_nonneg (a*(a - b)), sq_nonneg (b*(b - c)), sq_nonneg (c*(c - a)),
    sq_nonneg (a^2 - b*c), sq_nonneg (b^2 - a*c), sq_nonneg (c^2 - a*b),
    mul_self_nonneg (a^2 - b^2), mul_self_nonneg (b^2 - c^2),
    sq_nonneg ((a-b)*(b-c)), sq_nonneg ((a-b)*(a-c)), sq_nonneg ((b-c)*(a-c)),
    sq_nonneg ((a-b)*(b-c)*(a-c)),
    Real.sq_sqrt (show (0:ℝ) ≤ 2 by norm_num)]"""))
        candidates.append(("imo2006_polyrith", f"""{base}
  polyrith"""))

    # For amc12a_2010_p22 (sum of |kx-1| ≥ 49)
    if problem == "amc12a_2010_p22":
        candidates.append(("amc2010_calc", f"""{base}
  -- Split Icc 1 119 into {1..84} and {85..119}
  -- ∑₁⁸⁴ k = ∑₈₅¹¹⁹ k = 3570
  -- |3570x - 84| + |3570x - 35| ≥ |84 - 35| = 49
  -- and ∑|kx-1| ≥ |∑₁⁸⁴(kx-1)| + |∑₈₅¹¹⁹(kx-1)|
  calc ∑ k ∈ Finset.Icc (1:ℤ) 119, |↑k * x - 1|
      ≥ |∑ k ∈ Finset.Icc (1:ℤ) 84, (↑k * x - 1)| + |∑ k ∈ Finset.Icc (85:ℤ) 119, (↑k * x - 1)| := by
        have hsplit : Finset.Icc (1:ℤ) 119 = Finset.Icc 1 84 ∪ Finset.Icc 85 119 := by
          ext k; simp [Finset.mem_Icc, Finset.mem_union]; omega
        have hdisj : Disjoint (Finset.Icc (1:ℤ) 84) (Finset.Icc 85 119) := by
          simp [Finset.disjoint_Icc]; omega
        rw [hsplit, Finset.sum_union hdisj]
        exact abs_add_le _ _ |>.le |>.trans (le_refl _) |>.symm ▸ le_abs_self _ |>.symm ▸ sorry
      _ ≥ 49 := by sorry"""))

    # For imo_1988_p6 (famous Vieta jumping)
    if problem == "imo_1988_p6":
        candidates.append(("imo1988_sqrt", f"""{base}
  exact ⟨Nat.sqrt ((a ^ 2 + b ^ 2) / (a * b + 1)), by
    have := Nat.sqrt_eq ((a ^ 2 + b ^ 2) / (a * b + 1))
    push_cast
    rw [this]
    field_simp⟩"""))

    # For amc12a_2020_p21 (lcm condition)
    if problem == "amc12a_2020_p21":
        candidates.append(("amc2020_ext", f"""{base}
  have : S = Finset.filter (fun n => 5 ∣ n ∧ Nat.lcm (5!) n = 5 * Nat.gcd (10!) n) (Finset.range 100000) := by
    ext n
    simp [h₀]
    sorry
  sorry"""))

    return candidates


def test_proof(problem, desc, lean_code, timeout=120):
    tmp = f"/tmp/lean_proof_{problem}_{os.getpid()}.lean"
    try:
        # Ensure maxHeartbeats is high
        if "maxHeartbeats 0" in lean_code:
            lean_code = lean_code.replace("maxHeartbeats 0", "maxHeartbeats 800000")
        elif "maxHeartbeats" not in lean_code:
            lean_code = lean_code.replace("import Mathlib\n", "import Mathlib\nset_option maxHeartbeats 800000\n", 1)

        with open(tmp, "w") as f:
            f.write(lean_code)

        result = subprocess.run(
            ["bash", "-c", f"cd '{MINIF2F_DIR}' && lake env lean '{tmp}'"],
            capture_output=True, text=True, timeout=timeout
        )

        passed = result.returncode == 0 and "sorry" not in lean_code
        error = ""
        if not passed:
            for line in (result.stderr + result.stdout).split("\n"):
                if "error:" in line:
                    error = line.strip()[:120]
                    break
            if not error and result.returncode != 0:
                error = f"exit code {result.returncode}"
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:100]
    finally:
        try: os.unlink(tmp)
        except: pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--problems", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="exp105")
    args = parser.parse_args()

    problems = args.problems.split(",") if args.problems else UNSOLVED
    output_dir = ATTEMPTS_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pass = 0
    total_tested = 0
    results = {}

    for problem in problems:
        if problem in IMPOSSIBLE:
            print(f"SKIP (impossible): {problem}")
            continue

        header = read_theorem_header(problem)
        if header is None:
            print(f"SKIP (no header): {problem}")
            continue

        candidates = generate_candidates(problem, header)
        print(f"\n{'='*60}")
        print(f"Problem: {problem} ({len(candidates)} candidates)")
        print(f"{'='*60}")

        found = False
        for desc, lean_code in candidates:
            total_tested += 1
            sys.stdout.write(f"  Testing {desc}... ")
            sys.stdout.flush()

            passed, error = test_proof(problem, desc, lean_code, args.timeout)

            if passed:
                print(f"PASS!")
                total_pass += 1
                found = True
                out_path = output_dir / f"{problem}.lean"
                with open(out_path, "w") as f:
                    f.write(lean_code)
                print(f"  -> Written to {out_path}")
                results[problem] = desc
                break
            else:
                if error == "TIMEOUT":
                    print(f"TIMEOUT")
                else:
                    print(f"fail")

        if not found:
            print(f"  NO PROOF FOUND for {problem}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pass}/{len(problems)} solved, {total_tested} tested")
    print(f"{'='*60}")
    for prob, desc in results.items():
        print(f"  {prob}: {desc}")


if __name__ == "__main__":
    main()
