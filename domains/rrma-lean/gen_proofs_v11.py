#!/usr/bin/env python3
"""gen_proofs_v11.py — Fixed automated proof search."""
import subprocess, os, sys, time
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
    "amc12a_2009_p25", "amc12a_2019_p21", "amc12a_2020_p21",
    "amc12b_2021_p21", "imo_1962_p4", "imo_1967_p3", "imo_1978_p5",
    "imo_1979_p1", "imo_1987_p6", "imo_1988_p6", "imo_1990_p3",
    "imo_1993_p5", "imo_2006_p3",
]


def read_header(problem):
    path = VALID_DIR / f"{problem}.lean"
    if not path.exists(): return None
    text = path.read_text()
    idx = text.find("by sorry")
    if idx == -1: return None
    return text[:idx + 2]


def generate(problem, header):
    base = header.rstrip()
    cands = []

    # Single tactics
    for t in ["omega", "norm_num", "simp", "ring", "linarith", "nlinarith",
              "decide", "native_decide", "positivity", "aesop", "simp_all",
              "norm_num [Nat.factorial]", "polyrith"]:
        cands.append((f"s_{t[:20]}", f"{base}\n  {t}"))

    # Chains
    for c in [
        "simp; omega", "simp; ring", "simp; linarith", "simp; nlinarith",
        "simp; norm_num", "norm_num; omega", "field_simp; ring",
        "field_simp; linarith", "field_simp; nlinarith",
        "push_cast; omega", "push_cast; ring", "push_cast; norm_num",
        "zify; omega", "zify; ring", "simp_all; omega", "simp_all; ring",
        "simp_all; linarith", "simp_all; nlinarith",
        "norm_num; simp", "norm_num; native_decide",
        "simp; native_decide", "push_cast; simp; omega",
        "simp [Finset.sum]; norm_num", "simp [Finset.sum]; ring",
        "push_cast; norm_num; omega", "norm_num; simp; omega",
        "simp only [Set.ext_iff]; intro x; constructor <;> intro h <;> simp_all",
    ]:
        cands.append((f"c_{c[:30]}", f"{base}\n  {c}"))

    # Constructor patterns
    for s in [
        "constructor <;> linarith", "constructor <;> omega",
        "constructor <;> norm_num", "constructor <;> simp",
        "constructor <;> nlinarith", "constructor <;> native_decide",
        "refine \\u27e8?_, ?_\\u27e9 <;> norm_num",
        "refine \\u27e8?_, ?_\\u27e9 <;> omega",
    ]:
        cands.append((f"sp_{s[:25]}", f"{base}\n  {s}"))

    # Multi-line patterns
    multi = [
        "  constructor\n  \\u00b7 norm_num\n  \\u00b7 intro x hx\n    omega",
        "  constructor\n  \\u00b7 norm_num\n    native_decide\n  \\u00b7 intro x hx\n    omega",
        "  constructor\n  \\u00b7 constructor\n    \\u00b7 norm_num\n    \\u00b7 native_decide\n  \\u00b7 intro x hx\n    omega",
        "  simp only [Set.mem_setOf_eq]\n  constructor\n  \\u00b7 norm_num\n  \\u00b7 intro x hx\n    omega",
        "  ext x\n  simp\n  omega",
        "  ext x\n  simp\n  constructor <;> omega",
    ]
    for i, m in enumerate(multi):
        cands.append((f"m_{i}", f"{base}\n{m}"))

    # Problem-specific
    if problem == "imo_1990_p3":
        for bnd in [10, 20, 50]:
            cands.append((f"ic_{bnd}", base + f"""
  have hle : n \\u2264 {bnd} := by
    by_contra h
    push_neg at h
    have h1 : {bnd} < n := by omega
    have h2 : n ^ 2 \\u2264 2 ^ n + 1 := Nat.le_of_dvd (by positivity) h\\u2081
    nlinarith [Nat.lt_pow_self (show 1 < 2 by omega) n]
  interval_cases n <;> omega"""))

    if problem == "imo_2006_p3":
        cands.append(("sos", base + """
  nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (a - c),
    sq_nonneg (a*b - b*c), sq_nonneg (b*c - a*c), sq_nonneg (a*c - a*b),
    sq_nonneg (a*(a-b)*(a-c)), sq_nonneg (b*(b-a)*(b-c)), sq_nonneg (c*(c-a)*(c-b)),
    sq_nonneg ((a-b)*(b-c)), sq_nonneg ((a-b)*(a-c)), sq_nonneg ((b-c)*(a-c)),
    sq_nonneg ((a-b)*(b-c)*(a-c)),
    Real.sq_sqrt (show (0:ℝ) \\u2264 2 by norm_num)]"""))
        cands.append(("polyrith", base + "\n  polyrith"))

    if problem == "imo_1993_p5":
        # Construct the Fibonacci-like function
        cands.append(("fib", base + """
  refine \\u27e8fun n => if n = 0 then 0 else n + 1, ?_, ?_, ?_\\u27e9
  \\u00b7 simp
  \\u00b7 intro n
    constructor
    \\u00b7 simp; omega
    \\u00b7 intro m; simp; omega
  \\u00b7 intro n; simp; omega"""))

    if "IsGreatest" in header:
        cands.append(("ig_nd", base + """
  constructor
  \\u00b7 simp only [Set.mem_setOf_eq]
    constructor
    \\u00b7 omega
    \\u00b7 native_decide
  \\u00b7 intro x hx
    simp only [Set.mem_setOf_eq] at hx
    omega"""))

    return cands


def test(problem, desc, code, timeout=120):
    tmp = f"/tmp/lean_{problem}_{os.getpid()}.lean"
    # Fix heartbeats
    code = code.replace("maxHeartbeats 0", "maxHeartbeats 800000")
    if "maxHeartbeats" not in code:
        code = code.replace("import Mathlib\n", "import Mathlib\nset_option maxHeartbeats 800000\n", 1)
    try:
        with open(tmp, "w") as f: f.write(code)
        r = subprocess.run(
            ["bash", "-c", f"cd '{MINIF2F_DIR}' && lake env lean '{tmp}'"],
            capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0 and "sorry" not in code
        err = ""
        if not ok:
            for line in (r.stderr + r.stdout).split("\n"):
                if "error:" in line:
                    err = line.strip()[:100]; break
            if not err: err = f"exit {r.returncode}"
        return ok, err
    except subprocess.TimeoutExpired: return False, "TIMEOUT"
    except Exception as e: return False, str(e)[:80]
    finally:
        try: os.unlink(tmp)
        except: pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--problems", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="exp105")
    args = p.parse_args()
    problems = args.problems.split(",") if args.problems else UNSOLVED
    outdir = ATTEMPTS_DIR / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    total_pass = 0; total = 0; results = {}
    for prob in problems:
        if prob in IMPOSSIBLE:
            print(f"SKIP: {prob}"); continue
        hdr = read_header(prob)
        if not hdr:
            print(f"SKIP (no hdr): {prob}"); continue
        cands = generate(prob, hdr)
        print(f"\n{'='*50}\n{prob} ({len(cands)} candidates)\n{'='*50}")
        found = False
        for desc, code in cands:
            total += 1
            sys.stdout.write(f"  {desc}... "); sys.stdout.flush()
            ok, err = test(prob, desc, code, args.timeout)
            if ok:
                print("PASS!")
                total_pass += 1; found = True
                (outdir / f"{prob}.lean").write_text(code)
                print(f"  -> saved")
                results[prob] = desc; break
            else:
                print(f"{'TIMEOUT' if err=='TIMEOUT' else 'fail'}")
        if not found: print(f"  NONE found")
    print(f"\n{'='*50}\nSUMMARY: {total_pass}/{len(problems)} solved, {total} tested\n{'='*50}")
    for k,v in results.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
