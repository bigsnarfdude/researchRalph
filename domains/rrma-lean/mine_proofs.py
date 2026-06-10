#!/usr/bin/env python3
"""
mine_proofs.py — Mine proofs from all experiment directories.
For each failing problem in the target, try proofs from other experiments.
Only keep proofs that compile and have no sorry.
"""
import os, subprocess, sys, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
ELAN_PATH = os.path.expanduser("~/.elan/bin")
ATTEMPTS_DIR = os.path.join(DOMAIN_DIR, "attempts")

TARGET = sys.argv[1] if len(sys.argv) > 1 else "exp105"
TIMEOUT = 120
MAX_WORKERS = 2

KNOWN_IMPOSSIBLE = {
    "aime_1984_p5", "aime_1988_p3", "amc12a_2002_p21", "amc12a_2020_p13",
    "mathd_algebra_433", "mathd_algebra_437", "mathd_numbertheory_126", "mathd_algebra_282"
}

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
            with open(filepath) as f:
                content = f.read()
            if "sorry" in content:
                return False, "SORRY"
            return True, ""
        return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def find_failures(target_dir):
    """Find which problems fail in the target directory."""
    files = sorted(Path(target_dir).glob("*.lean"))
    print(f"Checking {len(files)} problems in {target_dir}...")

    failures = set()
    passes = set()

    def check_one(f):
        return f.stem, check_lean(str(f))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(check_one, f): f for f in files}
        for future in as_completed(futures):
            name, (ok, err) = future.result()
            if ok:
                passes.add(name)
            else:
                failures.add(name)
                if name not in KNOWN_IMPOSSIBLE:
                    print(f"  FAIL: {name} ({err[:60]})")

    print(f"Result: {len(passes)} pass, {len(failures)} fail")
    return failures, passes

def find_candidate_proofs(problem_name, target_dir):
    """Find all alternative proofs for a problem from other experiments."""
    candidates = []
    target_file = os.path.join(target_dir, f"{problem_name}.lean")

    if os.path.exists(target_file):
        with open(target_file) as f:
            target_content = f.read()
    else:
        target_content = ""

    for exp_dir in sorted(os.listdir(ATTEMPTS_DIR)):
        exp_path = os.path.join(ATTEMPTS_DIR, exp_dir)
        if not os.path.isdir(exp_path) or exp_dir == os.path.basename(target_dir):
            continue
        if exp_dir.startswith("_tmp"):
            continue

        candidate = os.path.join(exp_path, f"{problem_name}.lean")
        if not os.path.exists(candidate):
            continue

        with open(candidate) as f:
            content = f.read()

        # Skip if same as target or has sorry
        if content == target_content:
            continue
        if "sorry" in content:
            continue

        candidates.append((exp_dir, candidate, content))

    return candidates

def main():
    target_dir = os.path.join(ATTEMPTS_DIR, TARGET)

    # Step 1: Find failures
    failures, passes = find_failures(target_dir)
    solvable_failures = failures - KNOWN_IMPOSSIBLE
    print(f"\nSolvable failures: {len(solvable_failures)}")

    # Step 2: For each failure, try proofs from other experiments
    mined = []
    for name in sorted(solvable_failures):
        candidates = find_candidate_proofs(name, target_dir)
        if not candidates:
            continue

        print(f"\n{name}: {len(candidates)} candidate(s) from {[c[0] for c in candidates[:5]]}")

        for exp_dir, candidate_path, content in candidates:
            # Write to a temp file and test
            tmp = os.path.join(DOMAIN_DIR, "_tmp_mine.lean")
            with open(tmp, 'w') as f:
                f.write(content)

            ok, err = check_lean(tmp)
            if ok:
                # Copy to target
                dst = os.path.join(target_dir, f"{name}.lean")
                shutil.copy2(candidate_path, dst)
                print(f"  ✓ MINED from {exp_dir}")
                mined.append((name, exp_dir))
                break
            else:
                print(f"  ✗ {exp_dir}: {err[:60]}")

        if os.path.exists(os.path.join(DOMAIN_DIR, "_tmp_mine.lean")):
            os.remove(os.path.join(DOMAIN_DIR, "_tmp_mine.lean"))

    # Summary
    print(f"\n{'='*60}")
    print(f"Mined {len(mined)} proofs:")
    for name, src in mined:
        print(f"  {name} ← {src}")
    print(f"Estimated new total: {len(passes) + len(mined)}/244 = {(len(passes)+len(mined))/244:.4f}")

if __name__ == "__main__":
    main()
