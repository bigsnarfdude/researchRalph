#!/usr/bin/env python3
"""Automated tactic search for remaining unsolved MiniF2F problems.
Targets problems that are NOT known impossible."""

import subprocess, os, sys, time

MINIF2F = "/home/vincent/miniF2F-lean4"
ATTEMPTS = "/home/vincent/researchRalph/domains/rrma-lean/attempts/exp141"
TIMEOUT = 180  # seconds per attempt

# Known impossible/broken problems (don't waste time)
IMPOSSIBLE = {
    "aime_1988_p3", "mathd_algebra_282", "mathd_algebra_433",
    "amc12a_2002_p21", "amc12a_2020_p13", "imo_1962_p4"
}

# Problems we've already solved
SOLVED = {
    "mathd_algebra_437", "amc12a_2009_p25", "imo_2006_p3"
}

# Remaining hard problems to try
TARGETS = [
    "aime_1997_p11",  # trig sum
    "amc12b_2021_p21",  # rpow equation
    "imo_1967_p3",  # product divisibility
    "imo_1978_p5",  # rearrangement inequality
    "imo_1979_p1",  # alternating harmonic sum
    "imo_1987_p6",  # prime-generating polynomial
    "imo_1988_p6",  # Vieta jumping
    "imo_1990_p3",  # n²|2^n+1
    "imo_1993_p5",  # Wythoff sequence
]

# Tactic combinations to try (more sophisticated than single tactics)
TACTICS = [
    "omega",
    "norm_num",
    "decide",
    "native_decide",
    "simp_all",
    "aesop",
    "ring",
    "linarith",
    "nlinarith",
    "exact?",
    # With hypotheses
    "simp_all; omega",
    "simp_all; norm_num",
    "simp_all; nlinarith",
    # Field
    "field_simp; ring",
    "field_simp; norm_num",
    # Push cast
    "push_cast; ring",
    "push_cast; omega",
    # Interval cases (for bounded problems)
    "interval_cases n; omega",
    "interval_cases n; norm_num",
    "interval_cases n; simp_all",
]

def read_problem(name):
    """Read the theorem statement from the attempt file."""
    path = os.path.join(ATTEMPTS, f"{name}.lean")
    with open(path) as f:
        content = f.read()
    # Extract just the theorem header (up to ":= by")
    idx = content.find(":= by")
    if idx < 0:
        return None
    return content[:idx]

def try_tactic(name, header, tactic):
    """Try a single tactic on a problem."""
    proof = f"""{header}:= by
  {tactic}
"""
    tmpfile = f"/tmp/try_{name}.lean"
    with open(tmpfile, 'w') as f:
        f.write(proof)
    
    try:
        result = subprocess.run(
            ["bash", "-c", f"cd {MINIF2F} && lake env lean {tmpfile}"],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        if result.returncode == 0 and "sorry" not in result.stderr:
            return True
    except subprocess.TimeoutExpired:
        pass
    return False

def main():
    for name in TARGETS:
        if name in IMPOSSIBLE or name in SOLVED:
            continue
        header = read_problem(name)
        if header is None:
            print(f"SKIP {name}: can't parse")
            continue
        print(f"\n=== {name} ===")
        for tactic in TACTICS:
            print(f"  trying: {tactic}...", end=" ", flush=True)
            if try_tactic(name, header, tactic):
                print("PASS!")
                # Save the successful proof
                proof = f"""{header}:= by
  {tactic}
"""
                outpath = os.path.join(ATTEMPTS, f"{name}.lean")
                with open(outpath, 'w') as f:
                    f.write(proof)
                print(f"  SAVED to {outpath}")
                break
            else:
                print("fail")
        else:
            print(f"  No tactic worked for {name}")

if __name__ == "__main__":
    main()
