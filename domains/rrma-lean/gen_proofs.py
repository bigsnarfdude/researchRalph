#!/usr/bin/env python3
"""Generate batch proof attempts for remaining MiniF2F problems."""
import os, re, subprocess

MINIF2F = "/home/vincent/miniF2F-lean4"
VALID_DIR = f"{MINIF2F}/MiniF2F/Valid"
OUTPUT_DIR = "/home/vincent/researchRalph/domains/rrma-lean/attempts/exp076_merge"

# Problems that are still failing
FAILS = """aime_1987_p8 aime_1991_p6 aime_1994_p4 aime_1997_p11
amc12a_2009_p15 amc12a_2009_p25 amc12a_2010_p22 amc12a_2019_p21 amc12a_2020_p21 amc12b_2021_p21
imo_1962_p4 imo_1967_p3 imo_1978_p5 imo_1979_p1 imo_1987_p6 imo_1988_p6 imo_1990_p3 imo_1993_p5 imo_2006_p3
mathd_numbertheory_405 mathd_numbertheory_43 mathd_numbertheory_709""".split()

KNOWN_IMPOSSIBLE = {"aime_1984_p5", "aime_1988_p3", "amc12a_2002_p21", "amc12a_2020_p13", 
                     "mathd_algebra_433", "mathd_algebra_437", "mathd_numbertheory_126", "mathd_algebra_282"}

# Advanced tactic cascade
TACTICS = [
    "norm_num",
    "omega",
    "decide",
    "native_decide",
    "simp_all",
    "ring",
    "linarith",
    "nlinarith [sq_nonneg (a), sq_nonneg (b), sq_nonneg (c)]",
    "constructor <;> (norm_num <;> omega)",
    "constructor <;> linarith",
    "constructor <;> nlinarith",
    "refine ⟨?_, ?_⟩ <;> norm_num",
    "intro h; cases h <;> simp_all",
    "ext x; simp_all; omega",
    "simp only [Set.ext_iff]; intro x; simp_all",
]

for p in FAILS:
    if p in KNOWN_IMPOSSIBLE:
        continue
    src = f"{VALID_DIR}/{p}.lean"
    if not os.path.exists(src):
        continue
    with open(src) as f:
        content = f.read()
    # Just print what we'd try — don't overwrite working proofs
    print(f"Would attempt: {p}")

print(f"\nTotal: {len([p for p in FAILS if p not in KNOWN_IMPOSSIBLE])} solvable problems remaining")
