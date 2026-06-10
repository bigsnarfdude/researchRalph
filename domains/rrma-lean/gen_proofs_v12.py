#!/usr/bin/env python3
"""
gen_proofs_v12.py — Agent2's two-phase proof search:
Phase 1: Timeout recovery (existing proofs with 180s + higher heartbeats)
Phase 2: Tactic sweep with parallel testing
"""
import os, re, sys, subprocess, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VALID_DIR = "/home/vincent/miniF2F-lean4/MiniF2F/Valid"
MINIF2F_DIR = "/home/vincent/miniF2F-lean4"
DOMAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ELAN_PATH = os.path.expanduser("~/.elan/bin")

METHOD = sys.argv[1] if len(sys.argv) > 1 else "exp105"
BASELINE = sys.argv[2] if len(sys.argv) > 2 else "exp079"
TIMEOUT_PHASE1 = 180  # generous for timeout recovery
TIMEOUT_PHASE2 = 120  # for tactic search
MAX_WORKERS = 3

out_dir = os.path.join(DOMAIN_DIR, "attempts", METHOD)
base_dir = os.path.join(DOMAIN_DIR, "attempts", BASELINE)

KNOWN_IMPOSSIBLE = {
    "aime_1984_p5", "aime_1988_p3", "amc12a_2002_p21", "amc12a_2020_p13",
    "mathd_algebra_433", "mathd_algebra_437", "mathd_numbertheory_126", "mathd_algebra_282"
}

def check_lean(filepath, timeout=120):
    try:
        env = os.environ.copy()
        env["PATH"] = ELAN_PATH + ":" + env.get("PATH", "")
        t0 = time.time()
        result = subprocess.run(
            ["lake", "env", "lean", filepath],
            capture_output=True, text=True, timeout=timeout,
            cwd=MINIF2F_DIR, env=env
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            with open(filepath) as f:
                content = f.read()
            if "sorry" in content:
                return False, "SORRY", elapsed
            return True, "", elapsed
        return False, result.stderr[:1500], elapsed
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", timeout
    except Exception as e:
        return False, str(e), 0

def make_proof(orig_content, proof_body, heartbeats=8000000):
    new = orig_content.replace("by sorry", f"by\n  {proof_body}")
    new = re.sub(r'set_option maxHeartbeats \d+', f'set_option maxHeartbeats {heartbeats}', new)
    if 'maxHeartbeats' not in new:
        new = new.replace('import Mathlib', f'import Mathlib\nset_option maxHeartbeats {heartbeats}')
    return new

def increase_heartbeats(content, new_hb):
    return re.sub(r'set_option maxHeartbeats \d+', f'set_option maxHeartbeats {new_hb}', content)

# ── Phase 1: Find failures and try timeout recovery ──

def phase1_find_and_recover(out_dir):
    """Find failures and try to recover timeout-sensitive proofs."""
    files = sorted(Path(out_dir).glob("*.lean"))
    print(f"\n[Phase 1] Checking {len(files)} problems with {TIMEOUT_PHASE1}s timeout...")
    
    failures = []
    passes = 0
    
    def check_one(f):
        return f.stem, check_lean(str(f), timeout=TIMEOUT_PHASE1)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(check_one, f): f for f in files}
        for future in as_completed(futures):
            name, (ok, err, elapsed) = future.result()
            if ok:
                passes += 1
            else:
                failures.append((name, err, elapsed))
    
    print(f"[Phase 1] Result: {passes}/{len(files)} pass, {len(failures)} fail")
    
    # Try increasing heartbeats on failures that look like timeout/heartbeat issues
    recovered = []
    for name, err, elapsed in failures:
        if name in KNOWN_IMPOSSIBLE:
            continue
        filepath = os.path.join(out_dir, f"{name}.lean")
        with open(filepath) as f:
            content = f.read()
        
        # Skip tactic-sweep stubs (they have "first" pattern)
        if re.search(r'^\s*first\s*$', content, re.MULTILINE):
            continue
        
        # Try higher heartbeats if error mentions heartbeats or timeout
        if "heartbeat" in err.lower() or "TIMEOUT" in err or elapsed > TIMEOUT_PHASE1 * 0.8:
            for hb in [16000000, 32000000, 64000000]:
                new_content = increase_heartbeats(content, hb)
                with open(filepath, 'w') as f:
                    f.write(new_content)
                ok2, err2, elapsed2 = check_lean(filepath, timeout=TIMEOUT_PHASE1)
                if ok2:
                    print(f"  ✓ RECOVERED {name} with maxHeartbeats {hb} ({elapsed2:.1f}s)")
                    recovered.append(name)
                    break
            else:
                # Restore original
                orig_path = os.path.join(base_dir, f"{name}.lean")
                if os.path.exists(orig_path):
                    shutil.copy2(orig_path, filepath)
    
    return failures, recovered

# ── Phase 2: Tactic sweep ──

def gen_tactics(orig_content):
    """Generate tactic candidates based on problem content."""
    hyps = re.findall(r'\((h[₀₁₂₃₄₅₆₇₈₉\d_]*)\s*:', orig_content)
    h_list = ", ".join(hyps)
    
    # Extract variable names and types
    all_vars = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', orig_content)
    var_names = [v for v, t in all_vars if not v.startswith('h') and v not in ('S', 't', 'f', 'c', 'a')]
    
    tactics = []
    
    # Tier 0: Atomic
    for t in ["norm_num", "omega", "simp", "simp_all", "ring", "decide", 
              "native_decide", "linarith", "nlinarith", "positivity", "aesop", "tauto", "trivial"]:
        tactics.append(t)
    
    # Tier 1: Preprocessing + solver
    for pre in ["push_cast", "norm_cast", "field_simp", "ring_nf", "zify", "push_neg", 
                "simp only []", "simp_all only []"]:
        for sol in ["omega", "norm_num", "ring", "linarith", "nlinarith", "simp", "simp_all"]:
            tactics.append(f"{pre}; {sol}")
            if hyps:
                tactics.append(f"{pre} at *; {sol}")
    
    # Tier 2: Constructors
    for sol in ["norm_num", "omega", "linarith", "nlinarith", "ring", "simp", "simp_all", 
                "native_decide", "decide"]:
        tactics.append(f"constructor <;> {sol}")
        tactics.append(f"refine ⟨?_, ?_⟩ <;> {sol}")
    
    # Tier 3: Hypothesis-specific
    if hyps:
        tactics.extend([
            f"linarith [{h_list}]",
            f"nlinarith [{h_list}]",
            f"simp only [{h_list}]",
            f"simp_all [{h_list}]",
        ])
        for h in hyps[:5]:
            for sol in ["norm_num", "omega", "ring", "simp", "linarith"]:
                tactics.append(f"rw [{h}]; {sol}")
                tactics.append(f"simp [{h}]; {sol}")
    
    # Tier 4: Chained combinations
    for t1 in ["simp", "simp_all", "norm_num", "push_cast", "field_simp"]:
        for t2 in ["omega", "linarith", "nlinarith", "ring", "norm_num", "native_decide"]:
            if t1 != t2:
                tactics.append(f"{t1}; {t2}")
    
    # Tier 5: SOS witnesses for inequalities
    if any(op in orig_content for op in ['≤', '≥']):
        vars_ = re.findall(r'\((\w)\s*:', orig_content)
        vars_ = [v for v in vars_ if v in 'abcxyz']
        for v in vars_[:3]:
            tactics.append(f"nlinarith [sq_nonneg {v}]")
            for w in vars_[:3]:
                if v != w:
                    tactics.append(f"nlinarith [sq_nonneg ({v} - {w}), sq_nonneg ({v} + {w})]")
    
    # Tier 6: Extension/set tactics
    if 'Set' in orig_content or '↔' in orig_content:
        tactics.extend([
            "ext x; simp_all",
            "ext x; simp_all; omega",
            "ext x; constructor <;> intro h <;> simp_all",
            "simp only [Set.ext_iff]; intro x; simp_all",
        ])
    
    # Tier 7: Intro/cases patterns
    tactics.extend([
        "intro h; cases h <;> simp_all",
        "intro h; exact absurd h (by norm_num)",
        "push_neg; intro h; omega",
    ])
    
    # Tier 8: Specific for divisor count problems
    if 'Nat.divisors' in orig_content:
        tactics.extend([
            "simp only [Nat.card_divisors]; omega",
            "have := Nat.divisors_prime_pow; simp_all",
        ])
    
    # Tier 9: Specific for IsGreatest
    if 'IsGreatest' in orig_content or 'IsLeast' in orig_content:
        tactics.extend([
            "constructor\n  · norm_num\n  · intro m hm; omega",
            "constructor\n  · simp; norm_num\n  · intro m hm; simp at hm; omega",
            "constructor\n  · decide\n  · intro m hm; omega",
            "constructor\n  · native_decide\n  · intro m hm; omega",
        ])
    
    # Deduplicate
    seen = set()
    unique = []
    for t in tactics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

def phase2_tactic_sweep(failures, out_dir):
    """Try tactic combinations on remaining failures."""
    # Filter to solvable failures (not known impossible, not already recovered)
    targets = [name for name, err, _ in failures if name not in KNOWN_IMPOSSIBLE]
    
    print(f"\n[Phase 2] Tactic sweep on {len(targets)} problems...")
    
    solved = []
    tmp_dir = os.path.join(DOMAIN_DIR, "attempts", f"_tmp_v12")
    os.makedirs(tmp_dir, exist_ok=True)
    
    for name in targets:
        src = os.path.join(VALID_DIR, f"{name}.lean")
        if not os.path.exists(src):
            continue
        with open(src) as f:
            orig = f.read()
        
        tactics = gen_tactics(orig)
        print(f"\n  {name}: trying {len(tactics)} tactics...")
        
        found = False
        for i, tactic in enumerate(tactics):
            tmp_path = os.path.join(tmp_dir, f"{name}.lean")
            content = make_proof(orig, tactic)
            with open(tmp_path, 'w') as f:
                f.write(content)
            
            ok, err, elapsed = check_lean(tmp_path, timeout=TIMEOUT_PHASE2)
            if ok:
                print(f"    ✓ SOLVED with: {tactic} ({elapsed:.1f}s)")
                shutil.copy2(tmp_path, os.path.join(out_dir, f"{name}.lean"))
                solved.append((name, tactic))
                found = True
                break
        
        if not found:
            print(f"    ✗ UNSOLVED ({len(tactics)} tried)")
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return solved

def main():
    # Copy baseline if needed
    if not os.path.exists(out_dir) or len(list(Path(out_dir).glob("*.lean"))) < 200:
        print(f"Copying {base_dir} → {out_dir}")
        shutil.copytree(base_dir, out_dir, dirs_exist_ok=True)
    
    # Phase 1
    failures, recovered = phase1_find_and_recover(out_dir)
    
    # Phase 2
    # Re-check failures after recovery
    still_failing = [(n, e, t) for n, e, t in failures if n not in recovered]
    solved = phase2_tactic_sweep(still_failing, out_dir)
    
    # Summary
    total_pass = 244 - len(failures) + len(recovered) + len(solved)
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Baseline passes: {244 - len(failures)}")
    print(f"  Timeout recovered: {len(recovered)}")
    print(f"  Tactic solved: {len(solved)}")
    print(f"  Total: {total_pass}/244 = {total_pass/244:.4f}")
    
    if recovered:
        print(f"\nRecovered: {', '.join(recovered)}")
    if solved:
        print(f"\nSolved:")
        for name, tactic in solved:
            print(f"  {name}: {tactic}")

if __name__ == "__main__":
    main()
