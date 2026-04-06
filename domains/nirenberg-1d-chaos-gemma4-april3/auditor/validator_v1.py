import auditor_v1
import sys
from pathlib import Path

def run_validation(data_dir):
    print(f"=== GEMMA 4 SCIENTIFIC INTEGRITY VALIDATOR (v1.0) ===")
    
    summary = auditor_v1.run_audit(data_dir, quiet=True)
    if not summary:
        print(f"ERROR: Validation check failed for {data_dir}.")
        return

    desires_path = Path(data_dir) / "DESIRES.md"
    
    nudge = ""
    if summary['status'] == 'BLIND SPOT':
        nudge = f"""# SCIENTIFIC INTEGRITY NUDGE: GLOBAL SYMMETRY CHECK
**Priority:** MEDIUM (Integrity Check)
**Context:** The Auditor has completed a global scan of the current run results.

### VALIDATION SUMMARY:
- **Distribution:** Strong coverage of the positive and trivial branches.
- **Blind Spot Detected:** The mathematically expected symmetric negative branch ($u \\approx -1$) is currently unmapped (V-Asym Index: {summary['v_asym']:.2f}).
- **Observation:** Collective focus is currently centered on the oscillatory branch ({summary['novel_count']} experiments).

### SUGGESTED RESEARCH ACTION:
To ensure the integrity of the final consensus, we recommend a brief validation sweep of the negative region (`u_offset = -1.0`). This will verify the system's invariance and ground our findings in the full physical landscape.

**"Trust the discovery, validate the symmetry."**
"""
    elif summary['status'] == 'BIASED':
        nudge = f"""# SCIENTIFIC INTEGRITY NUDGE: DIVERSITY OF EXPLORATION
**Priority:** LOW
**Observation:** The swarm is showing a slight optimization bias toward the current best config.
**Suggestion:** Consider a small percentage of 'Scout' experiments in the negative u_offset basin to maintain global coverage.
"""
    else:
        print("STATUS: BALANCED. Swarm is performing well.")
        return

    desires_path.write_text(nudge)
    print(f"\nVALIDATION ALERT: Nudge written to {desires_path}")
    print(f"VERDICT: {summary['status']}")
    print(f"SUGGESTION: Verify Symmetric Invariance (-1 branch)")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/01_nigel_baseline"
    run_validation(target_dir)
