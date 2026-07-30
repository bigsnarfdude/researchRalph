#!/usr/bin/env python3
"""verify_filter_ml.py — independent re-verification gate for ML/agentic SFT traces.

The ML analog of verify_filter.py's Lean compile gate. For Lean, "does it
compile" is unfakeable ground truth, so a reward-hacked trace can't enter the
SFT corpus. ML/agentic domains have no such gate: the only quality signal is
the results.tsv score, which is exactly what the agents hack. Feeding those
traces to SFT (v5.2) would train the hack in and select for it.

This gate re-earns the score instead of trusting it. For each logged experiment:
  1. verify the oracle itself is unchanged (guard oracle-verify) — else the
     re-run is meaningless. A poisoned oracle aborts the whole gate.
  2. re-run the oracle on the experiment's ARCHIVED artifact in a throwaway
     copy of the domain (no side effects on the real island).
  3. admit the trace only if the independent score reproduces the claimed
     score within tolerance AND the session trace has no out-of-scope actions
     (guard scan-trace).

Anything that fails reproduction was a hack, a non-determinism, or a lost
artifact — quarantined to the reject log, never trained on.

Usage:
  python3 bootstrap/verify_filter_ml.py --island domains/sae-island-isl-a \
      --out sft_verified.jsonl --reject sft_rejected.jsonl [--tol 0.01]
      [--sample N] [--timeout 3000] [--guard v5/guard.sh]

Exit 0 = ran (see summary); 2 = oracle poisoned, gate aborted.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def read_rows(results_tsv):
    """SFT candidates: agent-produced experiments only.

    Skips `smoke` rows and the loop-owned `anchor` (an untouched-seed rerun the
    loop scores itself). The anchor has no agent session behind it, so there is
    no reasoning to train on — it used to be admitted with an empty assistant
    message, which is junk in the corpus.
    """
    rows = []
    with open(results_tsv) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 7 and p[3] != "smoke" and p[6] != "anchor":
                rows.append({"exp": p[0], "score": p[1], "status": p[3],
                             "desc": p[4], "agent": p[5], "design": p[6]})
    return rows


def editable_name(island):
    m = re.search(r"^editable:\s*(\S+)", (island / "config.yaml").read_text(), re.M)
    return m.group(1) if m else "sae.py"


def extract_reasoning(session_log):
    """Concatenate assistant text/thinking from a session trace (for the SFT target)."""
    if not session_log.exists():
        return ""
    out = []
    for line in session_log.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        for b in ((d.get("message") or {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") in ("text", "thinking"):
                t = b.get("text") or b.get("thinking") or ""
                if t.strip():
                    out.append(t.strip())
    return "\n".join(out)


def rerun_oracle(island, artifact_path, editable, timeout):
    """Re-run the oracle on artifact in a throwaway copy of the domain. -> float|None."""
    with tempfile.TemporaryDirectory(prefix="verifyml_") as tmp:
        tmp = Path(tmp)
        # copy oracle machinery (files only), leave out results/logs/boards/runs
        for item in island.iterdir():
            if item.is_file() and item.name not in ("results.tsv", "blackboard.md", ".oracle_hash"):
                shutil.copy2(item, tmp / item.name)
        (tmp / "results.tsv").write_text("exp_id\tscore\ttime\tstatus\tdescription\tagent\tdesign\n")
        ws = tmp / "workspace" / "verify"
        ws.mkdir(parents=True)
        shutil.copy2(artifact_path, ws / editable)
        env = {"CLAUDE_AGENT_ID": "verify", "ORACLE_WAIT": "999999", "PATH":
               __import__("os").environ.get("PATH", "")}
        # carry HOME/venv so the oracle finds python/deps
        for k in ("HOME", "VENV_PYTHON"):
            v = __import__("os").environ.get(k)
            if v:
                env[k] = v
        try:
            r = sh(["bash", "run.sh", "verify", f"reverify"], cwd=str(tmp), env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        m = None
        for line in r.stdout.splitlines():
            mm = re.match(r"\s*SCORE:\s*([0-9.]+)\s*$", line)
            if mm:
                m = mm.group(1)
        return float(m) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--island", required=True, type=Path)
    ap.add_argument("--out", default="sft_verified.jsonl", type=Path)
    ap.add_argument("--reject", default="sft_rejected.jsonl", type=Path)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--sample", type=int, default=0, help="verify only the N highest-scoring rows (0=all)")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--guard", default=None, help="path to v5/guard.sh (default: sibling of this repo)")
    ap.add_argument("--allow-unsnapshotted", action="store_true",
                    help="audit a legacy island that was never oracle-snapshotted (UNSAFE: the "
                         "re-run cannot be vouched for; only for pre-guard archives)")
    args = ap.parse_args()

    island = args.island.resolve()
    repo = island.parent.parent
    guard = Path(args.guard) if args.guard else repo / "v5" / "guard.sh"
    editable = editable_name(island)

    # ---- 1. oracle integrity: without this, re-verification is theater.
    # Fail CLOSED on ANY nonzero. This used to warn-and-proceed on rc!=3, which
    # meant `rm .oracle_hash` downgraded the abort to a warning and the hacked
    # traces sailed into the SFT corpus — the exact thing this gate exists to stop.
    gv = sh(["bash", str(guard), "oracle-verify", str(island)]
            + (["--allow-missing"] if args.allow_unsnapshotted else []))
    if gv.returncode != 0:
        print(f"ABORT: oracle integrity unverifiable (rc={gv.returncode}) — re-verification "
              f"is meaningless until it is restored.\n{gv.stderr}", file=sys.stderr)
        return 2
    if args.allow_unsnapshotted:
        print("WARNING: --allow-unsnapshotted — admitted traces are NOT integrity-vouched.",
              file=sys.stderr)

    rows = read_rows(island / "results.tsv")
    if args.sample:
        rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:args.sample]

    passed, rejected = [], []
    for r in rows:
        exp = r["exp"]
        artifact = island / "runs" / exp / editable
        n = int(re.sub(r"\D", "", exp) or 0)
        session_log = island / "logs" / f"exp{n:03d}_session.jsonl"

        if not artifact.exists():
            rejected.append({**r, "reject": "no-archived-artifact",
                             "detail": f"missing {artifact.relative_to(island)}"})
            continue

        # ---- 3. trace hygiene (cheap; do before the expensive re-run)
        # A MISSING trace is not a clean trace. This used to default to clean,
        # so deleting a session log laundered an escaping session into the corpus
        # (and the resulting sample would carry empty reasoning anyway).
        if not session_log.exists():
            rejected.append({**r, "reject": "no-session-trace",
                             "detail": f"missing {session_log.relative_to(island)} — cannot clear this session"})
            continue
        gs = sh(["bash", str(guard), "scan-trace", str(session_log), str(island)])
        trace_clean = (gs.returncode == 0)
        trace_detail = gs.stderr.strip()

        # ---- 2. independent re-run
        indep = rerun_oracle(island, artifact, editable, args.timeout)
        claimed = float(r["score"])
        if indep is None:
            rejected.append({**r, "reject": "oracle-rerun-failed", "detail": "no SCORE / timeout"})
            continue
        reproduced = abs(indep - claimed) <= args.tol

        reasons = []
        if not reproduced:
            reasons.append(f"score-mismatch(claimed={claimed} independent={indep:.4f})")
        if not trace_clean:
            reasons.append(f"trace-violation({trace_detail.splitlines()[-1] if trace_detail else '?'})")

        if reasons:
            rejected.append({**r, "reject": "+".join(reasons), "independent_score": indep})
        else:
            passed.append({
                "messages": [
                    {"role": "user", "content": (island / "program.md").read_text()
                                     if (island / "program.md").exists() else r["desc"]},
                    {"role": "assistant", "content": extract_reasoning(session_log)
                                     + "\n\n<artifact>\n" + artifact.read_text() + "\n</artifact>"},
                ],
                "meta": {"exp": exp, "verified_score": indep, "design": r["design"],
                         "island": island.name},
            })

    args.out.write_text("".join(json.dumps(p) + "\n" for p in passed))
    args.reject.write_text("".join(json.dumps(r) + "\n" for r in rejected))

    print(f"verify_filter_ml: {len(passed)} admitted, {len(rejected)} rejected "
          f"(of {len(rows)} candidates)")
    for r in rejected:
        print(f"  REJECT {r['exp']}: {r['reject']}")
    print(f"  -> {args.out}  (+ rejects: {args.reject})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
