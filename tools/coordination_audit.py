#!/usr/bin/env python3
"""Cross-agent coordination audit over domains/*/blackboard.md.

WHAT THIS COUNTS
  A PEER REFERENCE is a blackboard line whose author names a DIFFERENT agent.

WHY THAT QUALIFIER EXISTS
  The blackboard protocol prefixes every line with the writer's own id
  ("CLAIM agent0: residual=..."). A naive `\\bagent\\d\\b` match therefore counts
  the LOGGING FORMAT, not coordination, and scores ~every line as a reference.
  That error produced a spurious 26-28% "peer reference" rate and a spurious
  model effect (2026-08-19). Author is stripped before counting.

LIMITS -- read before quoting any number from this.
  1. Author detection is regex on line prefix. Lines with no recognisable
     prefix are counted with NO author stripped, which OVERCOUNTS. The
     --strict flag drops those lines instead; report which you used.
  2. Agents may refer to peers by role or by design name rather than id.
     Those are missed. This is a floor.
  3. Prose about "the other agent" without an id is missed (see YIELD, which
     is a separate and even rougher lexical proxy).
  4. Blackboards are the PUBLISHED surface. Reasoning that never gets written
     down is invisible here. Do not read a zero as absence of the behaviour.
  6. Model detection searches domain/logs/ AND the repo .run-logs/<domain>-*/ dir,
     because the fixed launcher writes session logs outside the domain.
  5. Domains whose stoplight.md contains "Design 'agentN' ... abandon this
     approach" are flagged: the scaffold is telling agents to abandon their
     peers, which contaminates any coordination measure. Flagged as BUG.
"""
import argparse, glob, os, re, collections, statistics, sys

AUTHOR = re.compile(r"^\s*(?:#+\s*)?(?:CLAIM(?:ED)?|RESPONSE|NOTE|RESULT|EXP[-\w]*)?\s*\(?(agent\s?\d+)\b", re.I)
ANYAG  = re.compile(r"\bagent[ _-]?(\d+)\b", re.I)
YIELD  = re.compile(r"already (ran|queued|did|tried)|avoid collision|skip.*already|someone else ran", re.I)
SLBUG  = re.compile(r"Design 'agent\d+'.*abandon", re.I)

def model_of(dom):
    ms = collections.Counter()
    # launch-agents-chaos-clean.sh writes session logs OUTSIDE the domain (.run-logs/),
    # so domain/logs/ is empty for every post-2026-08-19 run and this reported "unknown".
    # Search both, newest run dir first.
    import os
    cands = sorted(glob.glob(f"{dom}/logs/*.jsonl"))
    root = os.path.abspath(os.path.join(dom, "..", ".."))
    cands += sorted(glob.glob(f"{root}/.run-logs/{os.path.basename(dom)}-*/*.jsonl"), reverse=True)
    for f in cands[:6]:
        try:
            for i, l in enumerate(open(f, errors="ignore")):
                if i > 4000: break
                m = re.search(r'"model":"(claude-[a-z0-9.-]+)"', l)
                if m: ms[m.group(1)] += 1
        except OSError: pass
    return ms.most_common(1)[0][0] if ms else "unknown"

def audit_domain(dom, strict):
    bb = os.path.join(dom, "blackboard.md")
    if not os.path.exists(bb): return None
    lines = [l.strip() for l in open(bb, errors="ignore") if l.strip()]
    if not lines: return None
    peer = yld = unattributed = 0
    for l in lines:
        a = AUTHOR.match(l)
        if a is None:
            unattributed += 1
            if strict: continue
            auth = None
        else:
            auth = re.sub(r"\s", "", a.group(1)).lower()
        others = {f"agent{m}" for m in ANYAG.findall(l)}
        if auth: others.discard(auth)
        if others: peer += 1
        if YIELD.search(l): yld += 1
    sl = os.path.join(dom, "stoplight.md")
    bug = os.path.exists(sl) and bool(SLBUG.search(open(sl, errors="ignore").read()))
    n = len(lines)
    return dict(domain=os.path.basename(dom), model=model_of(dom), lines=n,
                peer=peer, peer_pct=100.0*peer/n, yield_n=yld,
                unattributed_pct=100.0*unattributed/n, stoplight_bug=bug)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="domains/*")
    p.add_argument("--min-lines", type=int, default=25)
    p.add_argument("--strict", action="store_true", help="drop lines with no detectable author (undercounts, no overcount)")
    p.add_argument("--by-model", action="store_true")
    a = p.parse_args()

    rows = [r for d in sorted(glob.glob(a.glob)) if os.path.isdir(d)
            for r in [audit_domain(d, a.strict)] if r and r["lines"] >= a.min_lines]
    if not rows: print("no domains"); sys.exit(1)

    print(f"mode: {'STRICT' if a.strict else 'lenient (unattributed lines counted, may overcount)'}\n")
    print(f"{'domain':<46}{'model':<26}{'lines':>6}{'peer%':>8}{'n':>5}{'yield':>7}{'unattr%':>9}  flag")
    for r in sorted(rows, key=lambda x: -x["peer_pct"]):
        print(f"  {r['domain']:<44}{r['model']:<26}{r['lines']:>6}{r['peer_pct']:>7.1f}%"
              f"{r['peer']:>5}{r['yield_n']:>7}{r['unattributed_pct']:>8.0f}%  {'STOPLIGHT-BUG' if r['stoplight_bug'] else ''}")
    if a.by_model:
        by = collections.defaultdict(list)
        for r in rows: by[r["model"]].append(r)
        print(f"\n{'model':<26}{'doms':>5}{'lines':>8}{'peer% (line-weighted)':>24}{'range':>16}")
        for m, v in sorted(by.items(), key=lambda x: -sum(y['lines'] for y in x[1])):
            L = sum(x["lines"] for x in v); pr = [x["peer_pct"] for x in v]
            print(f"{m:<26}{len(v):>5}{L:>8}{sum(x['peer']for x in v)/L*100:>23.1f}%"
                  f"{f'{min(pr):.1f}-{max(pr):.1f}%':>16}")
    nb = sum(1 for r in rows if r["stoplight_bug"])
    if nb: print(f"\n!! {nb} domains carry the stoplight agent-as-dead-end bug — coordination numbers there are contaminated.")
