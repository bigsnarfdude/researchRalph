#!/usr/bin/env python3
"""Cross-agent collision + redundancy audit over domains/*/results.tsv.

A COLLISION here means: one experiment description logged by two or more distinct
agents in the same domain. It is a proxy, and its limits are stated in --notes.

Columns are resolved BY HEADER NAME, never by position: results.tsv schemas differ
across domains (5 to 10 columns, description/agent at varying indices). Domains
without a usable header are reported as unmeasurable rather than silently dropped.
"""
import argparse, csv, os, glob, statistics, sys

DESC_KEYS  = ("description", "desc")
AGENT_KEYS = ("agent", "agent_id")

def resolve(path):
    """-> (rows, desc_idx, agent_idx) or (None, reason)."""
    with open(path, errors="ignore") as fh:
        rows = [r for r in csv.reader(fh, delimiter="\t")]
    if not rows:
        return None, "empty"
    hdr = [c.strip().lower() for c in rows[0]]
    d = next((i for i, c in enumerate(hdr) if c in DESC_KEYS), None)
    a = next((i for i, c in enumerate(hdr) if c in AGENT_KEYS), None)
    if d is None or a is None:
        return None, "no header"
    body = [r for r in rows[1:] if len(r) > max(d, a) and r[d].strip() and r[a].strip()]
    return (body, d, a), None

def audit(domain_glob, min_rows, min_agents):
    ok, skipped = [], []
    for f in sorted(glob.glob(domain_glob)):
        name = os.path.basename(os.path.dirname(f))
        res, why = resolve(f)
        if res is None:
            skipped.append((name, why)); continue
        body, d, a = res
        agents = {r[a].strip() for r in body}
        if len(body) < min_rows:   skipped.append((name, f"<{min_rows} rows"));   continue
        if len(agents) < min_agents: skipped.append((name, "single-agent"));      continue
        by = {}
        for r in body:
            by.setdefault(r[d].strip(), set()).add(r[a].strip())
        coll = sum(1 for v in by.values() if len(v) > 1)
        ok.append(dict(domain=name, n=len(body), agents=len(agents),
                       collisions=coll, rate=100.0 * coll / len(body)))
    return ok, skipped

NOTES = """\
LIMITS OF THIS PROXY -- read before quoting any number from it.
 1. Exact string match on description. Two agents running the same config but
    wording it differently are MISSED; two logging different work as "baseline"
    are a FALSE POSITIVE.
 2. It does not detect file-level collisions (agents overwriting each other's
    train.py). Those occur in this corpus and are invisible here.
 3. Rate denominator is experiment count, which varies ~20x across domains.
    Two duplicates is 22% at n=9 and 1% at n=180. Rates are NOT comparable
    across domains; do not average them across differing n without saying so.
 4. Domains lacking a header cannot be resolved and are excluded, not zero.
"""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="domains/*/results.tsv")
    p.add_argument("--min-rows", type=int, default=5)
    p.add_argument("--min-agents", type=int, default=2)
    p.add_argument("--notes", action="store_true")
    p.add_argument("--per-domain", action="store_true")
    args = p.parse_args()
    if args.notes: print(NOTES); sys.exit(0)

    ok, skipped = audit(args.glob, args.min_rows, args.min_agents)
    if not ok: print("no measurable domains"); sys.exit(1)
    rates = [d["rate"] for d in ok]
    tc = sum(d["collisions"] for d in ok)
    print(f"measurable domains : {len(ok)}   experiments: {sum(d['n'] for d in ok)}")
    print(f"collisions         : {tc} across {sum(1 for d in ok if d['collisions'])} domains")
    print(f"redundancy rate    : median {statistics.median(rates):.1f}%  "
          f"mean {statistics.mean(rates):.1f}%  max {max(rates):.1f}%")
    print(f"zero-collision     : {sum(1 for r in rates if r == 0)}/{len(ok)} domains")
    from collections import Counter
    print(f"excluded           : {len(skipped)}  " + str(Counter(w for _, w in skipped).most_common()))
    if args.per_domain:
        print()
        for d in sorted(ok, key=lambda x: -x["rate"]):
            print(f"  {d['domain']:<46} {d['collisions']:>3}/{d['n']:<5} {d['rate']:>5.1f}%  ({d['agents']} agents)")
