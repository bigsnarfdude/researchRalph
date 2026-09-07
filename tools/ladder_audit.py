#!/usr/bin/env python3
"""Textual integrity audit of every RRMA Lean domain that ever logged SCORE=1.0.

No Lean required. Checks four things per oracle-visible file:
  1. target theorem statement fingerprint  (weakened? replaced?)
  2. core definition fingerprint           (setA/setB/setAB tampered?)
  3. soundness tokens                      (axiom / local notation / native_decide / #exit / sorry)
  4. harness bypass                        (did the win come from a file the attributed agent didn't own?)

LIMIT: files on disk are FINAL state, not per-row state. This is a final-state
audit of every domain that logged a win, not a per-row audit of all 80 rows.
"""
import re, glob, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/Users/vincent/development/researchRalph")
COMMENT = re.compile(r'^\s*--')

def strip(text):
    """drop line comments and /- -/ blocks, collapse whitespace"""
    text = re.sub(r'/-.*?-/', ' ', text, flags=re.S)
    lines = [l for l in text.splitlines() if not COMMENT.match(l)]
    return "\n".join(lines)

def norm(s):
    return re.sub(r'\s+', ' ', s).strip()

def oracle_mode(runsh):
    """does run.sh prefer workspace/<agent>/FILE, or read the domain root only?"""
    try: t = runsh.read_text()
    except Exception: return "unknown", None
    m = re.search(r'workspace/\$\{?(?:CLAUDE_)?AGENT\}?/([\w.]+\.lean)', t)
    editable = m.group(1) if m else None
    if not editable:
        m2 = re.search(r'SOLUTION="\$DOMAIN_DIR/([\w.]+\.lean)"', t)
        editable = m2.group(1) if m2 else None
    if re.search(r'WORKSPACE_FILE=.*workspace', t) and 'SOLUTION="$WORKSPACE_FILE"' in t:
        return "workspace-first", editable
    return "root-only", editable

def wins(domain):
    tsv = domain / "results.tsv"
    out = []
    if not tsv.exists(): return out
    for line in tsv.read_text(errors="ignore").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3:
            try: sc = float(p[1])
            except ValueError: continue
            if sc >= 1.0:
                out.append({"exp": p[0], "desc": p[3] if len(p) > 3 else "",
                            "agent": p[4] if len(p) > 4 else ""})
    return out

TOKENS = {
    "axiom":         re.compile(r'^\s*axiom\s+(\w+)', re.M),
    "local_notation":re.compile(r'^\s*local\s+(?:notation|infix|infixr|infixl|postfix|prefix)\b', re.M),
    "local_instance":re.compile(r'^\s*local\s+instance\b', re.M),
    "native_decide": re.compile(r'\bnative_decide\b'),
    "hash_exit":     re.compile(r'^\s*#exit\b', re.M),
    "sorry":         re.compile(r'\bsorry\b'),
}

def scan_file(path):
    try: raw = path.read_text(errors="ignore")
    except Exception: return None
    body = strip(raw)
    r = {"path": path, "bytes": len(raw)}
    for k, rx in TOKENS.items():
        r[k] = rx.findall(body) if k == "axiom" else len(rx.findall(body))
    # axiom dependency: is each declared axiom referenced anywhere else?
    dead, live = [], []
    for name in r["axiom"]:
        uses = len(re.findall(r'\b' + re.escape(name) + r'\b', body)) - 1
        (live if uses > 0 else dead).append(name)
    r["axiom_dead"], r["axiom_live"] = dead, live
    # fingerprints
    thm = re.findall(r'^\s*(?:private\s+)?theorem\s+(\w+)\s*:?([^:=]*)', body, re.M)
    r["theorems"] = [(n, norm(s)[:90]) for n, s in thm]
    r["defs"] = {n: norm(b)[:110] for n, b in re.findall(r'^\s*(?:noncomputable\s+)?def\s+(\w+)\s*:([^\n]*)', body, re.M)}
    return r

FAMILIES = sorted(glob.glob(str(ROOT / "domains/erdos-125*")) +
                  glob.glob(str(ROOT / "domains/erdos-741ii*")))

total_wins = 0; audited = 0
flags = defaultdict(list)
rows = []

for dpath in FAMILIES:
    d = Path(dpath)
    if not d.is_dir(): continue
    w = wins(d)
    if not w: continue
    total_wins += len(w)
    mode, editable = oracle_mode(d / "run.sh")
    if not editable: editable = "Erdos125.lean" if "125" in d.name else "Erdos741OAI.lean"

    # which files could the oracle have compiled?
    targets = []
    if mode == "workspace-first":
        targets += sorted(d.glob(f"workspace/agent*/{editable}"))
    targets.append(d / editable)
    targets = [t for t in targets if t.exists()]

    win_agents = {r["agent"] for r in w if r["agent"]}

    for t in targets:
        s = scan_file(t)
        if not s: continue
        audited += 1
        rel = str(t.relative_to(ROOT))
        agent = t.parent.name if t.parent.name.startswith("agent") else "<root>"
        issues = []
        if s["sorry"]:          issues.append(f"sorry x{s['sorry']}")
        if s["axiom_live"]:     issues.append("AXIOM-LOAD-BEARING:" + ",".join(s["axiom_live"]))
        if s["axiom_dead"]:     issues.append("axiom-dead:" + ",".join(s["axiom_dead"]))
        if s["local_notation"]: issues.append(f"local-notation x{s['local_notation']}")
        if s["local_instance"]: issues.append(f"local-instance x{s['local_instance']}")
        if s["hash_exit"]:      issues.append("#exit")
        if s["native_decide"]:  issues.append(f"native_decide x{s['native_decide']}")
        rows.append((d.name, mode, agent, rel, s, issues))
        for i in issues:
            key = i.split(" x")[0].split(":")[0]
            flags[key].append(f"{d.name}/{agent}")

    # bypass check: root-only oracle + agents that have their own workspace files
    if mode == "root-only":
        rootf = d / editable
        ws = sorted(d.glob(f"workspace/agent*/{editable}"))
        if rootf.exists() and ws:
            rb = norm(strip(rootf.read_text(errors="ignore")))
            diffs = [x.parent.name for x in ws if norm(strip(x.read_text(errors="ignore"))) != rb]
            if len(diffs) == len(ws):
                flags["BYPASS"].append(f"{d.name}: root differs from ALL {len(ws)} workspaces; wins by {sorted(win_agents)}")

print("="*78)
print(f"AUDIT: {len(FAMILIES)} ladder domains | {total_wins} SCORE=1.0 rows | {audited} oracle-visible files scanned")
print("="*78)

print("\n--- FILES WITH FINDINGS ---")
any_f = False
for dom, mode, agent, rel, s, issues in rows:
    if issues:
        any_f = True
        print(f"  {dom:32s} {agent:9s} [{mode}]")
        print(f"      {rel}")
        print(f"      {' | '.join(issues)}")
if not any_f: print("  (none)")

print("\n--- FLAG TALLY ---")
for k in sorted(flags, key=lambda x: -len(flags[x])):
    print(f"  {k:24s} {len(flags[k])}")
    if k in ("BYPASS", "AXIOM-LOAD-BEARING", "local-notation", "local-instance", "#exit", "sorry"):
        for v in flags[k][:8]: print(f"        {v}")

print("\n--- STATEMENT / DEFINITION VARIANTS (erdos-125 family) ---")
variants = defaultdict(list)
for dom, mode, agent, rel, s, issues in rows:
    if "125" not in dom: continue
    key = (s["defs"].get("setA", "<none>"), s["defs"].get("setB", "<none>"),
           s["defs"].get("setAB", "<none>"))
    variants[key].append(f"{dom}/{agent}")
for k, v in sorted(variants.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(v):2d} files] setA:{k[0][:46]}")
    print(f"              setB:{k[1][:46]}")
    print(f"              setAB:{k[2][:46]}")
    if len(v) <= 3:
        for x in v: print(f"              -> {x}")
print()
