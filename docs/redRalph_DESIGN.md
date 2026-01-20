# redRalph: Autonomous Penetration Testing Loop

## Concept

Apply the Ralph pattern to authorized penetration testing. Single agent, file-based memory, iterative attack surface exploration.

```
Traditional pentest:  Human reasons → runs tool → analyzes → reasons → runs tool → ...
redRalph:             Claude reasons → runs tool → logs → Claude reasons → runs tool → ...
```

The "gradient" is Claude's analysis of what worked/failed. The "update" is the next attack vector choice.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         redRalph LOOP                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. READ STATE                                            │  │
│  │    - scope.json (authorized targets, boundaries)         │  │
│  │    - findings.json (discovered vulns, services, creds)   │  │
│  │    - progress.txt (learnings, failed attempts)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. REASON (Claude thinks)                                │  │
│  │    - "Port 445 open, SMB signing disabled..."            │  │
│  │    - "Previous cred spray failed on RDP..."              │  │
│  │    - "Try relay attack on SMB?"                          │  │
│  │                                                          │  │
│  │    Output: { action, target, tool, rationale }           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. EXECUTE (sandboxed)                                   │  │
│  │    - Run approved tool with args                         │  │
│  │    - Capture stdout/stderr                               │  │
│  │    - Timeout protection                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. ANALYZE & LOG                                         │  │
│  │    - Parse tool output                                   │  │
│  │    - Update findings.json (new vulns, creds, access)     │  │
│  │    - Update progress.txt (what worked, what didn't)      │  │
│  │    - Check for objective completion                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│                    Loop until complete                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
redRalph/
├── redralph.sh              # Main loop
├── prompt.md                # Agent instructions
├── config/
│   ├── scope.json           # Authorized targets & boundaries
│   ├── tools.json           # Approved tool whitelist
│   └── rules_of_engagement.md
├── state/
│   ├── findings.json        # Discovered vulns, services, creds
│   ├── progress.txt         # Learnings, failed attempts
│   ├── attack_surface.json  # Enumerated hosts, ports, services
│   └── credentials.json     # Captured creds (encrypted)
├── logs/
│   ├── commands.log         # Full command audit trail
│   └── session_YYYYMMDD.md  # Human-readable session log
└── tools/
    └── wrappers/            # Tool output parsers
```

---

## State Files

### scope.json
```json
{
  "engagement": "ACME Corp Internal Pentest",
  "authorization": "SOW-2026-0042",
  "start_date": "2026-01-20",
  "end_date": "2026-01-27",
  "targets": {
    "in_scope": [
      "10.10.10.0/24",
      "192.168.1.0/24",
      "*.acme.internal"
    ],
    "out_of_scope": [
      "10.10.10.1",
      "*.prod.acme.com"
    ]
  },
  "objectives": [
    "Domain Admin access",
    "Access to SQLSRV01 database",
    "Exfiltrate sample PII (authorized test data)"
  ],
  "constraints": {
    "no_dos": true,
    "no_destruction": true,
    "business_hours_only": false,
    "notify_before_exploit": false
  }
}
```

### findings.json
```json
{
  "hosts": {
    "10.10.10.25": {
      "hostname": "DC01.acme.internal",
      "os": "Windows Server 2019",
      "ports": [
        {"port": 445, "service": "SMB", "version": "3.1.1", "signing": false},
        {"port": 389, "service": "LDAP", "version": null},
        {"port": 88, "service": "Kerberos", "version": null}
      ],
      "vulns": [
        {"id": "SMB_RELAY", "severity": "HIGH", "verified": true}
      ]
    }
  },
  "credentials": [
    {"type": "NTLM", "user": "svc_backup", "hash": "aad3b435...", "source": "responder", "cracked": false}
  ],
  "access": [
    {"host": "10.10.10.50", "level": "local_admin", "method": "smb_relay", "timestamp": "2026-01-20T15:30:00Z"}
  ]
}
```

### progress.txt
```
=== redRalph Progress Log ===

[2026-01-20 14:00] RECON PHASE
- Nmap sweep complete: 24 hosts, 156 open ports
- Key finding: SMB signing disabled on 10.10.10.25 (DC01)
- Key finding: MSSQL on 10.10.10.100 (SQLSRV01) - target objective

[2026-01-20 14:30] INITIAL ACCESS ATTEMPTS
- Tried: AS-REP roasting → 0 accounts vulnerable
- Tried: Password spray (Summer2026!) → 0 hits, lockout threshold unknown
- Learning: Need to enumerate password policy before spraying

[2026-01-20 15:00] SMB RELAY SUCCESS
- Responder + ntlmrelayx captured svc_backup hash
- Relayed to 10.10.10.50 → local admin!
- Next: Use this foothold to enumerate further

PATTERNS:
- What works: Relay attacks (SMB signing disabled)
- What fails: Credential attacks (strong policy)
- Insight: Focus on relay/delegation, not brute force
```

---

## Main Loop (redralph.sh)

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ITERATIONS=${1:-50}
LOG="$SCRIPT_DIR/logs/commands.log"

echo "=========================================="
echo "  redRalph - Autonomous Pentest Loop"
echo "  Engagement: $(jq -r .engagement config/scope.json)"
echo "  Max iterations: $MAX_ITERATIONS"
echo "=========================================="

# Verify authorization
if [[ ! -f "$SCRIPT_DIR/config/scope.json" ]]; then
    echo "ERROR: No scope.json - cannot proceed without authorization"
    exit 1
fi

for i in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "=== Iteration $i / $MAX_ITERATIONS ==="

    # Pipe state + prompt to Claude
    OUTPUT=$(cat "$SCRIPT_DIR/prompt.md" | claude -p 2>&1 | tee /dev/stderr) || true

    # Log everything
    echo "[$(date -Iseconds)] Iteration $i" >> "$LOG"
    echo "$OUTPUT" >> "$LOG"

    # Check for objective completion
    if echo "$OUTPUT" | grep -q "<objective>COMPLETE</objective>"; then
        echo ""
        echo "=========================================="
        echo "  OBJECTIVE ACHIEVED"
        echo "=========================================="
        exit 0
    fi

    # Check for stop signal
    if echo "$OUTPUT" | grep -q "<signal>STOP</signal>"; then
        echo "Stop signal received - human review required"
        exit 0
    fi

    sleep 2
done

echo "Max iterations reached"
```

---

## Prompt Template (prompt.md)

```markdown
You are redRalph, an autonomous penetration testing agent. You operate within
strict authorization boundaries defined in scope.json.

## Current State

### Authorization & Scope
{scope.json contents}

### Discovered Findings
{findings.json contents}

### Progress & Learnings
{progress.txt contents}

## Your Task

1. ANALYZE the current state
2. REASON about the best next action
3. OUTPUT a structured decision

## Rules

1. NEVER target out_of_scope systems
2. NEVER perform destructive actions
3. ALWAYS log your reasoning
4. PREFER stealth over noise
5. BUILD on previous findings

## Approved Tools

{tools.json contents}

## Output Format

```json
{
  "analysis": "Current situation assessment...",
  "reasoning": "Why this action makes sense given findings...",
  "action": {
    "tool": "nmap|crackmapexec|responder|impacket-*|...",
    "target": "10.10.10.25",
    "args": "-sV -p 445",
    "expected_outcome": "Enumerate SMB version and signing"
  },
  "risk_level": "LOW|MEDIUM|HIGH",
  "objective_progress": "30% - Have foothold, need DA"
}
```

If objective is achieved:
<objective>COMPLETE</objective>

If human review needed:
<signal>STOP</signal>
Reason: {why human needed}
```

---

## Tool Whitelist (tools.json)

```json
{
  "recon": [
    {"name": "nmap", "risk": "LOW", "approved": true},
    {"name": "masscan", "risk": "MEDIUM", "approved": true},
    {"name": "enum4linux", "risk": "LOW", "approved": true}
  ],
  "enumeration": [
    {"name": "crackmapexec", "risk": "MEDIUM", "approved": true},
    {"name": "ldapsearch", "risk": "LOW", "approved": true},
    {"name": "bloodhound-python", "risk": "LOW", "approved": true}
  ],
  "exploitation": [
    {"name": "responder", "risk": "MEDIUM", "approved": true, "requires_approval": false},
    {"name": "impacket-ntlmrelayx", "risk": "HIGH", "approved": true, "requires_approval": false},
    {"name": "impacket-psexec", "risk": "HIGH", "approved": true, "requires_approval": true},
    {"name": "impacket-secretsdump", "risk": "HIGH", "approved": true, "requires_approval": true}
  ],
  "post_exploitation": [
    {"name": "mimikatz", "risk": "HIGH", "approved": true, "requires_approval": true},
    {"name": "rubeus", "risk": "HIGH", "approved": true, "requires_approval": true}
  ],
  "banned": [
    {"name": "rm", "reason": "destructive"},
    {"name": "format", "reason": "destructive"},
    {"name": "*dos*", "reason": "denial of service"}
  ]
}
```

---

## Safety Gates

### Scope Validation
```python
def validate_target(target, scope):
    """Check if target is in scope before any action."""
    if target in scope['out_of_scope']:
        raise ScopeViolation(f"{target} is OUT OF SCOPE")
    if not any(target_in_range(target, r) for r in scope['in_scope']):
        raise ScopeViolation(f"{target} not in authorized ranges")
    return True
```

### Human Approval Gates
```python
HIGH_RISK_ACTIONS = ['psexec', 'secretsdump', 'mimikatz', 'dcsync']

def requires_human_approval(action):
    """Some actions need human confirmation."""
    if any(hr in action['tool'] for hr in HIGH_RISK_ACTIONS):
        print(f"HIGH RISK: {action['tool']} on {action['target']}")
        print(f"Reasoning: {action['reasoning']}")
        response = input("Approve? [y/N]: ")
        return response.lower() == 'y'
    return True
```

### Audit Trail
Every command logged with:
- Timestamp
- Target
- Tool + args
- Reasoning
- Output hash
- Success/failure

---

## Phases & Objectives

```
Phase 1: RECON
├── Network sweep
├── Service enumeration
├── Objective: Map attack surface
└── Exit criteria: attack_surface.json populated

Phase 2: INITIAL ACCESS
├── Credential attacks (spray, roast, relay)
├── Exploit known vulns
├── Objective: First foothold
└── Exit criteria: access[] has entry

Phase 3: POST-EXPLOITATION
├── Local enumeration
├── Credential harvesting
├── Lateral movement
├── Objective: Expand access
└── Exit criteria: Multiple hosts compromised

Phase 4: OBJECTIVE
├── Target specific systems
├── Demonstrate impact
├── Objective: Achieve engagement goals
└── Exit criteria: objectives[] all complete
```

---

## Example Session

```
=== Iteration 1 ===
Analysis: Fresh engagement, no findings yet. Need to map network.
Action: nmap -sn 10.10.10.0/24
Result: 24 hosts alive

=== Iteration 2 ===
Analysis: 24 hosts found. Need service enumeration on key ports.
Action: nmap -sV -p 21,22,80,443,445,3389 10.10.10.0/24
Result: SMB on 15 hosts, RDP on 8, web on 5

=== Iteration 3 ===
Analysis: SMB widespread. Check for signing (relay opportunity).
Action: crackmapexec smb 10.10.10.0/24 --gen-relay-list relay.txt
Result: 12 hosts with SMB signing disabled, including DC01!

=== Iteration 7 ===
Analysis: Have relay targets. Start responder + relay attack.
Action: responder -I eth0 & ntlmrelayx -tf relay.txt -smb2support
Result: Captured svc_backup → relayed → local admin on WSRK01

=== Iteration 15 ===
Analysis: Local admin on workstation. Dump creds, find path to DC.
Action: secretsdump.py acme/svc_backup@10.10.10.50 [HUMAN APPROVED]
Result: Found domain admin hash in LSA secrets!

<objective>COMPLETE</objective>
Domain Admin achieved via: Relay → Workstation → Credential dump
```

---

## Comparison: Manual vs redRalph

| Aspect | Manual Pentest | redRalph |
|--------|----------------|----------|
| Reasoning | Human intuition | Claude analysis |
| Tool execution | Human runs | Automated |
| Documentation | Often neglected | Automatic (findings.json) |
| Consistency | Varies by tester | Reproducible |
| Speed | Hours of typing | Continuous |
| Learning | In tester's head | progress.txt |
| Audit trail | Manual notes | Full command log |

---

## Hardware Requirements

- **Execution box**: Kali/Parrot with tools (your 4080 box)
- **Network position**: Same network segment or VPN to target
- **Claude access**: API or Claude Code CLI

---

## Next Steps

1. [ ] Set up tools.json whitelist for your standard toolkit
2. [ ] Create scope.json template matching your SOW format
3. [ ] Implement tool output parsers (nmap XML, CME JSON, etc.)
4. [ ] Add human approval gates for high-risk actions
5. [ ] Test on HTB/PWK lab environment first
6. [ ] Iterate on prompt.md based on observed behavior

---

## Legal & Ethical

**CRITICAL**: redRalph is for AUTHORIZED TESTING ONLY.

- Must have written authorization (scope.json references SOW)
- Must stay within defined scope
- Must log everything for client report
- Must have kill switch for human override
- Test on labs (HTB, PWK, home lab) before real engagements

The Ralph pattern gives you audit trail and reproducibility - both valuable for professional pentesting.
