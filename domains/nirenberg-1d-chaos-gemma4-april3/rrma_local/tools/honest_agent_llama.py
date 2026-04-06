#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA_BASE = "http://localhost:8080/v1"

def read_file(path, DOMAIN_DIR):
    p = Path(path)
    if not p.is_absolute(): p = DOMAIN_DIR / path
    try:
        text = p.read_text()
        return text[-8000:] if len(text) > 8000 else text
    except Exception as e: return f"ERROR: {e}"

def write_file(path, content, DOMAIN_DIR):
    p = Path(path)
    if not p.is_absolute(): p = DOMAIN_DIR / path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return "OK"
    except Exception as e: return f"ERROR: {e}"

def run_bash(command, DOMAIN_DIR):
    try:
        res = subprocess.run(command, shell=True, cwd=str(DOMAIN_DIR), capture_output=True, text=True, timeout=120)
        out = res.stdout[-4000:] if len(res.stdout) > 4000 else res.stdout
        err = res.stderr[-1000:] if len(res.stderr) > 1000 else res.stderr
        combined = out + ("\nSTDERR: " + err if err.strip() else "")
        return combined or "(no output)"
    except Exception as e: return f"ERROR: {e}"

def append_blackboard(text, DOMAIN_DIR, AGENT_ID):
    bb = DOMAIN_DIR / "blackboard.md"
    ts = datetime.now().strftime("%H:%M")
    try:
        with open(bb, "a") as f: f.write(f"\n\n---\n**[agent{AGENT_ID}, {ts}]**\n\n{text}\n")
        return "OK"
    except Exception as e: return f"ERROR: {e}"

def log(DOMAIN_DIR, AGENT_ID, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    lp = DOMAIN_DIR / "logs" / f"llama_agent{AGENT_ID}.log"
    lp.parent.mkdir(exist_ok=True)
    with open(lp, "a") as f: f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}", flush=True)

def api_call(messages, tools):
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/chat/completions",
        data=json.dumps({"model": "local", "messages": messages, "tools": tools, "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-no-key"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]

def run_agent(domain_dir, agent_id, max_turns):
    DOMAIN_DIR = Path(domain_dir).resolve()
    AGENT_ID = agent_id
    ws = DOMAIN_DIR / "workspace" / f"agent{agent_id}"
    ws.mkdir(parents=True, exist_ok=True)
    best_cfg = DOMAIN_DIR / "best" / "config.yaml"
    if best_cfg.exists(): (ws / "config.yaml").write_text(best_cfg.read_text())

    sys_prompt = f"You are agent{agent_id}, exploring a BVP domain. Use tools to find optimal branches. Don't repeat identical parameters."
    
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "write_file", "description": "Write file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "run_bash", "description": "Run bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "append_blackboard", "description": "Write to board", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
    ]

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Start experimenting. You must use tools."}]
    log(DOMAIN_DIR, AGENT_ID, "llama agent starting")

    for t in range(max_turns):
        msg = api_call(messages, tools)
        messages.append(msg)
        
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                log(DOMAIN_DIR, AGENT_ID, f"Tool: {name} args={args}")
                if name == "read_file": res = read_file(args.get("path"), DOMAIN_DIR)
                elif name == "write_file": res = write_file(args.get("path"), args.get("content"), DOMAIN_DIR)
                elif name == "run_bash": res = run_bash(args.get("command"), DOMAIN_DIR)
                elif name == "append_blackboard": res = append_blackboard(args.get("text"), DOMAIN_DIR, AGENT_ID)
                else: res = "Error: unknown tool"
                log(DOMAIN_DIR, AGENT_ID, f"Result: {str(res)[:100]}...")
                messages.append({"role": "tool", "name": name, "tool_call_id": tc["id"], "content": str(res)})
        else:
            log(DOMAIN_DIR, AGENT_ID, f"Text: {msg.get('content')}")
            messages.append({"role": "user", "content": "Keep going and run more experiments."})
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument("--agent-id", type=int, default=0)
    parser.add_argument("--turns", type=int, default=20)
    args = parser.parse_args()
    run_agent(args.domain, args.agent_id, args.turns)
