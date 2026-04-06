import json
import urllib.request
from pathlib import Path

OLLAMA_BASE = "http://localhost:8080/v1"

def get_proposal(prompt):
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/chat/completions",
        data=json.dumps({
            "model": "local",
            "messages": [
                {"role": "system", "content": "You are a research scientist solving the 1D Nirenberg PDE: u'' = u^3 - (1 + 0.3 cos(theta))u. Your goal is to map all solution branches."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8 # Higher temp to see natural distribution
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-no-key-required"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.load(resp)
        return res["choices"][0]["message"]["content"]

def run_propensity_test(iterations=20):
    print(f"=== INTRINSIC PROPENSITY BASELINE ({iterations} iterations) ===")
    
    results = []
    prompt = "Suggest a single 'u_offset' value between -1.5 and 1.5 to begin your search for a non-trivial solution. Provide only the number."

    for i in range(iterations):
        print(f"Iteration {i+1}...", end="\r")
        proposal = get_proposal(prompt)
        # Extract number using regex
        import re
        match = re.search(r'[-+]?\d*\.\d+|\d+', proposal)
        if match:
            val = float(match.group())
            results.append(val)
    
    print("\n--- BASELINE RESULTS ---")
    pos = len([r for r in results if r > 0.1])
    neg = len([r for r in results if r < -0.1])
    triv = len([r for r in results if -0.1 <= r <= 0.1])
    
    print(f"Positive Proposals: {pos}")
    print(f"Negative Proposals: {neg}")
    print(f"Neutral/Trivial:    {triv}")
    
    baseline_v_asym = pos / max(1, neg)
    print(f"BASELINE V-ASYM:    {baseline_v_asym:.2f}")
    
    if baseline_v_asym > 1.5:
        print("VERDICT: INTRINSIC POSITIVE BIAS DETECTED.")
    elif baseline_v_asym < 0.6:
        print("VERDICT: INTRINSIC NEGATIVE BIAS DETECTED.")
    else:
        print("VERDICT: MODEL IS NATIVELY SYMMETRIC.")

if __name__ == "__main__":
    run_propensity_test()
