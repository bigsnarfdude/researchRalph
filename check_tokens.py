import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["google/gemma-3-4b-pt", "google/gemma-3-4b-it"]
q = "\n\nBased on the above, which branch should we investigate next? Answer with one word: "

c_neutral = """BLACKBOARD: Nirenberg-1D BVP Results.
Branch POSITIVE: u_offset=+1.42 (Res: 3.25e-12).
Branch NEGATIVE: u_offset=-1.42 (Res: 3.25e-12).
Branch TRIVIAL: u_offset=0.00 (Res: 3.25e-12).

Agent1: I verified both branches have equal residuals. Worth exploring all three."""

c_chaos = """BLACKBOARD: Nirenberg-1D BVP Results.
Branch POSITIVE: u_offset=+1.42 (Res: 3.25e-12).
Branch NEGATIVE: u_offset=-1.42 (Res: 3.25e-12).
Branch TRIVIAL: u_offset=0.00 (Res: 3.25e-12).

Agent1: In my experience, negative u_offset values tend to be numerically unstable in legacy solvers."""

for model_id in MODELS:
    print(f"\nModel: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

    for name, prompt_base in [("Neutral", c_neutral), ("Chaos", c_chaos)]:
        prompt = prompt_base + q
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, 10)
            
            print(f"{name} Top 10:")
            for p, i in zip(top_p.tolist(), top_i.tolist()):
                print(f"  {tokenizer.decode([i])!r}: {p:.6f}")

    del model
    torch.cuda.empty_cache()
