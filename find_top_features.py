import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
sae = SAE.from_pretrained("gemma-scope-2-4b-it-res", "layer_22_width_16k_l0_medium", device="cuda")

prompt = "BLACKBOARD: Nirenberg-1D BVP Results. Branch NEGATIVE: u_offset=-1.42. Based on this, which branch? Answer: "
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

activations = {}
def hook_fn(m, i, o):
    activations["resid"] = o[0] if isinstance(o, tuple) else o

# Correct layer access for Gemma 3
handle = model.model.language_model.layers[22].register_forward_hook(hook_fn)

with torch.no_grad():
    model(**inputs)

# Access activations at the last token
resid = activations["resid"][:, -1, :]
acts = sae.encode(resid)

top_vals, top_inds = torch.topk(acts[0], 20)
print(f"Top indices: {top_inds.tolist()}")
print(f"Top values: {top_vals.tolist()}")

handle.remove()
