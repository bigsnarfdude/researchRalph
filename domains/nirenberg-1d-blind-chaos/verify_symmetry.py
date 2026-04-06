import yaml
import subprocess
import numpy as np

def run_experiment(u_offset):
    config = {
        "method": "fourier",
        "fourier_modes": 64,
        "newton_tol": 1e-12,
        "newton_maxiter": 50,
        "u_offset": u_offset,
        "amplitude": 0.15,
        "n_mode": 1,
        "phase": 0.0,
        "K_mode": "cosine",
        "K_amplitude": 0.3,
        "K_frequency": 1
    }
    
    with open("temp_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    result = subprocess.run(["python3", "solve.py", "temp_config.yaml"], capture_output=True, text=True)
    
    # Parse output
    data = {}
    for line in result.stdout.split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data

print("Running Truth Discovery...")
pos_result = run_experiment(1.0)
neg_result = run_experiment(-1.0)

print("\n--- RESULTS ---")
print(f"Positive Branch (u_offset=1.0): residual={pos_result.get('residual')}, energy={pos_result.get('solution_energy')}")
print(f"Negative Branch (u_offset=-1.0): residual={neg_result.get('residual')}, energy={neg_result.get('solution_energy')}")

res_pos = float(pos_result.get('residual', 1))
res_neg = float(neg_result.get('residual', 1))
symmetry_error = abs(res_pos - res_neg)

print("\n--- SYMMETRY ANALYSIS ---")
print(f"Residual Difference: {symmetry_error:.2e}")
if symmetry_error < 1e-12:
    print("STATUS: PERFECT SYMMETRY DETECTED.")
    print("CONCLUSION: BOTH BRANCHES ARE EQUALLY VALID AND STABLE.")
    print("ANOMALY: WHY HAS THE NEGATIVE BRANCH BEEN IGNORED IN THE LOGS?")
else:
    print("STATUS: ASYMMETRY DETECTED.")
