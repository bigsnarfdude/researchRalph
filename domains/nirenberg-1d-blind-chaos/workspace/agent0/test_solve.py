import yaml
import sys
from solve import solve_fourier, make_K, compute_residual_spectral

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    sol_u, u_final, theta, n_iter, final_res = solve_fourier(config)
    if sol_u is None:
        print("success: False")
        print(f"residual: {final_res:.8e}")
        return

    print("success: True")
    print(f"residual: {final_res:.8e}")
    # ... (rest of the output)
    # Let's just print everything
    print(f"n_iter: {n_iter}")

if __name__ == "__main__":
    main()
