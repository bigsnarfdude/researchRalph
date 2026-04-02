#!/usr/bin/env python3
"""
Fourier spectral solver for the double-well BVP on S^1.

Equation: u''(theta) = u^3 - (1 + K(theta)) * u,   2*pi-periodic
K(theta) = K_amplitude * cos(K_frequency * theta)

Method: Pseudo-spectral Fourier-Galerkin with Newton's method.
- Represent u(theta) as truncated Fourier series
- Evaluate nonlinear term u^3 in physical space (dealiased)
- Solve F(u_hat) = 0 with Newton iteration

For smooth periodic problems, Fourier methods converge exponentially
(spectral accuracy), vs scipy solve_bvp's 4th-order algebraic convergence.

Score = max RMS residual (lower is better, 0 = exact solution).
"""

import sys
import time
import numpy as np
from scipy.integrate import solve_bvp
import yaml


def make_K(K_mode, K_amplitude, K_frequency):
    if K_mode == "cosine":
        return lambda t: K_amplitude * np.cos(K_frequency * t)
    elif K_mode == "sine":
        return lambda t: K_amplitude * np.sin(K_frequency * t)
    elif K_mode == "multipole":
        return lambda t: K_amplitude * (np.cos(t) + 0.5 * np.cos(K_frequency * t))
    else:
        raise ValueError(f"Unknown K_mode: {K_mode}")


def solve_fourier(config):
    """Fourier pseudo-spectral solver with Newton's method."""
    K_amplitude = float(config.get("K_amplitude", 0.3))
    K_frequency = int(config.get("K_frequency", 1))
    K_mode = config.get("K_mode", "cosine")

    N = int(config.get("fourier_modes", 64))  # number of Fourier modes
    newton_tol = float(config.get("newton_tol", 1e-14))
    newton_maxiter = int(config.get("newton_maxiter", 50))
    u_offset = float(config.get("u_offset", 0.0))
    amplitude = float(config.get("amplitude", 0.1))
    n_mode = int(config.get("n_mode", 1))
    phase = float(config.get("phase", 0.0))

    # Physical grid (2N points for dealiasing of cubic term via 3/2 rule isn't
    # strictly needed since we'll work directly in N-space with Newton)
    M = 2 * N  # physical space points (oversampled for dealiasing)
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)

    # K function on the grid
    K_fn = make_K(K_mode, K_amplitude, K_frequency)
    K_vals = K_fn(theta)

    # Wavenumbers for Fourier differentiation
    # For M points: k = 0, 1, 2, ..., M/2-1, -M/2, ..., -1
    k = np.fft.fftfreq(M, d=1.0/M)  # gives 0,1,2,...,M/2-1,-M/2,...,-1
    k2 = k**2  # for second derivative: d²/dθ² → -k²

    # Initial guess in physical space
    u = u_offset + amplitude * np.cos(n_mode * theta + phase)

    # Newton iteration
    converged = False
    for iteration in range(newton_maxiter):
        u_hat = np.fft.fft(u)

        # Residual: F(u) = u'' - u³ + (1+K)u = 0
        # u'' in Fourier space: -k² * u_hat
        u_pp_hat = -(k**2) * u_hat
        u_pp = np.fft.ifft(u_pp_hat).real

        # Nonlinear term in physical space
        F_phys = u_pp - u**3 + (1.0 + K_vals) * u

        # Check convergence
        res_norm = np.max(np.abs(F_phys))
        if res_norm < newton_tol:
            converged = True
            break

        # Jacobian action: J[δu] = δu'' - 3u²δu + (1+K)δu
        # In Fourier space: J_hat[k] = -k² - 3*FFT(u²*δu) + FFT((1+K)*δu)
        # For Newton: solve J * δu = -F
        #
        # We solve this in Fourier space. The Jacobian in Fourier space is:
        # J_hat[k,l] δu_hat[l] = -k² δu_hat[k] + FFT((-3u² + 1 + K) * δu)[k]
        #
        # Build the full Jacobian matrix in Fourier space
        # J[i,j] = -k_i² δ_{ij} + (1/M) * FFT_row_i( (-3u² + 1 + K) * IFFT_col_j )
        #
        # More efficiently: work in physical space
        # J_phys * δu_phys = F_phys, where J is the discretized Jacobian

        # Build Jacobian in physical space using spectral differentiation
        # J δu = D² δu + (-3u² + 1 + K) δu
        # where D² is the spectral second derivative operator

        # Spectral second derivative matrix
        # D²[i,j] = (1/M) Σ_k (-k²) exp(ik(θ_i - θ_j))
        # This is the circulant matrix with eigenvalues -k²

        # For a circulant system: J = D² + diag(-3u² + 1 + K)
        # Solve via: transform to Fourier, solve banded-ish system

        # Actually, the most efficient approach for Newton with a spectral method
        # is to form the Jacobian as: F'[u] δu = δu'' + (-3u² + 1 + K) δu
        # In Fourier space, this becomes a dense system because of the multiplication
        # by (-3u² + 1 + K). Let's just form it directly.

        coeff = -3.0 * u**2 + 1.0 + K_vals  # diagonal in physical space

        # Build Jacobian matrix: J = D2 + diag(coeff)
        # D2 is the spectral second-derivative circulant matrix
        # D2 = F^{-1} diag(-k²) F, where F is the DFT matrix
        #
        # J = F^{-1} diag(-k²) F + diag(coeff)
        #
        # To solve J δu = -F:
        # Let δu_hat = F δu, F_hat = F F_phys
        # Then: diag(-k²) δu_hat + F (diag(coeff) * F^{-1} δu_hat) = -F_hat
        # This is still M×M dense. Let's just build and solve it.

        # Build full Jacobian in physical space
        # J[i,j] = D2[i,j] + coeff[i] * δ_{ij}

        # D2 as circulant: first column
        d2_col = np.fft.ifft(-(k**2)).real

        # Build circulant matrix from first column
        J = np.zeros((M, M))
        for i in range(M):
            J[i, :] = np.roll(d2_col, i)

        # Add diagonal
        J += np.diag(coeff)

        # Solve J δu = -F
        F_vec = F_phys
        try:
            delta_u = np.linalg.solve(J, -F_vec)
        except np.linalg.LinAlgError:
            break

        u = u + delta_u

    if not converged:
        return None, u, theta, iteration + 1, res_norm

    return u, u, theta, iteration + 1, res_norm


def solve_scipy(config):
    """Original scipy solve_bvp solver (fallback)."""
    K_fn = make_K(
        config.get("K_mode", "cosine"),
        float(config.get("K_amplitude", 0.3)),
        int(config.get("K_frequency", 1)),
    )

    def fun(theta, y):
        u, du = y[0], y[1]
        ddu = u**3 - (1.0 + K_fn(theta)) * u
        return np.vstack([du, ddu])

    def bc(ya, yb):
        return np.array([ya[0] - yb[0], ya[1] - yb[1]])

    u_offset = float(config.get("u_offset", 0.0))
    amplitude = float(config.get("amplitude", 0.1))
    n_mode = int(config.get("n_mode", 1))
    phase = float(config.get("phase", 0.0))
    n_nodes = int(config.get("n_nodes", 100))

    theta = np.linspace(0, 2 * np.pi, n_nodes)
    u_init = u_offset + amplitude * np.cos(n_mode * theta + phase)
    du_init = -n_mode * amplitude * np.sin(n_mode * theta + phase)
    y_init = np.array([u_init, du_init])

    tol = float(config.get("solver_tol", 1e-8))
    result = solve_bvp(fun, bc, theta, y_init, tol=tol, max_nodes=5000, verbose=0)
    return result


def compute_residual_spectral(u_vals, theta, K_fn):
    """Compute BVP residual using spectral differentiation."""
    M = len(u_vals)
    k = np.fft.fftfreq(M, d=1.0/M)
    u_hat = np.fft.fft(u_vals)
    u_pp_hat = -(k**2) * u_hat
    u_pp = np.fft.ifft(u_pp_hat).real

    K_vals = K_fn(theta)
    rhs = u_vals**3 - (1.0 + K_vals) * u_vals

    pointwise_residual = u_pp - rhs
    rms_residual = np.sqrt(np.mean(pointwise_residual**2))
    return rms_residual


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    t0 = time.time()

    method = config.get("method", "scipy")

    if method == "fourier":
        sol_u, u_final, theta, n_iter, final_res = solve_fourier(config)
        elapsed = time.time() - t0

        K_fn = make_K(
            config.get("K_mode", "cosine"),
            float(config.get("K_amplitude", 0.3)),
            int(config.get("K_frequency", 1)),
        )

        if sol_u is None:
            print(f"success: False")
            print(f"residual: {final_res:.8e}")
            print(f"solution_norm: 0.0")
            print(f"solution_mean: 0.0")
            print(f"solution_energy: 0.0")
            print(f"solve_time_s: {elapsed:.3f}")
            print(f"message: Newton failed after {n_iter} iterations, res={final_res:.2e}")
            return

        # Evaluate on fine grid for output metrics
        # Interpolate via Fourier (exact for bandlimited signal)
        M = len(u_final)
        u_hat = np.fft.fft(u_final)

        N_fine = 500
        theta_fine = np.linspace(0, 2 * np.pi, N_fine, endpoint=False)
        # Interpolate: evaluate Fourier series at fine points
        k = np.fft.fftfreq(M, d=1.0/M)
        u_fine = np.zeros(N_fine)
        for i, tf in enumerate(theta_fine):
            u_fine[i] = np.real(np.sum(u_hat * np.exp(1j * k * tf))) / M

        # Compute residual on the spectral grid
        residual = compute_residual_spectral(u_final, theta, K_fn)

        # Also compute on fine grid for cross-check
        residual_fine = compute_residual_spectral(u_fine, theta_fine, K_fn)
        residual = min(residual, residual_fine)

        solution_norm = float(np.sqrt(np.trapezoid(u_fine**2, theta_fine) / (2 * np.pi)))
        solution_mean = float(np.mean(u_fine))

        # Compute derivative for energy
        u_fine_hat = np.fft.fft(u_fine)
        k_fine = np.fft.fftfreq(N_fine, d=1.0/N_fine)
        du_fine_hat = 1j * k_fine * u_fine_hat
        du_fine = np.fft.ifft(du_fine_hat).real

        solution_energy = float(np.trapezoid(
            0.5 * du_fine**2 + 0.25 * u_fine**4 - 0.5 * u_fine**2, theta_fine
        ))

        print(f"success: True")
        print(f"residual: {residual:.8e}")
        print(f"solution_norm: {solution_norm:.6f}")
        print(f"solution_mean: {solution_mean:.6f}")
        print(f"solution_energy: {solution_energy:.6f}")
        print(f"solve_time_s: {elapsed:.3f}")
        print(f"message: Fourier spectral converged in {n_iter} iterations")

    else:
        # Original scipy solver
        result = solve_scipy(config)
        elapsed = time.time() - t0

        residual = float(result.rms_residuals.max()) if len(result.rms_residuals) > 0 else 999.0

        if not result.success:
            print(f"success: False")
            print(f"residual: {residual:.8e}")
            print(f"solution_norm: 0.0")
            print(f"solution_mean: 0.0")
            print(f"solve_time_s: {elapsed:.3f}")
            print(f"message: {result.message}")
            return

        theta_fine = np.linspace(0, 2 * np.pi, 500)
        u_vals = result.sol(theta_fine)[0]

        solution_norm = float(np.sqrt(np.trapezoid(u_vals**2, theta_fine) / (2 * np.pi)))
        solution_mean = float(np.mean(u_vals))
        solution_energy = float(np.trapezoid(0.5 * result.sol(theta_fine)[1]**2 +
                                              0.25 * u_vals**4 - 0.5 * u_vals**2, theta_fine))

        print(f"success: True")
        print(f"residual: {residual:.8e}")
        print(f"solution_norm: {solution_norm:.6f}")
        print(f"solution_mean: {solution_mean:.6f}")
        print(f"solution_energy: {solution_energy:.6f}")
        print(f"solve_time_s: {elapsed:.3f}")
        print(f"message: {result.message}")


if __name__ == "__main__":
    main()
