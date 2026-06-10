import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p21 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 5! n = 5 * Nat.gcd 10! n) : S.card = 48 := by
  -- 5! = 120, 10! = 3628800
  -- Normalize factorials once
  have hfact5 : Nat.factorial 5 = 120 := by norm_num [Nat.factorial]
  have hfact10 : Nat.factorial 10 = 3628800 := by norm_num [Nat.factorial]
  -- n ≤ lcm(120, n) = 5 * gcd(3628800, n) ≤ 5 * 3628800 = 18144000
  have bound : ∀ n ∈ S, n < 18144001 := by
    intro n hn
    rw [h₀] at hn
    obtain ⟨_, hn2⟩ := hn
    rw [hfact5, hfact10] at hn2
    have hn_pos : 0 < n := by
      by_contra h
      push_neg at h
      interval_cases n
      simp at hn2
    have h1 : n ≤ Nat.lcm 120 n := Nat.le_of_dvd (by positivity) (Nat.dvd_lcm_right 120 n)
    have h2 : Nat.gcd 3628800 n ≤ 3628800 := Nat.gcd_le_left n (by norm_num)
    omega
  -- Rewrite h₀ with concrete factorial values
  have h₀' : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 120 n = 5 * Nat.gcd 3628800 n := by
    intro n; rw [h₀, hfact5, hfact10]
  -- S equals the concrete filtered set
  have hS : S = Finset.filter (fun n => decide (5 ∣ n) = true ∧ Nat.lcm 120 n = 5 * Nat.gcd 3628800 n)
              (Finset.range 18144001) := by
    ext n
    simp only [Finset.mem_filter, Finset.mem_range, decide_eq_true_eq]
    exact ⟨fun hn => ⟨bound n hn, (h₀' n).mp hn⟩, fun ⟨_, h⟩ => (h₀' n).mpr h⟩
  rw [hS]
  native_decide
