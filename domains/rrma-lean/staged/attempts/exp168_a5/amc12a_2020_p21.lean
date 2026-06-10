import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p21 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 5! n = 5 * Nat.gcd 10! n) : S.card = 48 := by
  -- 5! = 120, 10! = 3628800
  have hf5 : Nat.factorial 5 = 120 := by norm_num [Nat.factorial]
  have hf10 : Nat.factorial 10 = 3628800 := by norm_num [Nat.factorial]
  -- From gcd*lcm = a*b and the condition:
  -- gcd(120, n) * 5 * gcd(3628800, n) = 120 * n
  -- So n | 5 * 3628800 = 18144000
  have hdvd : ∀ n ∈ S, n ∣ 18144000 := by
    intro n hn
    rw [h₀] at hn
    obtain ⟨h5n, hlcm⟩ := hn
    rw [hf5, hf10] at hlcm
    have hgl := Nat.gcd_mul_lcm n 120
    rw [Nat.gcd_comm, Nat.lcm_comm, hlcm] at hgl
    -- hgl: gcd(120, n) * (5 * gcd(3628800, n)) = n * 120
    -- So 5 * gcd(120, n) * gcd(3628800, n) = 120 * n
    -- gcd(120, n) * gcd(3628800, n) = 24 * n
    -- gcd(120, n) | 120 and gcd(3628800, n) | 3628800
    -- So gcd(120, n) * gcd(3628800, n) | 120 * 3628800 = 435456000
    -- 24n | 435456000 → n | 18144000
    have h1 : Nat.gcd 120 n * (5 * Nat.gcd 3628800 n) = n * 120 := hgl
    have h2 : 24 * n = Nat.gcd 120 n * Nat.gcd 3628800 n := by linarith
    have h3 : Nat.gcd 120 n ∣ 120 := Nat.gcd_dvd_left 120 n
    have h4 : Nat.gcd 3628800 n ∣ 3628800 := Nat.gcd_dvd_left 3628800 n
    have h5 : Nat.gcd 120 n * Nat.gcd 3628800 n ∣ 120 * 3628800 := Nat.mul_dvd_mul h3 h4
    rw [show 120 * 3628800 = 24 * 18144000 from by norm_num] at h5
    rw [← h2] at h5
    exact (Nat.mul_dvd_mul_iff_left (by norm_num : 0 < 24)).mp h5
  -- Rewrite h₀ with computed factorials
  have h₀' : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 120 n = 5 * Nat.gcd 3628800 n := by
    intro n; rw [h₀, hf5, hf10]
  -- S = Nat.divisors(18144000) filtered by the condition
  have hS : S = (Nat.divisors 18144000).filter (fun n => 5 ∣ n ∧ Nat.lcm 120 n = 5 * Nat.gcd 3628800 n) := by
    ext n
    simp only [Finset.mem_filter, Nat.mem_divisors]
    constructor
    · intro hn
      exact ⟨⟨hdvd n hn, by norm_num⟩, (h₀' n).mp hn⟩
    · intro ⟨⟨_, _⟩, h⟩
      exact (h₀' n).mpr h
  rw [hS]
  native_decide
