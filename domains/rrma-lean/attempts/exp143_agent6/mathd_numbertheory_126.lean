import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_126 (x a : ℕ) (h₀ : 0 < x ∧ 0 < a) (h₁ : Nat.gcd a 40 = x + 3)
  (h₂ : Nat.lcm a 40 = x * (x + 3))
  (h₃ : ∀ b : ℕ, 0 < b → Nat.gcd b 40 = x + 3 ∧ Nat.lcm b 40 = x * (x + 3) → a ≤ b) : a = 8 := by
  -- Key identity: gcd * lcm = a * 40
  have hgcdlcm := Nat.gcd_mul_lcm a 40
  rw [h₁, h₂] at hgcdlcm
  -- So (x+3) * (x * (x+3)) = a * 40
  -- i.e. x * (x+3)^2 = 40 * a
  -- Also x+3 | 40 (since gcd(a,40) = x+3 ≤ 40)
  have hx3_le : x + 3 ≤ 40 := by
    have := Nat.gcd_le_right a (by omega : 0 < 40)
    omega
  -- gcd(a,40) divides 40
  have hx3_div : (x + 3) ∣ 40 := by
    rw [← h₁]; exact Nat.gcd_dvd_right a 40
  -- x+3 divides 40, so x+3 ∈ {1,2,4,5,8,10,20,40}
  -- Since x > 0, x+3 ≥ 4, so x+3 ∈ {4,5,8,10,20,40}
  -- From hgcdlcm: x*(x+3)^2 = 40*a, so 40 | x*(x+3)^2
  -- For a to be a positive integer, 40 | x*(x+3)^2
  -- Testing x+3=8 (x=5): a = 5*64/40 = 8. gcd(8,40)=8 ✓, lcm(8,40)=40=5*8 ✓
  -- We need to show x=5 and a=8
  -- From the equation and divisibility constraints
  have hx_le : x ≤ 37 := by omega
  interval_cases x <;> simp_all <;> omega
