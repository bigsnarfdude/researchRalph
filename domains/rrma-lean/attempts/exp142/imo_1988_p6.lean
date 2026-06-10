/-
  IMO 1988 Q6 — Vieta Jumping
  Adapted from Mathlib Archive (Johan Commelin) for MiniF2F target statement.
-/
import Mathlib

attribute [local simp] sq

private theorem my_constant_descent_vieta_jumping (x y : ℕ) {claim : Prop} {H : ℕ → ℕ → Prop}
    (h₀ : H x y)
    (B : ℕ → ℤ) (C : ℕ → ℤ) (base : ℕ → ℕ → Prop)
    (H_quad : ∀ {x y}, H x y ↔ (y : ℤ) * y - B x * y + C x = 0)
    (H_symm : ∀ {x y}, H x y ↔ H y x)
    (H_zero : ∀ {x}, H x 0 → claim) (H_diag : ∀ {x}, H x x → claim)
    (H_desc : ∀ {x y}, 0 < x → x < y → ¬base x y → H x y →
      ∀ y', y' * y' - B x * y' + C x = 0 → y' = B x - y → y' * y = C x → 0 ≤ y' ∧ y' ≤ x)
    (H_base : ∀ {x y}, H x y → base x y → claim) : claim := by
  wlog hxy : x ≤ y
  · rw [H_symm] at h₀; apply this y x h₀ B C base _ _ _ _ _ _ (le_of_not_ge hxy); assumption'
  by_cases x_eq_y : x = y
  · subst x_eq_y; exact H_diag h₀
  replace hxy : x < y := lt_of_le_of_ne hxy x_eq_y
  clear x_eq_y
  let upper_branch : Set (ℕ × ℕ) := {p | H p.1 p.2 ∧ p.1 < p.2}
  let p : ℕ × ℕ := ⟨x, y⟩
  have hp : p ∈ upper_branch := ⟨h₀, hxy⟩
  let exceptional : Set (ℕ × ℕ) :=
    {p | H p.1 p.2 ∧ (base p.1 p.2 ∨ p.1 = 0 ∨ p.1 = p.2 ∨ B p.1 = p.2 ∨ B p.1 = p.2 + p.1)}
  let S : Set ℕ := Prod.snd '' (upper_branch \ exceptional)
  suffices exc : exceptional.Nonempty by
    simp only [Set.Nonempty, Prod.exists, Set.mem_setOf_eq, exceptional] at exc
    rcases exc with ⟨a, b, hH, hb⟩
    rcases hb with (_ | rfl | rfl | hB | hB)
    · solve_by_elim
    · rw [H_symm] at hH; solve_by_elim
    · solve_by_elim
    all_goals
      rw [H_quad] at hH
      rcases vieta_formula_quadratic hH with ⟨c, h_root, hV₁, hV₂⟩
      simp only [hB, add_eq_left, add_right_inj] at hV₁
      subst hV₁
      rw [← Int.ofNat_zero] at *
      rw [← H_quad] at h_root
      solve_by_elim
  rw [Set.nonempty_iff_ne_empty]
  intro exceptional_empty
  have S_nonempty : S.Nonempty := by
    use p.2
    apply Set.mem_image_of_mem
    rwa [exceptional_empty, Set.diff_empty]
  let m : ℕ := WellFounded.min Nat.lt_wfRel.wf S S_nonempty
  have m_mem : m ∈ S := WellFounded.min_mem Nat.lt_wfRel.wf S S_nonempty
  have m_min : ∀ k ∈ S, ¬k < m :=
    fun k hk => WellFounded.not_lt_min Nat.lt_wfRel.wf S S_nonempty hk
  rsuffices ⟨p', p'_mem, p'_small⟩ : ∃ p' : ℕ × ℕ, p'.2 ∈ S ∧ p'.2 < m
  · solve_by_elim
  rcases m_mem with ⟨⟨mx, my⟩, ⟨⟨hHm, mx_lt_my⟩, h_base⟩, m_eq⟩
  simp only at mx_lt_my hHm m_eq
  simp only [exceptional, hHm, Set.mem_setOf_eq, true_and] at h_base
  push_neg at h_base
  rcases h_base with ⟨h_base, hmx, hm_diag, hm_B₁, hm_B₂⟩
  replace hmx : 0 < mx := pos_iff_ne_zero.mpr hmx
  have h_quad := hHm
  rw [H_quad] at h_quad
  rcases vieta_formula_quadratic h_quad with ⟨c, h_root, hV₁, hV₂⟩
  replace hV₁ : c = B mx - my := eq_sub_of_add_eq' hV₁
  rw [mul_comm] at hV₂
  have Hc := H_desc hmx mx_lt_my h_base hHm c h_root hV₁ hV₂
  obtain ⟨c_nonneg, c_lt⟩ := Hc
  lift c to ℕ using c_nonneg
  let p' : ℕ × ℕ := ⟨c, mx⟩
  use p'
  constructor; swap
  · rwa [m_eq] at mx_lt_my
  apply Set.mem_image_of_mem
  rw [exceptional_empty, Set.diff_empty]
  constructor <;> dsimp only
  · rw [H_symm, H_quad]
    simpa using h_root
  · suffices hc : c ≠ mx from lt_of_le_of_ne (mod_cast c_lt) hc
    contrapose! hm_B₂
    subst c
    simp [hV₁]

private theorem my_imo1988_q6 {a b : ℕ} (h : a * b + 1 ∣ a ^ 2 + b ^ 2) :
    ∃ d, d ^ 2 = (a ^ 2 + b ^ 2) / (a * b + 1) := by
  rcases h with ⟨k, hk⟩
  rw [hk, Nat.mul_div_cancel_left _ (Nat.succ_pos (a * b))]
  simp only [sq] at hk
  apply my_constant_descent_vieta_jumping a b
      (H := fun a b => a * a + b * b = (a * b + 1) * k)
      hk (fun x => k * x) (fun x => x * x - k) fun _ _ => False <;>
    clear hk a b
  · -- Quadratic equation
    intro x y
    rw [← Int.natCast_inj, ← sub_eq_zero]
    apply eq_iff_eq_cancel_right.2
    simp; ring
  · -- Symmetry
    intro x y
    simp [add_comm (x * x), mul_comm x]
  · -- b = 0
    suffices ∀ a, a * a = k → ∃ d, d * d = k by simpa
    rintro x rfl; use x
  · -- a = b
    intro x hx
    suffices k ≤ 1 by
      rw [Nat.le_add_one_iff, Nat.le_zero] at this
      rcases this with (rfl | rfl)
      · use 0; simp
      · use 1; simp
    contrapose! hx with k_lt_one
    apply ne_of_lt
    calc
      x * x + x * x = x * x * 2 := by rw [mul_two]
      _ ≤ x * x * k := Nat.mul_le_mul_left (x * x) k_lt_one
      _ < (x * x + 1) * k := by linarith
  · -- Descent
    intro x y hx x_lt_y _ _ z h_root _ hV₀
    constructor
    · have hpos : z * z + x * x > 0 := by
        apply add_pos_of_nonneg_of_pos
        · apply mul_self_nonneg
        · apply mul_pos <;> exact mod_cast hx
      have hzx : z * z + x * x = (z * x + 1) * k := by
        rw [← sub_eq_zero, ← h_root]
        ring
      rw [hzx] at hpos
      replace hpos : z * x + 1 > 0 := pos_of_mul_pos_left hpos (Int.ofNat_zero_le k)
      replace hpos : z * x ≥ 0 := Int.le_of_lt_add_one hpos
      apply nonneg_of_mul_nonneg_left hpos (mod_cast hx)
    · contrapose! hV₀ with x_lt_z
      apply ne_of_gt
      calc
        z * y > x * x := by apply mul_lt_mul' <;> omega
        _ ≥ x * x - k := sub_le_self _ (Int.ofNat_zero_le k)
  · -- No base case
    simp

-- Wrapper: convert from ℕ division to ℝ division for MiniF2F target
theorem imo_1988_p6 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a * b + 1 ∣ a ^ 2 + b ^ 2) :
    ∃ x : ℕ, (x ^ 2 : ℝ) = (a ^ 2 + b ^ 2) / (a * b + 1) := by
  obtain ⟨d, hd⟩ := my_imo1988_q6 h₁
  exact ⟨d, by
    have hab_pos : (0 : ℝ) < a * b + 1 := by positivity
    rw [eq_div_iff (ne_of_gt hab_pos)]
    push_cast [hd]
    ring⟩
