import Mathlib
set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat Finset

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  by_contra h; push_neg at h
  have hf_shift : ∀ n, f (n + 1987) = f n + 1987 := by
    intro n; have h1 := congr_arg f (h n); rw [h (f n)] at h1; omega
  have hf_mul : ∀ n k, f (n + 1987 * k) = f n + 1987 * k := by
    intro n k; induction k with
    | zero => simp
    | succ k ih =>
      rw [show n + 1987 * (k + 1) = (n + 1987 * k) + 1987 by ring, hf_shift, ih]; ring
  have hf_no_fix : ∀ n, n < 1987 → f n % 1987 ≠ n := by
    intro n hn hfix
    have hdm := Nat.div_add_mod (f n) 1987; rw [hfix] at hdm
    have : f (f n) = f n + 1987 * (f n / 1987) := by
      conv_lhs => rw [show f n = n + 1987 * (f n / 1987) by omega]
      exact hf_mul n (f n / 1987)
    rw [h] at this; omega
  let σ : Fin 1987 → Fin 1987 := fun ⟨n, _⟩ => ⟨f n % 1987, Nat.mod_lt _ (by omega)⟩
  have hinv : ∀ x : Fin 1987, σ (σ x) = x := by
    intro ⟨n, hn⟩; ext; simp only [σ, Fin.val_mk]
    have : f (f n) = f (f n % 1987) + 1987 * (f n / 1987) := by
      conv_lhs => rw [show f n = f n % 1987 + 1987 * (f n / 1987) by
        have := Nat.div_add_mod (f n) 1987; omega]
      exact hf_mul _ _
    rw [h] at this; omega
  have hfpf : ∀ x : Fin 1987, σ x ≠ x := by
    intro ⟨n, hn⟩ heq; exact hf_no_fix n hn (Fin.mk.inj heq)
  have σ_inj : Function.Injective σ := by
    intro a b hab; have := congr_arg σ hab; rwa [hinv, hinv] at this
  -- x < σ(x) or σ(x) < x (no fixed points)
  have key : ∀ x : Fin 1987, x < σ x ∨ σ x < x := by
    intro x; rcases Ne.lt_or_lt (hfpf x) with h | h
    · right; exact h
    · left; exact h
  let S := univ.filter (fun x : Fin 1987 => x < σ x)
  -- σ maps S to its complement bijectively
  have hS_image : S.image σ = univ.filter (fun x : Fin 1987 => σ x < x) := by
    ext y; simp only [mem_image, mem_filter, mem_univ, true_and, S]
    constructor
    · rintro ⟨x, hx, rfl⟩; rwa [hinv]
    · intro hy; exact ⟨σ y, by rwa [hinv], by rw [hinv]⟩
  have hS_disj : Disjoint S (S.image σ) := by
    rw [hS_image, Finset.disjoint_filter]
    intro x _ h1 h2; exact absurd (lt_trans h2 h1) (lt_irrefl _)
  have hS_union : S ∪ S.image σ = univ := by
    rw [hS_image]; ext x
    simp only [mem_union, mem_filter, mem_univ, true_and, S]
    exact ⟨fun _ => trivial, fun _ => key x⟩
  have : 2 * S.card = 1987 := by
    have hcard_img : (S.image σ).card = S.card := card_image_of_injective S σ_inj
    have := card_union_of_disjoint hS_disj
    rw [hS_union, Finset.card_univ, Fintype.card_fin] at this
    omega
  omega
