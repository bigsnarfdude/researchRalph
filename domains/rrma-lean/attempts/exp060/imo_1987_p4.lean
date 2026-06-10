import Mathlib
set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  by_contra h
  push_neg at h
  have hf_shift : ∀ n, f (n + 1987) = f n + 1987 := by
    intro n; have h1 := congr_arg f (h n); rw [h (f n)] at h1; omega
  have hf_mul_shift : ∀ n k, f (n + 1987 * k) = f n + 1987 * k := by
    intro n k; induction k with
    | zero => simp
    | succ k ih => rw [show n + 1987 * (k + 1) = (n + 1987 * k) + 1987 by ring, hf_shift, ih]; ring
  have hf_inj : Function.Injective f := by
    intro a b hab; have := congr_arg f hab; rw [h, h] at this; omega
  have hf_no_fix : ∀ n, n < 1987 → f n % 1987 ≠ n := by
    intro n hn hfix
    have hmod : f n = n + 1987 * (f n / 1987) := by
      have := Nat.div_add_mod (f n) 1987; omega
    have : f (f n) = f n + 1987 * (f n / 1987) := by
      rw [hmod, hf_mul_shift]
    rw [h] at this; omega
  let σ : Fin 1987 → Fin 1987 := fun ⟨n, hn⟩ => ⟨f n % 1987, Nat.mod_lt _ (by omega)⟩
  have hinv : ∀ x : Fin 1987, σ (σ x) = x := by
    intro ⟨n, hn⟩
    simp only [σ]
    ext
    simp only [Fin.val_mk]
    have hmod_lt : f n % 1987 < 1987 := Nat.mod_lt _ (by omega)
    have hk : f n = f n % 1987 + 1987 * (f n / 1987) := by
      have := Nat.div_add_mod (f n) 1987; omega
    have step : f (f n) = f (f n % 1987) + 1987 * (f n / 1987) := by
      conv_lhs => rw [hk]
      exact hf_mul_shift _ _
    rw [h] at step
    omega
  have hfpf : ∀ x : Fin 1987, σ x ≠ x := by
    intro ⟨n, hn⟩ heq
    have : f n % 1987 = n := by
      have := congr_arg Fin.val heq
      simp [σ] at this
      exact this
    exact hf_no_fix n hn this
  -- σ is a fixed-point-free involution on Fin 1987.
  -- Partition Fin 1987 into orbits of size 2. But 1987 is odd → contradiction.
  -- Use: σ² = id and σ has no fixed points → all orbits have size 2 → 2 | 1987.
  have : 2 ∣ 1987 := by
    have key : ∀ x : Fin 1987, x < σ x ∨ σ x < x := by
      intro x
      rcases lt_trichotomy x (σ x) with h | h | h
      · exact Or.inl h
      · exact absurd h (hfpf x ∘ Eq.symm ∘ Fin.val_eq_val_iff.mp ∘ le_antisymm (le_of_eq h) ∘ le_of_eq ∘ Eq.symm)
      · exact Or.inr h
    let S := Finset.univ.filter (fun x : Fin 1987 => x < σ x)
    have hσ_swap : ∀ x ∈ S, σ x ∉ S := by
      intro x hx hsx
      simp only [S, Finset.mem_filter, Finset.mem_univ, true_and] at hx hsx
      have : σ (σ x) = x := by rw [hinv]
      rw [this] at hsx
      exact absurd (lt_trans hsx hx) (lt_irrefl _)
    have hσ_map : ∀ x ∈ S, σ x ∈ Finset.univ \ S := by
      intro x hx
      simp only [Finset.mem_sdiff, Finset.mem_univ, true_and]
      exact hσ_swap x hx
    have hσ_inj_on : Set.InjOn σ ↑S := by
      intro a ha b hb hab
      have := congr_arg σ hab
      rw [hinv, hinv] at this
      exact this
    have hcard_le : S.card ≤ (Finset.univ \ S).card := by
      exact Finset.card_le_card_of_injOn σ hσ_map (fun a ha b hb hab => by
        have := congr_arg σ hab; rw [hinv, hinv] at this; exact this)
    have hcard_ge : (Finset.univ \ S).card ≤ S.card := by
      -- Every element not in S has σ x in S
      have : ∀ x ∈ Finset.univ \ S, σ x ∈ S := by
        intro x hx
        simp only [S, Finset.mem_sdiff, Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at hx
        simp only [S, Finset.mem_filter, Finset.mem_univ, true_and]
        have hle := hx.2
        have hne : σ x ≠ x := hfpf x
        have : σ x < x := lt_of_le_of_ne hle (fun h => hne h.symm)
        rw [hinv]
        exact this
      exact Finset.card_le_card_of_injOn σ this (fun a _ b _ hab => by
        have := congr_arg σ hab; rw [hinv, hinv] at this; exact this)
    have hS_eq : 2 * S.card = Fintype.card (Fin 1987) := by
      have := Finset.card_sdiff_add_card_eq_card (Finset.subset_univ S)
      rw [Fintype.card_fin]
      omega
    rw [Fintype.card_fin] at hS_eq
    exact ⟨S.card, by omega⟩
  omega
