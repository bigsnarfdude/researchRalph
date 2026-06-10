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
    have ⟨k, hk⟩ : ∃ k, f n = n + 1987 * k := ⟨f n / 1987, by omega⟩
    have : f (f n) = f n + 1987 * k := by rw [hk, hf_mul_shift]
    rw [h, hk] at this; omega
  let σ : Fin 1987 → Fin 1987 := fun ⟨n, hn⟩ => ⟨f n % 1987, Nat.mod_lt _ (by omega)⟩
  have hinv : ∀ x : Fin 1987, σ (σ x) = x := by
    intro ⟨n, hn⟩
    simp only [σ, Fin.mk.injEq]
    obtain ⟨k, hk⟩ : ∃ k, f n = f n % 1987 + 1987 * k := ⟨f n / 1987, by omega⟩
    have : f (f n) = f (f n % 1987) + 1987 * k := by rw [hk, hf_mul_shift]
    rw [h] at this; omega
  have hfpf : ∀ x : Fin 1987, σ x ≠ x := by
    intro ⟨n, hn⟩ heq; simp only [σ, Fin.mk.injEq] at heq; exact hf_no_fix n hn heq
  let S := Finset.univ.filter (fun x : Fin 1987 => x < σ x)
  let T := Finset.univ.filter (fun x : Fin 1987 => σ x < x)
  have hST_disj : Disjoint S T := by
    simp only [S, T, Finset.disjoint_filter]
    intro x _ h1 h2; exact absurd (lt_trans h2 h1) (lt_irrefl _)
  have hST_cover : S ∪ T = Finset.univ := by
    ext x; simp only [S, T, Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and]
    constructor
    · intro _; trivial
    · intro _; rcases lt_trichotomy x (σ x) with h | h | h
      · exact Or.inl h
      · exact absurd h.symm (hfpf x)
      · exact Or.inr h
  have hcard : S.card = T.card := by
    apply le_antisymm
    · exact Finset.card_le_card_of_injOn σ
        (by intro x hx; simp only [S, T, Finset.mem_filter, Finset.mem_univ, true_and] at *; rw [hinv]; exact hx)
        (fun a _ b _ h => by have := congr_arg σ h; rwa [hinv, hinv] at this)
    · exact Finset.card_le_card_of_injOn σ
        (by intro x hx; simp only [S, T, Finset.mem_filter, Finset.mem_univ, true_and] at *; rw [hinv]; exact hx)
        (fun a _ b _ h => by have := congr_arg σ h; rwa [hinv, hinv] at this)
  have htotal : S.card + T.card = 1987 := by
    rw [← Finset.card_union_of_disjoint hST_disj, hST_cover, Finset.card_univ, Fintype.card_fin]
  omega
