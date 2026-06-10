import Mathlib

set_option maxHeartbeats 6400000

open Nat List

private def zeckShift (n : ℕ) : ℕ :=
  (n.zeckendorf.map (fun k => Nat.fib (k + 1))).sum

private lemma isZeckendorfRep_map_succ {l : List ℕ} (hl : l.IsZeckendorfRep) :
    (l.map (· + 1)).IsZeckendorfRep := by
  unfold IsZeckendorfRep at *
  induction l with
  | nil => simpa
  | cons a l ih =>
    specialize ih hl.tail
    simp only [map_cons, cons_append] at *
    apply ih.cons'
    intro y hy
    cases l with
    | nil =>
      simp only [map_nil, nil_append, head?_cons, Option.mem_some_iff] at hy
      have h0 := hl.rel_head? (show (0 : ℕ) ∈ head? ([0] : List ℕ) from by simp)
      omega
    | cons b l' =>
      simp only [map_cons, cons_append, head?_cons, Option.mem_some_iff] at hy
      have hb := hl.rel_head? (show b ∈ head? (b :: l' ++ [0]) from by simp)
      omega

private lemma zeckendorf_zeckShift (n : ℕ) :
    (zeckShift n).zeckendorf = n.zeckendorf.map (· + 1) := by
  unfold zeckShift
  have heq : (map (fun k => fib (k + 1)) (zeckendorf n)) =
      (map fib (map (· + 1) (zeckendorf n))) := by
    rw [List.map_map]; rfl
  rw [heq]
  exact zeckendorf_sum_fib (isZeckendorfRep_map_succ (isZeckendorfRep_zeckendorf n))

private lemma list_sum_fib_add_two (l : List ℕ) :
    (l.map (fun k => fib (k + 2))).sum =
    (l.map (fun k => fib (k + 1))).sum + (l.map fib).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [map_cons, sum_cons]; rw [ih, fib_add_two]; omega

private lemma zeckShift_zeckShift (n : ℕ) : zeckShift (zeckShift n) = zeckShift n + n := by
  show (map (fun k => fib (k + 1)) (zeckShift n).zeckendorf).sum = zeckShift n + n
  rw [zeckendorf_zeckShift, List.map_map]
  change (map (fun k => fib (k + 1 + 1)) (zeckendorf n)).sum =
    (map (fun k => fib (k + 1)) (zeckendorf n)).sum + n
  have : (fun k => fib (k + 1 + 1)) = (fun k => fib (k + 2)) := by ext k; ring_nf
  rw [this, list_sum_fib_add_two, sum_zeckendorf_fib]

private lemma zeckShift_one : zeckShift 1 = 2 := by native_decide

private lemma zeckShift_injective : Function.Injective zeckShift := by
  intro a b hab
  have ha := zeckShift_zeckShift a
  have hb := zeckShift_zeckShift b
  rw [hab] at ha; omega

-- Key monotonicity lemma on Zeckendorf representations
private lemma zeckendorf_shift_mono :
    ∀ (l₁ l₂ : List ℕ), l₁.IsZeckendorfRep → l₂.IsZeckendorfRep →
    (l₁.map fib).sum < (l₂.map fib).sum →
    (l₁.map (fun k => fib (k + 1))).sum < (l₂.map (fun k => fib (k + 1))).sum := by
  intro l₁ l₂ h₁ h₂ hlt
  induction l₁ generalizing l₂ with
  | nil =>
    simp only [map_nil, sum_nil] at hlt ⊢
    cases l₂ with
    | nil => simp at hlt
    | cons a l =>
      simp only [map_cons, sum_cons]
      have : 0 < fib (a + 1) := fib_pos.2 (by omega)
      omega
  | cons a₁ l₁ ih₁ =>
    cases l₂ with
    | nil => simp at hlt
    | cons a₂ l₂ =>
      simp only [map_cons, sum_cons] at hlt ⊢
      by_cases heq : a₁ = a₂
      · subst heq
        have h₁' : l₁.IsZeckendorfRep := by
          unfold IsZeckendorfRep at h₁; exact h₁.tail
        have h₂' : l₂.IsZeckendorfRep := by
          unfold IsZeckendorfRep at h₂; exact h₂.tail
        have := ih₁ l₂ h₁' h₂' (by omega)
        omega
      · by_cases hlt_idx : a₁ < a₂
        · -- a₁ < a₂: use the Zeckendorf upper bound
          have shifted₁ := isZeckendorfRep_map_succ h₁
          have hbound : fib (a₁ + 1) + ((l₁.map (fun k => fib (k + 1))).sum) < fib (a₁ + 2) := by
            -- The shifted rep (a₁+1) :: l₁.map (·+1) has sum < fib((a₁+1)+1)
            have key := shifted₁.sum_fib_lt (n := a₁ + 2) (by
              simp only [map_cons, cons_append, head?_cons, Option.mem_some_iff]
              intro y hy; subst hy; omega)
            simp only [map_cons, List.map_map, Function.comp, sum_cons] at key
            convert key using 2
          have hfib_le : fib (a₁ + 2) ≤ fib (a₂ + 1) := fib_mono (by omega)
          omega
        · -- a₁ > a₂: contradicts hlt via Zeckendorf upper bound
          push_neg at heq hlt_idx
          have hgt : a₂ < a₁ := lt_of_le_of_ne (by omega) (Ne.symm heq)
          exfalso
          have hbound₂ := h₂.sum_fib_lt (n := a₂ + 1) (by
            simp only [map_cons, cons_append, head?_cons, Option.mem_some_iff]
            intro y hy; subst hy; omega)
          simp only [map_cons, sum_cons] at hbound₂
          have hfib_le : fib (a₂ + 1) ≤ fib a₁ := fib_mono (by omega)
          omega

private lemma zeckShift_strictMono : StrictMono zeckShift := by
  intro a b hab
  show zeckShift a < zeckShift b
  unfold zeckShift
  exact zeckendorf_shift_mono _ _ (isZeckendorfRep_zeckendorf a) (isZeckendorfRep_zeckendorf b)
    (by simp only [sum_zeckendorf_fib]; exact hab)

theorem imo_1993_p5 : ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) :=
  ⟨zeckShift, zeckShift_one, fun n => ⟨zeckShift_zeckShift n,
    fun m => zeckShift_strictMono (lt_add_one m)⟩⟩
