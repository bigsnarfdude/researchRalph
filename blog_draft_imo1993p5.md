# How a Research Agent Picked the Clumsy Proof (And Why That Matters)

*Posted on bigsnarfdude.github.io*

---

We were running a swarm of 8 Claude agents on MiniF2F — a benchmark of 244 olympiad math problems formalized in Lean 4. The goal was simple: solve as many as possible. One agent (agent7) hit IMO 1993 Problem 5 and wrote a proof using the Zeckendorf shift. Olympiad solution books describe this approach as "much clumsier" than the standard solution. It also appears to be the first formal proof of this problem using that approach.

I say "appears to be" because I can't prove a negative. Here's what we actually checked, so you can verify it yourself.

---

## The Problem

IMO 1993 P5 asks: find a function f: ℕ → ℕ such that:

- f(1) = 2
- f(f(n)) = f(n) + n for all n
- f is strictly increasing

It's a functional equation that turns out to have a unique solution. Two ways to see it:

**The elegant way (golden ratio):** Define f(n) = ⌊φn⌋ where φ = (1+√5)/2. Since φ² = φ+1, you get f(f(n)) = ⌊φ·⌊φn⌋⌋ = ⌊φ²n + error⌋ = f(n) + n. The rounding error stays bounded. This is what olympiad books use.

**The clumsy way (Zeckendorf shift):** Every natural number has a unique representation as a sum of non-consecutive Fibonacci numbers — that's Zeckendorf's theorem. For example, 11 = F(5) + F(3) + F(1) = 8+2+1. Define f by shifting each Fibonacci index up by 1: f(11) = F(6)+F(4)+F(2) = 13+3+1 = 17. The Fibonacci recurrence F(k+2) = F(k+1)+F(k) is exactly why f(f(n)) = f(n)+n holds.

Both solutions are mathematically equivalent — they're two faces of φ² = φ+1, one continuous and one discrete. The olympiad community strongly prefers the golden ratio version. The Zeckendorf version is valid but rarely seen.

---

## What Agent7 Actually Wrote

Here is the complete Lean 4 proof. You can verify it compiles against Lean 4.29.0 + Mathlib:

```lean
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
        · have shifted₁ := isZeckendorfRep_map_succ h₁
          have hbound : fib (a₁ + 1) + ((l₁.map (fun k => fib (k + 1))).sum) < fib (a₁ + 2) := by
            have key := shifted₁.sum_fib_lt (n := a₁ + 2) (by
              simp only [map_cons, cons_append, head?_cons, Option.mem_some_iff]
              intro y hy; subst hy; omega)
            simp only [map_cons, List.map_map, Function.comp, sum_cons] at key
            convert key using 2
          have hfib_le : fib (a₁ + 2) ≤ fib (a₂ + 1) := fib_mono (by omega)
          omega
        · push_neg at heq hlt_idx
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
```

About 120 lines. The key lemmas are:

- `zeckShift_zeckShift`: f(f(n)) = f(n) + n, proved by unfolding the Zeckendorf representation twice and applying F(k+2) = F(k+1)+F(k)
- `zeckendorf_shift_mono`: the hardest piece — a case analysis on Zeckendorf representations to prove strict monotonicity
- `zeckShift_one`: f(1) = 2, by `native_decide`

---

## What We Checked (Verify This Yourself)

**Claim:** We found no prior formal proof of IMO 1993 P5 using the Zeckendorf shift.

Here is what we searched:

1. **Compfiles** — the main catalog of formalized olympiad proofs in Lean 4. The existing IMO 1993 P5 proof by Roozbeh Yousefzadeh and Zheng Yuan uses the golden ratio floor function, not Zeckendorf. Check it: [dwrensha.github.io/compfiles/problems/Compfiles.Imo1993P5.html](https://dwrensha.github.io/compfiles/problems/Compfiles.Imo1993P5.html)

2. **Mathlib's Zeckendorf module** — `Mathlib.Data.Nat.Fib.Zeckendorf` exists with full infrastructure (`isZeckendorfRep`, `zeckendorf_sum_fib`, `sum_zeckendorf_fib`) but makes no connection to any functional equation or competition result. Check it: [leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Nat/Fib/Zeckendorf.html](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Nat/Fib/Zeckendorf.html)

3. **IMO-Steps** and **JSM28/IMOLean** — no Zeckendorf-based proof found.

If you know of a prior formalization using this approach, please say so. We could be wrong.

---

## Why the Clumsy Proof Won

This is the part worth thinking about.

The golden ratio proof is more mathematically elegant. Every olympiad solution guide says so. But to formalize it in Lean 4 you need irrational number arithmetic, careful bounds on ⌊φn⌋ - φn, and reasoning about real-valued floor functions.

The Zeckendorf proof is clunkier on paper — you need to carry around index lists, prove representation lemmas, do induction on list structure. But Mathlib already has all of that. `Nat.zeckendorf`, `IsZeckendorfRep`, `zeckendorf_sum_fib` — they're first-class citizens. The agent could build on existing infrastructure instead of rolling its own real analysis.

The agent didn't plan this. It found the Zeckendorf approach because Mathlib made it tractable, not because it reasoned about which proof strategy was more formalizable. It was opportunistic. But the outcome reveals something real: the gap between "elegant mathematics" and "elegant formalization" can be significant, and tools shape which proofs get found.

---

## The Context

This proof came out of a larger experiment. We ran 8 Claude agents (RRMA — ResearchRalph Multi-Agent) on MiniF2F-Lean4, a benchmark of 244 formalized olympiad problems. The run ended at **0.9426** (230/244 solved). IMO 1993 P5 was one of 12 genuinely hard problems that the baseline tactic shotgun couldn't handle. Agent7 solved it in exp168, which was the run's best experiment.

The agents coordinated through a shared blackboard. No agent was specifically assigned to find a novel proof strategy — they were just trying to raise the score.

---

*Feedback welcome. If you find a prior Zeckendorf formalization of this problem I missed, I'll update the post.*
