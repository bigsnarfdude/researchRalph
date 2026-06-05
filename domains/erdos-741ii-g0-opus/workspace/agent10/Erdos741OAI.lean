import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  -- CONSTRUCTION (gappy multi-scale blocks):
  --   A = ⋃_{k≥0} [2·3^k, 4·3^k]   (B_0=[2,4], B_1=[6,12], B_2=[18,36], …)
  -- Why NOT a cofinite/dense A (e.g. {0}∪{n|4≤n}): such A is a basis but the
  -- parity coloring A₁=A∩evens, A₂=A∩odds makes BOTH sumsets ⊆ evens and
  -- cofinite-in-evens, hence both syndetic — part 2 is FALSE for any A that
  -- contains a full-density residue structure. The basis property must instead
  -- come from CROSS-scale sums so that within any single color the sumset has
  -- unbounded gaps (the genuine content of Erdős #741(ii)).
  -- Coverage: within-block [4·3^k,8·3^k] ∪ cross [8·3^k,16·3^k] ∪
  --           within-B_{k+1} [12·3^k,24·3^k] = [4·3^k, 24·3^k], and these bands
  --           overlap across k, tiling [4,∞).
  refine ⟨⋃ k : ℕ, Set.Icc (2 * 3 ^ k) (4 * 3 ^ k), ?_, ?_⟩
  · -- PART 1 (basis): every n ≥ 4 decomposes via within-block or cross-block sums.
    intro n hn
    have hm : n / 4 ≠ 0 := by omega
    set L := Nat.log 3 (n / 4) with hLdef
    have h1 : 3 ^ L ≤ n / 4 := Nat.pow_log_le_self 3 hm
    have h2 : n / 4 < 3 ^ (L + 1) := Nat.lt_pow_succ_log_self (by norm_num) _
    have hpow : (3 : ℕ) ^ (L + 1) = 3 * 3 ^ L := by rw [pow_succ]; ring
    rw [hpow] at h2
    set q := 3 ^ L with hqdef
    have hq1 : 1 ≤ q := Nat.one_le_pow _ _ (by norm_num)
    have hlow : 4 * q ≤ n := by omega
    have hhigh : n < 12 * q := by omega
    by_cases hc : n ≤ 8 * q
    · -- within block L: a = (n+1)/2, b = n/2, both in [2q, 4q]
      refine ⟨(n + 1) / 2, ?_, n / 2, ?_, ?_⟩
      · refine Set.mem_iUnion.mpr ⟨L, ?_⟩
        simp only [Set.mem_Icc]; rw [← hqdef]; omega
      · refine Set.mem_iUnion.mpr ⟨L, ?_⟩
        simp only [Set.mem_Icc]; rw [← hqdef]; omega
      · omega
    · -- cross block L, L+1: a = 2q ∈ [2q,4q], b = n - 2q ∈ [6q,12q]
      push_neg at hc
      refine ⟨2 * q, ?_, n - 2 * q, ?_, ?_⟩
      · refine Set.mem_iUnion.mpr ⟨L, ?_⟩
        simp only [Set.mem_Icc]; rw [← hqdef]; omega
      · refine Set.mem_iUnion.mpr ⟨L + 1, ?_⟩
        simp only [Set.mem_Icc, hpow]; omega
      · omega
  · -- PART 2 (partition / all colorings) — OPEN. This construction is INSUFFICIENT.
    -- OBSTRUCTION (parity coloring): with A₁ = A∩evens, A₂ = A∩odds, both sumsets
    -- lie in the evens (even+even and odd+odd are both even). Each block
    -- [2·3^k,4·3^k] contains both parities at gap 2, so A₁+A₁ and A₂+A₂ each cover
    -- ALL evens ≥ 4 with gap 2 — BOTH syndetic. Hence part 2 is FALSE for this A.
    -- The same parity attack kills ANY A whose blocks are parity-dense intervals
    -- (incl. the cofinite A). A correct construction must make one residue class
    -- SPARSE so its self-sumset is non-syndetic, while routing the basis property
    -- through cross-class sums — the genuine hard content of Erdős #741(ii).
    -- See LEARNINGS.md / MISTAKES.md for the full analysis.
    sorry

end Erdos741OAI
