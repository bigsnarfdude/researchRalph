import Mathlib

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

open scoped Pointwise
open Set

namespace Erdos741Work

open scoped BigOperators Classical

private def IsSyndeticW (S : Set ℕ) : Prop :=
  ∃ p : ℕ, ∀ x : ℕ, ∃ y, y ∈ S ∧ y ∈ Icc x (x + p)

private def IsAddBasisOfOrderW (S : Set ℕ) (h : ℕ) : Prop :=
  ∀ n : ℕ, n ∈ h • S

private def HasLargeGapsW (S : Set ℕ) : Prop :=
  ∀ C : ℕ, ∃ N : ℕ, ∀ x, N ≤ x → x ≤ N + C → x ∉ S

private lemma not_syndetic_of_large_gapsW (S : Set ℕ) :
    (∀ k : ℕ, ∃ x : ℕ, Icc x (x + k) ⊆ Sᶜ) → ¬IsSyndeticW S := by
  intro h_gaps h_syn
  unfold IsSyndeticW at h_syn
  rcases h_syn with ⟨p, hp⟩
  obtain ⟨x, hx⟩ := h_gaps p
  obtain ⟨y, hy_S, hy_Icc⟩ := hp x
  exact hx hy_Icc hy_S

private lemma has_gaps_monoW (S : Set ℕ) (C1 C2 : ℕ) (h : C1 ≤ C2) :
    (∃ N, ∀ x, N ≤ x → x ≤ N + C2 → x ∉ S) →
    (∃ N, ∀ x, N ≤ x → x ≤ N + C1 → x ∉ S) := by
  rintro ⟨N, hN⟩
  exact ⟨N, fun x hx_ge hx_le => hN x hx_ge (hx_le.trans (Nat.add_le_add_left h _))⟩

private lemma infinite_orW (P Q : ℕ → Prop)
    (h_monoP : ∀ c1 c2, c1 ≤ c2 → P c2 → P c1)
    (h_monoQ : ∀ c1 c2, c1 ≤ c2 → Q c2 → Q c1)
    (h_or : ∀ c, P c ∨ Q c) :
    (∀ c, P c) ∨ (∀ c, Q c) := by
  by_cases hP : ∀ c, P c
  · left; exact hP
  · right
    push_neg at hP
    rcases hP with ⟨c0, hc0⟩
    intro c
    by_cases h_le : c ≤ c0
    · cases h_or c0 with
      | inl hP_c0 => contradiction
      | inr hQ_c0 => exact h_monoQ c c0 h_le hQ_c0
    · push_neg at h_le
      cases h_or c with
      | inl hP_c => exact absurd (h_monoP c0 c (le_of_lt h_le) hP_c) hc0
      | inr hQ_c => exact hQ_c

private def StateW := { p : Set ℕ × ℕ // ∀ x ∈ p.1, x ≤ p.2 }

private def step_propW (prev : StateW) (C : ℕ) (next : StateW) : Prop :=
  prev.val.1 ⊆ next.val.1 ∧
  prev.val.2 ≤ next.val.2 ∧
  (∀ x ∈ next.val.1 \ prev.val.1, x > prev.val.2) ∧
  (∃ N, N + C ≤ next.val.2 ∧ ∀ F₁ F₂, next.val.1 = F₁ ∪ F₂ → Disjoint F₁ F₂ →
    (∀ x, N ≤ x → x ≤ N + C → x ∉ F₁ + F₁) ∨ (∀ x, N ≤ x → x ≤ N + C → x ∉ F₂ + F₂)) ∧
  ((∀ n ≤ prev.val.2, ∃ a b, a ∈ prev.val.1 ∪ {0} ∧ b ∈ prev.val.1 ∪ {0} ∧ a + b = n) →
   (∀ n ≤ next.val.2, ∃ a b, a ∈ next.val.1 ∪ {0} ∧ b ∈ next.val.1 ∪ {0} ∧ a + b = n))

private lemma valid_ext_existsW (prev : StateW) (C : ℕ) : ∃ next, step_propW prev C next := by
  let G := prev.val.2
  let M := 2 * G + C + 1
  let W := 3 * G + 2 * C + 2
  let next_f := prev.val.1 ∪ Icc (G + 1) M ∪ {W}
  let next_gap := W + G + C + 1
  have h_bound : ∀ x ∈ next_f, x ≤ next_gap := by
    intro x hx
    simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hx
    rcases hx with (hx_prev | hx_icc) | hx_W
    · have hx_le := prev.property x hx_prev
      have hG : prev.val.2 = G := rfl
      have hGap : next_gap = W + G + C + 1 := rfl
      omega
    · have hM : M = 2 * G + C + 1 := rfl
      have hGap : next_gap = W + G + C + 1 := rfl
      omega
    · have hW : W = 3 * G + 2 * C + 2 := rfl
      have hGap : next_gap = W + G + C + 1 := rfl
      omega
  let next_state : StateW := ⟨(next_f, next_gap), h_bound⟩
  use next_state
  unfold step_propW
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro x hx
    simp only [next_state, next_f, mem_union, mem_Icc, mem_singleton_iff]
    left; left; exact hx
  · simp only [next_state, next_gap]
    have hG : prev.val.2 = G := rfl
    have hGap : next_gap = W + G + C + 1 := rfl
    omega
  · intro x hx
    simp only [next_state, next_f, mem_union, mem_Icc, mem_singleton_iff, mem_diff] at hx
    rcases hx with ⟨(hx_prev | hx_icc) | hx_W, hx_not⟩
    · contradiction
    · have hG : prev.val.2 = G := rfl
      have hM : M = 2 * G + C + 1 := rfl
      omega
    · have hG : prev.val.2 = G := rfl
      have hW : W = 3 * G + 2 * C + 2 := rfl
      omega
  · use W + G + 1
    refine ⟨?_, ?_⟩
    · have hGap : next_state.val.2 = W + G + C + 1 := rfl
      omega
    · intros F₁ F₂ h_union h_disj
      have h_union' : next_f = F₁ ∪ F₂ := h_union
      by_cases hW : W ∈ F₁
      · right
        intros x hx_ge hx_le hx_sum
        rcases hx_sum with ⟨a, ha, b, hb, hab⟩
        change a + b = x at hab
        have hW_not_F2 : W ∉ F₂ := by
          intro h
          have h_inter : W ∈ F₁ ∩ F₂ := ⟨hW, h⟩
          have h_empty : F₁ ∩ F₂ ⊆ ∅ := Set.disjoint_iff.mp h_disj
          exact h_empty h_inter
        have ha_next : a ∈ next_f := by
          have h_sub : F₂ ⊆ next_f := by rw [h_union']; exact Set.subset_union_right
          exact h_sub ha
        have hb_next : b ∈ next_f := by
          have h_sub : F₂ ⊆ next_f := by rw [h_union']; exact Set.subset_union_right
          exact h_sub hb
        have ha_le_M : a ≤ M := by
          simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at ha_next
          rcases ha_next with (ha_prev | ha_icc) | ha_W
          · have ha_le_G := prev.property a ha_prev
            have hG : prev.val.2 = G := rfl
            have hM : M = 2 * G + C + 1 := rfl
            omega
          · exact ha_icc.2
          · exfalso; apply hW_not_F2; rw [ha_W] at ha; exact ha
        have hb_le_M : b ≤ M := by
          simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hb_next
          rcases hb_next with (hb_prev | hb_icc) | hb_W
          · have hb_le_G := prev.property b hb_prev
            have hG : prev.val.2 = G := rfl
            have hM : M = 2 * G + C + 1 := rfl
            omega
          · exact hb_icc.2
          · exfalso; apply hW_not_F2; rw [hb_W] at hb; exact hb
        have hM : M = 2 * G + C + 1 := rfl
        have hW_def : W = 3 * G + 2 * C + 2 := rfl
        omega
      · left
        intros x hx_ge hx_le hx_sum
        rcases hx_sum with ⟨a, ha, b, hb, hab⟩
        change a + b = x at hab
        have ha_next : a ∈ next_f := by
          have h_sub : F₁ ⊆ next_f := by rw [h_union']; exact Set.subset_union_left
          exact h_sub ha
        have hb_next : b ∈ next_f := by
          have h_sub : F₁ ⊆ next_f := by rw [h_union']; exact Set.subset_union_left
          exact h_sub hb
        have ha_le_M : a ≤ M := by
          simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at ha_next
          rcases ha_next with (ha_prev | ha_icc) | ha_W
          · have ha_le_G := prev.property a ha_prev
            have hG : prev.val.2 = G := rfl
            have hM : M = 2 * G + C + 1 := rfl
            omega
          · exact ha_icc.2
          · exfalso; apply hW; rw [ha_W] at ha; exact ha
        have hb_le_M : b ≤ M := by
          simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hb_next
          rcases hb_next with (hb_prev | hb_icc) | hb_W
          · have hb_le_G := prev.property b hb_prev
            have hG : prev.val.2 = G := rfl
            have hM : M = 2 * G + C + 1 := rfl
            omega
          · exact hb_icc.2
          · exfalso; apply hW; rw [hb_W] at hb; exact hb
        have hM : M = 2 * G + C + 1 := rfl
        have hW_def : W = 3 * G + 2 * C + 2 := rfl
        omega
  · intro h_prev_cov n hn
    have hG : prev.val.2 = G := rfl
    have hM : M = 2 * G + C + 1 := rfl
    have hW : W = 3 * G + 2 * C + 2 := rfl
    have hGap : next_gap = W + G + C + 1 := rfl
    change n ≤ W + G + C + 1 at hn
    by_cases h1 : n ≤ G
    · have h_cov := h_prev_cov n h1
      rcases h_cov with ⟨a, b, ha, hb, hab⟩
      use a, b
      constructor
      · rcases ha with ha_prev | ha_0
        · left; left; left; exact ha_prev
        · right; exact ha_0
      · constructor
        · rcases hb with hb_prev | hb_0
          · left; left; left; exact hb_prev
          · right; exact hb_0
        · exact hab
    · push_neg at h1
      by_cases h2 : n ≤ M
      · use n, 0
        constructor
        · left; left; right; exact ⟨by omega, h2⟩
        · constructor
          · right; exact Set.mem_singleton 0
          · omega
      · push_neg at h2
        by_cases h3 : n ≤ 2 * M
        · let a := n / 2
          let b := n - n / 2
          use a, b
          constructor
          · left; left; right
            have ha_ge : G + 1 ≤ a := by omega
            have ha_le : a ≤ M := by omega
            exact ⟨ha_ge, ha_le⟩
          · constructor
            · left; left; right
              have hb_ge : G + 1 ≤ b := by omega
              have hb_le : b ≤ M := by omega
              exact ⟨hb_ge, hb_le⟩
            · omega
        · push_neg at h3
          let c := n - W
          use W, c
          constructor
          · left; right; exact Set.mem_singleton W
          · constructor
            · left; left; right
              have hc_ge : G + 1 ≤ c := by omega
              have hc_le : c ≤ M := by omega
              exact ⟨hc_ge, hc_le⟩
            · omega

noncomputable def seq_stepW : ℕ → StateW
  | 0 => ⟨(∅, 0), by intro x hx; contradiction⟩
  | n + 1 => Classical.choose (valid_ext_existsW (seq_stepW n) n)

noncomputable def f_seqW (n : ℕ) : Set ℕ := (seq_stepW n).val.1
noncomputable def gap_seqW (n : ℕ) : ℕ := (seq_stepW n).val.2

private lemma seq_step_propW (n : ℕ) : step_propW (seq_stepW n) n (seq_stepW (n + 1)) :=
  Classical.choose_spec (valid_ext_existsW (seq_stepW n) n)

private lemma f_seq_coversW (n : ℕ) :
    ∀ m ≤ gap_seqW n, ∃ a b, a ∈ f_seqW n ∪ {0} ∧ b ∈ f_seqW n ∪ {0} ∧ a + b = m := by
  induction n with
  | zero =>
    intro m hm
    have hm0 : m = 0 := Nat.eq_zero_of_le_zero hm
    use 0, 0
    have h0_in : 0 ∈ f_seqW 0 ∪ {0} := Or.inr (Set.mem_singleton 0)
    exact ⟨h0_in, h0_in, by rw [hm0]⟩
  | succ n ih =>
    have h := seq_step_propW n
    rcases h with ⟨_, _, _, _, h_cov⟩
    intro m hm
    exact h_cov ih m hm

private lemma f_seqW_mono {m k : ℕ} (h : m ≤ k) : f_seqW m ⊆ f_seqW k := by
  induction h with
  | refl => rfl
  | step _ ih => exact ih.trans (seq_step_propW _).1

private lemma gap_seqW_mono {m k : ℕ} (h : m ≤ k) : gap_seqW m ≤ gap_seqW k := by
  induction h with
  | refl => exact le_refl _
  | step _ ih => exact ih.trans (seq_step_propW _).2.1

private lemma subset_f_k_of_leW (k : ℕ) :
    ∀ y ∈ (⋃ n, f_seqW n), y ≤ gap_seqW k → y ∈ f_seqW k := by
  intros y hy hy_le
  have hy_ex : ∃ n, y ∈ f_seqW n := Set.mem_iUnion.mp hy
  by_cases hy_fk : y ∈ f_seqW k
  · exact hy_fk
  · exfalso
    have h_min : ∃ m, y ∈ f_seqW m ∧ ∀ j < m, y ∉ f_seqW j := by
      let P := fun m => y ∈ f_seqW m
      have h_ex : ∃ m, P m := hy_ex
      exact ⟨Nat.find h_ex, Nat.find_spec h_ex, fun j hj => Nat.find_min h_ex hj⟩
    rcases h_min with ⟨m, hm_in, hm_min⟩
    have h_m_gt_k : m > k := by
      by_contra h_not_gt
      push_neg at h_not_gt
      exact hy_fk (f_seqW_mono h_not_gt hm_in)
    have h_m_pos : m > 0 := by linarith
    have h_m_minus_1 : y ∉ f_seqW (m - 1) := hm_min (m - 1) (Nat.pred_lt (ne_of_gt h_m_pos))
    have h_diff : y ∈ f_seqW m \ f_seqW (m - 1) := ⟨hm_in, h_m_minus_1⟩
    have h_gap : y > gap_seqW (m - 1) := by
      have h_m_eq : m = (m - 1) + 1 := by omega
      have h_diff' : y ∈ f_seqW ((m - 1) + 1) \ f_seqW (m - 1) := by rw [← h_m_eq]; exact h_diff
      exact (seq_step_propW (m - 1)).2.2.1 y h_diff'
    have h_gap_mono : gap_seqW k ≤ gap_seqW (m - 1) := gap_seqW_mono (by omega)
    linarith

private lemma erdos_gap_set_existsW : ∃ A : Set ℕ, IsAddBasisOfOrderW (A ∪ {0}) 2 ∧
    ∀ A₁ A₂, A = A₁ ∪ A₂ → Disjoint A₁ A₂ →
      HasLargeGapsW (A₁ + A₁) ∨ HasLargeGapsW (A₂ + A₂) := by
  let A := ⋃ n, f_seqW n
  use A
  constructor
  · intro n
    have h_step := seq_step_propW n
    rcases h_step with ⟨_, _, _, ⟨N, hN_le, _⟩, _⟩
    have hn : n ≤ gap_seqW (n + 1) := le_trans (Nat.le_add_left n N) hN_le
    have h_cov := f_seq_coversW (n + 1) n hn
    rcases h_cov with ⟨a, b, ha, hb, hab⟩
    have hsum : n ∈ (A ∪ {0}) + (A ∪ {0}) := by
      use a
      constructor
      · cases ha with
        | inl ha_f => left; exact Set.mem_iUnion.mpr ⟨n + 1, ha_f⟩
        | inr ha_0 => right; exact ha_0
      · use b
        constructor
        · cases hb with
          | inl hb_f => left; exact Set.mem_iUnion.mpr ⟨n + 1, hb_f⟩
          | inr hb_0 => right; exact hb_0
        · exact hab
    have h_two_smul : (A ∪ {0}) + (A ∪ {0}) = 2 • (A ∪ {0}) := (two_nsmul (A ∪ {0})).symm
    rw [← h_two_smul]; exact hsum
  · intros A₁ A₂ h_part h_disj
    have h_or_C : ∀ C : ℕ,
        (∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₁ + A₁) ∨
        (∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₂ + A₂) := by
      intro C
      have h_step := seq_step_propW C
      rcases h_step with ⟨_, _, _, ⟨N, hN_le, h_gap_k⟩, _⟩
      have hN_le' : N + C ≤ gap_seqW (C + 1) := hN_le
      have h_F_part : f_seqW (C + 1) = (A₁ ∩ f_seqW (C + 1)) ∪ (A₂ ∩ f_seqW (C + 1)) := by
        ext x
        simp only [mem_union, mem_inter_iff]
        constructor
        · intro hx
          have hxA : x ∈ A := Set.mem_iUnion.mpr ⟨C + 1, hx⟩
          have hx_part : x ∈ A₁ ∪ A₂ := by rw [← h_part]; exact hxA
          rcases hx_part with h1 | h2
          · left; exact ⟨h1, hx⟩
          · right; exact ⟨h2, hx⟩
        · rintro (⟨-, hx⟩ | ⟨-, hx⟩) <;> exact hx
      have h_F_disj : Disjoint (A₁ ∩ f_seqW (C + 1)) (A₂ ∩ f_seqW (C + 1)) := by
        rw [Set.disjoint_iff]
        intro x hx
        exact Set.disjoint_iff.mp h_disj ⟨hx.1.1, hx.2.1⟩
      have h_gap_F := h_gap_k (A₁ ∩ f_seqW (C + 1)) (A₂ ∩ f_seqW (C + 1)) h_F_part h_F_disj
      cases h_gap_F with
      | inl h_inl =>
        left; use N
        intros x hx_ge hx_le hx_in
        rcases hx_in with ⟨a, ha, b, hb, hab⟩
        have hab' : a + b = x := hab
        have ha_gap : a ≤ gap_seqW (C + 1) :=
          le_trans (le_trans (Nat.le_add_right a b) hab'.le) (le_trans hx_le hN_le')
        have hb_gap : b ≤ gap_seqW (C + 1) :=
          le_trans (le_trans (Nat.le_add_left b a) hab'.le) (le_trans hx_le hN_le')
        have ha_A : a ∈ A := by rw [h_part]; left; exact ha
        have hb_A : b ∈ A := by rw [h_part]; left; exact hb
        have ha_fk := subset_f_k_of_leW (C + 1) a (Set.mem_iUnion.mpr (Set.mem_iUnion.mp ha_A)) ha_gap
        have hb_fk := subset_f_k_of_leW (C + 1) b (Set.mem_iUnion.mpr (Set.mem_iUnion.mp hb_A)) hb_gap
        exact h_inl x hx_ge hx_le ⟨a, ⟨ha, ha_fk⟩, b, ⟨hb, hb_fk⟩, hab⟩
      | inr h_inr =>
        right; use N
        intros x hx_ge hx_le hx_in
        rcases hx_in with ⟨a, ha, b, hb, hab⟩
        have hab' : a + b = x := hab
        have ha_gap : a ≤ gap_seqW (C + 1) :=
          le_trans (le_trans (Nat.le_add_right a b) hab'.le) (le_trans hx_le hN_le')
        have hb_gap : b ≤ gap_seqW (C + 1) :=
          le_trans (le_trans (Nat.le_add_left b a) hab'.le) (le_trans hx_le hN_le')
        have ha_A : a ∈ A := by rw [h_part]; right; exact ha
        have hb_A : b ∈ A := by rw [h_part]; right; exact hb
        have ha_fk := subset_f_k_of_leW (C + 1) a (Set.mem_iUnion.mpr (Set.mem_iUnion.mp ha_A)) ha_gap
        have hb_fk := subset_f_k_of_leW (C + 1) b (Set.mem_iUnion.mpr (Set.mem_iUnion.mp hb_A)) hb_gap
        exact h_inr x hx_ge hx_le ⟨a, ⟨ha, ha_fk⟩, b, ⟨hb, hb_fk⟩, hab⟩
    let P := fun C => ∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₁ + A₁
    let Q := fun C => ∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₂ + A₂
    cases infinite_orW P Q
      (fun c1 c2 hc => has_gaps_monoW (A₁ + A₁) c1 c2 hc)
      (fun c1 c2 hc => has_gaps_monoW (A₂ + A₂) c1 c2 hc)
      h_or_C with
    | inl h => left; intro C; exact h C
    | inr h => right; intro C; exact h C

end Erdos741Work

-- Public workspace definitions
def IsSyndetic' (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ n : ℕ, ∃ m ∈ S, n ≤ m ∧ m ≤ n + C

def IsAddBasis2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ (A ∪ {0}), ∃ b ∈ (A ∪ {0}), a + b = n

def sumset' (S : Set ℕ) : Set ℕ := {n | ∃ a ∈ S, ∃ b ∈ S, a + b = n}

open Set Filter

theorem erdos_741_ii :
    ∃ A : Set ℕ,
      IsAddBasis2 A ∧
      ∀ A₁ A₂ : Set ℕ, A = A₁ ∪ A₂ → Disjoint A₁ A₂ →
        ¬(IsSyndetic' (sumset' A₁) ∧ IsSyndetic' (sumset' A₂)) := by
  open Erdos741Work in
  obtain ⟨A, h_basis, h_gaps⟩ := erdos_gap_set_existsW
  use A
  refine ⟨?_, ?_⟩
  · -- Bridge: IsAddBasisOfOrderW (A ∪ {0}) 2 → IsAddBasis2 A
    intro n
    have hn := h_basis n
    rw [two_nsmul] at hn
    rw [Set.mem_add] at hn
    exact hn
  · intros A₁ A₂ h_union h_disj
    rintro ⟨h_syn1, h_syn2⟩
    -- Bridge: sumset' Aᵢ = Aᵢ + Aᵢ
    have h_s1 : A₁ + A₁ = sumset' A₁ := by
      ext n; simp [sumset', Set.mem_add]
    have h_s2 : A₂ + A₂ = sumset' A₂ := by
      ext n; simp [sumset', Set.mem_add]
    -- Bridge: IsSyndetic' → HasLargeGapsW fails
    cases h_gaps A₁ A₂ h_union h_disj with
    | inl h1 =>
      -- h1 : HasLargeGapsW (A₁ + A₁)
      -- h_syn1 : IsSyndetic' (sumset' A₁)
      rw [← h_s1] at h_syn1
      -- h_syn1 : IsSyndetic' (A₁ + A₁)
      obtain ⟨p, hp⟩ := h_syn1
      obtain ⟨N, hN⟩ := h1 p
      obtain ⟨m, hm_mem, hm_ge, hm_le⟩ := hp N
      exact hN m hm_ge hm_le hm_mem
    | inr h2 =>
      rw [← h_s2] at h_syn2
      obtain ⟨p, hp⟩ := h_syn2
      obtain ⟨N, hN⟩ := h2 p
      obtain ⟨m, hm_mem, hm_ge, hm_le⟩ := hp N
      exact hN m hm_ge hm_le hm_mem
