import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 40000
set_option synthInstance.maxSize 128

open scoped Pointwise BigOperators Classical
open Set

def IsSyndetic' (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ n : ℕ, ∃ m ∈ S, n ≤ m ∧ m ≤ n + C

def IsAddBasis2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ (A ∪ {0}), ∃ b ∈ (A ∪ {0}), a + b = n

def sumset' (S : Set ℕ) : Set ℕ := {n | ∃ a ∈ S, ∃ b ∈ S, a + b = n}

private lemma sumset'_eq (S : Set ℕ) : sumset' S = S + S := by
  ext n
  simp only [sumset', mem_setOf_eq, Set.mem_add]

private def HasLargeGaps (S : Set ℕ) : Prop :=
  ∀ C : ℕ, ∃ N : ℕ, ∀ x, N ≤ x → x ≤ N + C → x ∉ S

private lemma not_syndetic_of_large_gaps' (S : Set ℕ) :
    (∀ k : ℕ, ∃ x : ℕ, Icc x (x + k) ⊆ Sᶜ) → ¬IsSyndetic' S := by
  intro h_gaps ⟨p, hp⟩
  obtain ⟨x, hx⟩ := h_gaps p
  obtain ⟨m, hm_mem, hm_ge, hm_le⟩ := hp x
  exact hx ⟨hm_ge, hm_le⟩ hm_mem

private lemma has_gaps_mono (S : Set ℕ) (C1 C2 : ℕ) (h : C1 ≤ C2) :
    (∃ N, ∀ x, N ≤ x → x ≤ N + C2 → x ∉ S) →
    (∃ N, ∀ x, N ≤ x → x ≤ N + C1 → x ∉ S) := by
  rintro ⟨N, hN⟩
  exact ⟨N, fun x hx_ge hx_le => hN x hx_ge (hx_le.trans (Nat.add_le_add_left h _))⟩

private lemma infinite_or (P Q : ℕ → Prop)
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

private lemma icc_subset_complement {A : Set ℕ} {x k : ℕ}
    (h : ∀ y ∈ Icc x (x + k), y ∉ A) : Icc x (x + k) ⊆ Aᶜ :=
  fun y hy => h y hy

private def State' := { p : Set ℕ × ℕ // ∀ x ∈ p.1, x ≤ p.2 }

private def step_prop' (prev : State') (C : ℕ) (next : State') : Prop :=
  prev.val.1 ⊆ next.val.1 ∧
  prev.val.2 ≤ next.val.2 ∧
  (∀ x ∈ next.val.1 \ prev.val.1, x > prev.val.2) ∧
  (∃ N, N + C ≤ next.val.2 ∧ ∀ F₁ F₂, next.val.1 = F₁ ∪ F₂ → Disjoint F₁ F₂ →
    (∀ x, N ≤ x → x ≤ N + C → x ∉ F₁ + F₁) ∨ (∀ x, N ≤ x → x ≤ N + C → x ∉ F₂ + F₂)) ∧
  ((∀ n ≤ prev.val.2, ∃ a b, a ∈ prev.val.1 ∪ {0} ∧ b ∈ prev.val.1 ∪ {0} ∧ a + b = n) →
   (∀ n ≤ next.val.2, ∃ a b, a ∈ next.val.1 ∪ {0} ∧ b ∈ next.val.1 ∪ {0} ∧ a + b = n))

private lemma valid_ext_exists' (prev : State') (C : ℕ) : ∃ next, step_prop' prev C next := by
  let G := prev.val.2
  let M := 2 * G + C + 1
  let W := 3 * G + 2 * C + 2
  let next_f := prev.val.1 ∪ Icc (G + 1) M ∪ {W}
  let next_gap := W + G + C + 1
  have h_bound : ∀ x ∈ next_f, x ≤ next_gap := by
    intro x hx
    simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hx
    rcases hx with (hx_prev | hx_icc) | hx_W
    · exact le_trans (prev.property x hx_prev) (by omega)
    · omega
    · omega
  let next_state : State' := ⟨(next_f, next_gap), h_bound⟩
  use next_state
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro x hx
    simp only [next_state, next_f, mem_union, mem_Icc, mem_singleton_iff]
    left; left; exact hx
  · simp only [next_state]; omega
  · intro x hx
    simp only [next_state, next_f, mem_union, mem_Icc, mem_singleton_iff, mem_diff] at hx
    rcases hx with ⟨(hx_prev | hx_icc) | hx_W, hx_not⟩
    · contradiction
    · omega
    · omega
  · use W + G + 1
    refine ⟨show W + G + 1 + C ≤ next_gap from by omega, ?_⟩
    intros F₁ F₂ h_union h_disj
    have h_next_eq : next_f = F₁ ∪ F₂ := h_union
    by_cases hW : W ∈ F₁
    · right
      intros x hx_ge hx_le hx_sum
      simp only [Set.mem_add] at hx_sum
      obtain ⟨a, ha, b, hb, hab⟩ := hx_sum
      have hW_not_F2 : W ∉ F₂ := fun h =>
        Set.disjoint_iff.mp h_disj ⟨hW, h⟩
      have ha_next : a ∈ next_f := by rw [h_next_eq]; exact Set.mem_union_right _ ha
      have hb_next : b ∈ next_f := by rw [h_next_eq]; exact Set.mem_union_right _ hb
      have ha_le_M : a ≤ M := by
        simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at ha_next
        rcases ha_next with (ha_prev | ha_icc) | ha_W
        · exact le_trans (prev.property a ha_prev) (by omega)
        · exact ha_icc.2
        · exfalso; apply hW_not_F2; rw [ha_W] at ha; exact ha
      have hb_le_M : b ≤ M := by
        simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hb_next
        rcases hb_next with (hb_prev | hb_icc) | hb_W
        · exact le_trans (prev.property b hb_prev) (by omega)
        · exact hb_icc.2
        · exfalso; apply hW_not_F2; rw [hb_W] at hb; exact hb
      omega
    · left
      intros x hx_ge hx_le hx_sum
      simp only [Set.mem_add] at hx_sum
      obtain ⟨a, ha, b, hb, hab⟩ := hx_sum
      have ha_next : a ∈ next_f := by rw [h_next_eq]; exact Set.mem_union_left _ ha
      have hb_next : b ∈ next_f := by rw [h_next_eq]; exact Set.mem_union_left _ hb
      have ha_le_M : a ≤ M := by
        simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at ha_next
        rcases ha_next with (ha_prev | ha_icc) | ha_W
        · exact le_trans (prev.property a ha_prev) (by omega)
        · exact ha_icc.2
        · exfalso; apply hW; rw [ha_W] at ha; exact ha
      have hb_le_M : b ≤ M := by
        simp only [next_f, mem_union, mem_Icc, mem_singleton_iff] at hb_next
        rcases hb_next with (hb_prev | hb_icc) | hb_W
        · exact le_trans (prev.property b hb_prev) (by omega)
        · exact hb_icc.2
        · exfalso; apply hW; rw [hb_W] at hb; exact hb
      omega
  · intro h_prev_cov n hn
    change n ≤ W + G + C + 1 at hn
    by_cases h1 : n ≤ G
    · obtain ⟨a, b, ha, hb, hab⟩ := h_prev_cov n h1
      exact ⟨a, b,
        (by rcases ha with ha_prev | ha_0
            · left; left; left; exact ha_prev
            · right; exact ha_0),
        (by rcases hb with hb_prev | hb_0
            · left; left; left; exact hb_prev
            · right; exact hb_0),
        hab⟩
    · push_neg at h1
      by_cases h2 : n ≤ M
      · exact ⟨n, 0, by left; left; right; exact ⟨by omega, h2⟩,
          by right; exact mem_singleton 0, by omega⟩
      · push_neg at h2
        by_cases h3 : n ≤ 2 * M
        · exact ⟨n / 2, n - n / 2,
            by left; left; right; exact ⟨by omega, by omega⟩,
            by left; left; right; exact ⟨by omega, by omega⟩,
            by omega⟩
        · push_neg at h3
          exact ⟨W, n - W,
            by left; right; exact mem_singleton W,
            by left; left; right; exact ⟨by omega, by omega⟩,
            by omega⟩

noncomputable def seq_step' : ℕ → State'
  | 0 => ⟨(∅, 0), by intro x hx; contradiction⟩
  | n + 1 => Classical.choose (valid_ext_exists' (seq_step' n) n)

noncomputable def f_seq' (n : ℕ) : Set ℕ := (seq_step' n).val.1
noncomputable def gap_seq' (n : ℕ) : ℕ := (seq_step' n).val.2

private lemma seq_step_prop' (n : ℕ) : step_prop' (seq_step' n) n (seq_step' (n + 1)) :=
  Classical.choose_spec (valid_ext_exists' (seq_step' n) n)

private lemma f_seq_covers' (n : ℕ) :
    ∀ m ≤ gap_seq' n, ∃ a b, a ∈ f_seq' n ∪ {0} ∧ b ∈ f_seq' n ∪ {0} ∧ a + b = m := by
  induction n with
  | zero =>
    intro m hm
    have hm0 : m = 0 := Nat.eq_zero_of_le_zero hm
    exact ⟨0, 0, Or.inr (mem_singleton 0), Or.inr (mem_singleton 0), by rw [hm0]⟩
  | succ n ih =>
    exact (seq_step_prop' n).2.2.2.2 ih

private lemma f_seq_mono' {m k : ℕ} (h : m ≤ k) : f_seq' m ⊆ f_seq' k := by
  induction h with
  | refl => rfl
  | step _ ih => exact ih.trans (seq_step_prop' _).1

private lemma gap_seq_mono' {m k : ℕ} (h : m ≤ k) : gap_seq' m ≤ gap_seq' k := by
  induction h with
  | refl => exact le_refl _
  | step _ ih => exact ih.trans (seq_step_prop' _).2.1

private lemma subset_f_k_of_le' (k : ℕ) :
    ∀ y ∈ (⋃ n, f_seq' n), y ≤ gap_seq' k → y ∈ f_seq' k := by
  intros y hy hy_le
  by_cases hy_fk : y ∈ f_seq' k
  · exact hy_fk
  · exfalso
    have hex : ∃ n, y ∈ f_seq' n := Set.mem_iUnion.mp hy
    have h_min : ∃ m, y ∈ f_seq' m ∧ ∀ j < m, y ∉ f_seq' j :=
      ⟨Nat.find hex, Nat.find_spec hex, fun j hj => Nat.find_min hex hj⟩
    obtain ⟨m, hm_in, hm_min⟩ := h_min
    have h_m_gt_k : m > k := by
      by_contra h
      push_neg at h
      exact absurd (f_seq_mono' h hm_in) hy_fk
    have h_m_pos : 0 < m := by omega
    have h_diff : y ∈ f_seq' m \ f_seq' (m - 1) :=
      ⟨hm_in, hm_min (m - 1) (Nat.sub_lt h_m_pos Nat.one_pos)⟩
    have h_gap : y > gap_seq' (m - 1) := by
      have heq : m = (m - 1) + 1 := by omega
      exact (seq_step_prop' (m - 1)).2.2.1 y (heq ▸ h_diff)
    have hkm : k ≤ m - 1 := by omega
    linarith [gap_seq_mono' hkm]

private lemma subset_union_of_f_seq (n : ℕ) : f_seq' n ⊆ ⋃ m, f_seq' m :=
  fun x hx => Set.mem_iUnion.mpr ⟨n, hx⟩

private lemma erdos_gap_set_exists' : ∃ A : Set ℕ, IsAddBasis2 A ∧
    ∀ A₁ A₂, A = A₁ ∪ A₂ → Disjoint A₁ A₂ →
      HasLargeGaps (A₁ + A₁) ∨ HasLargeGaps (A₂ + A₂) := by
  let A := ⋃ n, f_seq' n
  use A
  constructor
  · intro n
    have hn : n ≤ gap_seq' (n + 1) := by
      obtain ⟨N, hN, _⟩ := (seq_step_prop' n).2.2.2.1
      show n ≤ (seq_step' (n + 1)).val.2
      linarith
    obtain ⟨a, b, ha, hb, hab⟩ := f_seq_covers' (n + 1) n hn
    exact ⟨a, (by cases ha with
      | inl h => left; exact subset_union_of_f_seq (n + 1) h
      | inr h => right; exact h),
      b, (by cases hb with
      | inl h => left; exact subset_union_of_f_seq (n + 1) h
      | inr h => right; exact h),
      hab⟩
  · intros A₁ A₂ h_part h_disj
    have h_or_C : ∀ C : ℕ, (∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₁ + A₁) ∨
                            (∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₂ + A₂) := by
      intro C
      obtain ⟨N, hN_bound, h_gap_k⟩ := (seq_step_prop' C).2.2.2.1
      have h_F_part : f_seq' (C + 1) = (A₁ ∩ f_seq' (C + 1)) ∪ (A₂ ∩ f_seq' (C + 1)) := by
        ext x
        simp only [mem_union, mem_inter_iff]
        constructor
        · intro hx
          have hxA : x ∈ A := Set.mem_iUnion.mpr ⟨C + 1, hx⟩
          rcases (h_part ▸ hxA) with h1 | h2
          · left; exact ⟨h1, hx⟩
          · right; exact ⟨h2, hx⟩
        · rintro (⟨-, hx⟩ | ⟨-, hx⟩) <;> exact hx
      have h_F_disj : Disjoint (A₁ ∩ f_seq' (C + 1)) (A₂ ∩ f_seq' (C + 1)) :=
        Set.disjoint_iff.mpr fun x hx => Set.disjoint_iff.mp h_disj ⟨hx.1.1, hx.2.1⟩
      rcases h_gap_k (A₁ ∩ f_seq' (C + 1)) (A₂ ∩ f_seq' (C + 1)) h_F_part h_F_disj with
          h_inl | h_inr
      · left
        refine ⟨N, fun x hx_ge hx_le hx_in => ?_⟩
        simp only [Set.mem_add] at hx_in
        obtain ⟨a, ha, b, hb, hab⟩ := hx_in
        have ha_le : a ≤ N + C := by linarith [Nat.le_add_right a b]
        have hb_le : b ≤ N + C := by linarith [Nat.le_add_left b a]
        have ha_gap : a ≤ gap_seq' (C + 1) := le_trans ha_le hN_bound
        have hb_gap : b ≤ gap_seq' (C + 1) := le_trans hb_le hN_bound
        have ha_A : a ∈ A := by rw [h_part]; exact Set.mem_union_left _ ha
        have hb_A : b ∈ A := by rw [h_part]; exact Set.mem_union_left _ hb
        have ha_fk : a ∈ f_seq' (C + 1) := subset_f_k_of_le' (C + 1) a ha_A ha_gap
        have hb_fk : b ∈ f_seq' (C + 1) := subset_f_k_of_le' (C + 1) b hb_A hb_gap
        have : x ∈ (A₁ ∩ f_seq' (C + 1)) + (A₁ ∩ f_seq' (C + 1)) := by
          simp only [Set.mem_add]
          exact ⟨a, ⟨ha, ha_fk⟩, b, ⟨hb, hb_fk⟩, hab⟩
        exact h_inl x hx_ge hx_le this
      · right
        refine ⟨N, fun x hx_ge hx_le hx_in => ?_⟩
        simp only [Set.mem_add] at hx_in
        obtain ⟨a, ha, b, hb, hab⟩ := hx_in
        have ha_le : a ≤ N + C := by linarith [Nat.le_add_right a b]
        have hb_le : b ≤ N + C := by linarith [Nat.le_add_left b a]
        have ha_gap : a ≤ gap_seq' (C + 1) := le_trans ha_le hN_bound
        have hb_gap : b ≤ gap_seq' (C + 1) := le_trans hb_le hN_bound
        have ha_A : a ∈ A := by rw [h_part]; exact Set.mem_union_right _ ha
        have hb_A : b ∈ A := by rw [h_part]; exact Set.mem_union_right _ hb
        have ha_fk : a ∈ f_seq' (C + 1) := subset_f_k_of_le' (C + 1) a ha_A ha_gap
        have hb_fk : b ∈ f_seq' (C + 1) := subset_f_k_of_le' (C + 1) b hb_A hb_gap
        have : x ∈ (A₂ ∩ f_seq' (C + 1)) + (A₂ ∩ f_seq' (C + 1)) := by
          simp only [Set.mem_add]
          exact ⟨a, ⟨ha, ha_fk⟩, b, ⟨hb, hb_fk⟩, hab⟩
        exact h_inr x hx_ge hx_le this
    let P := fun C => ∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₁ + A₁
    let Q := fun C => ∃ N, ∀ x, N ≤ x → x ≤ N + C → x ∉ A₂ + A₂
    cases infinite_or P Q
      (fun c1 c2 hc => has_gaps_mono (A₁ + A₁) c1 c2 hc)
      (fun c1 c2 hc => has_gaps_mono (A₂ + A₂) c1 c2 hc)
      h_or_C with
    | inl h => left; intro C; exact h C
    | inr h => right; intro C; exact h C

theorem erdos_741_ii :
    ∃ A : Set ℕ,
      IsAddBasis2 A ∧
      ∀ A₁ A₂ : Set ℕ, A = A₁ ∪ A₂ → Disjoint A₁ A₂ →
        ¬(IsSyndetic' (sumset' A₁) ∧ IsSyndetic' (sumset' A₂)) := by
  obtain ⟨A, h_basis, h_gaps⟩ := erdos_gap_set_exists'
  use A, h_basis
  intro A₁ A₂ h_union h_disj h_syn
  obtain ⟨h_syn1, h_syn2⟩ := h_syn
  have h_syn1' : IsSyndetic' (A₁ + A₁) := (sumset'_eq A₁) ▸ h_syn1
  have h_syn2' : IsSyndetic' (A₂ + A₂) := (sumset'_eq A₂) ▸ h_syn2
  rcases h_gaps A₁ A₂ h_union h_disj with h1 | h2
  · exact not_syndetic_of_large_gaps' (A₁ + A₁)
      (fun k => by
        obtain ⟨N, hN⟩ := h1 k
        exact ⟨N, icc_subset_complement (fun y hy => hN y hy.1 hy.2)⟩)
      h_syn1'
  · exact not_syndetic_of_large_gaps' (A₂ + A₂)
      (fun k => by
        obtain ⟨N, hN⟩ := h2 k
        exact ⟨N, icc_subset_complement (fun y hy => hN y hy.1 hy.2)⟩)
      h_syn2'
