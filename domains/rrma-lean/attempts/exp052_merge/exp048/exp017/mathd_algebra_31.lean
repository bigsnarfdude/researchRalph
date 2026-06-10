import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_31 (x : NNReal) (u : ℕ → NNReal) (h₀ : ∀ n, u (n + 1) = NNReal.sqrt (x + u n))
  (h₁ : Filter.Tendsto u Filter.atTop (𝓝 9)) : 9 = NNReal.sqrt (x + 9) := by
  have hlim : Filter.Tendsto (fun n => u (n + 1)) Filter.atTop (𝓝 9) :=
    h₁.comp (Filter.tendsto_atTop_atTop.mpr fun n => ⟨n, fun m hm => by omega⟩)
  simp only [h₀] at hlim
  have hlim2 : Filter.Tendsto (fun n => NNReal.sqrt (x + u n)) Filter.atTop (𝓝 (NNReal.sqrt (x + 9))) := by
    apply (NNReal.continuous_sqrt.tendsto _).comp
    exact tendsto_const_nhds.add h₁
  exact tendsto_nhds_unique hlim hlim2
