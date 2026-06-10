import Mathlib
set_option maxHeartbeats 3200000
open BigOperators Real Nat Topology Rat

-- IMO 1990 P3: n ≥ 2, n² | 2ⁿ + 1 ⟹ n = 3
-- Direct proof: show n is odd, minFac = 3, all prime factors are 3, n = 3^k, k ≤ 1.

-- n is odd
private lemma n_odd {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1) : ¬ 2 ∣ n := by
  intro h2n; have : 2 ∣ n ^ 2 := dvd_pow h2n (by omega); omega

-- For any prime p | n with p ∤ 2: p = 3
-- Uses ZMod order theory: ord_p(2) = 2 (from minimality argument)
private lemma prime_factor_eq_three {n : ℕ} (hn : 2 ≤ n) (hdvd : n ^ 2 ∣ 2 ^ n + 1)
    {p : ℕ} (hp : Nat.Prime p) (hp_dvd : p ∣ n) : p = 3 := by
  have h_odd := n_odd hn hdvd
  have hp2 : p ≠ 2 := fun h => by rw [h] at hp_dvd; exact h_odd hp_dvd
  have hp3 : 3 ≤ p := by omega
  -- p | 2^n + 1
  have hn_dvd : n ∣ 2 ^ n + 1 := dvd_trans (dvd_pow_self n (by omega)) hdvd
  have hpp : p ∣ 2 ^ n + 1 := dvd_trans hp_dvd hn_dvd
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have h2nz : (2 : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_zmod_eq_zero_iff_dvd]
    intro h; have := hp.eq_one_or_self_of_dvd 2 h; omega
  -- 2^n ≡ -1 mod p, so 2^(2n) ≡ 1
  have h2n1 : (2 : ZMod p) ^ n = -1 := by
    have : ((2 ^ n + 1 : ℕ) : ZMod p) = 0 := (ZMod.natCast_zmod_eq_zero_iff_dvd _ _).mpr hpp
    have : (2 : ZMod p) ^ n + 1 = 0 := by push_cast at this ⊢; exact this; linarith
  have h22n : (2 : ZMod p) ^ (2 * n) = 1 := by rw [pow_mul, h2n1]; ring
  -- ord | 2n but ∤ n
  have hd2n : orderOf (2 : ZMod p) ∣ 2 * n := orderOf_dvd_of_pow_eq_one h22n
  have hdn : ¬ orderOf (2 : ZMod p) ∣ n := by
    intro h; rw [orderOf_dvd_iff_pow_eq_one.mp h] at h2n1
    have : (0 : ZMod p) = 2 := by linarith
    rw [eq_comm, ZMod.natCast_zmod_eq_zero_iff_dvd] at this; exact by have := hp.eq_one_or_self_of_dvd 2 this; omega
  -- ord is even
  have hd_even : 2 ∣ orderOf (2 : ZMod p) := by
    by_contra hd_odd
    exact hdn ((Nat.Prime.coprime_iff_not_dvd Nat.prime_two).mpr hd_odd).dvd_of_dvd_mul_right hd2n
  obtain ⟨e, he⟩ := hd_even
  set d := orderOf (2 : ZMod p) with hd_def
  -- e | n
  have he_dvd_n : e ∣ n := (Nat.mul_dvd_mul_iff_left (by omega : 0 < 2)).mp (he ▸ hd2n)
  -- d | p-1, so d ≤ p-1, e ≤ (p-1)/2 < p
  have hd_dvd_pm1 : d ∣ p - 1 := orderOf_dvd_of_pow_eq_one (ZMod.pow_card_sub_one_eq_one h2nz)
  have he_lt_p : e < p := by have := Nat.le_of_dvd (by omega) hd_dvd_pm1; omega
  -- Key: all prime factors of e divide n and are < p.
  -- We claim e = 1 (proved by well-founded recursion on the size of e).
  -- If e > 1, let q = minFac(e). q is prime, q | e | n, q ≤ e < p.
  -- But by THIS SAME LEMMA applied to q (since q is prime, q | n, q ≠ 2):
  --   q = 3. So 3 | e | n.
  -- And the minFac of e is 3, but also the minFac of n is 3 (by our argument).
  -- So e = 3^j for some j. Then d = 2·3^j | p-1.
  -- If j = 0: d = 2, 4 ≡ 1 mod p, p | 3, p = 3.
  -- If j ≥ 1: d ≥ 6. 2^(2·3^j) ≡ 1 mod p. And 2^(3^j) ≡ -1 mod p.
  --   But 2^3 = 8 and p | 8+1 = 9 only if p | 9, i.e. p = 3. But p ≥ 7. ✗
  --   Wait, 2^(3^j) ≡ -1 mod p, not 2^3 ≡ -1 mod p.
  --   Since e = 3^j and j ≥ 1: 3 | e | n. Also 3^j < p.
  --   d = 2·3^j ≤ p-1. And p | 2^(3^j)+1.
  --   Now 2^(3^j)+1 = (2^(3^(j-1)))^3 + 1 = (2^(3^(j-1))+1)(2^(2·3^(j-1))-2^(3^(j-1)}+1).
  --   The first factor: 2^(3^(j-1))+1. Note 2^(3^(j-1)) has order 2·3^(j-1) or 2·3^j mod p.
  --   Actually ord = 2·3^j (unchanged), and if p | 2^(3^(j-1))+1, then 2^(3^(j-1)) ≡ -1,
  --   so 2^(2·3^(j-1)) ≡ 1, meaning ord | 2·3^(j-1). But ord = 2·3^j ∤ 2·3^(j-1). Contradiction.
  --   So p | second factor: 2^(2·3^(j-1))-2^(3^(j-1)}+1.
  --   For j=1: 2^2-2+1=3. p|3, p=3, but p≥7. ✗
  --   For j=2: 2^6-2^3+1=57=3·19. p|57, p prime ≥ 7. p=19.
  --     ord_19(2) should be 2·9=18. Check: 18|19-1=18. ✓
  --     But p=19 | n means 19 is a prime factor of n.
  --     We need p=19→ by this lemma → 19=3. Contradiction!
  -- The recursion works! For any prime p | n, this lemma says p = 3.
  -- The recursion is well-founded on p (we only invoke the lemma for primes q < p).
  --
  -- But Lean doesn't support this kind of recursive lemma easily without
  -- explicit well-founded recursion. And the "all prime factors of e" step
  -- requires knowing that q < p when q | e.
  --
  -- SIMPLIFICATION: instead of full recursion, observe that if e > 1,
  -- then e has a prime factor q with q | n and q < p.
  -- By the minFac argument (same as for p = minFac(n)):
  -- q's minFac argument gives ord_q(2) = 2, so q | 3, q = 3.
  -- But that argument ALSO needs all prime factors of the order's "e'" to be 3...
  -- The recursion bottoms out at p = 3 = minFac(n), where e = 1.
  --
  -- Actually, the minFac case IS the base case:
  -- For p = minFac(n): e < p = minFac(n). All prime factors of e are < minFac(n).
  -- But minFac(n) ≤ every prime factor of n, and e | n. So e has no prime factors
  -- that are < minFac(n) while dividing n. Hence e = 1. d = 2. p | 3. p = 3.
  --
  -- For general p > 3 with p | n:
  -- e | n and e < p. Prime factors of e divide n and are < p.
  -- Since minFac(n) = 3 (proved above), these prime factors are ≥ 3.
  -- And they're < p. By strong induction on p: they're all = 3.
  -- So e = 3^j. Then d = 2·3^j.
  -- For j ≥ 1: p | 2^(3^j)+1, and (by the factorization argument above)
  -- p | 2^(2·3^(j-1))-2^(3^(j-1))+1. Evaluate at j=1: p|3, p=3<p. ✗
  -- For j ≥ 2: iterate the factorization to get p | Φ_{2·3^j}(2).
  -- Then p | Φ_{2·3^j}(2) and p ≠ 3. But Φ_6(2) = 3, so p ∤ Φ_6(2).
  -- By the recursive structure: Φ_{2·3^j}(2) = (2^(3^j)+1) / (2^(3^(j-1))+1).
  -- p | numerator, p ∤ denominator (proved above). So p | Φ_{2·3^j}(2).
  -- Hmm, I need to show this leads to a contradiction for p > 3.
  -- For j=1: Φ_6(2) = 3. p > 3 doesn't divide 3. ✗
  -- For j≥2: we showed p | second factor but p ∤ first factor.
  --   Iterate on j: 2^(2·3^(j-1))-2^(3^(j-1)}+1 = (2^(3^(j-1)})²-2^(3^(j-1)}+1 = Φ_6(2^(3^(j-1)}).
  --   And Φ_6(x) = x²-x+1. For x = 2^(3^(j-1)}, ord_p(x) = 6 (by the order argument).
  --   Then p | x²-x+1 = Φ_6(x).
  --   Further: Φ_6(x) = (x³+1)/(x+1). And x³+1 = (x+1)(x²-x+1).
  --   So x²-x+1 = (x³+1)/(x+1). And x+1 = 2^(3^(j-1)}+1. p ∤ x+1 (proved).
  --   So p | x²-x+1.
  --   Now x²-x+1 = (x-ω)(x-ω²) where ω = e^{2πi/3}.
  --   In ZMod p: x²-x+1 ≡ 0 mod p means x is a primitive 6th root of unity.
  --   The number of such roots is φ(6)=2. So at most 2 primes divide Φ_6(x) with
  --   multiplicity 1 (generically). But we don't have such bounds in Lean.
  --
  -- I think this approach is too complex. Let me try the computational bound.
  -- For n = 3^k: v₃(2^(3^k)+1) = k+1. So for 3^(2k) | 2^(3^k)+1, need 2k ≤ k+1, k ≤ 1.
  -- The LTE part: v₃(2^(3^k)+1) = k+1.
  -- Proof by induction on k:
  --   k=1: v₃(9) = 2 = 1+1. ✓
  --   k→k+1: 2^(3^(k+1))+1 = (2^(3^k))³+1 = (2^(3^k)+1)(2^(2·3^k)-2^(3^k)+1)
  --   v₃ of first factor = k+1 (by IH).
  --   v₃ of second factor: let a=2^(3^k). a≡-1 mod 3.
  --     a²-a+1 ≡ 1+1+1 = 3 ≡ 0 mod 3. So v₃ ≥ 1.
  --     a²-a+1 = (a+1)²-3a. v₃(a+1)=k+1≥2 for k≥1, so v₃((a+1)²)≥4.
  --     v₃(3a)=1 (since gcd(a,3)=1). So v₃(a²-a+1)=1 (ultrametric).
  --   Total: v₃(2^(3^(k+1))+1) = (k+1)+1 = k+2. ✓
  sorry

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  -- Step 1: n is a power of 3 (from prime_factor_eq_three)
  -- Step 2: n = 3^k with k ≤ 1 (from LTE)
  -- Step 3: n = 3 (k = 0 gives n=1, contradiction; k=1 gives n=3)
  sorry
