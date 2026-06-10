#!/usr/bin/env python3
"""Generate Lean 4 proof for amc12a_2009_p25 via period-24 computation."""

# All values as (p, q) meaning p + q*s where s = sqrt(3)
# Represented as rational tuples (p_num/p_den, q_num/q_den) but we'll use exact fractions
from fractions import Fraction

# a(n) = p + q*s where s = sqrt(3), s^2 = 3
# Store as (p, q) with Fraction components
vals = {}
vals[1] = (Fraction(1), Fraction(0))  # 1
vals[2] = (Fraction(0), Fraction(1,3))  # s/3 = 1/s (since 1/s = s/3)

def add_vals(v1, v2):
    return (v1[0]+v2[0], v1[1]+v2[1])

def mul_vals(v1, v2):
    # (p1+q1*s)(p2+q2*s) = p1p2+3q1q2 + (p1q2+p2q1)s
    p = v1[0]*v2[0] + 3*v1[1]*v2[1]
    q = v1[0]*v2[1] + v1[1]*v2[0]
    return (p, q)

def neg_val(v):
    return (-v[0], -v[1])

def sub_vals(v1, v2):
    return add_vals(v1, neg_val(v2))

def inv_val(v):
    # 1/(p+qs) = (p-qs)/(p^2 - 3q^2)
    denom = v[0]**2 - 3*v[1]**2
    if denom == 0:
        raise ValueError("Division by zero")
    return (v[0]/denom, -v[1]/denom)

def div_vals(v1, v2):
    return mul_vals(v1, inv_val(v2))

ONE = (Fraction(1), Fraction(0))

# Compute a(n) for n = 3..26 using recurrence
# a(n+2) = (a(n) + a(n+1)) / (1 - a(n)*a(n+1))
for n in range(1, 25):
    if n+2 not in vals:
        an = vals[n]
        an1 = vals[n+1]
        num = add_vals(an, an1)
        prod = mul_vals(an, an1)
        denom = sub_vals(ONE, prod)
        if denom == (Fraction(0), Fraction(0)):
            # 0/0 case shouldn't happen
            raise ValueError(f"Denom zero at n={n}")
        vals[n+2] = div_vals(num, denom)

# Print values
for n in range(1, 27):
    p, q = vals[n]
    parts = []
    if p != 0: parts.append(str(p))
    if q != 0:
        if q == 1: parts.append("s")
        elif q == -1: parts.append("-s")
        elif q > 0: parts.append(f"{q}*s")
        else: parts.append(f"({q})*s")
    if not parts: parts.append("0")
    print(f"a({n}) = {' + '.join(parts)}")

# Verify periodicity
assert vals[25] == vals[1], f"a(25)={vals[25]} != a(1)={vals[1]}"
assert vals[26] == vals[2], f"a(26)={vals[26]} != a(2)={vals[2]}"
assert vals[17] == (Fraction(0), Fraction(0)), f"a(17)={vals[17]} != 0"
print(f"\n2009 mod 24 = {2009 % 24}")
print(f"a(17) = {vals[17]} = 0 ✓")
print(f"Period 24 verified: a(25)=a(1), a(26)=a(2) ✓")

# Now generate the Lean proof
def val_to_lean(v):
    """Convert (p, q) to Lean expression string."""
    p, q = v
    if p == 0 and q == 0:
        return "0"
    parts = []
    if p != 0:
        if p.denominator == 1:
            parts.append(str(p.numerator))
        else:
            parts.append(f"({p.numerator} : ℝ) / {p.denominator}")
    if q != 0:
        if q == 1:
            parts.append("s")
        elif q == -1:
            parts.append("(-s)")
        elif q.denominator == 1:
            if q.numerator > 0:
                parts.append(f"{q.numerator} * s")
            else:
                parts.append(f"({q.numerator}) * s")
        else:
            parts.append(f"({q.numerator} : ℝ) / {q.denominator} * s")
    if not parts:
        return "0"
    return " + ".join(parts).replace("+ (-", "- (").replace("+ -", "- ")

# Generate lean proof steps
print("\n--- LEAN PROOF STEPS ---\n")
for n in range(3, 27):
    v = vals[n]
    lean_val = val_to_lean(v)
    print(f"  -- a({n}) = {lean_val}")
    nm2 = n - 2
    print(f"  have ha{n} : a {n} = {lean_val} := by")
    print(f"    have h := h₂ {nm2} (by omega)")
    print(f"    rw [show ({nm2}:ℕ) + 2 = {n} from rfl, show ({nm2}:ℕ) + 1 = {n-1} from rfl] at h")
    if n == 3:
        print(f"    rw [h₀, h₁] at h; rw [h]")
    else:
        print(f"    rw [ha{nm2}, ha{n-1}] at h; rw [h]")

    # Determine proof strategy
    num = add_vals(vals[nm2], vals[n-1])
    if num == (Fraction(0), Fraction(0)):
        # Numerator is 0, so result is 0
        print(f"    simp [show ({val_to_lean(vals[nm2])}) + ({val_to_lean(vals[n-1])}) = 0 from by ring]")
    elif vals[nm2] == (Fraction(0), Fraction(0)) or vals[n-1] == (Fraction(0), Fraction(0)):
        # One of them is 0, so denominator is 1
        print(f"    simp; ring")
    else:
        print(f"    sorry  -- field_simp + nlinarith [hs2]")
    print()
