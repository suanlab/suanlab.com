## A recurrence needs initial conditions

The rule aₙ=paₙ₋₁+qaₙ₋₂ does not identify a unique sequence until initial conditions are supplied. Here a₀=0,a₁=1. With p=q=1 it gives 0,1,1,2,3,5,8,13,21,…, so a₈=21.

Dynamic programming stores smaller answers and builds larger ones using such dependencies. The index may represent problem size or an algorithmic step rather than physical time.

## Package coefficients in a formal series

Define G(x)=Σₙ₌₀∞aₙxⁿ. Initially x is a bookkeeping symbol. Formal power-series equalities compare coefficients; they do not yet assert numerical convergence.

$$
G(x)-x=pxG(x)+qx^2G(x),\qquad
G(x)=\frac{x}{1-px-qx^2}.
$$

Removing a₀ and a₁x leaves the n≥2 terms. Multiplication by x and x² aligns the preceding indices. The denominator has constant term one, so it has a formal inverse. Changing the initial conditions also changes the numerator.

## Substitution introduces convergence

To interpret the formal series as a numerical infinite sum, check convergence. For nonnegative p,q not both zero, the dominant growth rate is ρ=(p+√(p²+4q))/2, and the radius is R=1/ρ. The chosen initial conditions do not cancel that dominant component.

For p=q=1, ρ=φ≈1.618 and R≈0.618. At x=0.5 the infinite sum is G(0.5)=2. The N=8 partial sum includes only exponents zero through eight and equals 1.65234375. Increasing N illustrates approximation; a finite table alone is not a convergence proof.

At x=0.8 the rational expression equals −20/11≈−1.818. All series terms are nonnegative and fail to tend to zero, so the series diverges rather than summing to that negative value. The rational continuation and the original series sum are different objects.

## Boundaries and degenerate cases

The lab checks the interior condition |x|<R. In these nonpolynomial nonnegative-coefficient examples, the general term fails to vanish on |x|=R. Other power series can have different boundary behavior.

If p=q=0, the generating function is simply x, a polynomial with unrestricted convergence radius. The finite radius readout is —, with this interpretation explained in the lab. A zero denominator makes the rational evaluation undefined. N=0 legitimately gives a₀=0 and partial sum zero.

## Read the computation

The plot shows coefficients aₙ; the table also shows weighted terms aₙxⁿ. They answer different questions. Small integer coefficients are computed directly through N≤12. This does not guarantee exact floating-point behavior for arbitrarily large recurrences.

```python
p, q, x, N = 1, 1, 0.5, 8
values = [0, 1]
for n in range(2, N + 1):
    values.append(p * values[-1] + q * values[-2])
partial = sum(values[n] * x**n for n in range(N + 1))
print(values[N], partial)  # 21, 1.65234375
```

## AI connections and exercises

Recurrences can count structured objects, characterize search costs and encode grammar coefficients. Neural hidden-state updates are also recurrences, but nonlinear vector updates do not generally admit this scalar linear generating-function solution. Check linearity, initial conditions and whether coefficients depend on the step.

1. With p=1,q=0: **0,1,1,1,… and G=x/(1−x)**.
2. With p=q=1,N=3, the partial polynomial is **x+x²+2x³**.
3. A finite rational value at x=0.8 establishes **only that substitution value**, not convergence of the power series.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). The derivation uses exactly the stated initial conditions.
