## Functions need a chosen notion of distance

Functions can be added and scaled, so they form vector spaces. To say an approximation is close, choose a norm. L1 integrates absolute magnitude, L2 takes the square root of integrated squared magnitude, and L∞ measures essential supremum. Lp identifies functions differing only on null sets.

$$
\|f\|_1=\int|f|,\qquad \|f\|_2^2=\int|f|^2.
$$

## Completeness differs from inner-product geometry

A Banach space contains limits of all Cauchy sequences in its norm. A Hilbert space is complete in an inner-product norm. L1 and L∞ are Banach but generally not Hilbert. For real L2 functions, the inner product is integral fg.

Inner-product norms satisfy the parallelogram law. For disjoint unit vectors, its left side minus right side is four in L1, zero in L2 and minus two in L∞. Completeness alone does not supply orthogonal geometry.

$$
\|u+v\|^2+\|u-v\|^2=2\|u\|^2+2\|v\|^2.
$$

## Compute a sequence and its limit

On [0,1], a geometric-series identity gives

$$
\begin{aligned}
f_m(x)&=\sum_{k=0}^{m}\frac{x^k}{2^{k+1}},\\
f(x)-f_m(x)&=\frac{(x/2)^{m+1}}{2-x},\\
\|f-f_m\|_\infty&=2^{-(m+1)},\quad f(x)=\frac1{2-x}.
\end{aligned}
$$

Degrees two and six give uniform errors 1/8 and 1/128. Since the interval has unit measure, L1 and L2 errors are bounded by the uniform error. Limit norms are log 2, 1/√2 and one. Switching norms changes the measurement, not the blue partial polynomial or orange limit curve.

## The wrong space can lose the limit

Polynomials with uniform norm are incomplete: this Cauchy sequence has a nonpolynomial limit. Continuous functions with uniform norm are complete. Continuous functions with L1 norm are not: a continuous ramp rising from zero to one over width 1/m after x=.5 approaches a step in L1 with error 1/(2m), for m≥2. No continuous function equals that step almost everywhere. **Specify both the set and its norm.** The lab's Lp selections refer to complete Lp spaces, while plotted examples have continuous representatives.

For AI approximation, small average loss differs from uniformly small error. State the input measure and evaluation norm.

## Prove completeness of ℓ² using coordinates and tails

Take a Cauchy sequence x⁽ⁿ⁾ in real ℓ². Each coordinate difference is bounded by the full norm difference, so every coordinate has a real limit x_k. Cauchy sequences have a common norm bound M. Taking coordinate limits over finitely many indices gives Σk≤K|x_k|²≤M². Increasing K proves x∈ℓ².

For ε>0 choose N such that ‖x⁽ⁿ⁾−x⁽ᵐ⁾‖₂≤ε for n,m≥N. Fix n, first let m→∞ in finitely many coordinates, then increase K:

$$
\sum_{k=1}^K|x_k^{(n)}-x_k|^2\le\varepsilon^2
\quad\Longrightarrow\quad
\|x^{(n)}-x\|_2\le\varepsilon\qquad(n\ge N).
$$

The limit lies in the space and convergence holds in norm, proving completeness. The inner product Σx_ky_k induces this norm, so ℓ² is Hilbert. The Cauchy condition controls the infinite tail; coordinate convergence alone would not suffice.

If only finitely supported sequences c₀₀ are allowed, truncations of x_k=1/k are ℓ²-Cauchy but their limit lies outside c₀₀. Every ℓ² vector is approximated by its own finite truncations, so c₀₀ is dense nevertheless. Density and completeness are different properties.

## Inspect constants in finite-dimensional norm equivalence

This table uses counting-measure sequence norms, rather than the probability measure on [0,1]. Set the first N coordinates to 1/√N and all later coordinates to zero.

| N | ℓ¹ norm | ℓ² norm | ℓ∞ norm |
|---|---|---|---|
| 4 | 2 | 1 | .5 |
| 16 | 4 | 1 | .25 |
| 64 | 8 | 1 | .125 |

The norms are equivalent at each fixed N, but no single resolution-independent C gives ℓ¹≤Cℓ² or ℓ²≤Cℓ∞ throughout this family. Specify the measure and dimension dependence when discussing neural-operator or mesh-refinement stability. The code checks finite vector values; the infinite-sum argument above proves completeness.

Small query errors also need an appropriate formula. At m=8,x=10⁻⁵, the actual tail is about 9.77×10⁻⁴⁹, but subtracting function values previously produced about 1.11×10⁻¹⁶. The calculator now directly uses the exact tail formula. Positive tails below numeric representation range show a dash and explanation, while the exact zero at x=0 remains zero.

**Exercise:** Does coordinatewise convergence to zero imply ℓ²-norm convergence? **Explanation:** No. Each fixed coordinate of e_n tends to zero while its norm stays one. The completeness proof additionally used the Cauchy condition.

```python
from math import sqrt

for N in [4,16,64]:
    # Counting-measure sequence norms, not unit-interval Lp norms.
    vector=[1/sqrt(N)]*N
    l1=sum(map(abs,vector))
    l2=sqrt(sum(x*x for x in vector))
    linf=max(map(abs,vector))
    assert abs(l2-1)<1e-12
    assert abs(l1-sqrt(N))<1e-12
    assert abs(linf-1/sqrt(N))<1e-12
    print('N, l1, l2, linfinity:',N,l1,l2,linf)
```


## Exercises with explanations

1. What is the degree-zero L1 norm? **1/2:** the function is constant 1/2.
2. Does a nonpolynomial limit mean convergence failed? **No.** Convergence holds in a larger complete space, while the polynomial space lacks the limit.
3. Does changing one point alter an L2 norm? **No.** This leads to the point-evaluation problem in the dual-space chapter.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
