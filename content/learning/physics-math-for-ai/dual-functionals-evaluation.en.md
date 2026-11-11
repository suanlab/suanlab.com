## Continuous scalar measurements of functions

A linear functional sends vectors or functions to scalars. The continuous dual X* contains bounded linear functionals, with norm given by the largest output magnitude on the unit ball. Arbitrary algebraic linear maps need not belong to this dual.

$$
\|\ell\|_{X^*}=\sup_{\|f\|_X\le1}|\ell(f)|.
$$

## Riesz representation in a Hilbert space

Every continuous linear functional on a real Hilbert space has a unique inner-product representative g. In L2[0,1], it is integral gf. Cauchy–Schwarz supplies the upper bound; for nonzero g, input g/‖g‖ attains it.

$$
\ell_g(f)=\int_0^1gf,\qquad \|\ell_g\|=\|g\|_2.
$$

For g=1+a t and f=1+b t, the pairing is 1+(a+b)/2+ab/3. At a=b=1, pairing and Cauchy–Schwarz bound equal 7/3, while the dual norm is √(7/3). A functional's value differs from its norm.

## Two problems with point evaluation

An L2 equivalence class ignores null-set changes, so f(0) is not well defined. Even restricting to continuous representatives does not make point evaluation bounded in L2 norm.

$$
v_m(t)=\max(1-mt,0),\quad v_m(0)=1,\quad \|v_m\|_2=\frac1{\sqrt{3m}}.
$$

The orange triangle narrows without losing its height. At m=4 and sixteen, norms are about .289 and .144. The point-value-to-norm ratio √(3m) grows without bound. No common C can guarantee |f(0)|≤C‖f‖₂.

## Connect to kernels and sensors

Point evaluation can become continuous under a different norm or stronger regularity. In an RKHS it has a representative kₓ with f(x)=⟨f,kₓ⟩. General L2 is not automatically an RKHS. Sobolev point evaluation also depends on dimension and differentiation order.

An AI observation model must specify the function space and measurement stability. A finite-width weighted integral is continuous when its weight belongs to L2, as the Riesz example illustrates.

## Where the Riesz representative comes from

Let ℓ≠0 be a continuous linear functional on a real Hilbert space. Its kernel is closed, so the projection theorem proved in the next lesson applies. Choose u with ℓ(u)≠0 and set v=u−Pkerℓ u. Then v≠0, v⊥ker ℓ and ℓ(v)=ℓ(u). For any f, the vector f−[ℓ(f)/ℓ(v)]v lies in that kernel. Hence

$$
\langle f,v\rangle=\frac{\ell(f)}{\ell(v)}\|v\|^2,
\qquad g=\frac{\ell(v)}{\|v\|^2}v,
\qquad \ell(f)=\langle f,g\rangle.
$$

If two representatives differ by a vector orthogonal to every f, choosing that difference as f proves its norm is zero. For ℓ=0 choose g=0. Complex spaces require conjugates consistent with the chosen inner-product convention; this derivation is real.

## Narrower sensors have larger measurement norms

A width-h average sensor ℓ_h(f)=h⁻¹∫₀ʰf has representative g_h=h⁻¹1_[0,h] and norm h⁻¹/². Each fixed h>0 gives an L²-continuous measurement, but the norms are not uniformly bounded as h→0. Pairing the spike v_m with h=1/m gives point value one and average exactly one half. Convergence of averages to point values for each fixed continuous f is not uniform over this changing family.

| m | Spike L² norm | Averaging sensor dual norm | Spike H¹ norm |
|---|---|---|---|
| 1 | .577350 | 1 | 1.154701 |
| 4 | .288675 | 2 | 2.020726 |
| 16 | .144338 | 4 | 4.002603 |

Its derivative's squared integral is m, so the spike's H¹ norm grows as sqrt(m+1/(3m)). For the absolutely continuous representative of a one-dimensional H¹ function, the fundamental theorem and Cauchy–Schwarz give, for every x,

$$
|u(x)|\le\|u\|_{L^2(0,1)}+\|u'\|_{L^2(0,1)}
\le\sqrt2\,\|u\|_{H^1(0,1)}.
$$

Average |u(x)|≤|u(y)|+∫|u'| over y to get the first inequality. Bounded evaluation under this stronger norm therefore agrees with the L² spike counterexample. The weak-derivative lesson explains this H¹ representative. Do not transfer the same bound unchanged to arbitrary dimensions.

**Exercise:** If every ℓ_h is continuous, must limiting point evaluation be L²-continuous? **Explanation:** No. The continuity constants h⁻¹/² diverge.

```python
from fractions import Fraction as F
from math import sqrt

for m in [1,4,16]:
    h=F(1,m)
    spike_integral=h/2
    sensor_value=spike_integral/h
    signal_l2_squared=F(1,3*m)
    sensor_dual_squared=F(1,h)
    derivative_l2_squared=F(m)
    assert sensor_value==F(1,2)
    assert sensor_value**2<=signal_l2_squared*sensor_dual_squared
    assert 1<=2*(signal_l2_squared+derivative_l2_squared)
    print('m, spike L2 norm, averaging dual norm, spike H1 norm:',
          m,*(round(sqrt(float(v)),6) for v in
              [signal_l2_squared,sensor_dual_squared,
               signal_l2_squared+derivative_l2_squared]))
```


## Exercises with explanations

1. Norm for g=1? **One**, since the interval has unit length.
2. Is the pairing of 1+t with itself its norm? **It is the squared norm:** distinguish 7/3 from √(7/3).
3. Does the m=16 picture prove unboundedness? **No.** The exact ratio as m tends to infinity supplies the proof.

Further reading: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
