## Corners can obey integration by parts

A weak derivative v is locally integrable and satisfies the integration-by-parts identity for every smooth compactly supported test. A classical derivative need not exist at every point; weak derivatives are unique up to null sets.

$$
\int u\varphi'=-\int v\varphi.
$$

For u=|x|, v=sign(x) is a weak derivative despite the corner. Choosing v(0)=0 changes no integral. A step function instead has a delta distributional derivative, not a locally integrable derivative, and is not generally in W¹,¹ across its jump.

## Sobolev membership also needs integrability

H¹=W¹,² requires u and its first weak derivatives in L2. For uα=|x|ᵅ with α>0, u is always L2. Away from zero the weak derivative is α sign(x)|x|ᵅ⁻¹.

$$
\begin{aligned}
\|u_\alpha\|_2^2&=\frac2{2\alpha+1},\\
\|v_\alpha\|_2^2&=\frac{2\alpha^2}{2\alpha-1}\quad(\alpha>1/2).
\end{aligned}
$$

At or below .5 the squared derivative integral diverges; equality gives logarithmic divergence. Its L1 norm is still two for every selected exponent, so weak differentiability survives outside H1.

## Check the identity independently

The lab uses endpoint-zero tests φ₀=1−x² and φ₁=x(1−x²). They are not compactly supported tests themselves. They check an extended integration-by-parts identity with vanishing boundary terms for these continuous u and L1 derivatives. Two checks do not prove the universal test-function condition.

Even φ₀ gives zero by odd symmetry. For φ₁:

$$
\int u_\alpha\varphi_1'=-\int v_\alpha\varphi_1=\frac{-4\alpha}{(\alpha+1)(\alpha+3)}.
$$

At α=1 both sides equal −.5 and derivative norm is √2. At α=2 the derivative is 2x with norm √(8/3).

## Read the sampled graph carefully

Blue is u and orange the chosen weak representative. Derivative curves sample only |x|≥1/60. A finite graph cannot establish boundedness near the singularity. The zero marker is an arbitrary null-set representative; undefined classical derivatives display —.

Weak and pointwise PDE losses impose different assumptions. Check boundary conditions, integrability and dimension-dependent embeddings.

## Prove weak differentiability separately from Lᵖ membership

For u=|x|ᵅ,α>0, the candidate v=α sign(x)|x|ᵅ⁻¹ belongs to L¹. Integrate by parts on the two intervals excluding zero, then shrink the omitted width η. The boundary contribution ηᵅ[φ(−η)−φ(η)] tends to zero. Integrability of v permits passage to the limit, proving the identity for every compact smooth test rather than only two sampled tests.

For 1≤p<∞, the exponent of |v|ᵖ near zero is pα−p. Integrability requires this exponent to exceed −1:

$$
u=|x|^\alpha\in W^{1,p}(-1,1)
\quad\Longleftrightarrow\quad \alpha>1-\frac1p
\qquad(\alpha>0),
$$
$$
\|v\|_p^p=\frac{2\alpha^p}{p\alpha-p+1}
\quad\text{when }p\alpha-p+1>0.
$$

Equality gives logarithmic divergence. For p=∞, treat the endpoint separately: v is essentially bounded exactly when α≥1. The screen displays the p=2 H¹ condition; varying p below is a separate example.

| p, α=3/4 | ∫\|v\|ᵖ | In W¹,ᵖ |
|---|---|---|
| 1 | 2 | Yes |
| 2 | 9/4 | Yes |
| 3 | 27/8 | Yes |
| 4 | ∞, logarithmic | No |

## One-dimensional H¹ representatives and boundary traces

If u′=v∈L²(0,1), F(x)=∫₀ˣv is absolutely continuous with weak derivative v. The distributional derivative of u−F is zero, so u−F is almost everywhere constant on the connected interval. To see the last fact, express a zero-integral compact test as another compact test's derivative. Thus u has a unique absolutely continuous representative, constant+F, to which the previous lesson's evaluation bound applies.

Continuous traces do not mean all H¹ functions have zero boundary values. H¹₀ is the H¹ closure of compact smooth functions; on (−1,1), returning to the interval used for |x|ᵅ, it consists of functions with both endpoint traces zero. The current |x|ᵅ has endpoint values one, so even when it belongs to H¹ it is not in H¹₀. For α>1/2, subtracting one preserves the derivative and makes both traces zero. At α=3/4 the resulting squared L² norm is 18/35. Fraction arithmetic checks this integral and the p-dependent thresholds.

**Exercise:** Does an L¹ weak derivative automatically give H¹ membership? **Explanation:** No. At α=1/2 the derivative is L¹ but not L².

```python
from fractions import Fraction as F

alpha=F(3,4)
for p in [1,2,3,4]:
    exponent=p*alpha-p
    integral=2*alpha**p/(exponent+1) if exponent>-1 else None
    expected={1:F(2),2:F(9,4),3:F(27,8),4:None}[p]
    assert integral==expected
    print('p, derivative Lp norm to power p:',p,
          str(integral) if integral is not None else 'infinity (logarithmic at p=4)')
# Subtracting the endpoint trace leaves the weak derivative unchanged.
zero_trace_norm_squared=2/(2*alpha+1)-4/(alpha+1)+2
assert zero_trace_norm_squared==F(18,35)
print('L2 squared norm of |x|^(3/4)-1:',zero_trace_norm_squared)
```


## Exercises with explanations

1. Is α=.75 in H1? **Yes**, with squared derivative norm 2.25.
2. Change v(0) to 100? **The same Lp class and integrals remain.**
3. Does a corner alone exclude weak PDE use? **No.** Check the regularity and boundary conditions actually required.

Further reading: [NTNU MA8502, Function Spaces](https://www.math.ntnu.no/emner/MA8502/2024h/lectures/html/chapter_01/function_spaces.html).
