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

## Exercises with explanations

1. Is α=.75 in H1? **Yes**, with squared derivative norm 2.25.
2. Change v(0) to 100? **The same Lp class and integrals remain.**
3. Does a corner alone exclude weak PDE use? **No.** Check the regularity and boundary conditions actually required.

Further reading: [NTNU MA8502, Function Spaces](https://www.math.ntnu.no/emner/MA8502/2024h/lectures/html/chapter_01/function_spaces.html).
