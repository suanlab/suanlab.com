## Compare measures against one reference

Q is absolutely continuous with respect to P if every P-null event is also Q-null. The direction Q≪P matters. For probability measures satisfying this condition, a nonnegative measurable function w converts Q into weighted P integration. This Radon–Nikodym derivative is a relative mass density, rather than an ordinary coordinate derivative.

$$
Q\ll P\iff P(A)=0\Rightarrow Q(A)=0,\qquad Q(A)=\int_A w\,dP.
$$

## Ratios on a finite space

The defaults are P=(1/2,1/2,0,0) and Q=(1/4,3/4,0,0). Ratios at the first two points are one half and three halves. At the final points, the table leaves the ratio undefined instead of evaluating 0/0. The chosen global version sets those values to zero. Other nonnegative values on P-null points give the same Q. The chosen derivative integrates to one under P.

$$
w_i=\frac{Q_i}{P_i}\quad(P_i>0),\qquad\int w\,dP=Q(\Omega)=1.
$$

## Change the measure of an expectation

For φ(ω)=ω, the default Q expectation is 0.75, equal to the P expectation of wφ. The full-support example gives 1.5 by both calculations. The identity applies to nonnegative functions or signed functions satisfying the relevant integrability conditions. General sigma-finite versions exist, but this lab works with finite probability measures.

$$
E_Q[\varphi]=E_P\!\left[\frac{dQ}{dP}\varphi\right].
$$

## Weights cannot create missing support

In failure mode Q puts mass one quarter at point two while P assigns zero. No weighting of P can create mass there, so no global derivative exists. Ratios on positive-P points remain candidate entries only; the global derivative integral and changed expectation stay undefined. Importance sampling must check this support condition. Even when it holds, finite or small estimator variance needs separate assumptions.

$$
P(\{2\})=0<Q(\{2\})\implies Q\not\ll P.
$$

## Exercises with explanations

1. What is the derivative value on a P-null point? **It depends on the chosen version.**
2. Do ratios on some points guarantee a global derivative? **No: absolute continuity is required.**
3. With densities p,q against the same reference, what is the ratio? **q/p where p>0, subject to the support condition.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
