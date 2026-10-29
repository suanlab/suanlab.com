## Multiplication has an additive coordinate

Repeated multiplication appears in growth, decay, probability products and numerical scale. A logarithm asks which exponent produced a positive value. It is an inverse only under explicit conditions on the base and domain; the symbol log does not make every input admissible.

For a>0, integer powers begin with repeated multiplication, a⁰=1 and a⁻ⁿ=1/aⁿ. Rational powers use positive roots. Extension to real exponents requires continuity or an equivalent analytic construction, developed more formally in analysis.

## The conditions for an inverse

For a>0,a≠1, the function aˣ maps the real line bijectively onto positive reals. Its inverse satisfies

$$
\log_a y=x\iff a^x=y,\qquad y>0.
$$

At a=2,x=2, the output is 4 and log₂4=2. When 0<a<1, the exponential is decreasing but still bijective onto positive reals. For a=1/2, (1/2)²=1/4 and log₀.₅(1/4)=2.

At a=1 every exponent gives one, so the inverse is not a function. At y=0 no real exponent produces that output for a valid positive base. A limiting approach toward zero is not an actual zero output at a finite exponent.

## Derive the logarithm rules from powers

Let u=logₐx and v=logₐy for positive x,y. Then xy=aᵘaᵛ=aᵘ⁺ᵛ, so

$$
\log_a(xy)=\log_a x+\log_a y.
$$

Similarly a quotient gives a difference, and logₐ(xʳ)=r logₐx for positive x and the real-power convention. These laws do not distribute over addition: log(x+y) is generally not log x+log y. Taking x=y=1 supplies an immediate counterexample.

Changing base gives logₐy=ln(y)/ln(a). The denominator must be nonzero, exactly the exclusion a≠1. A base close to one can make the inverse numerically sensitive even when it is mathematically defined.

## Natural logarithms and local rates

The natural base e is characterized by the exponential’s derivative matching its value. Writing aˣ=exp(x ln a) and applying the chain rule gives

$$
\frac{d}{dx}a^x=(\ln a)a^x.
$$

The sign of ln a explains increasing versus decreasing behavior. For base one, this derivative is zero, consistent with a constant function; a finite derivative does not restore invertibility.

At a=2,x=2, the derivative is 4 ln2≈2.772589. The lab reports the derivative as a number, not as a claim that the screen’s curve angle has a fixed geometric scale.

## Limits, real domains and dimensional inputs

For a>1, logₐy tends without a finite lower bound as y→0⁺. Writing ln0=−∞ would confuse a limit with an ordinary real value. Negative real inputs also have no real logarithm. Complex logarithms require branch choices and are a different construction.

Arguments of physical exponentials and logarithms need dimensionless combinations. For decay exp(−kt), k has inverse-time units. For a measured length, ln(L/L₀) uses an explicit reference length rather than taking a logarithm of meters by themselves.

## Read the experiment and connect to AI

The graph deliberately shows only the exponential curve. The logarithm and inverse reconstruction are numeric readouts, so unequal plotting scales are not mistaken for a reflection diagram. Changing the probe does not change the underlying function. Base one or log input zero produces an explicit undefined value.

Probability products become sums of log probabilities, which can improve numerical handling. This does not eliminate all underflow or precision issues. Stable log-sum-exp evaluates the logarithm of a sum using a shifted expression; it is not the invalid identity log(x+y)=log x+log y.

For model comparisons, state whether reported likelihoods use natural logs or another base. Changing the log base rescales values; it does not automatically change the maximizing parameter for the same positive likelihood.

## Exercises and reference

1. Compute log₂(1/8). **Answer:** −3.
2. Explain why base 1/2 is valid but base one is not.
3. Refute ln(x+y)=ln x+ln y with positive inputs.
4. Rewrite a physical decay factor so its exponent is dimensionless, identifying every unit.

For exponents and logarithms see [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e). The identities above are derived under the stated real-domain conditions.
