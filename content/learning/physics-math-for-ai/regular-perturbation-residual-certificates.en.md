## 1. Start with a solvable problem, then ask what changed

Imagine a spring that becomes harder to stretch as its displacement grows. A perfectly linear spring has a displacement proportional to the applied force. A weak nonlinear term changes that answer. Perturbation theory asks us to use the solvable spring as a starting point, then compute successive corrections.

The important question is not just whether we can calculate another term. We also need to know which parameter is small, which solution branch we follow, and how far our approximation can be from the actual answer. This chapter builds those checks into a single model.

Our variable x is a real displacement or a one-dimensional model parameter. The coefficients k, λ and h represent stiffness, nonlinear coupling and applied source. The interactive experiment restricts k to positive values and λ to nonnegative values. It describes one convex equilibrium problem, not an evolving field or a phase transition.

## 2. Energy gives a concrete equation

Consider the energy and its derivative:

$$
V(x)=\frac{k}{2}x^2+\frac{\lambda}{4}x^4-hx,
\qquad F(x)=V'(x)=kx+\lambda x^3-h.
$$

Equilibrium means F(x)=0. At λ=0 the solution is x=h/k. When λ is positive, the quartic term penalizes large displacements, so the equilibrium has the same sign as h and a magnitude no larger than |h/k|.

This is also a simple regularized optimization problem: a quadratic cost plus a quartic penalty and a linear reward. The source is an external coefficient, not a randomly sampled observation. We will not identify a stationary point with a probability distribution or a thermal average.

## 3. Why there is exactly one real equilibrium

The derivative of F is uniformly positive:

$$
F'(x)=V''(x)=k+3\lambda x^2\ge k>0.
$$

Thus F increases strictly, runs from negative to positive infinity, and has exactly one real zero x*. The same inequality says V is k-strongly convex, so this equilibrium is its unique global minimum.

For h>0 the root lies between 0 and h/k. For h<0 it lies between h/k and 0. For h=0 it is exactly zero. These are mathematical brackets, not guesses from a graph. The calculator repeatedly bisects the bracket with exact rational arithmetic, using the sign of F to keep the half containing the root.

The final rational interval encloses the root. Its midpoint is only an approximation; the interval endpoints and their residual signs carry the guarantee. The implementation performs at most 48 bisections and may stop earlier when a midpoint is an exact root.

## 4. Small compared with what?

A numerical value of λ does not define weak coupling by itself. If h is large or k is small, even a seemingly small quartic coefficient can matter greatly. When h is nonzero, introduce a linear-response scale s=h/k and write x=s y. The equation becomes

$$
y+g y^3=1,\qquad g=\frac{\lambda h^2}{k^3}.
$$

The dimensionless quantity g compares the cubic force with the linear force at the unperturbed displacement. Two experiments with the same g have the same normalized equilibrium y, even if their physical scales differ. At h=0 we use the exact zero solution instead of dividing by s.

For k=1, λ=1/100 and h=4, the coupling is g=0.16. For k=1/10, λ=1/1000 and h=1, it is g=1. In both cases the printed λ is small; neither normalized problem lies inside the power-series disk derived below.

## 5. Match coefficients rather than solve everything again

Hold k and h fixed and seek a power series in λ. Substitute it into the equilibrium equation and collect equal powers:

$$
x(\lambda)=\sum_{n\ge0}a_n\lambda^n,\qquad
a_0=\frac{h}{k},\qquad
a_n=-\frac1k\sum_{i+j+\ell=n-1}a_i a_j a_\ell\quad(n\ge1).
$$

The first equation reproduces the linear spring. At first order, the cubic term is evaluated using the zeroth-order solution. At second order, corrections from each of the three factors contribute. Every new coefficient is determined by coefficients we already know.

For k=h=1 the expansion begins

$$
x(\lambda)=1-\lambda+3\lambda^2-12\lambda^3+55\lambda^4+\cdots.
$$

The coefficients alternate here because an added positive nonlinear stiffness initially reduces the displacement. Alternating signs alone do not guarantee a useful finite approximation at every λ.

## 6. A hand calculation at second order

Take k=h=1 and λ=1/10. Truncation after the λ² term gives P₂=1−0.1+0.03=0.93. Substitution into the original equation produces

$$
F(P_2)=0.93+0.1(0.93)^3-1=0.0104357.
$$

This is the equation residual. It is not the error P₂−x*. Because F increases, the positive residual tells us that P₂ lies above the root. A plot should show the true crossing to the left of the approximate marker.

The zeroth approximation is 1, the first is 0.9, the second is 0.93, the third is 0.918, and the fourth is 0.9235. Watching this sequence is informative, but a sequence of nearby values does not yet prove accuracy. We next derive a certificate from the original equation.

## 7. Turn the residual into a guaranteed displacement error

Apply the mean value theorem between any real approximation p and the root. Since F(x*)=0 and F'≥k,

$$
|p-x_*|\le\frac{|F(p)|}{k}.
$$

This bound works for every real p in the declared model, including a poor perturbative approximation outside the convergence disk. For the second-order example it guarantees |0.93−x*|≤0.0104357.

Geometrically, F cannot rise more slowly than a line of slope k. A vertical residual therefore limits the horizontal distance to its zero. If k becomes small, the same residual allows a larger displacement error. A residual of 0.001 gives a bound of 0.001 when k=1 and 0.01 when k=0.1.

This argument is specific to a known lower derivative bound. In a general neural network loss or nonlinear field equation, a small gradient may coexist with a flat direction or another stationary point. One must prove an appropriate stability estimate before transferring the claim.

## 8. Energy accuracy is a different quantity

Strong convexity also bounds the energy gap:

$$
0\le V(p)-V(x_*)\le\frac{F(p)^2}{2k}.
$$

To see this, use the strong-convexity lower bound for V(x*) based at p. Rearrangement gives an upper quadratic expression in p−x*, whose maximum is F(p)²/(2k).

The displacement bound and the energy-gap bound have different units and meanings. A small energy difference does not by itself identify every coordinate accurately in a general high-dimensional problem. Our displayed energy certificate follows from this model's explicit curvature assumption.

## 9. Check the algebra at every order

For the truncated polynomial P_N(λ), define its symbolic residual before choosing a numerical coupling:

$$
\mathcal R_N(\lambda)=kP_N(\lambda)+\lambda P_N(\lambda)^3-h.
$$

Coefficient matching makes all powers from λ⁰ through λᴺ vanish exactly. The remaining polynomial has degree at most 3N+1. Its value at the chosen λ must equal the residual computed by direct substitution.

This gives two useful checks. A wrong recurrence usually leaves a nonzero low-order coefficient. A wrong numerical evaluation can disagree with evaluation of the independently expanded residual polynomial. Exact fractions avoid confusing a tiny rounding remnant with an algebraic cancellation.

## 10. The convergence radius comes from the analytic branch

An independent coefficient formula is

$$
a_n=\frac{(-1)^n}{2n+1}\binom{3n}{n}\frac{h^{2n+1}}{k^{3n+1}}.
$$

It follows by Lagrange inversion of y+g y³=1. The ratio of consecutive positive combinatorial factors tends to 27/4. Hence the normalized Taylor series has radius 4/27 for nonzero h.

The same obstruction can be found from the inverse map g=(1−y)/y³. Its derivative vanishes at y=3/2, where g=−4/27. The branch selected by y(0)=1 encounters a complex-analytic obstruction on the negative real coupling axis, even though our experiment only uses nonnegative coupling.

Inside |g|<4/27 the power series converges to this branch. For g>4/27 it diverges as the order tends to infinity. A smooth, unique real solution can therefore exist where its Taylor series about zero fails. The interface labels the boundary separately; this classification does not provide a finite-order error bound there.

## 11. Distinguish two limits

Fix N and let λ approach zero with k and h fixed. The remainder is O(λᴺ⁺¹). This is an asymptotic statement: for sufficiently small coupling, the retained powers capture successive orders.

Now fix λ and increase N. Convergence asks whether the sequence of partial sums approaches the exact solution. This is a different question, controlled here by the radius in g. Neither O notation nor an appealing first few terms proves convergence at the selected input.

The experiment shows exact terms, partial sums, equation residuals, root-distance intervals and proven residual bounds separately. The distance interval comes from the independently enclosed root; it is not an exact symbolic expression for the root error.

## 12. Linear response and conditioning

Differentiate the equilibrium equation with respect to the source:

$$
\frac{dx_*}{dh}=\frac{1}{k+3\lambda x_*^2},\qquad
0<\frac{dx_*}{dh}\le\frac1k.
$$

This susceptibility measures how much the true equilibrium changes when h changes. The nonlinear stiffness reduces that response relative to the linear model. Rational root endpoints give bounds on |x*| and hence bounds on the susceptibility.

The limit k→0 deserves a new analysis. At k=0 and λ>0 the answer scales as the cube root of h/λ, and the formulas dividing by k are invalid. The input intentionally rejects k=0. A boundary of a method's assumptions is an opportunity to change scales, not a reason to keep using a broken formula.

## 13. Experiments to make the distinctions visible

First use the default parameters and increase the truncation order. Compare the marker's distance from the root interval with the residual certificate. Next set λ=1 while keeping k=h=1. The root remains unique, but high-order partial sums eventually grow instead of converging.

Then compare the default input with k=2, λ=1/5, h=2. The entire equilibrium equation has been multiplied by two, so every partial displacement and g stay unchanged. Residuals double; dividing by k leaves the displacement certificate unchanged. This is a useful invariance test.

Finally reverse the source sign. The equilibrium and every partial displacement change sign, while g and the magnitude bounds stay the same. Set h=0 and check the exactly zero branch, then set λ=0 and recover the linear model.

## 14. What to inspect in a visualization

A force graph should show F(x), its unique zero bracket, the chosen partial sum and the vertical residual. An order graph should show signed terms and errors with an explicit scale; a large high-order term must not silently clip the plotted domain.

The energy curve should use the actual quartic energy, with a separate label for an illustrative schematic. Moving a camera or changing a display option must not change the underlying computed answer. Large rational tables should scroll within their panel while preserving the full values.

Use the exact tables to decide the mathematical claims, and use the drawings to see their geometry. A large or small visual marker does not replace the reported interval or certificate.

## 15. Connection to AI research

Suppose a learned equilibrium solver returns p. Report its equation residual, the assumptions establishing stability, and the resulting error bound. Comparing only losses can hide the dependence on k. Varying k tests whether a method handles increasingly soft directions.

A perturbative baseline also exposes what a model learns beyond a local expansion. Compare against equal truncation orders and independent root enclosures, include couplings outside the Taylor disk, and distinguish interpolation accuracy from a mathematical certificate. Keep λ, k, h, order and the evaluation precision in the research record.

This scalar example does not establish global guarantees for neural-network optimization or nonlinear PDEs. Its transferable habit is to connect an approximation to a declared equation, assumptions, a stability estimate and an independent reference.

## 16. Explained exercises

1. **Why does λ=0.001 not always imply weak coupling?** With k=0.1 and h=1, g=1. The ratio of nonlinear to linear force at the unperturbed displacement is order one.
2. **Find the first correction for arbitrary k and h.** Substitute a₀=h/k into the cubic term: a₁=−h³/k⁴. The correction is a₁λ, so coefficients and evaluated terms are different objects.
3. **Does a positive residual place p above or below the root?** Above, because F is strictly increasing. This inference uses monotonicity, not the sign pattern of the series.
4. **What is certified by residual 0.002 at k=1/4?** The displacement error is at most 0.008. The energy gap is at most 0.000008. Neither statement gives the exact error.
5. **Can a series diverge while the physical answer remains well defined?** Yes. At k=h=λ=1 the equation is strictly monotone, but g=1 exceeds the Taylor radius 4/27.
6. **What changes under simultaneous doubling of k, λ and h?** The equation and residual double. The root, every partial displacement, g and |F(p)|/k stay unchanged.
7. **Why not accept k=0 and reuse the certificate?** The proof needs a strictly positive lower slope and divides by k. A new dominant balance is required at the excluded limit.

## 17. References and what comes next

[NIST DLMF §2.1](https://dlmf.nist.gov/2.1) defines asymptotic notation; [§2.2](https://dlmf.nist.gov/2.2) discusses inverse expansions; [§2.11](https://dlmf.nist.gov/2.11) explains why remainder control matters. The recurrence, quartic certificates and concrete radius calculation above are derived for this chapter's declared model.

Next we study a finite integral whose perturbation series diverges for every nonzero positive coupling, then boundary layers and long-time secular effects. Completing this regular example alone does not complete the broader perturbation topic.
