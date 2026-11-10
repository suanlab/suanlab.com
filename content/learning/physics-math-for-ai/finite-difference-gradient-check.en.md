## A derivative check is another numerical algorithm

A finite-difference check compares a proposed derivative with changes in independently evaluated function values. It is valuable for detecting a missing factor, incorrect chain-rule path or wrong reduction. However, the check itself has truncation error, rounding error and assumptions about differentiability. A green check is evidence under those assumptions, not a proof of every gradient implementation.

The lab uses exp(x), a smooth reference, and |x|, which is nonsmooth at zero. A bias control deliberately corrupts the proposed gradient. For |x| at zero, the proposed zero is explicitly an implementation convention, while the mathematical derivative readout remains absent.

## Balance truncation against cancellation

The forward and centered formulas are

$$
D_f(h)=\frac{f(x+h)-f(x)}h,\qquad
D_c(h)=\frac{f(x+h)-f(x-h)}{2h}.
$$

Taylor expansion gives first-order forward and second-order centered truncation error at sufficiently smooth points. Decreasing h initially helps. Yet subtracting nearly equal rounded function values amplifies evaluation errors after division by h. Eventually making h smaller can make the check less accurate.

The plot uses log10(h) horizontally and shows the actual two derivative estimates vertically. It does not label a symmetric estimate as a true derivative when differentiability has failed. Consult the readout for mathematical derivative, actual represented steps and the discrepancy against the proposed gradient.

## A nominal step can disappear

At x=1 in binary64, adding 10^-18 produces exactly the same represented x. Subtracting that step also leaves the represented value unchanged. Both difference numerators vanish, so the computed derivatives are zero even though exp'(1)=e.

This is why the lab reports (x+h)-x and x-(x-h), not only the requested h. Dividing by the nominal h cannot recover a perturbation that never reached the function evaluation. Useful gradient checks should try a range of steps and consider absolute as well as relative tolerances, especially near a true zero derivative.

## A nonsmooth false pass

At x=0, |h|=|-h|, making the centered difference zero for every positive h. It agrees perfectly with the proposed zero convention. Nevertheless,

$$
\lim_{h\downarrow0}\frac{|h|-0}{h}=1,\qquad
\lim_{h\uparrow0}\frac{|h|-0}{h}=-1.
$$

No classical derivative exists. A symmetric stencil can therefore hide a kink rather than validate a gradient. At a nonzero x, a sufficiently large step can cross the kink and invalidate the local smooth approximation even though the derivative at x exists.

## Exercise: diagnose before changing the code

Use exp(0.5) with h=10^-4 and add gradient bias 0.25. The centered check discrepancy is approximately 0.25, identifying the injected error. Next use exp at one with h=10^-18: a large discrepancy now comes from the checker arithmetic rather than a wrong analytic derivative. Finally use |x| at zero and obtain exact agreement. Explain why these three outcomes require different diagnoses and why testing only one step at one point is inadequate.

Further reading: [JAX differentiation and numerical checks](https://docs.jax.dev/en/latest/automatic-differentiation.html). The vanished-step and kink examples are independently evaluated here.
