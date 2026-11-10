## Learn from another policy's actions

A behavior policy beta generated logged trajectories. We want the expected return of a different target policy pi. Simply averaging logged rewards estimates the behavior value. Importance sampling changes the contribution of each observation according to how likely its actions would have been under the target. This requires both a probability calculation and a support condition; a plausible numerical answer alone cannot establish validity.

The model here is fully observed, with known common transitions and rewards. It deliberately differs from the earlier hidden-confounding example. It isolates sequential weighting rather than claiming to solve hidden confounding or unknown logging propensities.

## Use the same two-step world

Start at s0=0, choose a0, receive r0=1{a0=0}, transition to s1, choose a1, receive r1=1{a1=s1}, then terminate. Probability of s1=1 is alpha after a0=1 and 1-alpha otherwise. Return is r0+gamma*r1. The target action-one probabilities are p0,p1; behavior probabilities are b0,b1. The path code 4*a0+2*s1+a1 identifies one of eight rows.

Because the environment is given, a separate model-based oracle is available. Writing p=p0, q=p1 and w=1-alpha+p*(2*alpha-1), its value is J=1-p+gamma*((1-w)*(1-p)+w*q). Losing support in the logged data does not erase knowledge supplied by this model. It does prevent certifying some data-reweighting estimators.

## Factor the path ratio

For a target-positive path with positive behavior probability, define rho_t=pi(a_t|s_t)/beta(a_t|s_t). Common transition factors cancel. Full-trajectory importance sampling uses

$$
Z_{\mathrm{IS}}=\rho_0\rho_1(r_0+\gamma r_1).
$$

If every target-positive path has behavior support, summing under beta replaces its path probability Q with Q*(P/Q)=P, proving E_beta[Z_IS]=J. A behavior-only path has target weight zero. A target-positive path with zero behavior probability cannot be repaired by division, epsilon smoothing or silently removing that row.

## Weight each reward by its own prefix

Future action probabilities sum to one conditional on the history. The first reward needs only the first action ratio:

$$
Z_{\mathrm{PDIS}}=\rho_0r_0+\gamma\rho_0\rho_1r_1.
$$

The implementation gives sufficient support certificates separately. Full IS requires full trajectory support. Per-decision IS requires first-action support and, when gamma is nonzero, full support. This conservative certificate is not a necessary characterization for every special deterministic reward: unsupported paths with zero relevant reward can sometimes leave a numerical mean unchanged. Agreement with the oracle still does not prove the displayed generic support condition.

At gamma=0, future support is unnecessary for PDIS. For p0=.8, p1=.6, b0=.5, b1=0, alpha=.8, the oracle and certified PDIS mean are .2. Full-trajectory support fails and its certified mean is unavailable. Summing full weights only over observed support gives diagnostic .176, a different quantity. This is a concrete reason to stop a reward's ratio at its own prefix.

## A worked clipping calculation

With p0=4/5, p1=3/5, b0=b1=1/2, alpha=4/5, gamma=1, direct value and both importance means are 84/125. Their one-trajectory variances are 7184/15625 for full IS and 6994/15625 for PDIS. The ordering in this example is not a universal variance theorem.

Explicit clipping replaces the full weight by min(c,rho0*rho1). At c=1 its expectation is 61/125, giving bias -23/125. Returns are nonnegative here, so the clipped mean cannot exceed the target value when support holds. At c=2 the mean already equals 84/125 even though the largest weight is 2.56: the path being clipped has zero return. Clipping does not always cause strictly negative bias. With missing support, a difference from the oracle includes missing target mass as well as any clipping effect.

## Read support and numerical diagnostics separately

Blue bars show raw path ratios, orange bars their explicit clipping. A dash means the path lacks behavior support; a zero bar can instead mean target probability zero. The marker “ovf” means positive support but a raw ratio beyond the floating-point range. Tiny positive factors remain positive for support checks even when their product rounds to zero.

The model calculates moments with log-scaled terms, avoiding unnecessary overflow from first forming a huge ratio. A genuinely nonrepresentable variance remains unavailable; clipping is never used as a hidden numerical fallback. The lower blue curve is the observed-support clipped diagnostic mean versus cap, while orange is the known target value. Neither is a sampled training curve.

## Exercise and solution

Exercise: if the target never visits state one, must a zero behavior probability for one action at that state invalidate full support?

Solution: no. Only target-positive paths need behavior support. For p0=0, alpha=1, the process stays at state zero; p1 describes an unreachable choice. With b0=0 the realized target path is supported even when b1=0. Its full IS mean is two at gamma=1. Cap .25 changes the clipped mean to .5.

Exercise: does a finite-sample self-normalized importance estimate equal the ratio of population weighted expectations?

Solution: generally no. An expectation of a random ratio differs from a ratio of expectations. This unit enumerates ordinary IS, PDIS and explicit clipping, not a self-normalized finite-sample estimator or a confidence interval.

## Further study

Sequential weighting is a foundation for off-policy reinforcement learning. [Precup, Sutton and Singh, ICML 2000](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf) studies off-policy evaluation with eligibility traces; it is a further-reading reference. The finite path identities, certificates and numerical examples here are derived explicitly above. Extending them to unknown behavior propensities, long horizons, estimated value functions or hidden confounding requires additional assumptions and error analysis.
