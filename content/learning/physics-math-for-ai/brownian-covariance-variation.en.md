## A finite path display represents an infinite-dimensional law

Standard Brownian motion starts at zero, has continuous paths and independent Gaussian increments with variance equal to elapsed time. The lab draws seeded Gaussian increments on a fine 128-step grid. A coarser grid aggregates those same increments, preserving every shared endpoint rather than generating an unrelated path on each refinement.

Blue joins sampled values. The straight connectors are a rendering choice, not a claim that Brownian paths have piecewise classical derivatives. Orange curves plus/minus sqrt(t) are pointwise one-standard-deviation references. They are neither a simultaneous confidence band nor a deterministic bound that an entire realization must obey.

## Covariance comes from shared increments

For s<=t, write W_t=W_s+(W_t-W_s). The later increment is independent of W_s and has mean zero. Therefore

$$
E[W_t]=0,\qquad \operatorname{Var}(W_t)=t,\qquad
\operatorname{Cov}(W_s,W_t)=\min(s,t).
$$

The covariance controls refer to time fractions of the displayed terminal time T. The reported value is a property of the Gaussian process law, not a covariance estimated from one observed trajectory. Shared noise history explains why values at two times are correlated even though disjoint increments are independent.

## Squared increments behave differently from ordinary variation

On N equal intervals, define QV=sum of squared increments. Each increment has variance T/N, so its square has mean T/N and variance 2(T/N)². Independence gives

$$
E[QV]=T,\qquad \operatorname{Var}(QV)=2T^2/N.
$$

The actual QV is computed from the shown path and generally differs from T. Refining the grid reduces the theoretical fluctuation scale, but the observed values need not approach T monotonically. A single sequence looking favorable is not a proof of the quadratic-variation theorem.

## Keep scaling and sampling separate

At N=16 and T=1, QV variance is 0.125. At N=64 it is 0.03125, with the same path endpoint under this coupling. Doubling T for a fixed seeded increment pattern multiplies path values by sqrt(2), QV by two, and its theoretical variance by four. These are distinct scaling powers.

## Exercise: explain the overlap

For time fractions one quarter and three quarters of T=2, covariance is one half. Derive it by listing which fine increments occur in both sums. Then compare coarse and fine QV values: their endpoint matches exactly, but their squared-increment sums need not. Expanding the square of a sum reveals cross terms, explaining why endpoint preservation does not preserve QV. Finally state why crossing an orange standard-deviation reference is not a model failure.

Further reading: [Lawler's Brownian-motion and variation notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The shared-grid visualization and scaling checks are independently constructed.
