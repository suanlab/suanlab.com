## Start with a choice that can actually be made

Suppose a system must choose one value from Z={0,1}, or from Z={0,1,3}. Given a target t, the cost of option i is its squared distance from that target. The exact problem is small enough to enumerate:

$$
c_i=(z_i-t)^2,\qquad c_* = \min_i c_i,\qquad z_{\rm hard}\in\arg\min_{z_i\in Z}c_i.
$$

The lab chooses the lowest index when several options tie. This makes the answer deterministic, but it does not make the optimum unique. Orange outlines mark every minimizing option; the orange dot marks the one returned by that tie rule. There is no sampling in this experiment.

A neural network may find it useful to produce a smooth output instead of a discontinuous argmin. That changes the mathematical object being computed. We will track that change rather than use the same word “solution” for every output.

## Relax the decision into a probability distribution

Let p be a distribution over the K options. The probability simplex requires nonnegative coordinates whose sum is one. With positive temperature tau, add an entropy reward to the expected cost:

$$
F_\tau(p)=\sum_i p_i c_i-\tau H(p),\qquad
H(p)=-\sum_i p_i\log p_i,\qquad 0\log0=0.
$$

Low expected cost favors good options; the entropy reward favors distributing mass. The optimum is the Gibbs distribution and its objective value is the soft minimum:

$$
p_i^\tau=\frac{e^{-c_i/\tau}}{\sum_j e^{-c_j/\tau}},\qquad
S_\tau=-\tau\log\sum_i e^{-c_i/\tau}.
$$

All logarithms here are natural. Costs and temperature must use compatible units so that c_i/tau is dimensionless. The teaching model uses dimensionless costs. Temperature is a smoothing parameter here, without asserting that the decision system is a physical thermal system.

## Why this is the optimum, rather than just a formula

For any distribution q, substitution of the Gibbs log probabilities gives

$$
F_\tau(q)-S_\tau=\tau\sum_i q_i\log\frac{q_i}{p_i^\tau}
=\tau D_{\rm KL}(q\Vert p^\tau)\ge0.
$$

Thus the Gibbs probabilities minimize the entropy-regularized objective. This certificate concerns a distributional problem. It does not say that the Gibbs mean is a feasible answer to the original discrete problem. In a larger optimization problem, one also has to establish whether an implemented relaxation is convex and solved accurately; a smooth expression alone proves neither.

[Boyd and Vandenberghe, Convex Optimization, Example 3.25](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) gives the log-sum-exp/negative-entropy conjugacy underlying this identity. The finite option sets, diagrams and numerical examples below are constructed for this lab.

## Four quantities that should not be merged

Write mu for the mean of the option value under p. The exact minimum c_*, the soft minimum S_tau, the expected discrete cost E[c], and the loss of the mean (mu−t)^2 answer different questions. For squared cost,

$$
\mathbb E[(Z-t)^2]=(\mathbb E[Z]-t)^2+\operatorname{Var}(Z).
$$

To derive it, write Z−t=(Z−mu)+(mu−t), expand the square, and use E[Z−mu]=0. The cross term vanishes. This makes the difference between an average prediction and an actual random decision explicit.

The weighted mean lies in the convex hull of the options, but need not lie in Z itself. For an affine cost, evaluating at the mean equals the expected cost. For this nonlinear squared cost, they differ by the variance. A mean may look attractive precisely because it has removed the variability that an actual discrete choice must retain.

## Worked example: zero mean loss is not zero discrete cost

Choose Z={0,1}, target t=0.5 and tau=0.5. Both choices have cost 0.25, so the Gibbs probabilities are exactly 1/2 and 1/2. The mean is 0.5, giving zero loss if the mean is inserted into the squared-error expression. However, 0.5 is not an available option. Choosing either actual option costs 0.25, and a random choice has the same expected cost.

The variance is 0.25, accounting for the entire difference. The entropy is log2. Consequently,

$$
S_{0.5}=0.25-0.5\log2\approx-0.096574.
$$

This negative number is an entropy-rewarded objective value. Squared errors have not become negative. The second diagram puts the four quantities on one horizontal scale, including a zero reference, so that this distinction is visible.

## Bounds and what cooling guarantees

Factoring out the minimum cost gives a stable expression and immediate bounds:

$$
S_\tau=c_*-\tau\log\sum_i e^{-(c_i-c_*)/\tau},\qquad
c_*-\tau\log K\le S_\tau\le c_*.
$$

Each exponential is at most one and at least one equals one. The sum is therefore between one and K. As tau tends to zero, the soft minimum approaches the exact minimum value. Convergence of this value does not, by itself, establish equality of decision outputs or their derivatives.

With a unique minimizer, its probability approaches one. With M tied minimizers, each retains mass 1/M. This is a limit of Gibbs distributions with fixed costs, not a statement about annealing schedules in a changing learned model. A simultaneous change in target and temperature can produce different limiting probabilities near a tie.

## Worked example: the zero-temperature tie

With Z={0,1,3} and t=2, the costs are 4,1,1. At positive temperature the probability of option0 becomes small as the temperature decreases. In the analytical zero-temperature mode the probabilities are exactly (0,1/2,1/2). The mean is2, outside the discrete option set. Its loss is zero, while expected cost and exact minimum are both1. Variance is1.

The deterministic hard answer is1. The Gibbs limiting mean is2. Both statements are consistent because they describe different objects. In limit mode the temperature field is ignored; the displayed curve still sweeps positive temperatures and has no current-temperature markers. This prevents a positive-temperature plot from masquerading as an exact zero-temperature computation.

## Gradients: useful information with a specific meaning

For positive temperature, differentiating the normalized probabilities with respect to the target gives

$$
\frac{\partial p_i}{\partial t}=\frac{2}{\tau}p_i(z_i-\mu),\qquad
\frac{\partial\mu}{\partial t}=\frac{2}{\tau}\operatorname{Var}(Z).
$$

These are derivatives of the relaxed probabilities and mean. The hard answer is locally constant away from decision boundaries, so its derivative is zero there. At a tie boundary it jumps and has no ordinary derivative. In analytical-limit mode the mean is also locally constant away from ties, and its derivative is unavailable at ties. The lab uses a missing value there instead of inventing a zero or an infinity.

Small temperature is not automatically an easy training regime. Near a tie the relaxed derivative can become large; away from ties mass can saturate and gradients can become tiny. A straight-through rule assigns a surrogate backward derivative to a hard forward operation. That surrogate should not be reported as the true derivative of argmin.

## Numerical implementation and reading the diagrams

The probability bars are located at the actual option values, including the wider spacing from1 to3. The hard choice and mean use the same horizontal coordinate system on separate rows. The four-cost diagram compares distinct quantities on a shared cost axis. The final plot shows expected cost versus temperature in blue and exact minimum in orange.

Minimum-cost shifting prevents large positive exponentials. Very small positive probabilities can still underflow to floating-point zero. We retain finite log probabilities at positive temperature and report an underflow count. Analytical-limit zero mass instead has an unavailable log probability. Labels are rounded; a mean displayed near1 is not necessarily exactly1 or a mathematical discrete choice. Numerical membership is a diagnostic of the computed number, not proof about all real arithmetic.

## Exercise and explained solution

Exercise: with Z={0,1,3}, target1.25 and tau0.1, identify the exact choice before using the controls. Then raise the temperature. Does rounding the mean to an integer always produce an allowed, optimal choice?

Solution: the three costs are1.5625,0.0625,3.0625, so option1 is uniquely optimal. Relative to its weight, the other weights are exp(−15) and exp(−30). The low-temperature mean is close to, but below,1. As temperature increases, mass spreads. Integer rounding is not a general repair for an irregular option set: an integer2 is not even available. Selecting a feasible nearest option and certifying its optimality are separate operations. This lab directly enumerates the exact optimum; it does not claim a general rounding guarantee.

Additional exercise: derive the soft-minimum lower bound using the shifted exponential sum. At a two-option tie, compute the relaxed mean derivative for tau0.5 and tau0.01. The variance is0.25, so the derivatives are1 and50. The hard-choice derivative at that same boundary is unavailable in both cases.

## Connecting to AI research

Relaxations appear in discrete latent variables, architecture search, routing, assignments and structured prediction. Record the original feasible set, the relaxed objective, what is executed at inference, and how a gradient is obtained. Measure the final hard decision rather than only the attractive relaxed loss. Claims about a larger constrained problem require its own feasibility and approximation analysis.

[Jang, Gu and Poole's Gumbel-Softmax paper](https://arxiv.org/abs/1611.01144) studies differentiable random relaxations of categorical sampling. Our experiment uses deterministic Gibbs probabilities from costs. It implements neither Gumbel noise nor a straight-through estimator; those methods require their own bias and sampling analysis.
