## Define the question before choosing an attack

Suppose a binary classifier assigns a score to a two-dimensional input. We want to know whether every allowed change of that input preserves the correct prediction. This universal question differs from asking whether one chosen attack succeeds. In the linear case we can solve the universal question exactly, making it a useful reference for later nonlinear examples.

The score is f(x)=w dot x+b. Prediction is+1 when the score is nonnegative and−1 otherwise. The true label y is either−1 or+1 and is assumed fixed throughout the threat set. This label assumption is part of the mathematical problem; a small numerical distance does not by itself prove that an image, measurement or physical state keeps its meaning.

## Specify the threat set and its units

Allowed perturbations satisfy

$$
\|\delta\|_p\le\epsilon.
$$

This laboratory uses an unconstrained ball in real two-dimensional space. It does not additionally clip the perturbed input to a pixel box. Adding an input box changes the optimization problem and can change the attaining perturbation, so such clipping must never be introduced silently.

Equal numerical epsilon values do not define equal threat sets. The L1 ball is a diamond, the L2 ball a disk and the infinity-norm ball a square. For equal radii in two dimensions they are nested in that order. Coordinate units and preprocessing also matter: changing the scale of one feature changes what a norm bound permits unless the threat model is transformed with it.

## Write the signed margin

The signed margin is

$$
m=y(w^\top x+b).
$$

A positive margin is a correct prediction away from the boundary. A negative margin is an error. At margin zero, the prediction depends on the declared tie convention and the true label. Here score zero predicts+1, so a+1 target is correct there while a−1 target is not.

The perturbed margin is m+y w transpose delta. Only the last term depends on the adversarial choice. The classifier weights, bias, input and true label are fixed during the inner minimization.

## Derive the dual-norm bound and prove it is attained

Holder's inequality bounds an inner product by a product of dual norms. If1/p+1/q=1, it implies

$$
yw^\top\delta\ge-\|w\|_q\|\delta\|_p\ge-\epsilon\|w\|_q.
$$

For the three supported norms we can exhibit a feasible perturbation attaining the lower bound. Therefore this is an equality for the worst-case optimization, not merely a sufficient estimate:

$$
\min_{\|\delta\|_p\le\epsilon}y f(x+\delta)=m-\epsilon\|w\|_q.
$$

For an infinity-norm threat, choose each coordinate as minus epsilon times y times the sign of the corresponding weight. The dual norm is L1. For an L2 threat, choose the negative normalized weight direction multiplied by epsilon y; the dual is L2. For an L1 threat, put the whole budget on a coordinate of maximal absolute weight; the dual is infinity.

If several coordinates tie for the maximal weight, multiple L1 optimizers may exist. The code selects the first maximal coordinate as one attaining witness. If all weights vanish, every feasible perturbation leaves the score unchanged and no normalization by the weight norm is performed.

## Work the unequal threat-set guide

Take w=(3,4),x=(0,0),b=4 and true label+1. The clean margin is4. At epsilon1, an infinity-norm perturbation may change both coordinates by−1, giving margin4−3−4=−3.

With an L2 ball the dual norm is5 and the worst perturbation is(−3/5,−4/5), producing margin−1. With an L1 ball the dual norm is4; the selected witness is(0,−1), producing margin zero. The different outcomes follow from different allowed sets, not from one attack being a more clever optimizer for the same set.

## Treat exact boundary ties honestly

The second guide uses the L1 case above. Its worst score is exactly zero. Every point in the closed ball has nonnegative score, so the+1 target remains correctly classified under the stated positive tie rule. Nevertheless the strict certificate that every signed margin is positive does not hold.

The third guide changes the bias to−4 and the target to−1. The clean signed margin is again4. The worst perturbation(0,1) reaches score zero, which predicts+1 and is now wrong. The same zero worst signed margin therefore has different actual correctness for the two target labels.

A strict positive-margin certificate is sufficient for either label. Actual robustness under a closed ball requires a nonnegative worst margin for a+1 target and a strictly positive worst margin for a−1 target. State the tie rule instead of rounding the boundary into an arbitrary answer.

## Compare radicals without rounding a certificate

L2 witness coordinates and norms may contain square roots. The graphics show decimal approximations, but the logical flags do not compare rounded numbers. All controls use bounded integer or quarter-grid encodings.

Write the encoded clean margin as M/4 and epsilon as E/4. For nonnegative M, comparing M with E times the square root of the sum of squared weights is equivalent to comparing their integer squares. If M is negative, the worst margin is already negative. These bounded integer products remain exactly representable, including equality. A plotted decimal point is not itself proof of a strict inequality.

The mathematical attaining L2 perturbation may be algebraic while its displayed coordinates are rounded. Classification of an exact boundary uses the algebraic margin relation; recomputing a dot product from rounded coordinates can create a spurious tiny sign.

## Interpret a radius and the exceptional cases

At a clean correct nonconstant point, the critical radius is m divided by the dual weight norm. For a+1 target its finite boundary is included by the tie rule; for a−1 target it is excluded. The radius denotes a supremum, so the endpoint convention still needs to be stated.

A correct constant classifier has no finite adversarial boundary and its radius is unbounded. This is displayed with an explicit flag rather than a fake zero or a nonfinite plot value. An already incorrect clean point has no certified correct neighborhood; its reported radius is undefined rather than a negative distance.

The formula concerns this fixed affine model. It does not establish that a trained nonlinear model can realize chosen weights, that the ground-truth label is stable under the threat, or that the perturbations are plausible natural data.

## Read the three diagrams

The geometry diagram draws the chosen ball, the affine decision boundary when one exists, the original input and one worst-case destination. A correct constant classifier has no drawn decision line. A zero-radius ball collapses to its input.

The certificate panel separates clean correctness, exact worst sign, strict certification, actual tie-aware robustness and the meaning of the radius. The last graph varies epsilon with all other parameters fixed. Its straight blue margin line and orange zero reference are not a training trajectory or a probability of error. The circle used to display an L2 ball is a graphical approximation; the formula and exact sign comparison determine the result.

## Exercises and worked solutions

1. With weights(3,4), what are the dual norms for L1,L2 and infinity threats? Solution:4,5 and7 respectively.
2. Why cannot the L1 threat use perturbation(−1,−1) at epsilon1? Solution: its L1 norm is2, so it is infeasible.
3. At zero worst score, does a strict margin certificate hold? Solution: no. Actual correctness still depends on the target and tie rule.
4. If both weights are zero and bias is positive, can changing the input flip the prediction? Solution: no. For a+1 target the correct radius is unbounded; for a−1 target the clean prediction is already wrong.
5. If two L1-optimal coordinates tie, does selecting one prove uniqueness? Solution: no. It only provides one attaining perturbation.
6. Why is clamping a witness to a pixel range a different problem? Solution: it adds constraints absent from the ball used in the derivation and can remove the original optimizer.

## Connect the calculation to research

[Goodfellow, Shlens and Szegedy](https://arxiv.org/abs/1412.6572) discuss linear mechanisms behind adversarial examples. The elementary dual-norm calculation here is derived explicitly for a fixed affine classifier. The next unit uses a small ReLU model to show why a feasible gradient attack and a sufficient safety bound need not solve the same global optimization problem.
