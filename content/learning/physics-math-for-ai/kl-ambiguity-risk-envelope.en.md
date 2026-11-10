## Change probabilities rather than individual inputs

A model can have a low average loss under one distribution and a much higher loss when difficult outcomes become more common. Distributional robustness asks how large the expected loss could become within a specified set of alternative distributions. The uncertainty set is therefore a set of probabilities, not a collection of perturbations applied separately to each input.

This laboratory fixes three outcome losses and the nominal distribution p=(0.2,0.3,0.5). It solves the inner worst-risk problem. In a full learning objective one might minimize this worst risk over model parameters, but the controls here do not secretly train or select a model. Keeping the losses fixed lets us inspect precisely what probability reweighting does.

## Define the direction of the ambiguity constraint

The selected uncertainty set is

$$
\mathcal Q_\rho=\{q\ge0:\sum_i q_i=1,\ D_{KL}(q\Vert p)\le\rho\},
\qquad D_{KL}(q\Vert p)=\sum_i q_i\log(q_i/p_i).
$$

The order q relative to p matters. Reversing that order changes the feasible set and its support behavior. Here every nominal probability is positive, so placing positive alternative mass on any of the three outcomes is permitted at some finite cost. A zero q term contributes zero to KL by its limiting convention.

Natural logarithms make the radius a quantity in nats. It is not the same numerical unit as a transport distance, a pointwise input radius or a differential-privacy parameter. Giving two different uncertainty models the same number does not make their sets comparable.

## State the risk envelope

For a fixed loss vector l, the worst expected risk is

$$
R_\rho(l)=\sup_{q\in\mathcal Q_\rho}\sum_i q_i l_i.
$$

The baseline itself is feasible, so worst risk cannot be below nominal risk. Enlarging the radius enlarges the feasible set, so worst risk cannot decrease. No expected loss can exceed the largest outcome loss. These facts give useful checks before any root finder or diagram is trusted.

The ambiguity set also needs a substantive interpretation. Selecting a radius by hand does not establish that an unknown future distribution belongs to it. A statistical coverage statement would require a sampling experiment and additional assumptions; an optimization solution alone cannot supply that statement.

## Derive a dual upper objective directly

For a positive temperature eta, define an exponential tilt

$$
q_{\eta,i}=\frac{p_i\exp(l_i/\eta)}{Z_\eta},\qquad
Z_\eta=\sum_i p_i\exp(l_i/\eta).
$$

Nonnegative KL between any q and this tilt gives

$$
0\le D_{KL}(q\Vert q_\eta)
=D_{KL}(q\Vert p)-\frac{q^\top l}{\eta}+\log Z_\eta.
$$

Rearrange and use the KL budget to obtain

$$
q^\top l\le\eta\rho+\eta\log Z_\eta.
$$

Thus every positive temperature gives an upper bound on every feasible risk. Minimizing this objective over temperature gives the entropy dual. If the tilted distribution uses exactly the KL budget, both inequalities above become equalities, providing an attaining primal witness and a matching dual value.

## Solve the intermediate-radius equation

Write tau=1/eta. The tilt is proportional to p times exp(tau l). Differentiating its KL along tau gives

$$
\frac{d}{d\tau}D_{KL}(q_\tau\Vert p)
=\tau\operatorname{Var}_{q_\tau}(l).
$$

For nonconstant losses and positive finite tau this derivative is positive. Therefore the entropy budget grows monotonically along the tilt until its limiting maximum-set cost. An intermediate radius can be matched by a one-dimensional bracketed root search rather than an unconstrained search over arbitrary probability vectors.

The implementation centers exponentials by subtracting the largest loss. This changes neither the normalized probabilities nor the mathematical objective and avoids unnecessarily large exponential values. Logarithms, probabilities and the root remain floating-point calculations. Their raw numerical primal/dual discrepancy is displayed rather than converted into an exact machine-proof badge.

## Handle radius zero as a limit

At radius zero, nonnegative KL can vanish only at q=p. The worst risk is exactly the nominal expectation as a mathematical statement. For nonconstant losses, the entropy-dual infimum is approached as eta tends to infinity. It need not have a finite-temperature minimizer.

The first guide uses losses(0,1,3). The nominal risk is0.2 times0 plus0.3 times1 plus0.5 times3, which is1.8. The probability bars remain at the baseline. The dual graph intentionally does not place a finite optimizer marker for the infinite-temperature limit.

## Work the small-radius guide

At radius0.1 nat with the same losses, the numerical maximizing tilt is approximately

$$
q\approx(0.08878080,\ 0.19664092,\ 0.71457829).
$$

Its expected loss is approximately2.34037578. The largest-loss outcome gains probability, while the other probabilities decrease. The nominal risk stays1.8 because the losses and baseline have not changed.

Inspect the calculated KL and raw dual-minus-primal difference together. A very small numerical gap is evidence of a good solve under the chosen numerical checks; it is not interval arithmetic or a statistical confidence level. A tiny negative displayed gap caused by floating arithmetic is retained visibly rather than silently clipped into zero.

## Derive saturation and discuss optimizer selection

Let A contain all outcomes with maximal loss. Restrict the nominal distribution to A and renormalize. This distribution has KL cost

$$
\rho_{sat}=-\log p(A).
$$

At or beyond that radius it attains the maximal possible loss, so worst risk saturates. For losses(0,1,3), the maximizing outcome has nominal probability0.5 and the threshold is log2. Radius1 is already large enough for q=(0,0,1) and risk3.

When several outcomes tie for largest loss, the renormalized nominal distribution is a convenient minimum-KL maximizing witness. Above its threshold other maximizing distributions may also be feasible. The display chooses one witness and does not list the whole optimizer set or establish uniqueness. If all losses are constant, every feasible distribution has the same risk, and the baseline remains a valid selection.

## Read the three visualizations

Blue and orange bars compare baseline and selected alternative probabilities for each outcome. The dual profile plots an upper objective against temperature, with the selected finite temperature marked when appropriate. Temperature zero is interpreted by its maximum-loss limit, not by substituting into a division by zero.

The final graph varies the ambiguity radius while keeping losses and baseline fixed. Blue shows numerical worst risk and orange nominal risk. These are risk envelopes, not training trajectories or probabilities that the model will fail. The radius axis is measured in nats, and interpolation between calculated curve points is a visual approximation.

## Exercises and worked solutions

1. Why can worst risk never be below nominal risk? Solution: q=p is feasible at every nonnegative radius.
2. At zero radius, can a nonconstant loss tilt put extra mass on the worst outcome? Solution: no. KL zero forces q=p.
3. For tied maximal losses with nominal total mass0.7, what is the saturation threshold? Solution: minus log0.7 nats; normalize the nominal probabilities on that set to construct a witness.
4. Does a displayed optimal witness prove uniqueness? Solution: no. Constant or saturated problems can have multiple feasible maximizers.
5. Does a radius0.1 nat imply every outcome probability changes by at most0.1? Solution: no. KL is a joint nonlinear divergence constraint, not a coordinatewise probability bound.
6. If every loss increases by2, what changes? Solution: every feasible expected loss and worst risk increases by2; the maximizing probabilities need not change.
7. What additional evidence is needed for deployment coverage? Solution: an explicit connection between the chosen ambiguity set, the data-generating process and the distribution shift of interest, with statistical assumptions where appropriate.

## Connect to broader research

[Duchi and Namkoong](https://arxiv.org/abs/1810.08750) study robust learning under distributional shifts. The finite entropy derivation here explains an inner risk envelope with fixed losses; it is not their complete learning or finite-sample theory. The next unit changes the ambiguity geometry to transport cost and obtains an exact finite-model primal/dual certificate.
