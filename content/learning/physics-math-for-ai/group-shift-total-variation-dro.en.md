## Changing group proportions changes risk

Define a world with group labels zero and one. Training group-one probability is alpha and target probability is beta. Conditional labels stay fixed; only group proportions change. Classifier zero always predicts zero, classifier one always predicts one, and classifier two predicts an independent fair coin. Its group loss .5 is expected randomized loss, not one realized coin outcome.

Actual training group counts select the sample ERM. Blue shows selected-classifier risk across target group proportions, orange the robust choice, and markers identify current beta. Distinguish oracle risk in the fully declared world from risk identifiable from observed samples.

## Importance weights require support

For training group probabilities P_g and target probabilities Q_g,

$$
R_Q(h)=E_P\left[\frac{Q_G}{P_G}\ell(h,G)\right]
$$

requires positive training mass wherever target mass is positive, together with the maintained conditional-label assumptions. When P_g=0<Q_g, the ratio is undefined and loss there is not identified from samples alone without additional model assumptions. Importance estimation is then unavailable. The oracle risk remains computable only because the whole teaching world is specified; it does not resolve the observational identification problem.

Even with support, large weights can create high estimation variance. One finite weighted average need not equal target risk exactly. Unbiasedness concerns expectation over repeated training samples.

## Total-variation ambiguity

For two-point distributions, TV distance is the absolute difference in group-one mass. Around a declared population center alpha,

$$
q\in[\max(0,\alpha-\rho),\min(1,\alpha+\rho)],\qquad
R_q(h)=(1-q)\ell_{h0}+q\ell_{h1}.
$$

Risk is affine, so its maximum occurs at an interval endpoint. Compute both endpoints and choose the least-worst of the three classifiers. This ambiguity set is a declared model, not an automatically data-certified confidence set. A target outside the interval is not covered by its robustness claim.

## Worked example and exercise

At alpha=.9, always-one has training population risk .1. Moving to beta=.1 raises target risk to .9. Radius .8 gives interval [.1,1]. Worst risks are one for always-zero, .9 for always-one and .5 for fair prediction, so the finite-class robust choice is fair prediction.

Exercise: compute weights for alpha=.9 and beta=.1. Solution: group zero receives .9/.1=9, while group one receives .1/.9=1/9. The rare training group gets a large weight. With alpha=1 and beta=0, the first denominator is zero and must not be replaced with an invented large number.

## Interpret AI robustness claims

Report the allowed shift, conditional-label assumptions and how the radius was obtained. Worst-case risk belongs to the declared ambiguity set and class, not every conceivable real-world change. Record training averages, target oracles, importance estimates, finite-sample uncertainty and minimax objectives separately. A transparent toy calculation helps identify those requirements without granting universal robustness.

References: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) and [MIT learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/).
