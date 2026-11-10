## Fisher information averages over the model

For an interior Bernoulli distribution, an outcome y∈{0,1} has probability pʸ(1−p)¹⁻ʸ. The score is the derivative of log probability with respect to the chosen parameter. In probability coordinates,

$$
s_p(y)=\frac{y-p}{p(1-p)},\qquad
\mathbb E_p[s_p]=0,\qquad
I_p=\mathbb E_p[s_p^2]=\frac{1}{p(1-p)}.
$$

The expectation uses the model distribution at the current parameter. A squared score from one observed label is not generally this expectation. At p=0.2, label one gives score five and square twenty-five; label zero gives score −1.25 and square 1.5625. The model-weighted average is 0.2·25+0.8·1.5625=6.25.

## Information changes coordinates with the tangent

Let θ=log[p/(1−p)] be the logit. Since dp/dθ=p(1−p), the logit score is y−p and its Fisher information is p(1−p). Different numerical information values do not mean different statistical geometry:

$$
I_\theta=\left(\frac{dp}{d\theta}\right)^2 I_p,
\qquad I_p\,dp^2=I_\theta\,d\theta^2.
$$

The metric coefficient must be read together with the tangent coordinate. A large probability-coordinate coefficient near a boundary reflects coordinate stretching; interpreting it as universally greater learning difficulty ignores the parameterization.

## Local KL supplies the quadratic metric

For a small probability displacement Δ, the KL expansion is

$$
D_{KL}(\operatorname{Ber}(p)\|\operatorname{Ber}(p+\Delta))
=\frac12 I_p\Delta^2+O(\Delta^3).
$$

The lab keeps query probabilities interior. Blue is exact n-sample KL for independent observations; orange is the Fisher quadratic. The diagram zooms to q within 0.01 of p and scales the vertical axis to the local KL values. Their agreement is local, not an identity at every displacement. A stable log remainder removes the analytically canceling linear terms before numerical evaluation.

At p=0.5,Δ=0.01,n=1, exact KL is approximately 0.000200040 and the quadratic is 0.0002. At p=0.2 the asymmetry of the distribution makes the cubic correction more visible.

## Independence and regularity matter

For n independent observations with the same parameter, scores add and zero-mean cross terms vanish, giving total information nI. Correlated observations require their joint likelihood instead. Support changes, boundary parameters and singular models can invalidate familiar score or Hessian identities unless their regularity conditions are checked.

The lab's label control changes the observed score and outer product. It does not change true Fisher or the KL curve. The count control scales independent information and KL, while still showing one illustrative observed label in the table.

## Exercise and explained answer

Set p=0.2 and n=10. Total probability-coordinate Fisher is 62.5, while per-sample logit Fisher is 0.16. Switch the label from one to zero and verify only observed-score quantities change. Explain why an empirical gradient outer-product matrix computed using dataset labels can differ from a model-expectation Fisher matrix, even when both are positive semidefinite.

Further reading: [Martens, New Insights and Perspectives on the Natural Gradient Method](https://www.jmlr.org/papers/v21/17-678.html).
