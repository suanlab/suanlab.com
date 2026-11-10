## The generating distribution and the scoring distribution

Suppose outcomes come from P, while a predictor reports probabilities Q. Assigning cost −log₂ Q(x) to observation x gives cross-entropy after averaging under P. Unlike entropy of P itself, this quantity depends on Q as well. Classification training connects this to increasing the probability assigned to correct labels.

$$
H(P,Q)=-\sum_xP(x)\log_2Q(x),\qquad D_{\rm KL}(P\Vert Q)=\sum_xP(x)\log_2\frac{P(x)}{Q(x)}.
$$

## Split cross-entropy into two terms

Expanding the log ratio gives H(P,Q)=H(P)+KL(P||Q). For fixed target P, entropy stays constant as Q changes, so minimizing cross-entropy minimizes this directed KL.

At P=Bernoulli(.8), Q=Bernoulli(.5), every outcome costs one bit. Cross-entropy is one, H(P)≈0.72193 and extra cost is about 0.27807 bits. Matching Q to P removes the extra cost, while entropy itself remains.

## Two rules for zero probabilities

A term with P(x)=0 contributes zero to the average. If P(x)>0 but Q(x)=0, cross-entropy and KL are mathematically infinite. Probability clipping replaces the original scoring rule with another one; this lab applies no such clipping.

At p=.8,q=0, the predictor excludes a success that occurs with positive target probability. At p=q=0, success is impossible under P too, and KL is zero. Inspect target support as well as zero predictive masses.

## Plot window versus full readouts

Blue is the fixed H(P); orange is cross-entropy across q. The curve uses q from .005 to .995, avoiding a finite-looking drawing of infinite endpoints. A selected q outside this window has no plotted dot, but its full value or infinity remains in the readout. Denominators are not silently replaced by small constants.

## Direction and assumptions

KL(P||Q) averages under P and generally differs from KL(Q||P). It is not a symmetric metric satisfying a general triangle inequality. Empirical dataset log loss also differs from population expected cross-entropy; generalization and calibration require separate evaluation.

## Exercises with explanations

1. Does H(P) change when only Q is trained? **No.**
2. Is cross-entropy zero whenever P=Q? **No: entropy can remain.**
3. Is an infinite KL necessarily numeric overflow? **No: support failure can produce mathematical infinity.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
