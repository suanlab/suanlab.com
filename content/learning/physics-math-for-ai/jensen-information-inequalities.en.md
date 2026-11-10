## Average before or after transforming?

Choose positive u and v with probabilities λ and 1−λ. Transforming their mean gives f(E[X]); averaging transformed values gives E[f(X)]. These differ for nonlinear functions. For a convex function, Jensen's inequality places the first quantity below the second.

$$
f(\lambda u+(1-\lambda)v)\leq\lambda f(u)+(1-\lambda)f(v),\quad 0\leq\lambda\leq1.
$$

## The minus-log curve and its chord

Here f(t)=−log₂t, with f″(t)=1/(t²ln2)>0 on positive inputs. The orange chord lies above the blue curve. At the mean input, chord height is E[f(X)] and curve height is f(E[X]).

For λ=.5,u=.5,v=2, transformed values are one and minus one, so E[f(X)]=0. The input mean is 1.25, giving f(E[X])≈−0.32193 and gap approximately 0.32193. Negative function values do not contradict convexity. Inputs are positive numbers, not necessarily probabilities.

## Equality conditions

Strict convexity gives a positive gap when both weights are positive and u differs from v. Equal inputs or a weight zero or one leave only one active value and produce equality. Negative weights or weights not summing to one do not define the probability average used here.

## Explain nonnegative KL

When Q is positive on positive P support, apply Jensen to Q(i)/P(i) under P. Q can place additional mass outside that support, so its average ratio can be below one. This gives:

$$
D_{\rm KL}(P\Vert Q)=E_P\!\left[-\log_2\frac QP\right]
\geq-\log_2\!\left(\sum_{i:P_i>0}Q_i\right)\geq0.
$$

If Q vanishes on a positive P state, KL is infinite and handled separately. Avoid evaluating ratios at zero P states. This is a general argument with domain and averaging conditions, not a proof inferred from a single graph.

## Connect to learning objectives

Log is concave, reversing the inequality direction. Jensen arguments for log evidence in latent-variable models therefore require care with signs and support. Freely exchanging averages and nonlinear transformations can produce incorrect losses or probabilities. Revisit the earlier ELBO and its KL gap with these conditions in mind.

## Exercises with explanations

1. Is minus log convex? **Strictly convex on positive inputs.**
2. What is the gap for equal inputs? **Zero.**
3. Do negative function values invalidate Jensen? **No: convexity is not the sign of the function.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
