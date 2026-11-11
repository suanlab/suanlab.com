## A finite answer can have an overflowing formula

Suppose three class logits are (c,c-g,c-2g), and divide them by a positive temperature T before forming probabilities. For c=1000, g=T=1, the answer is a perfectly ordinary distribution. Nevertheless, directly evaluating exp(1000) exceeds binary64 range. Dividing infinite exponential values by an infinite sum does not recover the desired probabilities.

The opposite failure occurs near c=-1000: all three naive exponentials round to zero, and their normalization becomes zero divided by zero. These are arithmetic failures of an expression, rather than evidence that the statistical model has no probability distribution.

## Move the reference before exponentiation

Write z for the scaled logits and m=max(z). Factoring out exp(m) gives

$$
\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m},\qquad
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}.
$$

All shifted exponents are nonpositive. At least one exponential is exactly exp(0)=1, so the denominator cannot vanish. This permits a stable log partition function without first constructing the potentially unrepresentable partition function itself.

Shifting does not make every tiny probability representable. When a nonmaximal logit is sufficiently far below the maximum, its shifted exponential can still underflow. The lab leaves such genuine zero rounded terms visible. An invalid naive calculation is shown as absent, rather than silently replaced by an apparently valid all-zero distribution.

## Work through the three-class example

For gap one and temperature one, shifted exponentials are (1,e^-1,e^-2). Their sum is approximately 1.503215, giving probabilities approximately (0.665241,0.244728,0.090031). The same probabilities appear for offsets zero, plus 1000 and minus 1000, although naive arithmetic fails in the last two cases.

The stable log-sum-exp changes with the offset: it is c+log(1+e^-1+e^-2). Thus common-shift invariance belongs to normalized probabilities, not to the log partition value itself. Keep the meaning of the displayed quantity explicit when comparing implementations.

## Temperature is a different operation

A smaller temperature increases relative logit gaps and concentrates the distribution. It is not another harmless common shift. At zero gap all three probabilities equal one third for every positive temperature and offset, which is a useful invariant for testing a classifier head.

For an observed class k, cross-entropy is LSE(z)-z_k. Computing the loss this way avoids taking the logarithm of a tiny probability that has already rounded to zero. Stable formulas should be applied before information is destroyed, rather than patched after an invalid result appears.

## Convexity and perturbation bounds from the probability weights

Differentiate L(z)=logΣexp zᵢ. Its gradient is p and Hessian is diag(p)−ppᵀ. For any vector v,

$$
v^T\nabla^2L\,v=\sum_i p_iv_i^2-(\sum_i p_iv_i)^2
=\operatorname{Var}_p(v_i)\ge0,\qquad \nabla^2L\,\mathbf1=0.
$$

So L is convex and has zero curvature along a common shift. This does not mean L itself is shift invariant: L(z+c1)=L(z)+c. If |Δzᵢ|≤ε, factoring exp(±ε) bounds |L(z+Δz)−L(z)|≤ε. The stable arithmetic and the mathematical conditioning are separate reasons that this computation can be reliable.

For these three structured logits, set r=g/T. Compute the relative logits (0,−r,−2r) before subtracting nearly equal absolute logits. The leading-class loss is log1p(exp(−r)+exp(−2r)). It can remain positive when both the displayed LSE and its largest logit round to the same number. Probabilities use the same requested relative gaps; raw absolute logits remain separate diagnostics.

| r | Leading-class loss |
|---|---|
| 0 | log3 |
| 1 | log(1+e⁻¹+e⁻²) |
| 50 | approximately e⁻⁵⁰ |

```python
from math import exp, log1p, fsum
for r in (0., 1., 50.):
    weights = (1., exp(-r), exp(-2*r))
    p = tuple(x/fsum(weights) for x in weights)
    loss = log1p(weights[1]+weights[2])
    v = (1., -2., .5)
    mean = fsum(a*b for a, b in zip(p, v))
    quadratic = fsum(a*(b-mean)**2 for a, b in zip(p, v))
    assert quadratic >= 0 and loss > 0
    if r == 50:
        assert 1000.+loss == 1000.
        assert abs(loss/exp(-50)-1) < 1e-14
    print(r, loss, "Hessian quadratic:", quadratic)
```

The code checks the Hessian quadratic form as a variance and retains a tiny loss at offset 1000. It does not claim probabilities are recoverable from already-rounded equal absolute logits. Also, finite raw exponentials can have an infinite sum; that naive normalization is invalid and must not be displayed as an all-zero probability vector.

## Exercise: normalized and unnormalized quantities

Set all three logits equal to 1000. The probabilities are exactly one third in the real-number model, while log-sum-exp equals 1000+log(3). Now add ten to every logit: probabilities stay fixed and the log-sum-exp increases by ten. With temperature two, adding ten to the unscaled logits instead increases log-sum-exp by five. Explain the change using the order of operations.

Further reading: [SciPy logsumexp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html). The controlled examples and failure-state distinctions are derived here.
