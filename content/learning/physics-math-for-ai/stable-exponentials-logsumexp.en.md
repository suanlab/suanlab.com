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

## Exercise: normalized and unnormalized quantities

Set all three logits equal to 1000. The probabilities are exactly one third in the real-number model, while log-sum-exp equals 1000+log(3). Now add ten to every logit: probabilities stay fixed and the log-sum-exp increases by ten. With temperature two, adding ten to the unscaled logits instead increases log-sum-exp by five. Explain the change using the order of operations.

Further reading: [SciPy logsumexp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html). The controlled examples and failure-state distinctions are derived here.
