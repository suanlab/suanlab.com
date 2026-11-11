## Learning structure differs from fitting parameters

Parameter learning fixes graph structure and estimates probability tables. Structure learning compares edges. This lab compares only two observed binary models: independent M0:p(A)p(B), with two free parameters, and M1:p(A)p(B|A), with three. It does not search over arbitrary graphs.

## MLE from sixteen rows

Default counts are (5,3,3,5). Both marginals are .5, so independent MLE gives four masses .25. The edge model fits the empirical joint counts divided by sixteen. Sum count times log probability only for positive counts; zero-count terms contribute zero.

$$
\ell(\theta)=\sum_{a,b}n_{ab}\log p_\theta(a,b).
$$

The larger model contains independence, so its optimized likelihood cannot be smaller. Better training likelihood alone establishes neither generalization nor correct structure.

## BIC penalizes additional parameters

Use BIC=−2ℓMLE+k ln N, where smaller is preferred. The default small dataset selects independence after the additional-parameter penalty. Repeating its proportions eight times produces an edge-model win because likelihood gains scale with N while the penalty scales with ln N. This constructed count input does not justify inflating independent sample size by reusing real observations.

## Keep smoothing separate from scoring

Pseudocount α is added to both outcomes in each conditional row. Thus P(B=1|A=1)=(n11+α)/(n10+n11+2α). The smoothed table illustrates prediction; BIC always uses MLE. Changing α should not change BIC. Do not substitute a smoothed fit while calling it the maximum-likelihood score.

## Direction and causal limits

A complete two-variable observed law can factorize in either direction. Association alone does not identify a causal arrow. BIC is an asymptotic approximation tied to candidate models and regularity/sample assumptions, not an automatically justified score for every boundary or latent-variable setting.

## Likelihood gain is empirical mutual information

For complete binary counts n_ab with N>0, let p̂_ab=n_ab/N. The independent MLE matches each empirical marginal separately, giving p̂_a p̂_b. The one-edge model can fit the full empirical joint. Assuming positive marginals and interpreting zero-count terms as zero, their log-likelihood difference is

$$
\ell_1-\ell_0
=\sum_{a,b}n_{ab}\ln\frac{\widehat p_{ab}}{\widehat p_a\widehat p_b}
=N\widehat I_{\mathrm{nats}}(A;B).
$$

Since these candidates differ by one free parameter, BIC₁−BIC₀=ln N−2NÎ. The edge model wins exactly when 2NÎ>ln N. If MI is measured in bits, multiply it by ln 2 first. Reversing the factorization direction preserves the same positive joint family and its three parameters, hence the same likelihood and BIC. This score cannot identify the arrow direction.

| N, proportions (5,3,3,5) | 2NÎ, nats | BIC₁−BIC₀ |
|---|---|---|
| 16 | 1.010686 | 1.761903 |
| 32 | 2.021372 | 1.444364 |
| 128 | 8.085489 | −3.233459 |

The code checks both these fair-marginal examples and unequal counts (1,2,3,7). Assuming an independent MLE of .25 for arbitrary data would produce the wrong identity and scores. All three count patterns on the screen are deliberately restricted to fair marginals.

## Check the approximation behind BIC

In a regular k-dimensional model, suppose log likelihood falls as a nondegenerate negative quadratic with scale N near an interior MLE, and the prior density is smooth and positive there. The local evidence integral has width N⁻¹/² in each direction. Its logarithm is ℓMLE−(k/2)ln N+O(1); multiplying by −2 yields BIC. This Laplace explanation uses fixed dimension, nondegenerate curvature, and sampling/prior assumptions. Singular latent models or boundary maxima do not automatically retain that dimension penalty. The screen's perfect-match counts give a boundary MLE, so distinguish evaluating the displayed formula from justifying it as an asymptotic Bayes approximation.

**Exercise:** May bit-valued MI=.1 and N=100 be inserted unchanged into 2NÎ? **Explanation:** No. Natural-log likelihood gives gain 2×100×.1×ln 2.

Additional source: [Schwarz (1978), the asymptotic model-dimension criterion](https://sites.stat.washington.edu/courses/stat527/s14/readings/ann_stat1978.pdf). The MI identity for these two candidates is expanded directly here.

```python
from math import log

def scores(counts):
    N = sum(counts)
    p = [n/N for n in counts]  # index a+2*b
    pa = [sum(p[a+2*b] for b in [0,1]) for a in [0,1]]
    pb = [sum(p[a+2*b] for a in [0,1]) for b in [0,1]]
    q = [pa[i%2]*pb[i//2] for i in range(4)]
    ll1 = sum(n*log(v) for n,v in zip(counts,p) if n)
    ll0 = sum(n*log(v) for n,v in zip(counts,q) if n)
    mi = sum(v*log(v/w) for v,w in zip(p,q) if v)
    assert abs((ll1-ll0)-N*mi) < 1e-11
    forward = [pa[i%2]*(p[i]/pa[i%2]) for i in range(4)]
    reverse = [pb[i//2]*(p[i]/pb[i//2]) for i in range(4)]
    assert max(abs(a-b) for a,b in zip(forward,reverse)) < 1e-15
    return N, 2*N*mi, log(N)-2*N*mi

for repetitions in [1,2,8]:
    N,gain,delta = scores([n*repetitions for n in [5,3,3,5]])
    print('N, twice log-likelihood gain, BIC(edge)-BIC(independent):',
          N, round(gain,6), round(delta,6))
scores([1,2,3,7])  # Unequal marginals: the independent MLE is not uniform.
```


## Exercises with explanations

1. How many edge-model parameters? **Three.**
2. Does α affect this BIC? **No: it affects only the separate smoothed table.**
3. Do perfect matching observations identify causal direction? **No: additional assumptions are required.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
