## Model selection changes the evaluation question

An interval constructed for a predictor fixed before evaluation can lose its advertised coverage when the same evaluation outcomes select the predictor. Trying many candidates and reporting the candidate with the fewest observed errors creates an optimistic selection rule. The effect can occur even when every candidate has exactly the same true error probability. This lesson isolates that mechanism with a finite, explicitly specified experiment.

The controls set sample size n, common hypothetical risk p, noncoverage budget alpha, and a predeclared finite candidate count M. A final control queries a possible winning error count. That query is not a simulated random observation and can even be impossible for p equal to zero or one. Requested counts greater than n clamp to n; the display reports the effective value and whether its probability is positive.

## Define the synthetic experiment precisely

Assume each candidate receives its own independent stream of n Bernoulli error indicators with common probability p. The candidate counts are independent and identically distributed. Select a candidate having the smallest count; ties can be resolved by any fixed rule because all true risks are equal.

$$
K_j\stackrel{\mathrm{iid}}\sim\mathrm{Bin}(n,p),\qquad
K_* = \min_{1\le j\le M}K_j.
$$

Real models tested on shared examples generally have correlated errors. Their counts need not satisfy this independence model, so the power formulas derived here cannot be transferred automatically. The construction is a transparent counterexample and an intuition builder, not an estimator of arbitrary benchmark-selection bias.

## Recall what a single upper endpoint guarantees

For a candidate fixed independently of its evaluation sample, a one-sided Clopper–Pearson endpoint U_alpha(K) has coverage at least one minus alpha. When k is less than n, the endpoint solves F_U(k) equal to alpha, where F is the binomial cumulative distribution. At k equal to n the endpoint is one. With no observations it is also one.

$$
\Pr_p\{p\le U_\alpha(K)\}\ge1-\alpha.
$$

At a fixed p, noncoverage is equivalent to the strict condition F_p(K) less than alpha. Equality is covered. This strict inequality matters because the binomial distribution is discrete and a boundary may carry positive probability. A displayed decimal endpoint rounded to four digits should not decide which side of an exact boundary the mathematical event belongs to.

## Derive selected coverage rather than assuming it

Let b_alpha be the probability, under the fixed binomial law, that F_p(K) is strictly below alpha. Because the endpoint increases with the count, the selected minimum has a covering endpoint exactly when every candidate count avoids that lower-tail failure set. Independence gives

$$
b_\alpha=\sum_{k:F_p(k)<\alpha}\Pr_p(K=k),\qquad
\Pr_p\{p\le U_\alpha(K_*)\}=(1-b_\alpha)^M.
$$

The per-candidate failure probability can be smaller than alpha because of discreteness. Nevertheless, raising the individual coverage to a large power can sharply reduce selected coverage. The blue curve shows this result as candidate count changes. It measures repeated-experiment coverage at the specified p, not the probability that a single observed interval is correct after seeing its endpoint.

## Follow the default numerical example

Use n equal to 40, p equal to 0.1, and alpha equal to 0.05. A fixed candidate has coverage approximately 0.985219. For a queried zero-error result, its numerical upper endpoint is about 0.0721575, below the true value 0.1. A valid confidence procedure can miss on this particular outcome; validity constrains the total probability of misses.

Now increase M from one to 20. Naive selected coverage falls to approximately 0.742432, despite the nominal 95 percent label on the individual construction. The selection process has made unusually low counts much more likely to be reported. This is a property of the repeated experiment, not a claim that every selected interval will miss.

## Correct a predeclared finite family

Construct each candidate's upper interval using noncoverage budget alpha divided by M. The union bound then gives simultaneous coverage, and therefore coverage for whichever candidate is selected:

$$
\Pr\{\exists j:p_j>U_{\alpha/M}(K_j)\}
\le\sum_{j=1}^{M}\alpha/M=\alpha.
$$

This theorem does not require independence between candidate errors or equality of candidate risks. Valid marginal intervals and a correctly accounted finite family suffice. By contrast, the orange curve's exact power calculation still uses the special independent, equal-risk experiment. Separate the general theorem from that example-specific evaluation of its conservatism.

In the default 20-candidate example, the adjusted zero-error upper endpoint is approximately 0.139108. At p equal to 0.1 the adjusted selected procedure has coverage one because no count belongs to its failure set. This is a discrete design-specific result, not a promise that Bonferroni correction always gives coverage one for every possible p.

## Compare with a genuinely fresh holdout

An alternative design selects a candidate using one set of records, then evaluates that fixed selected candidate on new independent records. In the equal-risk experiment, this fresh count again has the original binomial law, and its coverage is approximately 0.985219 for the default n, p, and alpha. This number describes a different experiment; querying another count in the original table does not create new data.

Repeatedly selecting, querying, tuning, and stopping based on the supposed holdout can destroy its independence role. The present finite-family correction does not cover unlimited adaptive queries or unspecified stopping rules. Specialized adaptive-data-analysis methods require their own algorithms and proofs; a larger number typed into the candidate control does not implement them.

## Read the count histogram and its boundaries

If S_k denotes the probability that a single count is at least k, then the minimum-count mass is

$$
\Pr(K_*=k)=S_k^M-S_{k+1}^M.
$$

Blue bars show the single-count distribution; orange bars show the selected minimum. Increasing M pushes probability toward small reported counts while leaving the true p unchanged. The expected selected empirical rate is the sum of k divided by n against this orange distribution, when n is positive. At n equal to zero that empirical quantity is unavailable, not zero; every upper endpoint is one.

The implementation uses exact integer binomial weights for rational percentage p and exact comparisons for failure events and advertised-coverage flags. Plotted probabilities, expected values, and confidence endpoints are numerical. This division prevents a rounded endpoint from silently changing a strict boundary, but it does not certify the sampling assumptions of a real experiment.

## Research extension: Adapt models while keeping each new evaluation conditional

A predeclared finite family is not the only valid design. At research round $j$, choose the model and a fixed batch size using past information $\mathcal F_{j-1}$. Evaluate on genuinely fresh records independent of that history. Conditional on the history, its risk $p_j$ is fixed and the new count is binomial. If the conditional endpoint has failure budget $\alpha_j$, then iterated expectation gives

$$
\Pr\{p_j>U_j\}
=\mathbb E\bigl[\Pr\{p_j>U_j\mid\mathcal F_{j-1}\}\bigr]\le\alpha_j.
$$

Choose $\alpha_j=\alpha/[j(j+1)]$. A countable union bound protects every round, so an arbitrary eventual choice among the evaluated models also has its own valid endpoint:

$$
\Pr\{\forall j\ge1:\ p_j\le U_j\}\ge1-\alpha.
$$

This permits adaptive model construction because each new evaluation is conditionally valid. It does not permit repeatedly tuning on the same supposed fresh batch. Peeking and stopping inside a batch requires an appropriate time-uniform procedure, such as the preceding construction, rather than the fixed-size interval used here.

A concrete reuse counterexample shows why conditioning matters. Let inputs be uniform on ten values, with true label zero everywhere. After seeing three evaluation inputs, choose a classifier that predicts zero exactly on those values and one elsewhere. Its evaluation error is always zero, but its true risk is $1-d/10\ge0.7$, where $d\le3$ is the number of distinct observed inputs. The naive 95% zero-error upper endpoint is $1-0.05^{1/3}\approx0.631597$. It misses in every one of the 1,000 possible input triples. This construction deliberately violates evaluation independence; it is not a failure of the binomial theorem for a fixed predictor.

For genuinely fresh three-record evaluations, conditional risks $0.7,0.8,0.9$ illustrate the valid schedule at $\alpha=1/20$:

| Research round $j$ | Allocated $\alpha_j$ | Largest conditional miss probability in this risk set |
|---|---:|---:|
| 1 | 1/40 | 1/125 |
| 2 | 1/120 | 1/125 |
| 10 | 1/2200 | 0 |

The code enumerates the reuse counterexample and independently calculates each fresh binomial failure probability with exact fractions. The general conditional-coverage theorem is broader than these three risks. The native finite-candidate power curves still describe their stated independent equal-risk experiment; this extension supplies a separate design, not a reinterpretation of those curves.

```python
from fractions import Fraction as F
from itertools import product
from math import comb
misses=0
for batch in product(range(10),repeat=3):
    memorized=set(batch)
    assert all(x in memorized for x in batch)
    risk=1-F(len(memorized),10)
    assert (1-risk)**3<F(1,20)
    misses+=1
assert misses==1000
alpha=F(1,20)
for j in [1,2,10]:
    allocated=alpha/F(j*(j+1)); failures=[]
    for p in [F(7,10),F(4,5),F(9,10)]:
        cdf=F(0); bad=F(0)
        for k in range(4):
            mass=comb(3,k)*p**k*(1-p)**(3-k); cdf+=mass
            if cdf<allocated:bad+=mass
        assert bad<=allocated
        failures.append(bad)
    print(j,allocated,max(failures))
assert sum(alpha/F(j*(j+1)) for j in range(1,11))==alpha*F(10,11)
```

## Worked exercises and solutions

1. **Exercise:** Set n to one, p to 0.9, alpha to 0.1, and M to one. Is zero errors a noncoverage event? **Solution:** No. Its cumulative probability is exactly alpha, its upper endpoint is 0.9, and equality includes the truth. The correct event uses a strict less-than comparison.
2. **Exercise:** Why does changing only the queried k not change the coverage curve? **Solution:** Coverage integrates across every possible sample. The query inspects one endpoint; it does not alter the generating law or selection rule.
3. **Exercise:** Does the union-bound correction fail when candidate counts are correlated? **Solution:** Not solely for that reason. Marginal validity still yields the finite union bound. However, the independent power formula generally fails under correlation.
4. **Exercise:** Query one error when p is zero and n is positive. **Solution:** The interface marks that queried count as impossible under the configured law. An endpoint can be calculated, but it cannot be interpreted as an outcome with positive probability in this experiment.
5. **Exercise:** Explain the difference between nominal coverage and true risk. **Solution:** Coverage is a probability over repeated datasets that an interval contains a fixed parameter. Risk is that parameter: the probability of a prediction error on a new example from the specified population.
6. **Exercise:** Does a new random seed for plotting create a fresh holdout? **Solution:** No. Fresh evaluation requires new independent records according to the sampling design. Neither a display change nor a new count query establishes that independence.

## Sources and research directions

The [NIST exact-binomial interval reference](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm) supports the interval inversion used here. [Dwork and colleagues on reusable holdouts](https://arxiv.org/abs/1506.02629) study a broader adaptive-data-analysis problem. Our finite Bonferroni experiment is not their reusable-holdout algorithm. Research extensions include correlated candidate errors, heterogeneous risks, justified sequential confidence procedures, and evaluation under deployment shift; each requires an explicit experimental design before a coverage claim can be assessed.
