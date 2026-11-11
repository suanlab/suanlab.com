## Which average is being taken?

Training error averages already observed examples. Population risk averages over the joint distribution of future inputs and labels. They remain different quantities even with identical loss functions. Here x=0,1,2,3,4 are equally likely. Inputs at least two usually have label one, while smaller inputs usually have label zero. The label flips with probability epsilon. Because this teaching world is fully specified, population risks can be computed exactly rather than estimated by a test simulation.

The six hypotheses are h_t(x)=1{x≥t}, for t=0,...,5. Thresholds zero and five are constant-one and constant-zero classifiers. Blue shows actual sample risk, orange population risk, and markers show a separately selected query hypothesis rather than necessarily the ERM.

## Define risk and ERM

$$
R(h)=E_{(X,Y)\sim P}[1\{h(X)\ne Y\}],\qquad
\widehat R_S(h)=\frac1N\sum_{i=1}^N1\{h(X_i)\ne Y_i\}.
$$

ERM minimizes empirical risk over the declared class. Ties choose the smallest threshold. Changing the class or tie rule can select a different model from the same data. A concentration inequality for one hypothesis fixed independently of data cannot simply be transferred to a data-selected ERM. A finite-class union bound or another complexity argument is needed.

## Population oracle in this world

Since noise is below one half, threshold two predicts the more likely label at every input. Direct enumeration gives

$$
R(h_t)=\epsilon+(1-2\epsilon)\frac{|t-2|}{5},\qquad R(h_2)=\epsilon.
$$

Even this oracle cannot remove intrinsic label noise. A realized sample ERM can nevertheless have empirical error below or above epsilon. The inequality E[min Rhat]≤min E[Rhat] concerns expectations, not every dataset. Forcing every observed generalization gap to be positive would distort the data. The readout keeps its signed value.

## From uniform control to an oracle inequality

Suppose the whole fixed class satisfies |R(h)−Rhat(h)|≤e on one event. Let h* minimize population risk and let the learner return an approximate ERM with empirical suboptimality at most a. Insert the two empirical risks between the population risks:

$$
R(\widehat h)-R(h^*)\le [R(\widehat h)-\widehat R(\widehat h)]+a+[\widehat R(h^*)-R(h^*)]\le2e+a.
$$

The event controls the data-selected hypothesis because it holds simultaneously over the class. For M fixed [0,1]-loss hypotheses and IID data, a two-sided Hoeffding union bound gives e=sqrt(log(2M/delta)/(2N)). This is an upper bound, possibly larger than the loss range, rather than the exact observed gap.

In the realizable case with a zero-risk member and exact consistency, a different argument is faster. A fixed hypothesis of risk above e survives N labels without a mistake with probability at most (1−e)^N≤exp(−Ne). Union-bound those bad hypotheses to obtain an excess-risk guarantee at scale log(M/delta)/N. Noise or failure to find a consistent member invalidates that step; it cannot be inferred merely from a low training loss.

The auxiliary two-input world below has Y=X, equal input masses and three increasing thresholds. The smallest-threshold ERM is wrong only when all observations equal one. Its failure probability is exactly 2^(−N), independently enumerated rather than inferred from a bound.

| N in the auxiliary world | Probability ERM has positive excess |
|---|---:|
| 1 | 1/2 |
| 2 | 1/4 |
| 4 | 1/16 |

```python
from itertools import product
from fractions import Fraction as F
for N in (1,2,4):
    failure=F(0)
    risk=[F(1,2),F(0),F(1,2)]
    for sample in product((0,1),repeat=N):
        empirical=[F(sum(int(x>=t)!=x for x in sample),N) for t in range(3)]
        chosen=min(range(3),key=empirical.__getitem__)
        uniform=max(abs(a-b) for a,b in zip(risk,empirical))
        assert risk[chosen]<=2*uniform
        if risk[chosen]>0:failure+=F(1,2**N)
    assert failure==F(1,2**N)
    print(N,failure)
```

The current six-threshold lab uses a different five-input world and actual seeded realizations. Its known population curve is a teaching oracle. Selecting a class, stopping rule or repeated test-set query after inspecting data requires its own simultaneous control or independent evaluation. The deterministic generator supplies reproducible IID-model illustrations; its finite seed set is not the probability space of these theorems.

## Worked example and exercise

Without noise, population risks from threshold zero through five are [0.4,0.2,0,0.2,0.4,0.6]. At N=1 and seed seven, the actual sample is (x=0,y=0). Thresholds one through five all have zero training error, so the tie rule selects one. Its population risk is 0.2. This is a minimal example of zero training error without zero generalization error.

Exercise: compute risks for t=0 and t=2 when noise is 0.1. Solution: threshold zero disagrees with the majority label on two inputs, giving 0.1+0.8·2/5=0.42. Threshold two has risk 0.1. A single sample's observed label proportions are not these exact expectations.

## Connection to AI evaluation

Real distributions are unknown, so evaluation requires held-out data, an explicit procedure and uncertainty assessment. This exact population curve is an oracle for a declared finite world. Repeatedly choosing classes or hyperparameters from test results creates further selection bias. Record the input law, fixed class, actual samples, selection rule and evaluation target before interpreting a learning curve or applying a generalization bound.

References: [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) and [MIT statistical learning materials](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/). Text and finite examples are original.
