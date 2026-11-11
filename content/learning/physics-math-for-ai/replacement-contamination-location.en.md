## Start with a target, not a suspicious point

Imagine that a sensor measures a location near zero. Most readings lie between minus two and two, but a recording fault changes some readings to one hundred. We want a location estimator that describes the original process. The mean of the corrupted observations is a legitimate summary of those observations, but it may be a poor estimate of the original location. Robustness starts by stating which target should remain meaningful after corruption.

This experiment fixes the clean target at zero and uses21 original observations. It is deterministic: no random sampling, confidence level or automatic outlier detector is hidden behind the diagrams. The clean lattice is symmetric, so its mean and median both equal zero.

## Specify the replacement mechanism

For indices i from0 through20, the original values are

$$
x_i=(i-10)/5.
$$

Replace the largest m ORIGINAL values by the same location z. Choosing negative z still replaces those originally largest observations. It does not switch to replacing the smallest ones. This distinction explains why positive and negative replacements can have different effects on the median.

An orange mark records a replacement operation, even when the replacement value happens to equal its original value. The mechanism and the resulting numerical difference are different facts. The displayed fraction m/21 is a finite sample replacement fraction; it is not the mixture probability in an iid population contamination model.

## Compare four estimators

The sample mean minimizes squared residual loss and gives each observation the same linear contribution. The sample median, here the eleventh ordered observation, minimizes absolute residual loss. A symmetric k-trimmed mean sorts the data, removes k values from EACH end, and averages the remaining21−2k observations.

Trimming is based on observed order. It does not know which records were corrupted. Contamination placed near the center can survive trimming, and removing too few upper extremes leaves other extreme values in the average. The clean target being known for this demonstration does not give the estimators oracle access to replacement labels.

## Derive the Huber score equation

For a positive fixed physical threshold c, define

$$
\rho_c(r)=\begin{cases}r^2/2,&|r|\le c,\\c|r|-c^2/2,&|r|>c.\end{cases}
$$

The Huber location minimizes the sum of these losses over residuals x_i−theta. Differentiating with respect to theta gives the negative score sum, so a minimizer satisfies

$$
\sum_i\operatorname{clip}(x_i-\theta,-c,c)=0.
$$

Small residuals act quadratically; large residuals contribute only a signed amount c to this equation. That limits the direct force of an individual distant reading. It does not remove that reading, identify its origin, or make the estimate insensitive to the number of replacements.

## Understand scale and uniqueness

The threshold has the same physical units as the observations. In this laboratory the scale is known and fixed. If all observations are multiplied by a positive factor, multiply c by the same factor to obtain a corresponding scaled estimate. Keeping c fixed while changing measurement units changes the procedure.

The general numerical helper retains both endpoints of the Huber minimizer interval. With an even sample, a zero-score plateau can exist: observations−3 and3 with c=1 have every location in[−2,2] as a minimizer. Our odd sample of21 has no such plateau because a region with all residuals clipped cannot have equally many positive and negative contributions. The displayed numerical interval therefore collapses to a unique root up to numerical precision.

## Work the one-replacement guide

Set m=1,z=100,k=1,c=2.5. The removed original value is2, and the remaining20 clean values sum to−2. The sample mean is

$$
\overline x=(100-2)/21=14/3.
$$

The eleventh ordered value remains zero. Trimming one value at each end removes−2 and100, leaving a symmetric clean subset with trimmed mean zero. For the Huber root, the outlier contributes2.5 while the20 retained residuals remain inside the quadratic region. Thus−2−20theta+2.5=0, giving theta=1/40. Check that every retained residual really is within2.5 before using this equation.

## Approach and cross the majority boundary

At m=10,z=100,c=2.5 the median is still zero and the Huber estimate is8/5. At that root, six retained residuals clip to−2.5 and five sum to−10. Their total−25 balances ten outlier contributions of2.5.

At m=11 the median becomes100. Every retained clean residual clips to−2.5, while the eleven repeated values have residual100−theta. Solving−25+11(100−theta)=0 gives theta=1075/11. A bounded individual score has not prevented a corrupted majority from moving the estimate close to the outliers.

## Prove breakdown rather than reading a bounded slider

Replacement breakdown asks for the smallest fraction of observations that can be replaced to make an estimator arbitrarily large under unrestricted replacement. One replacement sends the mean to infinity, giving breakdown1/21. The median needs eleven replacements: with at most ten, its middle order statistic remains between retained finite clean values; eleven identical replacements going to infinity move the median with them.

A fixed k-trimmed mean breaks at(k+1)/21. Send k+1 replacements to positive infinity: trimming k from the top leaves one divergent value. With at most k replacements, any untrimmed contaminated value is bounded between clean order statistics. The known-scale Huber location also needs an eleven-observation majority in this odd sample: a smaller contaminated side cannot balance the clipped force of the clean majority at a sufficiently distant candidate location.

The z slider stops at a finite bound. These algebraic paths and bounds establish breakdown; the finite drawing only illustrates them.

## Read the figures and numerical evidence

The sample diagram shows original-to-current displacement and the four location estimates on a shared adaptive position axis. The final graph fixes m,k,c and varies z; blue is the sample mean and orange the Huber location. The median and trimmed mean remain available in the diagram and table. The horizontal axis is a replacement position, not time or an optimization trajectory.

Means and trimmed means are summed on an exact integer lattice before division. This preserves a true symmetric zero without suppressing small nonzero estimates. Huber roots are numerical and accompanied by a score residual. An independent test solver splits the score into affine pieces between residual knots, rather than duplicating the production bisection algorithm.

## Research extension: Contamination can make two clean targets indistinguishable

A deterministic replacement path illustrates an estimator's behavior. A population lower bound asks a stronger question: can any estimator recover the clean target uniformly over a declared contamination class? Consider an auxiliary class of distributions on $\{-1,1\}$, with the clean mean as target, and contamination fraction $0<\varepsilon<1/2$. Set $d=\varepsilon/(1-\varepsilon)$ and

$$
P_0=((1+d)/2,(1-d)/2),\quad P_1=((1-d)/2,(1+d)/2),\quad
Q_0=(0,1),\quad Q_1=(1,0).
$$

The clean means are $-d$ and $d$, but direct substitution gives the same observed law:

$$
(1-\varepsilon)P_0+\varepsilon Q_0
=(1-\varepsilon)P_1+\varepsilon Q_1=(1/2,1/2).
$$

Consequently every iid sample size has the same observable distribution in the two worlds. For any estimate $T$, including a randomized one with the same randomization law in both worlds, $|T+d|+|T-d|\ge2d$ pointwise. Integrating under the common sample law proves

$$
\max_{j=0,1}\mathbb E_j|T-\mu_j|\ge d,\qquad
\max_{j=0,1}\mathbb E_j(T-\mu_j)^2\ge d^2.
$$

For squared loss, average the two squared expressions to get $T^2+d^2$. More observations cannot remove this identification floor. It does not say every robust problem has these constants: the distribution class, target and contamination model are part of the result.

| Population contamination | Absolute-error floor | Squared-error floor |
|---|---:|---:|
| 1/10 | 1/9 | 1/81 |
| 1/5 | 1/4 | 1/16 |
| 1/3 | 1/2 | 1/4 |

The code checks the two mixture laws and enumerates length-three samples for an illustrative sample-mean estimator. The universal lower bound comes from the pointwise inequalities, not from checking that one estimator. This auxiliary population experiment is distinct from the fixed 21-observation replacement mechanism and the later uniform-reference location functional.

```python
from fractions import Fraction as F
from itertools import product
for epsilon in [F(1,10),F(1,5),F(1,3)]:
    d=epsilon/(1-epsilon)
    P0=[(1+d)/2,(1-d)/2]; P1=P0[::-1]
    Q0=[F(0),F(1)]; Q1=Q0[::-1]
    R0=[(1-epsilon)*p+epsilon*q for p,q in zip(P0,Q0)]
    R1=[(1-epsilon)*p+epsilon*q for p,q in zip(P1,Q1)]
    assert R0==R1==[F(1,2),F(1,2)]
    means=[-P[0]+P[1] for P in [P0,P1]]
    assert means==[-d,d]
    absolute=[F(0),F(0)]; squared=[F(0),F(0)]
    for sample in product([-1,1],repeat=3):
        estimate=F(sum(sample),3); weight=F(1,8)
        assert abs(estimate+d)+abs(estimate-d)>=2*d
        for j,mean in enumerate(means):
            absolute[j]+=weight*abs(estimate-mean)
            squared[j]+=weight*(estimate-mean)**2
    assert max(absolute)>=d and max(squared)>=d*d
    print(epsilon,d,d*d)
```

## Exercises and worked solutions

1. How many values remain with k=3? Solution:21−6=15. Removing three from each tail is not removing three in total.
2. Why is the one-replacement mean14/3 rather than100/21? Solution: the original value2 was removed, so the net sum change is98.
3. With ten large negative replacements, must the median stay zero? Solution: no. Under the stated replacement mechanism it becomes−2, still bounded by the clean range. Resistance to divergence is not zero bias.
4. Does increasing c always improve robustness? Solution: no. A larger c permits a larger outlier score and approaches squared-loss behavior on a fixed bounded sample.
5. What is the breakdown of trimming one from each end? Solution:2/21, because two diverging upper replacements leave one after trimming.
6. Does a nonunique Huber root indicate a failed solver? Solution: not necessarily. The two-point example has a genuine flat interval of optimal values.

## Connect to research without overstating the example

[Chen, Gao and Ren](https://arxiv.org/html/1511.04144v2) study robust inference under an explicit population contamination model. Such statistical guarantees require distributional classes, a loss and a sampling experiment. This fixed one-dimensional demonstration supplies none of those general minimax conclusions. Regression leverage, unknown scale, heavy-tailed clean laws and high-dimensional structure require further theory. The next unit separates local influence, finite population bias and global breakdown.
