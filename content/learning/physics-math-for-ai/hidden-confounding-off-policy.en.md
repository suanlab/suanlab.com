## A policy value is an intervention question

Logged actions were chosen by a behavior policy. Evaluating another policy asks about outcomes under a changed assignment mechanism. Importance weighting can connect these distributions when its identification assumptions hold. Observing every action in the marginal log is not enough if hidden variables affect both action choice and outcomes.

The lab isolates a one-step decision. It does not implement all the assumptions or estimators needed for sequential reinforcement learning. Exact population expectations remove sampling noise so that structural bias and missing overlap can be examined separately.

## Specify logging and target policies

A hidden group U is Bernoulli(pi). Logging assigns X=1 with probability e_U. The conditional outcome mean is

$$
q(u,x)=0.1+0.6u+0.2x.
$$

The target policy assigns X=1 with probability r independently of U. Its true value in the declared SCM is

$$
V(r)=\sum_{u,x}P(u)\pi_r(x)q(u,x)
=0.1+0.6\,\pi+0.2r.
$$

The diagram shows the logging graph U→X,U→Y,X→Y. The target replaces the U-dependent assignment with a fresh independent policy coin. Merely predicting Y among logged X cases does not perform that replacement.

## What marginal propensity weights estimate

Let e=P_log(X=1). Weighting only by the marginal action probability yields

$$
V_{marg}=E_{log}\left[\frac{\pi_r(X)}{P_{log}(X)}Y\right]
=(1-r)E[Y\mid X=0]+rE[Y\mid X=1],
$$

where the relevant action probabilities are positive. This mixture retains the logged group's composition within each action. It generally differs from the intervention value when U is hidden.

At pi=0.5,e0=0.1,e1=0.9 and r=1, observed X=1 cases are 90% U=1. The marginal weighted value is 0.1(0.3)+0.9(0.9)=0.84. The true target value is 0.6. Infinite data can estimate 0.84 precisely without repairing its bias of 0.24.

## Conditional weighting is an oracle here

If U and its logging propensities were known, and conditional action support held, the correct identity would be

$$
V(r)=E_{log}\left[\frac{\pi_r(X)}{P_{log}(X\mid U)}Y\right].
$$

The conditional weighted readout is explicitly an oracle because this example calls U hidden. It is not an estimator available from action and outcome logs alone. In the declared model, independent outcome noise given U,X provides the required conditional exchangeability; a real dataset would need justification for that assumption and measurement of a sufficient adjustment set.

## Bound a policy value before claiming improvement

If only the exact binary logging law is known and consistency is assumed, the missing outcomes remain free. Its cells p_xy give E[Y0] in [p01,p01+P(X=1)] and E[Y1] in [p11,p11+P(X=0)]. For an external independent Bernoulli policy with probability r of action one,

$$
L(r)=(1-r)p_{01}+rp_{11},\quad
H(r)=L(r)+(1-r)P(X=1)+rP(X=0).
$$

Both endpoints are attainable by assigning every missing potential outcome zero or one; compatible mixtures fill the interval. This is not an assumption that the outcome mechanism is known. The live oracle selects one particular compatible model, whereas these bounds quantify all consistency-only completions.

The default logging law has cells (21/50,2/25,2/25,21/50). Its logging outcome mean is 1/2. Even with exact probabilities and both marginal actions observed, none of the external-coin policies below has a lower bound above that baseline.

| External r | Sharp policy-value interval | Known-model oracle |
|---|---|---:|
| 0 | [2/25,29/50] | 2/5 |
| 1/2 | [1/4,3/4] | 1/2 |
| 1 | [21/50,23/25] | 3/5 |

```python
from fractions import Fraction as F
joint=[F(21,50),F(2,25),F(2,25),F(21,50)]
logging=joint[1]+joint[3];assert logging==F(1,2)
for r in (F(0),F(1,2),F(1)):
    candidates=[]
    for mask in range(16):
        value=F(0)
        for cell,mass in enumerate(joint):
            x,y=divmod(cell,2);missing=(mask>>cell)&1
            y0,y1=(y,missing) if x==0 else (missing,y)
            value+=mass*((1-r)*y0+r*y1)
        candidates.append(value)
    lo=(1-r)*joint[1]+r*joint[3]
    hi=lo+(1-r)*sum(joint[2:])+r*sum(joint[:2])
    oracle=F(2,5)+r/F(5)
    assert (min(candidates),max(candidates))==(lo,hi)
    assert lo<=oracle<=hi and lo<logging
    print(r,lo,hi,oracle)
```

This does not establish that every new policy is harmful; it shows what cannot be certified without restricting the hidden confounding or collecting more information. A justified sensitivity set can narrow the policy-value set and support robust comparisons, but its scope must be explicit. The example is not the full optimizer in the cited confounding-robust policy research.

Marginal action support now uses normalized log weights over hidden strata. A positive but unrepresentable joint action probability no longer falsely becomes absent support; raw masses and nonrepresentable raw weights remain visible as numerical limitations. Conditional support still requires each target action in every positive stratum, and genuinely missing cells remain unavailable. Neither numerical normalization nor marginal overlap removes hidden-confounding bias.

## Exercise: marginal overlap without conditional overlap

Set pi=0.5,e0=0,e1=1 and r=1. Are both logging actions present? Can either weighting scheme recover the target without extra assumptions?

**Solution.** Marginally each action occurs with probability one half. Marginal IPS equals E[Y|X=1]=0.9, while true target value remains 0.6. Group U=0 never receives action one, so the conditional oracle identity lacks support and its readout is unavailable. A model-defined oracle outcome can still be computed, but observed conditional weighting cannot supply the missing cell.

Now set r=0 when both logging propensities are zero. The target uses only action zero, so action-one support is unnecessary. Both relevant weighting identities are supported. Requiring overlap for an action with target probability zero would be an unnecessary restriction.

## Reading the graph and table

Blue shows the exact target policy value; orange shows the marginal-weighted population value wherever defined. A missing portion is not zero. The table distinguishes joint logging masses, target action probabilities and marginal versus conditional weights. Zero-target actions use weight zero. An undefined or nonrepresentable raw weight displays an em dash; expectations are integrated algebraically before forming huge ratios to avoid spurious overflow.

## Connection to robust policy research

Sensitivity analyses can restrict the possible hidden confounding and produce policy-value bounds or robust improvements. Such guarantees depend on a declared uncertainty model and cannot identify effects under unrestricted confounding. Sequential settings additionally require a history/state model, appropriate sequential assumptions and horizon-specific support. This chapter establishes the one-step failure mechanism that those extensions must address.

## Primary reference

Kallus and Zhou, [Confounding-Robust Policy Improvement](https://papers.neurips.cc/paper_files/paper/2018/hash/3a09a524440d44d7f19870070a5ad42f-Abstract.html), NeurIPS 2018. The binary population, weighting calculations and support cases here are original teaching examples, not an implementation of that paper's full optimizer.
