## A mechanism can stay fixed while an average changes

A model validated in one population may be applied to another with a different mix of cases. If treatment effects vary by group, even invariant within-group mechanisms do not imply an invariant population-average effect. Transport asks which source information and target measurements suffice under explicitly stated similarities and differences.

The lab uses a measured binary group Z, treatment X and outcome Y. Environment E changes the distribution of Z. An optional E→Y edge represents a changed outcome mechanism. E is an environment indicator, not conditioning on a selected collider S=1.

## Specify source and target worlds

Let source and target group-one probabilities be s and t. Source treatment probabilities are e0,e1. The source outcome mechanism is

$$
q(z,x)=0.1+0.4z+(0.1+0.2z)x.
$$

The target uses q*(z,x)=q(z,x)+Delta x. At Delta=0 the conditional outcome mechanism is invariant. Independent outcome and treatment innovations given Z remove residual confounding within strata in this declared SCM. Z is assumed measured comparably in both environments.

The source group effects are 0.1 and 0.3. Thus source ATE=0.1+0.2s, while target oracle ATE=0.1+0.2t+Delta. An oracle uses the complete specified model; it is distinct from what source observations can identify.

## Transport from actual observed cells

For a target-positive stratum, calculate both source treatment means from source joint probabilities. Their difference can be averaged using target masses:

$$
\tau_{transport}=\sum_z
\{E_s[Y\mid Z=z,X=1]-E_s[Y\mid Z=z,X=0]\}P_t(Z=z).
$$

This formula needs source support for the stratum and both treatment levels whenever P_t(Z=z)>0, as well as the conditional exchangeability, consistency and outcome-invariance assumptions. The implementation leaves missing source conditional means unavailable. It does not read the oracle q to fill an empty cell or extrapolate a regression into an absent group.

## Unlabeled target data cannot verify outcome invariance

Source observations and target P(Z) determine the reweighting functional under the invariant-effect assumption. They cannot generally test that assumption. Two target worlds q_target(z,x)=q_source(z,x) and q_source(z,x)+Delta_z*x have identical source data and identical target Z distribution, yet different target ATEs. Additional target outcomes, experiments or justified restrictions are needed to distinguish them.

Instead of claiming exact invariance, suppose a substantive sensitivity model bounds |Delta_z|≤d_z. In this example the shifted outcome probabilities remain valid throughout those ranges. Then

$$
\tau_t=\tau_{transport}+\sum_zP_t(z)\Delta_z,\quad
\tau_t\in\left[\tau_{transport}-\sum_zP_t(z)d_z,\ \tau_{transport}+\sum_zP_t(z)d_z\right].
$$

These bounds are sharp for the declared independent shift intervals: choose every Delta_z at the matching endpoint. They are not a confidence interval learned automatically from the source. If probability constraints or coupled shift restrictions excluded those corners, sharpness would require a revised feasibility analysis.

For d0=1/20,d1=1/10, source stratum effects (1/10,3/10) and target group-one mass t, the interval changes with composition:

| Target t | Transported effect | Sensitivity interval |
|---|---:|---|
| 0 | 1/10 | [1/20,3/20] |
| 1/2 | 1/5 | [1/8,11/40] |
| 1 | 3/10 | [1/5,2/5] |

```python
from fractions import Fraction as F
from itertools import product
effect=[F(1,10),F(3,10)];bounds=[F(1,20),F(1,10)]
for t in (F(0),F(1,2),F(1)):
    Q=[1-t,t];center=sum(p*v for p,v in zip(Q,effect));radius=sum(p*v for p,v in zip(Q,bounds))
    candidates=[]
    for signs in product((-1,1),repeat=2):
        shifts=[s*d for s,d in zip(signs,bounds)]
        for z in (0,1):
            q1=F(1,10)+F(2,5)*z+effect[z]+shifts[z]
            assert 0<=q1<=1
        candidates.append(center+sum(p*d for p,d in zip(Q,shifts)))
    assert min(candidates)==center-radius and max(candidates)==center+radius
    print(t,center,min(candidates),max(candidates))
```

The live shift control remains one shared Delta, while this auxiliary sensitivity model permits separate bounded group shifts. Source conditional rates now normalize log cells, so positive source support is retained when a joint product is too small to represent. A genuinely absent source stratum or treatment is still unavailable; no outcome oracle fills it. Mechanism similarity, support, sensitivity restrictions and sampling uncertainty should be reported separately.

## Worked exercise and solution

Set s=0.2,t=0.8, with e0=e1=0.5 and Delta=0. Compute source, transported and target effects.

**Solution.** Source ATE=0.14. The transported effect is 0.2(0.1)+0.8(0.3)=0.26, agreeing with target oracle ATE. The source marginal average does not transfer unchanged, although both conditional mechanisms do. The difference comes entirely from heterogeneous effects and the changed group mixture.

Now set Delta=0.1. Source observations and transported value remain 0.26, but target oracle ATE becomes 0.36. Reweighting population composition cannot correct a changed treatment-response mechanism. The extra E→Y path marks this distinct failure.

## Missing support is another failure mode

Set s=0 and t=1. The source has no Z=1 cases. Even with randomized source treatment, the target group's conditional outcome means cannot be recovered by this nonparametric cell adjustment. The transported readout is an em dash, while the declared oracle remains defined. If the target instead has t=0, only source group zero is required; absence of an irrelevant target-zero group is not a support failure.

The curve varies target composition. Orange is drawn only where adjustment is supported, including an isolated endpoint when that is the only supported composition. Missing portions are not zero-effect estimates.

## Scope of the transport claim

This is a measured-covariate transport example under a specific selection diagram and available observational source cells. General transportability can combine source experiments and target observations in more complex graphs. This reweighting formula is not a universal algorithm. Invariance is a substantive assumption, not proved by two similar predictive scores. Record which mechanisms may change, which measurements are shared, the target population and the source support before transferring an AI evaluation.

## Primary reference

Pearl and Bareinboim, [Transportability across studies: A formal approach](https://ftp.cs.ucla.edu/pub/stat_ser/r372.pdf), report R-372, revised 2018. The binary heterogeneous-effect example and source-cell computations here are original constructions.
