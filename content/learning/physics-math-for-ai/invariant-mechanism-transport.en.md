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
