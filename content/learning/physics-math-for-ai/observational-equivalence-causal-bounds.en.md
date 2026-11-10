## More observations do not reveal every missing outcome

With binary treatment X and outcome Y, each case has potential outcomes Y0 and Y1. Consistency says that the observed outcome is the one corresponding to the treatment received: Y=Y_X. We do not observe the opposite-treatment outcome for that same case. Even an exact population table P(X,Y) need not determine its distribution.

An effect is point identified when every model satisfying the stated assumptions and observed distribution gives the same effect. A partially identified effect instead has a set of compatible values. This is a statement about information, before considering the sampling uncertainty of an estimated table.

## Write the observed constraints explicitly

Let pi=P(X=1), r0 be the outcome rate among X=0, and r1 the rate among X=1. In a zero-mass treatment group its declared rate has no observational contribution. The four observed cell probabilities, indexed by 2X+Y, are

$$
(p_{00},p_{01},p_{10},p_{11})=
((1-\pi)(1-r_0),(1-\pi)r_0,\pi(1-r_1),\pi r_1).
$$

These fix E[Y0|X=0] and E[Y1|X=1] wherever the corresponding groups have positive mass. They leave the opposite-treatment means free. No conditional exchangeability, random assignment or monotonicity is assumed in this chapter.

## Derive the sharp consistency bounds

Write a=E[Y1|X=0] and b=E[Y0|X=1]. Because outcomes are binary, each unknown mean lies in [0,1]. Therefore

$$
ATE=\pi r_1+(1-\pi)a-(1-\pi)r_0-\pi b.
$$

Minimize with a=0,b=1 and maximize with a=1,b=0:

$$
L=p_{11}-p_{01}-\pi,\qquad
H=p_{11}-p_{01}+1-\pi.
$$

The interval has width one on the binary outcome-difference scale. It can be narrower than the logical range [-1,1], but it is not generally a single value. These bounds are sharp: compatible populations actually attain both endpoints. They are not merely loose inequalities obtained from a numerical optimizer.

## Construct every value while keeping the table fixed

The lab sets hidden U equal to natural treatment X. Conditional on U=0, assign mean Y0=r0 and mean Y1=lambda. Conditional on U=1, assign mean Y0=1-lambda and mean Y1=r1. Generate the two potential outcomes from independent Bernoulli noises within each U stratum, then observe Y_U.

This is an explicit compatible completion, not an identification assumption we learned from data. Its observed table is unchanged for every lambda in [0,1], while

$$
ATE(\lambda)=L+\lambda.
$$

Thus the blue curve covers the entire sharp interval. The eight response-type rows (U,Y0,Y1) show how hidden population mass changes without changing observed cells. This particular coupling is one construction; the bounds allow other couplings too.

## Sixteen extreme completions

For each observed cell (X,Y), choose its missing opposite-treatment outcome to be either zero or one. Four binary choices give sixteen completion masks, including duplicates when cell masses vanish. A bit for an X=0 cell chooses Y1; a bit for an X=1 cell chooses Y0. The selected extreme ATE is computed from those assignments, not selected from the continuous curve by visual approximation.

The lower endpoint uses mask 12: missing Y1 is zero in untreated cells and missing Y0 is one in treated cells. The upper uses mask 3, reversing those choices. Intermediate mixtures of compatible extreme populations fill the interval as well.

## Worked exercise and solution

Take pi=0.4, r0=0.7 and r1=0.2. Calculate the crude difference and bounds.

**Solution.** The crude difference is -0.5. Cell masses are (0.18,0.42,0.32,0.08), giving L=0.08-0.42-0.4=-0.74 and H=0.26. Both negative and positive causal effects are compatible with the exact same observations. At lambda=0.5, the completion effect is -0.24. The crude contrast is not identified as the causal effect merely because both treatment groups occur.

As another example, observed X=Y with pi=0.5 gives bounds [0,1]. One completion has Y0=Y1=U and effect zero; another has Y0=0,Y1=1 and effect one. Both produce the same perfect observational relationship.

## What additional information changes

Randomization, justified exchangeability, restrictions on selection or response, and additional experiments can shrink an identified set. They require their own assumptions and evidence. A confidence interval addresses uncertainty in an estimated observed distribution; an identification interval addresses what that distribution can determine even when known exactly. In an AI study, report these separately rather than presenting precision in prediction as identification of intervention effects.

## Primary reference

Richardson, Hudgens, Gilbert and Fine, [Nonparametric Bounds and Sensitivity Analysis of Treatment Effects](https://arxiv.org/abs/1503.01598), Statistical Science, 2014: partial identification and sensitivity frameworks, with the classical Robins–Manski bounds discussed therein. The displayed construction, mask enumeration and numerical exercises are original teaching examples.
