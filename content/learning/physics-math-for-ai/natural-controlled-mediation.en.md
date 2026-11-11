## Different meanings of a direct effect

A treatment can change an outcome both directly and through a mediator M. Holding M fixed asks what happens under a joint intervention. Letting M take the value it would naturally have had under another treatment asks a cross-world question. These contrasts need not agree when treatment and mediator interact. Define the reference worlds before interpreting a regression coefficient as an effect.

The lab uses a known structural model with independent, mean-zero mediator noise E_M in {-0.5,0.5} and outcome noise E_Y in {-0.25,0.25}. Each noise pair has probability one quarter. It provides an explicit cross-world coupling; it is not a claim that arbitrary observational data identify all these effects.

## Build four potential outcomes

For treatment x and controlled mediator m,

$$
M_x=ax+E_M,\qquad Y(x,m)=cx+bm+dxm+E_Y.
$$

For each noise state compute Y(0,M0), Y(1,M0), Y(0,M1), Y(1,M1). Keep the same noise in all worlds. The total effect for changing treatment from zero to one compares Y(1,M1) with Y(0,M0).

## Pair the natural effects correctly

The pure natural direct effect holds the mediator at M0. The total natural indirect effect changes M0 to M1 while holding treatment at one:

$$
PNDE=E[Y(1,M_0)-Y(0,M_0)]=c,
$$

$$
TNIE=E[Y(1,M_1)-Y(1,M_0)]=a(b+d).
$$

Alternatively, change the mediator at treatment zero to obtain PNIE=ab, and then change treatment while retaining M1 to obtain TNDE=c+ad. Thus

$$
TE=c+ab+ad=PNDE+TNIE=PNIE+TNDE.
$$

These identities telescope within every noise state before averaging. Interaction makes the two indirect effects different; there is no universal indirect effect equal to ab without specifying its reference treatment.

## Controlled direct effects

A controlled mediator value m gives CDE(m)=c+dm. The blue curve varies this intervention value; orange shows PNDE=c. The latter averages over natural M0, whose mean is zero in this model. At nonzero interaction, a fixed m can yield a different magnitude or sign from the natural direct contrast.

The plot compares two estimands. It does not claim that CDE(m) plus a single universal indirect effect reconstructs TE for every m. The table and separate readouts carry the correctly paired natural decompositions.

## The mediation formula needs more than randomized treatment

For measured baseline C, the familiar nested-counterfactual functional is

$$
E[Y_{x,M_{x'}}]=\sum_{c,m}E[Y\mid X=x,M=m,C=c]P(M=m\mid X=x',C=c)P(C=c).
$$

One derivation conditions on C and M_x'. A cross-world independence such as Y_xm independent of M_x' given C separates their expectations; consistency, treatment and mediator exchangeability, and support then replace them by the displayed observational terms. Independent structural errors can supply the required relationships in an appropriate graph. Randomizing X alone does not supply mediator–outcome exchangeability or rule out a treatment-induced mediator–outcome confounder.

An explicit counterexample makes the missing information visible. Let U and randomized X be independent fair bits, M0=U and Y(0,m)=0. In world A, set M1=1−U and Y(1,m)=1{m=U}. In world B, set M1=U and Y(1,m)=1{m≠U}. Both worlds have uniform factual M and Y=0 in both treatment arms, including positive support for every (X,M) cell. Yet E[Y(1,M0)] is one in A and zero in B.

| Quantity from the common law or completion | Value |
|---|---:|
| Observational mediation functional | 0 |
| Nested mean in world A | 1 |
| Nested mean in world B | 0 |

```python
from fractions import Fraction as F
from itertools import product
observed=[];nested=[]
for world in (0,1):
    law={(x,m,y):F(0) for x,m,y in product((0,1),repeat=3)}
    target=F(0)
    for u in (0,1):
        m0=u;m1=1-u if world==0 else u
        y1=lambda m:int(m==u) if world==0 else int(m!=u)
        target+=F(1,2)*y1(m0)
        for x in (0,1):
            m=m0 if x==0 else m1;y=0 if x==0 else y1(m)
            law[x,m,y]+=F(1,4)
    observed.append(law);nested.append(target)
assert observed[0]==observed[1] and nested==[1,0]
assert all(sum(observed[0][x,m,y] for y in (0,1))>0 for x,m in product((0,1),repeat=2))
assert sum(mass*y for (x,m,y),mass in observed[0].items())==0
print('same observed law, different nested means:',nested)
```

This counterexample has shared mediator/outcome dependence and is distinct from the live independent-noise interaction model. It shows why a nested effect needs its identifying assumptions, not that the mediation formula is invalid when those assumptions hold. Interventional mediator-distribution effects ask different questions and need their own estimands and adjustment conditions.

In the current model, analytical integration of centered noises now computes c, a(b+d), ab, c+ad and c+dm directly. `expanded` retains averages of differences between the rounded atom outcomes. Tiny effects can survive even when those displayed statewise differences round to zero. The decomposition residuals remain independent arithmetic checks rather than being forced to zero.

## Worked exercise and solution

Set a=1,b=0.5,c=0.2,d=0.5. Compute both decompositions and CDE(-1).

**Solution.** TE=0.2+0.5+0.5=1.2. PNDE=0.2 and TNIE=1.0 form the first sum. PNIE=0.5 and TNDE=0.7 form the second. CDE(-1)=0.2-0.5=-0.3, despite positive PNDE. This is possible because fixing a negative mediator value differs from averaging its natural baseline value.

Now remove interaction by setting d=0. The two natural indirect effects both become 0.5, both natural direct effects become 0.2, and CDE(m)=0.2 for every m. The familiar linear path-product formula is recovered as a special case.

## Definition is not observational identification

The structural equations let us calculate nested counterfactuals directly. Estimating them from a study requires assumptions about treatment–outcome, treatment–mediator and mediator–outcome confounding, treatment-induced confounders, consistency, support, and the required cross-world relationships. Randomizing treatment alone does not automatically resolve every mediator–outcome problem. Measurement timing and what it means to set the mediator also matter.

For an AI system, a proposed internal representation is not automatically a manipulable mediator. Specify the intervention on M and which other computation mechanisms remain invariant before calling an ablation a controlled or natural effect.

## Primary reference

Pearl, [Direct and Indirect Effects](https://ftp.cs.ucla.edu/pub/stat_ser/r273-jsm05.pdf), ASA/JSM version of report R-273. The finite-noise interaction model and calculations here are original examples, not copied empirical results.
