## A question about the cases we actually observed

A population intervention asks how outcomes would behave if an assignment were replaced across the population. A counterfactual can ask what would have happened to cases with particular factual observations. That evidence changes which exogenous states are plausible. Repeating an intervention on a fresh random individual therefore need not answer the counterfactual question.

Use the binary SCM from the structural-model chapter. Independent exogenous bits U=(Z,E_X,E_Y) generate X=Z XOR E_X. The outcome mechanism is either X XOR E_Y, Z XOR E_Y, or X XOR Z XOR E_Y. These functions specify how the same underlying state responds to alternative actions. They supply more information than the marginal observational distribution alone.

## Abduction: which states explain the evidence?

Let factual evidence be e={X=x_f,Y=y_f}. For each of eight exogenous states, execute the natural assignments and compare them with the evidence. Retain and normalize compatible prior masses:

$$
P(u\mid e)=\frac{P(u)\,1\{X(u)=x_f,Y(u)=y_f\}}{P(e)}.
$$

This is uncertainty about the exogenous state of the observed cases. It is not a new draw from the prior. The table shows prior mass, posterior mass, and factual values for every state. Rows with prior probability zero remain visible but contribute no evidence. If P(e)=0, there is no posterior under this model; the lab displays an em dash instead of inventing a state distribution.

## Action and prediction: share the same U

For action a, replace X's assignment by X_a=a, keep the inferred U and retain the outcome function. Recompute Y_a(u), then average:

$$
E[Y_a\mid e]=\sum_u Y_a(u)P(u\mid e).
$$

The twin diagram shares one latent U between factual and action worlds. The factual X and Y squares are evidence, while X_a is assigned. Resampling E_Y after learning something about it would answer another question. The retained-state construction also implies consistency: applying the factual treatment again gives the factual outcome in every compatible state.

## Randomization identifies marginals, not every cross-world coupling

Suppose randomized treatment is independent of (Y0,Y1), so separate trial arms identify p0=P(Y0=1) and p1=P(Y1=1). They still do not observe both outcomes for one individual. Let t=P(Y0=1,Y1=1). Nonnegativity of all four response-type masses gives the sharp Fréchet interval

$$
\max(0,p_0+p_1-1)\le t\le\min(p_0,p_1),\quad
P(Y_1=1,Y_0=0)=p_1-t.
$$

For p1>0, random assignment also gives P(Y0=0|X=1,Y=1)=(p1−t)/p1. This probability of necessity concerns the actually treated successes under the specified randomized regime. An observationally assigned treatment would require further assumptions before dropping X from that conditional.

Take p0=3/10,p1=3/5. ATE stays 3/10 for every compatible coupling, but necessity ranges from one half to one. The code explicitly constructs all three example joint laws and verifies that their randomized factual (X,Y) tables agree.

| Cross-world t | Benefit probability | Necessity among treated successes |
|---|---:|---:|
| 0 | 3/5 | 1 |
| 3/20 | 9/20 | 3/4 |
| 3/10 | 3/10 | 1/2 |

![Benefit and harm probabilities across compatible cross-world couplings, with a fixed average treatment effect and unchanged randomized observations.](/assets/learning/causal-counterfactual-coupling-en.svg)

```python
from fractions import Fraction as F
p0,p1=F(3,10),F(3,5);baseline=None
for t in (F(0),F(3,20),F(3,10)):
    response={(0,0):1-p0-p1+t,(0,1):p1-t,(1,0):p0-t,(1,1):t}
    assert min(response.values())>=0 and sum(response.values())==1
    factual={(x,y):sum(mass/2 for (y0,y1),mass in response.items() if (y0 if x==0 else y1)==y) for x in (0,1) for y in (0,1)}
    if baseline is None:baseline=factual
    assert factual==baseline
    benefit=response[0,1];harm=response[1,0]
    assert benefit-harm==p1-p0
    print(t,benefit,benefit/p1)
```

Monotonicity Y1≥Y0 would remove harm and force t=p0 in this example. Ordered marginal risks alone do not establish monotonicity. The live XOR SCM supplies one particular cross-world coupling through shared noise; the figure instead varies couplings while retaining their trial marginals.

For extremely rare factual evidence, the implementation now keeps `logEvidence` and normalizes compatible log masses before exponentiating the posterior. Raw `evidence` can underflow to zero while posterior support is positive; the “posterior defined” flag then remains one. Truly impossible evidence has no finite log mass and still yields no posterior. The raw atom probabilities are retained for comparison, not silently floored or replaced by equally likely states.

## A complete numerical example

Choose the chain outcome Y=X XOR E_Y, with pi=0.5, assignment noise 0.1 and outcome noise 0.2. Observe X_f=1,Y_f=1. The evidence implies E_Y=0. Its probability is 0.5×0.8=0.4.

Two states have positive posterior mass: (Z,E_X,E_Y)=(0,1,0) has mass 0.1, and (1,0,0) has mass 0.9. Setting X_a=0 gives Y_a=0 in both. Thus

$$
E[Y_0\mid X=1,Y=1]=0,
\qquad E[Y\mid do(X=0)]=0.2.
$$

The latter average retains the population's 20% outcome-noise rate. The former conditions on evidence that identifies the outcome-noise bit. Their difference is not a computational error or a contradictory prediction.

## Another mechanism, another answer

Switch to the fork outcome Y=Z XOR E_Y. Changing X cannot change Y for a fixed exogenous state. Given factual Y=1, every compatible state therefore has Y_a=1 under either action. The population intervention mean can still be 0.5. This separates an individual-state response from a population composition, without claiming to know which latent state actually occurred.

## Exercise and solution

In the XOR outcome Y=X XOR Z XOR E_Y, set both noise probabilities to zero and pi=0.5. Observe X_f=1,Y_f=0. What happens under action a=0?

**Solution.** Naturally X=Z, so the evidence identifies Z=1. Retaining Z and E_Y=0 while assigning X_a=0 gives Y_a=1. Reapplying action a=1 instead gives Y_a=0, confirming consistency. If the selected factual event were impossible, neither answer could be formed by conditioning in this SCM.

## What a fitted predictor cannot supply by itself

A supervised model may estimate P(Y|X) accurately while leaving the joint coupling of potential outcomes unspecified. A counterfactual answer depends on structural functions, noise distributions and their cross-world sharing. In AI explanations, changing one input and reusing a prediction model is not automatically a causal counterfactual about a real person or system. State the evidence, action, retained exogenous assumptions and outcome mechanism explicitly. The next chapter constructs multiple compatible models to demonstrate observational non-identification.

## Primary references

- Judea Pearl, [Sufficient Causes: On Oxygen, Matches, and Fires](https://ftp.cs.ucla.edu/pub/stat_ser/r484.pdf), 2019: an explicit abduction–action–prediction example.
- Alexander Balke and Judea Pearl, [Counterfactuals and Policy Analysis in Structural Models](https://arxiv.org/abs/1302.4929): structural evaluation of counterfactual queries.

The binary examples, posterior weights and twin diagram here are original teaching constructions.
