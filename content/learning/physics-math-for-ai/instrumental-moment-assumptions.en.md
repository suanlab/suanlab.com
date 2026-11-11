## An external source of treatment variation

If an unobserved factor U affects both treatment X and outcome Y, regressing Y on X generally mixes the treatment response with selection. An instrument Z supplies variation whose relationship to the outcome must satisfy additional assumptions. Correlation with treatment is necessary but does not establish validity. This chapter examines an explicit linear population model; it does not automatically extend its constant-effect formula to arbitrary nonlinear or heterogeneous systems.

## State the structural assumptions

Independent centered innovations H,V,E_X,E_Y have variance one. Define

$$
Z=H,\quad U=\rho H+\sqrt{1-\rho^2}V,
$$

$$
X=\gamma Z+\alpha U+\sigma E_X,\qquad
Y=\beta X+\delta U+\kappa Z+E_Y.
$$

The structural treatment effect is beta. Relevance requires Cov(Z,X) to be nonzero. The direct coefficient kappa represents an exclusion violation, and rho represents association between the instrument and the outcome disturbance through U. At rho=0, Z and U are independent by construction. With kappa=rho=0 and independent E_Y, the instrument moment removes the confounding term.

## Derive the ratios

Using unit variances and the declared innovation independence,

$$
D=\operatorname{Cov}(Z,X)=\gamma+\alpha\rho,
\quad \operatorname{Cov}(Z,Y)=\beta D+\delta\rho+\kappa.
$$

For D nonzero,

$$
\beta_{IV}=\beta+\frac{\delta\rho+\kappa}{D}.
$$

The ordinary regression slope instead uses Cov(X,Y)/Var(X), where

$$
\operatorname{Var}(X)=\gamma^2+\alpha^2+2\gamma\alpha\rho+\sigma^2.
$$

Its extra numerator is delta(gamma rho+alpha)+kappa D. A large observational sample estimates these population moments more precisely; it does not eliminate a nonzero assumption-violation term.

## A heterogeneous binary instrument identifies a local effect

The live linear constant-effect model identifies beta when its moment assumptions hold. With binary randomized Z and heterogeneous responses, the Wald ratio can identify a different estimand. Assume Z is independent of all potential treatments and outcomes, exclusion Y=Y(X), consistency, relevance and monotonicity X1≥X0. Always-takers and never-takers do not change treatment when Z changes; only compliers do.

Subtracting the two assignment arms and cancelling unchanged responses proves

$$
E[X\mid Z=1]-E[X\mid Z=0]=P(C),\quad
E[Y\mid Z=1]-E[Y\mid Z=0]=P(C)E[Y_1-Y_0\mid C].
$$

Their ratio is the complier LATE, not automatically the whole-population ATE. Defiers would contribute with the opposite sign, destroying this simple interpretation unless additional restrictions were justified. Monotonicity is an individual-level restriction, not just a positive first-stage average.

| Principal stratum in the auxiliary population | Mass | Mean outcome effect |
|---|---:|---:|
| Never-takers | 3/10 | 1 |
| Compliers | 1/2 | 2/5 |
| Always-takers | 1/5 | −1 |

```python
from fractions import Fraction as F
# x0,x1,y0,y1,mass; the complier stratum has two outcome response types.
atoms=[(0,0,0,1,F(3,10)),(0,1,0,1,F(1,5)),(0,1,0,0,F(3,10)),(1,1,1,0,F(1,5))]
assert sum(row[-1] for row in atoms)==1
means=[]
for z in (0,1):
    ex=ey=F(0)
    for x0,x1,y0,y1,mass in atoms:
        x=x0 if z==0 else x1;ex+=mass*x;ey+=mass*(y0 if x==0 else y1)
    means.append((ex,ey))
first=means[1][0]-means[0][0];reduced=means[1][1]-means[0][1]
ate=sum(mass*(y1-y0) for x0,x1,y0,y1,mass in atoms)
assert first==F(1,2) and reduced==F(1,5)
assert reduced/first==F(2,5) and ate==F(3,10)
print(first,reduced,reduced/first,ate)
```

Here the first-stage difference is 1/2 and the reduced form is 1/5, so Wald is 2/5 while ATE is 3/10. The four response atoms give an explicit construction, separate from the continuous linear-noise controls.

The IV numerator and denominator are now evaluated as exact dyadic polynomials of the stored binary64 coefficients, then the ratio is rounded to binary64 with ties to even. This preserves tiny products and residuals after large cancellations. `expandedIV` retains the directly divided rounded moments; `factoredIV` retains the ordinary beta+(delta*rho+kappa)/D calculation. A mathematically zero denominator or a nonrepresentable final ratio remains unavailable. A rounded table moment can be zero even when the exact stored-input denominator is not. This is exact intermediate arithmetic for this small polynomial, not arbitrary-precision input data or an inference certificate. This numerical repair does not solve weak-instrument sampling inference or establish the exclusion and independence assumptions.

## Worked examples

With gamma=alpha=beta=delta=sigma=1 and kappa=rho=0, Var(X)=3. IV equals 1, while OLS equals 1+1/3. Both methods see the same confounded population, but only the valid instrument moment cancels the U contribution.

Now set gamma=0.1 and kappa=0.1, retaining rho=0. IV becomes 1+0.1/0.1=2. The instrument is relevant, yet a small excluded-path violation creates a large error. Alternatively, with gamma=alpha=delta=beta=1, kappa=0 and rho=0.5, D=1.5 and IV=4/3. Exclusion alone does not repair instrument–disturbance association.

## Exercise and solution

Set gamma=0, rho=0, alpha=sigma=1 and kappa=0. What does the IV ratio report? Does the outcome equation stop having a treatment effect?

**Solution.** D=0, so the instrument contains no treatment variation and the IV ratio is unavailable. The declared structural coefficient beta still defines an effect. Identification through this particular moment has failed; the effect itself has not become zero. OLS can remain defined because Var(X)>0.

## Reading the sensitivity plot

The horizontal axis varies kappa while retaining the other parameters. Blue shows the actual population ratio and orange shows beta. The display window is [-5,5]; blue points outside it are omitted, with an explicit omitted-point count. This is a bounded visualization, not clipping the ratio to five. A zero denominator or a nonfinite machine ratio displays an em dash; inspect the denominator separately. Individual moment values remain in the table.

## Population identification and research practice

A ratio of estimated moments is not generally finite-sample unbiased merely because the population ratio identifies beta. Weak instruments also create sampling and inference problems, beyond the assumption sensitivity isolated here. A real study needs a substantive account of instrument assignment, exclusion, measured populations, effect heterogeneity and uncertainty. Instrument relevance in a prediction model cannot supply that account.

## Primary reference

Hernán and Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook), chapter 16. The covariance model, diagrams and sensitivity examples here are original teaching constructions.
