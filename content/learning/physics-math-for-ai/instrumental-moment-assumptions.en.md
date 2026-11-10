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
