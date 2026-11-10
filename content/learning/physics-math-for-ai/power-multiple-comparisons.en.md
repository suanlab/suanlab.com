## Power refers to a specified alternative

The lab uses a one-sided Gaussian mean test with known σ=1, null mean zero and alternative difference δ. The statistic follows N(δ√n,1) under that alternative. Power is the probability of exceeding the threshold there, not a posterior probability that a parameter is true. The reported power concerns one alternative test, not simultaneous discovery of every hypothesis.

$$
Z\sim N(\delta\sqrt n,1),\qquad\operatorname{Power}(\delta)=\Phi(\delta\sqrt n-z_{1-\alpha}).
$$

## Roles of effect size and sample count

Larger effects or samples separate the distribution centers, making threshold exceedance more likely. Blue shows the null density, orange the alternative density and the vertical line the threshold. At n=16,δ=0.5 the shift is two. At zero effect, rejection probability equals test size. A small p-value is not itself an effect-size measurement.

$$
\lambda=\delta\sqrt n,\qquad\beta(\delta)=1-\operatorname{Power}(\delta).
$$

## Familywise error across tests

With all nulls true and independent tests, the chance of at least one false rejection is 1−(1−α)^m. Ten 5% tests give about 0.401. Dependence invalidates that exact product formula. The displayed familywise number explicitly assumes independent all-null tests. Selecting only favorable experimental outcomes also belongs in the analysis of the selection procedure.

$$
\operatorname{FWER}_{\mathrm{ind}}=1-(1-\alpha_{\mathrm{each}})^m.
$$

## Bonferroni’s guarantee and cost

Assigning each test 0.05/m gives FWER at most 0.05 by the union bound, without requiring independence. The higher threshold can reduce power at fixed effect and n. Distinguish the exact independent-null family error from its dependence-robust upper bound. This controls FWER; it is not an implementation of FDR control, sequential testing or arbitrary data-dependent family selection.

$$
P\left(\bigcup_{j=1}^m\{\text{false reject}_j\}\right)\le\sum_{j=1}^m\alpha_j,\qquad\alpha_j=0.05/m.
$$

## Exercises with explanations

1. What is the corrected level for ten tests? **0.005 each.**
2. Does dependence remove the Bonferroni bound? **No: the union bound remains valid.**
3. Does high power alone validate a study design? **No: model, selection and independence assumptions also matter.**

Further reading: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
