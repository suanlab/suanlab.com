## A result needs a precisely bounded claim

A new AI method can improve one score without establishing a general scientific contribution. A useful claim identifies the population, comparator, metric, compute budget and assumptions. Reproducibility means that the procedure and evidence can be inspected; repeating a single favorable number is not enough.

This lab uses synthetic paired score differences, not real experimental results. It isolates uncertainty arithmetic so that evaluation mistakes can be understood before working with costly experiments.

## Define the paired comparison

For matched condition $i$, define $d_i=\text{score(new,i)}-\text{score(base,i)}$, assuming larger scores are better. Pairing can control shared variation when the conditions are genuinely matched. It is not a license to pair unrelated observations after seeing their outcomes.

The four synthetic differences are $[a,a+b,a-b,a]$. The reference uses $a=b=0.1$, giving $[0.1,0.2,0,0.1]$. The mean is 0.1. The sample variance uses denominator three:

$$
s^2=\frac{\sum_i(d_i-\bar d)^2}{4-1}
=\frac{2b^2}{3},\qquad SE=\frac{s}{\sqrt4}=\frac b{\sqrt6}.
$$

Thus $s\approx0.081650$ and SE is approximately 0.040825. Standard deviation across differences and standard error of their mean are not interchangeable.

## An interval requires assumptions

If paired differences are independent normal draws with a common mean and unknown variance, a two-sided 95% Student-t interval is

$$
\bar d\pm t_{0.975,3}\frac{s}{\sqrt4},\qquad t_{0.975,3}\approx3.182446.
$$

The reference interval is approximately $[-0.029923,0.229923]$. It includes zero. That does not prove the effect is zero. It indicates that this procedure under its assumptions leaves considerable uncertainty at this sample size.

The generated symmetric differences are chosen teaching values, not evidence that actual differences were sampled independently from a normal population. The formula illustrates what an assumption-dependent calculation returns. It does not certify that its assumptions hold for a real benchmark.

## Interpret coverage correctly

A frequentist 95% confidence procedure covers the fixed population mean in 95% of repeated samples under its model. Once an interval is computed, the standard frequentist definition does not assign a posterior 95% probability to that fixed parameter. Bayesian credible intervals answer a differently defined question.

Four training seeds do not automatically represent four independent datasets, populations or scientific systems. Shared data, correlated runs, selected hyperparameters and repeated inspection of results can invalidate a naive uncertainty interpretation. Increasing the number of plotted points is not a substitute for identifying the sampling unit.

## Design the experiment before viewing results

Write the hypothesis, primary metric, direction of improvement, exclusions, budget and stopping rule first. Keep training, validation and test roles distinct. Compare strong baselines at matched relevant costs and report variability rather than selecting the best seed.

For ablation, remove or change one proposed mechanism while controlling alternatives. Add a negative control that should not benefit if the explanation is correct. An ablation difference alone may still confound parameter count, tuning budget or training duration; design the comparison to isolate the intended claim.

## Connect mathematical and physical validation

For a mathematical claim, state quantifiers, assumptions and conclusion, then supply a proof or counterexample. Numeric examples cannot prove a universal theorem. For a physical model, additionally check dimensions, boundaries, conservation and convergence with resolution. A small residual does not establish that all unobserved causes were identified.

For a learned scientific simulator, separate approximation error, solver error, observation noise and model discrepancy. Report where a result fails as well as where it works. A new paper’s abstract identifies its claimed scope but does not by itself verify its proof or experimental evidence.

## Read the lab and plan a stronger study

Blue shows four paired differences and orange their mean, not a probability density. The table contains the sample standard deviation, standard error and t interval. Increasing $a$ shifts the mean and interval; increasing $b$ widens it. Neither action repairs leakage or a biased comparator.

A stronger study would collect defensible independent units, repeat a predeclared analysis and compare sensitivity to assumptions. Multiple hypotheses and selective reporting require explicit treatment; the displayed interval does not include those corrections. Do not interpret zero variation in a contrived example as proof that real evaluation uncertainty vanished.

## Exercises and source

1. With $a=0.3,b=0.1$, compute the shifted interval. **Answer:** approximately $[0.170077,0.429923]$.
2. Explain why including zero is not proof of no effect.
3. Write a research question, matched baseline, negative control and failure criterion for low-rank compression.
4. For a scientific inverse problem, list which conclusions need new observations rather than more optimizer steps.

The t critical-value convention is documented in the [NIST Student’s-t table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm). All score differences here are synthetic. They illustrate analysis mechanics and do not support a claim that any actual AI method improved.
