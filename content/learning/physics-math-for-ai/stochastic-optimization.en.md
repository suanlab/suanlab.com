## Why use a noisy gradient?

A full training gradient can require evaluating every example. A sampled gradient can be much cheaper while having the correct conditional expectation. This makes stochastic gradient descent useful, but it does not turn each noisy update into a guaranteed loss reduction.

Use three fixed targets $z\in\{-1,1,3\}$, sampled uniformly with replacement. For a scalar parameter $w$, each example has loss $\ell_i(w)=(w-z_i)^2/2$. The lab compares the exact average gradient with reproducibly sampled updates. It is not minibatch training on a large network.

## Derive the objective and gradient noise

The target mean is 1 and its population variance is $8/3$. Expanding squared loss gives

$$
L(w)=\frac13\sum_i\frac12(w-z_i)^2
=\frac12(w-1)^2+\frac43.
$$

Thus $\nabla L(w)=w-1$ and the minimizer is 1. A sample gradient is $g_i(w)=w-z_i$. Conditional on the current $w$ and an independent uniform index,

$$
E[g_i(w)|w]=w-1,\qquad \operatorname{Var}(g_i(w)|w)=8/3.
$$

Unbiasedness is a statement about expectation. At the optimum, sample gradients can be 2, 0 or $-2$ even though the full gradient is zero.

## Calculate the first update

At $w_0=4$, the full gradient is 3. With $\eta=0.1$, full gradient descent moves to 3.7. A sampled target of $-1$ gives gradient 5 and new parameter 3.5; target 1 gives 3.7; target 3 gives 3.9. Their average is 3.7, but no individual sampled step has to equal it.

The seed selects a pseudorandom sequence so the displayed experiment can be repeated. It is not a distributional theorem and changing only the seed does not change the objective. The plot displays 30 updates for both algorithms, using the same initial value and learning rate.

## Analyze stability before generalizing

For full descent, subtract the minimizer to get $w_{t+1}-1=(1-\eta)(w_t-1)$. It converges for $0<\eta<2$. At 2 it can oscillate; above 2 it grows in magnitude.

For SGD the centered recurrence includes noise. With an independent sample each step, the mean follows full descent. Under the same stable range, the long-run variance for this scalar model is

$$
\frac{\eta^2(8/3)}{1-(1-\eta)^2}
=\frac{\eta(8/3)}{2-\eta}.
$$

A constant learning rate therefore leaves a noise floor rather than exact convergence of every iterate. This calculation assumes the sampling model stated above; it is not a universal formula for neural-network training. A finite seeded run need not equal the theoretical mean or variance.

## Batches, schedules and dependence

Independent averaging of $B$ gradients at the same point divides noise variance by $B$. Sampling without replacement changes the dependence and may require a finite-population correction. Reusing correlated examples can invalidate the simple independent averaging formula.

Decreasing learning rates, iterate averaging, momentum and adaptive methods modify the dynamics. Convergence guarantees require their own smoothness, noise and schedule assumptions. This lab isolates plain constant-rate SGD before adding those mechanisms; it does not claim that every modern optimizer behaves like this scalar recurrence.

## Read the experiment

Blue is full gradient descent and orange is one sampled path, both plotted as parameter versus update count. The final table evaluates the full objective for both terminal parameters, rather than comparing losses on different sampled targets. This avoids confusing stochastic loss noise with actual objective improvement.

Change the seed while keeping everything else fixed. Then change the learning rate. Report these as distinct interventions. A pleasing single path is not evidence of a robust improvement.

## Exercises and research protocol

1. Compute all possible gradients at $w=1$. **Answer:** 2, 0, $-2$.
2. At rate 0.1, calculate the theoretical stationary variance. **Answer:** $8/57\approx0.140351$.
3. Repeat many seeds and compare empirical mean and spread at a fixed iteration. Do not count iterations from one correlated trajectory as independent runs.
4. Compare methods at both equal update count and equal example-evaluation budget. Those are different costs.

For stochastic-gradient assumptions and analysis, see [MIT’s stochastic gradient descent lecture notes](https://www.mit.edu/~gfarina/2024/67220s24_L10_sgd/L10.pdf). The three-target recurrence is this textbook’s exact teaching example, not an empirical claim about large-scale AI.
