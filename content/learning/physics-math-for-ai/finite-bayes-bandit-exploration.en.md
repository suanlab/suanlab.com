## A reward can teach as well as pay

A bandit chooses which source of reward to try without knowing its mean. Exploitation favors an arm with a high current expected reward. Exploration can favor an arm whose result will improve later choices. Exploration is therefore not synonymous with choosing randomly: a finite Bayesian planner can deliberately choose an uncertain arm because of its future information value.

This chapter maximizes expected reward, unlike the cost-minimizing sensing example in the previous chapter. A higher value is better. The model is small enough to compute every conditional expectation, so no sampled reward path is used as an estimate of optimal value.

## Specify the uncertainty and what remains fixed

Two arms have fixed unknown Bernoulli means theta0 and theta1, with independent Beta priors. Conditional on an arm's mean, its rewards are IID. The mean is not redrawn from its prior at every pull. Marginally, rewards from the same arm can be dependent because they share that unknown mean.

A prior Beta(a,b) has density proportional to theta^(a-1)*(1-theta)^(b-1), predictive success probability a/(a+b), and variance ab/((a+b)²*(a+b+1)). The positive parameters are prior hyperparameters; they need not be actual observed successes or failures. A success changes (a,b) to (a+1,b), and a failure to (a,b+1). Only the chosen arm is updated.

The four Beta parameters and remaining pull budget H form the decision state. We use positive integer parameters and at most six pulls to make interactive exact expectation calculations practical. This is a bound on the teaching example, not a claim that bandit learning ends after six decisions.

## Put future learning into the action value

Let s contain both posterior parameter pairs. For each arm i, let p_i be its predictive success probability and s_i+ and s_i- its updated states. With V0=0,

$$
Q_{H,i}(s)=p_i+p_i V_{H-1}(s_i^+)+(1-p_i)V_{H-1}(s_i^-),
\qquad V_H(s)=\max_i Q_{H,i}(s).
$$

The first term is the immediate expected reward. The other terms value optimal decisions after observing the result. This is a belief-state Bellman calculation: choosing an arm changes what the controller will know. Memoization reuses posterior states reached by different histories. The result is exact finite-model expectation up to floating-point roundoff, rather than a Monte Carlo approximation.

The implemented policy uses H explicitly. With one pull left, information has no future use and the larger predictive mean wins. With more pulls, a lower immediate mean can be outweighed by better later decisions. Exact floating-point Q ties choose arm0; tiny differences close to a tie should not be interpreted as strong practical evidence.

## Give the greedy comparator the same data

Adaptive myopic greedy chooses the larger posterior mean at every step and then updates after the observed reward. It is not a policy that commits forever to the initially preferred arm. Its expected total reward satisfies its own recursion using the same outcome probabilities, but its action is selected by the current mean rather than future optimal value.

Both policies receive the same model and observation opportunities. They differ in how they value future information. Posterior sampling, introduced in the next chapter, is a third rule; drawing a plausible mean and choosing its best arm is not generally the same as solving this finite Bellman equation.

## Read the plots and hypothetical posterior correctly

The native plot shows Beta densities for the selected arm before and after the selected hypothetical reward. Blue is the prior to that update, orange the posterior. The hypothetical arm/outcome need not equal the planner's first action or a result that actually occurred. A density height can exceed one; probability is an area, not a curve's height.

The lower plot compares total expected rewards across budgets. Blue is the Bayesian optimum, orange adaptive greedy. These are not empirical cumulative reward paths. Their small difference is a real value difference, so the readouts show both action Q values and the total advantage without visually exaggerating it. Changing the hypothetical outcome changes the density comparison, not the initial decision problem.

## Worked exercise: two uniform arms

Give both arms Beta(1,1) priors and allow two pulls. Compute the expected reward if the first pull is arm0 and the second uses the best posterior mean.

**Solution.** The immediate mean is 1/2. A success occurs with probability 1/2 and makes arm0's mean 2/3, larger than arm1's 1/2. A failure makes arm0's mean 1/3, so the next choice is arm1. Total expectation is

$$
\frac12+\frac12\frac23+\frac12\frac12=\frac{13}{12}.
$$

A fixed commitment to one arm has expectation one over two pulls. Learning can improve on H times the initially largest mean, so that quantity is not an upper bound on an adaptive Bayesian policy. The universal reward upper bound in this example is H, because each reward is at most one.

## Worked exploration example: a lower mean wins

Now give arm0 Beta(4,3) and arm1 Beta(1,1). Their means are 4/7 and 1/2, and variances 3/98 and 1/12. With H=1, choose arm0. With H=4, the finite calculation gives

$$
Q_{4,0}=\frac{33}{14},\qquad Q_{4,1}=\frac{50}{21}.
$$

The second is larger by 1/42, so the first action is arm1 despite its smaller current mean. The optimal total value is 50/21. Adaptive greedy obtains 65/28, giving a total advantage of 5/84. The first-action Q difference and total advantage over an entire different policy are separate quantities.

The second guided example reproduces this comparison. Then select a hypothetical arm1 failure: its posterior becomes Beta(1,2), while arm0 remains Beta(4,3). The unknown arm's mean falls to 1/3. The calculation has used uncertainty to justify an experiment and then used its possible evidence to change later choices.

## Verify a planner without trusting a second copy of it

Tests enumerate all contingent action trees for small budgets. For a fixed tree and reward sequence, the arm-specific prior predictive likelihood can be integrated as a ratio of Beta functions. Summing rewards against those sequence probabilities evaluates the tree independently of the production posterior recursion. Exact rational fixtures check the four-pull example, while density integration checks normalization and moments.

These checks support this finite implementation. General bandit problems can have many arms, longer budgets, continuous outcomes or misspecified priors, making exact planning much harder. Gittins-index results for other formulations should not be imported as an optimality claim for this specific finite calculation.

## Connection to AI research

Exploration allocates data as well as rewards. A learned model or prior affects which data will ever be collected, so a poor prior is not automatically corrected by many observations of other arms. Bayesian expected reward also averages over the stated prior; it differs from regret at fixed true means. The next seeded experiments make that evaluation distinction explicit while comparing practical exploration rules.

## Primary reference

Lattimore and Szepesvari, [Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf), provides the broader Bayesian bandit framework. The finite two-arm example, rational fixtures, contingent-tree checks, density diagram and adaptive-greedy comparison here were constructed for this textbook.
