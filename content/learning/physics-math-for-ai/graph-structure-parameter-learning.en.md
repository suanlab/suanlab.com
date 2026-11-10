## Learning structure differs from fitting parameters

Parameter learning fixes graph structure and estimates probability tables. Structure learning compares edges. This lab compares only two observed binary models: independent M0:p(A)p(B), with two free parameters, and M1:p(A)p(B|A), with three. It does not search over arbitrary graphs.

## MLE from sixteen rows

Default counts are (5,3,3,5). Both marginals are .5, so independent MLE gives four masses .25. The edge model fits the empirical joint counts divided by sixteen. Sum count times log probability only for positive counts; zero-count terms contribute zero.

$$
\ell(\theta)=\sum_{a,b}n_{ab}\log p_\theta(a,b).
$$

The larger model contains independence, so its optimized likelihood cannot be smaller. Better training likelihood alone establishes neither generalization nor correct structure.

## BIC penalizes additional parameters

Use BIC=−2ℓMLE+k ln N, where smaller is preferred. The default small dataset selects independence after the additional-parameter penalty. Repeating its proportions eight times produces an edge-model win because likelihood gains scale with N while the penalty scales with ln N. This constructed count input does not justify inflating independent sample size by reusing real observations.

## Keep smoothing separate from scoring

Pseudocount α is added to both outcomes in each conditional row. Thus P(B=1|A=1)=(n11+α)/(n10+n11+2α). The smoothed table illustrates prediction; BIC always uses MLE. Changing α should not change BIC. Do not substitute a smoothed fit while calling it the maximum-likelihood score.

## Direction and causal limits

A complete two-variable observed law can factorize in either direction. Association alone does not identify a causal arrow. BIC is an asymptotic approximation tied to candidate models and regularity/sample assumptions, not an automatically justified score for every boundary or latent-variable setting.

## Exercises with explanations

1. How many edge-model parameters? **Three.**
2. Does α affect this BIC? **No: it affects only the separate smoothed table.**
3. Do perfect matching observations identify causal direction? **No: additional assumptions are required.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
