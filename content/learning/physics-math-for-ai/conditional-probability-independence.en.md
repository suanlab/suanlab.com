## Conditioning changes the denominator

Conditional probability restricts attention to an observed event B and renormalizes. The four default counts are one for both events, two for B without A, one for A without B and six for neither. Among ten outcomes, three belong to B, and only one of those belongs to A. Therefore P(A|B)=1/3. Keep the predicted event distinct from the conditioning event.

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.
$$

## Reverse the conditional with Bayes

For P(B|A), the denominator contains the two A outcomes, giving one half. Express the same joint probability in the two possible conditioning orders to obtain Bayes’ rule. Here (1/2)(1/5)/(3/10)=1/3. The denominator P(B) can also be obtained by summing the A and non-A alternatives. The small table exposes the role of the prior population proportion.

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)},\quad P(B)=P(B\mid A)P(A)+P(B\mid A^c)P(A^c).
$$

## Independence is not disjointness

Independence means that joint probability factors into the product of marginals. Disjoint events with positive probabilities cannot be independent: their joint probability is zero while the product is positive. The independent example has counts (1,2,2,4), total nine, marginal probabilities one third each and joint probability one ninth. Learning B leaves the probability of A unchanged in that model.

$$
A\perp B\iff P(A\cap B)=P(A)P(B).
$$

## When the denominator vanishes

If B has zero count, the elementary ratio cannot define P(A|B); the interface shows an undefined value rather than inventing zero. If every count is zero, even the probability distribution is undefined. Independence involving null events still makes sense through the product condition. Regular conditional laws for continuous variables require a separate construction. In selection effects or classifier evaluation, identify the observed population before interpreting a conditional.

## Exercises with explanations

1. What is the default P(A)? **Two out of ten, or 0.2.**
2. Can a null B be independent of A? **Yes: the product condition holds.**
3. Does independence always make the conditional ratio available? **No: its denominator must be positive.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
