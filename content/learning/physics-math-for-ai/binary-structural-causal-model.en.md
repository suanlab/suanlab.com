## Why prediction needs an additional language

Suppose a sensor records a switch X and an alarm Y. Predicting alarms among observations with X=1 asks about a selected group. Forcing the switch to one asks what a physical operation would change. A joint probability table answers the first question. To answer the second, specify how values are produced and which mechanism the operation replaces.

A structural causal model (SCM) consists of endogenous variables, exogenous inputs, structural assignments, and a joint distribution of the exogenous inputs. A directed acyclic graph (DAG) records allowed structural dependencies. The graph alone does not specify effect sizes, functions, or noise probabilities. An allowed edge can have no observable influence at special parameter settings. A zero correlation therefore does not automatically remove a causal possibility.

## Three explicit binary worlds

The lab uses independent Bernoulli bits Z, E_X, E_Y with probabilities pi, epsilon_X, epsilon_Y. XOR returns one when its bits differ. Evaluate the assignments in order:

$$
Z\sim\operatorname{Bern}(\pi),\qquad X=Z\oplus E_X.
$$

Choose the outcome mechanism:

| Choice | Assignment for Y | Graph |
|---|---|---|
| 0 | X XOR E_Y | Z → X → Y |
| 1 | Z XOR E_Y | X ← Z → Y |
| 2 | X XOR Z XOR E_Y | Z → X, Z → Y, X → Y |

These are alternative worlds, not three fitted explanations proven by the same observations. Independent exogenous inputs are an explicit assumption. Allowing hidden common causes through dependent inputs would require revising the graphical representation and subsequent identification arguments.

## Enumerating a population rather than sampling

There are eight exogenous states. Each state has probability

$$
p(z,e_x,e_y)=\pi^z(1-\pi)^{1-z}
\epsilon_X^{e_x}(1-\epsilon_X)^{1-e_x}
\epsilon_Y^{e_y}(1-\epsilon_Y)^{1-e_y}.
$$

Apply the assignments and add probabilities for states sharing the same (X,Y). This is an exact finite population calculation; the table is not eight equally likely observations. Zero-probability rows remain visible. The plot indexes joint cells by 2X+Y, so 0,1,2,3 mean 00,01,10,11. Each stem marks a discrete cell probability; there are no intermediate binary states between the indexed cells.

## Replace an equation, retain the population

An intervention do(X=a) replaces X's assignment by the constant a. Retain the exogenous distribution and the Y assignment, then recompute Y. The crossed incoming edge in the diagram marks that replacement. Outgoing edges remain because descendants still use X.

In the fork world with pi=0.5 and both noise probabilities 0.1, Bayes' rule gives P(Z=1|X=1)=0.9. Therefore

$$
P(Y=1\mid X=1)=0.9(0.9)+0.1(0.1)=0.82.
$$

Setting X does not alter Z or its influence on Y, so P(Y=1|do(X=1))=0.5. Observation updates information about a common cause; intervention changes an assignment. This numerical distinction follows from the declared SCM, not from the table alone.

## Worked exercise and solution

Set both noise probabilities to zero, pi=0.5, and choose mechanism 2. Predict Y before forcing X, then force X=1.

**Solution.** Naturally X=Z and Y=X XOR Z=0 for every state of positive probability. Under do(X=1), Y=1 XOR Z, so half the population has Y=1. The observational mean is zero while the intervention mean is one half. Deleting the Y equation would incorrectly remove the very mechanism through which the intervention acts.

## Connection to AI experiments

An input ablation, randomized interface, or policy change is an intervention only relative to a specified causal system. Replacing an observed feature in a prediction vector need not change the physical process that produced the target. Record the intervention target, assignment rule, retained population, and mechanisms assumed invariant. Later chapters use these assumptions to ask which intervention distributions are identifiable without knowing every structural equation.

## Primary references

- Judea Pearl, [A Causal Calculus for Statistical Research](https://proceedings.mlr.press/r0/pearl95a.html), 1995: structural-equation semantics and intervention notation.
- Miguel Hernán and James Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook): population interventions and identification assumptions.

The binary XOR worlds and exact eight-state calculations here are original teaching examples.
