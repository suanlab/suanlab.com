## From independences to a set of graphs

A covariance matrix describes statistical relationships. Discovering causal direction requires assumptions connecting those relationships to a structural graph. The causal Markov condition says graphical d-separation implies distributional conditional independence. Faithfulness adds the converse: extra independences do not arise through parameter cancellation. Neither statement follows from a good predictive fit alone.

This lab assumes acyclic linear Gaussian structural equations, three observed variables A,B,C, independent unit-variance innovations, no latent common causes and no selection. It enumerates all 25 three-node DAGs. It is a population demonstration, not a finite-sample discovery algorithm with calibrated testing error.

## Four generating structures

The controls choose chain A→B→C, fork A←B→C, collider A→B←C, or a triangle with A→B→C and A→C. Coefficients a,b are positive. In the triangle,

$$
A=E_A,\quad B=aA+E_B,\quad C=cA+bB+E_C.
$$

Each system is a linear transformation T of independent innovations, so Sigma=T Tᵀ. The diagrams show the declared generating graph, including allowed paths at special parameter values; they do not show a structure established by the displayed observational covariance.

## Gaussian conditional independence

For this positive-definite Gaussian family, zero correlation implies marginal independence, and zero partial correlation implies the corresponding conditional independence. Conditioning on B uses

$$
\Sigma_{AC\mid B}=\Sigma_{AC}-\Sigma_{AB}\Sigma_{BC}/\Sigma_{BB}.
$$

Divide by the square root of the two conditional variances to obtain partial correlation. The tool evaluates six queries: each variable pair, with either no conditioning or conditioning on the remaining variable. A numerical tolerance of 1e-9 labels zero in these exact-population calculations. It is not a significance threshold or evidence that nearly zero correlations prove causal absence in real data.

## Markov-compatible versus faithful-compatible

For every candidate DAG, enumerate d-separations. A Markov-CI-compatible candidate requires every implied independence to occur in the covariance. A faithful-CI-compatible candidate requires its independence pattern to match exactly. Complete DAGs imply no independences and can therefore remain Markov-compatible even when the data have more independences.

A chain and the two other non-collider orientations of the same skeleton share the independence A independent of C given B. Three faithful-compatible DAGs remain. This is Markov equivalence; observational CI information alone does not choose their directions. The coefficient values and innovation variances can be reparameterized when comparing observationally equivalent Gaussian models. Keeping identical unit-noise coefficients across orientations would not generally produce the same covariance.

## Exercise: the collider contrast

Choose collider mode with a=b=1. Compare marginal A,C correlation and partial correlation given B.

**Solution.** A and C are independent innovations, so their marginal correlation is zero. B=A+C+E_B selects on a common consequence. Conditional covariance is -1/3, each conditional variance is 2/3, and partial correlation is -1/2. The corresponding three-node collider is the sole faithful-compatible DAG for that independence pattern, under the declared assumptions. That conclusion still depends on causal sufficiency, Markov and faithfulness.

## Cancellation defeats the converse

Choose triangle mode with a=b=1,c=-1. Then C=E_B+E_C, and A is marginally independent of C even though both the direct and indirect paths are present in the declared graph. The partial correlation given B is -1/sqrt(3). Its independence pattern resembles the collider example, but the declared triangle is unfaithful. Blindly imposing faithfulness would select the wrong generating structure here.

## Research implications

CI-based discovery can return an equivalence class rather than a unique graph. Interventions, non-Gaussian models or justified functional restrictions can supply additional information, but require their own assumptions. Finite samples introduce testing error, multiple comparisons and instability near cancellations. Hidden causes, feedback and selection can invalidate this lab's DAG family altogether. Report the assumptions and candidate set before presenting a discovered arrow as an established causal mechanism.

## Primary reference

Peters, Janzing and Schölkopf, [Elements of Causal Inference](https://web.math.ku.dk/~peters/elements.html), 2017. The covariance examples and exhaustive three-node candidate comparison here are original implementations.
