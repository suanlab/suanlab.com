## Hidden states behind observations

State-space models separate observed y from latent x. An HMM uses discrete states. Here X₀ is fair, state-stay probability is r and emission agreement is s. Read the four observation bits from lowest to highest mask position.

$$
p(x_{0:3},y_{0:3})=p(x_0)\prod_{t=1}^3p(x_t\mid x_{t-1})\prod_{t=0}^3p(y_t\mid x_t).
$$

## Forward summarizes the past, backward the future

Forward αₜ(x)=p(y₀:ₜ,Xₜ=x) sums the previous forward weights through transitions and multiplies the emission. Backward βₜ(x)=p(yₜ₊₁:₃|Xₜ=x) summarizes future observations. Total evidence is Σxα₃(x).

$$
\alpha_t(x)=p(y_t\mid x)\sum_u p(x\mid u)\alpha_{t-1}(u),\qquad
\beta_t(x)=\sum_v p(v\mid x)p(y_{t+1}\mid v)\beta_{t+1}(v).
$$

Filtering normalizes α; smoothing normalizes αβ. In default observations 0,0,1,1, future ones can change the first-state posterior. At the final time filtering and smoothing coincide because no future observations remain.

## Log arithmetic and impossible evidence

Log-sum-exp avoids losing products of tiny probabilities. If r=s=1 but observations alternate 1,0,1,0, the model cannot produce the evidence: states never change and emissions are perfect. Evidence is mathematically zero, log evidence minus infinity and smoothing undefined.

With r=1,s=10⁻³²⁰, that alternating evidence instead has positive but extremely small probability. Its exponential can underflow to zero while log evidence remains finite and posterior ratios remain computable. The screen separates possibility from numeric-range failure.

## Continuous state space and a Kalman example

A linear Gaussian model has the same structure: Xₜ=FXₜ₋₁+wₜ, Yₜ=HXₜ+vₜ. For scalar prior N(m,P) and observation-noise variance R, K=PH/(H²P+R), posterior mean m+K(y−Hm), variance (1−KH)P. With m=0,P=1,H=R=y=1, posterior is N(.5,.5). This passes Gaussian moments rather than discrete tables. General nonlinear models do not automatically share these closed forms.

## Exercises with explanations

1. Which method uses future observations? **Smoothing.**
2. What if emissions have accuracy .5? **They do not distinguish states, leaving marginals .5.**
3. Do marginal modes give a joint MAP path? **Not generally: distinguish a path objective such as Viterbi.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
