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

## Every time cut gives the same evidence

Assume finitely many states, normalized transition and emission tables, and positive evidence. Fixing Xₜ splits the factorization into past and future observation weights αₜ and βₜ. Thus Z=Σᵢαₜ(i)βₜ(i) at every t, an implementation invariant. Applying the same argument to adjacent states gives the edge posterior:

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}Z,
\qquad
\xi_t(i,j)=\frac{\alpha_t(i)A_{ij}B_j(y_{t+1})\beta_{t+1}(j)}Z.
$$

Summing ξ over j must give γₜ(i); summing over i gives γₜ₊₁(j). These node and edge posteriors supply expected visits and transitions without committing to one latent path.

## Verify one EM step with exact fractions

Keep the fair initial distribution fixed and estimate only stay probability r and emission accuracy s in this symmetric binary model. For one observation sequence of length T=4, the updates are

$$
r_{\mathrm{new}}=\frac1{T-1}\sum_{t=0}^{T-2}\sum_i\xi_t(i,i),
\qquad
s_{\mathrm{new}}=\frac1T\sum_{t=0}^{T-1}\gamma_t(y_t).
$$

The denominators count three transitions and four observations. Hold the old posterior q(x)=pθold(x|y) fixed. Then log Znew−log Zold=log E_q[pθnew(x,y)/pθold(x,y)]≥E_q log[pθnew/pθold]. Finite states and positive parameters justify Jensen. These tied Bernoulli updates maximize the expected complete-data log likelihood, making the right side nonnegative. Exact EM therefore does not decrease the training sequence's evidence. It does not guarantee a global optimum or performance on new data.

$$
Q(\theta\mid\theta_{\mathrm{old}})
=\sum_x q(x)\log p_\theta(x,y),\qquad
\log Z_{\mathrm{new}}-\log Z_{\mathrm{old}}
\ge Q(\theta_{\mathrm{new}}\mid\theta_{\mathrm{old}})
-Q(\theta_{\mathrm{old}}\mid\theta_{\mathrm{old}})\ge0.
$$

| Stage, observations 0,0,1,1 | r | s | Evidence |
|---|---|---|---|
| Initial | .800000 | .900000 | .058276 |
| One EM step | .674656 | .909637 | .068564 |

The code checks γ/ξ consistency and independently sums all sixteen latent paths using fractions. The screen performs inference at supplied r,s; the EM update is a separate executable example. In an unrestricted HMM, a transition row with zero expected visits needs separate treatment because its update is unidentified. The tied total-count denominators here avoid that row-wise zero division.

The earlier Kalman formula also has assumptions: independent Gaussian prior and observation noise with P>0,R>0. The posterior quadratic precision is P⁻¹+H²/R and its linear coefficient is m/P+Hy/R. Completing the square yields the stated mean and variance. A nonlinear observation need not preserve a Gaussian posterior.

**Exercise:** Does substituting Viterbi path counts for ξ retain this EM proof? **Explanation:** No. Replacing expectation over the old posterior by one path no longer maximizes the same Jensen lower bound.

Additional source: [Rabiner (1989), §III on forward/backward and reestimation](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf). The tied two-parameter update and fraction example are derived here.

```python
from fractions import Fraction as F
from itertools import product

y = [0,0,1,1]
def infer(r,s):
    transition = lambda i,j: r if i==j else 1-r
    emission = lambda t,i: s if i==y[t] else 1-s
    alpha = [[F(1,2)*emission(0,i) for i in [0,1]]]
    for t in range(1,4):
        alpha.append([emission(t,j)*sum(alpha[t-1][i]*transition(i,j)
                      for i in [0,1]) for j in [0,1]])
    beta = [[F(1),F(1)] for _ in range(4)]
    for t in range(2,-1,-1):
        beta[t] = [sum(transition(i,j)*emission(t+1,j)*beta[t+1][j]
                       for j in [0,1]) for i in [0,1]]
    Z = sum(alpha[-1])
    gamma = [[alpha[t][i]*beta[t][i]/Z for i in [0,1]] for t in range(4)]
    xi = [[[alpha[t][i]*transition(i,j)*emission(t+1,j)*beta[t+1][j]/Z
            for j in [0,1]] for i in [0,1]] for t in range(3)]
    for t in range(4):
        assert sum(alpha[t][i]*beta[t][i] for i in [0,1]) == Z
    for t in range(3):
        for i in [0,1]:
            assert sum(xi[t][i]) == gamma[t][i]
            assert sum(xi[t][j][i] for j in [0,1]) == gamma[t+1][i]
    # Independent complete-path enumeration checks both node and edge marginals.
    paths = []
    for x in product([0,1],repeat=4):
        w = F(1,2)
        for t in range(4):
            w *= emission(t,x[t])
            if t: w *= transition(x[t-1],x[t])
        paths.append((x,w))
    assert sum(w for x,w in paths) == Z
    for t in range(4):
        assert sum(w*x[t] for x,w in paths)/Z == gamma[t][1]
    for t in range(3):
        for i,j in product([0,1],repeat=2):
            assert sum(w for x,w in paths if x[t]==i and x[t+1]==j)/Z == xi[t][i][j]
    return Z,gamma,xi

r,s = F(4,5),F(9,10)
Z,gamma,xi = infer(r,s)
r_new = sum(xi[t][i][i] for t in range(3) for i in [0,1])/3
s_new = sum(gamma[t][y[t]] for t in range(4))/4
new_Z,_,_ = infer(r_new,s_new)
assert new_Z >= Z
for label,rr,ss,zz in [('old',r,s,Z),('one EM step',r_new,s_new,new_Z)]:
    print(label, 'r,s,evidence:', *(round(float(v),6) for v in [rr,ss,zz]))
```


## Exercises with explanations

1. Which method uses future observations? **Smoothing.**
2. What if emissions have accuracy .5? **They do not distinguish states, leaving marginals .5.**
3. Do marginal modes give a joint MAP path? **Not generally: distinguish a path objective such as Viterbi.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
