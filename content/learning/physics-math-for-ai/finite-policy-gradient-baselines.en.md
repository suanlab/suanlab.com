## Why differentiate a policy?

A stochastic policy maps a state to an action distribution. Improving its parameters means increasing expected reward, rather than merely increasing the probability of actions that happened to succeed. A high observed reward can reflect a lucky transition. We therefore distinguish the expected gradient from the random gradient estimate obtained from one trajectory. The interactive diagram displays the entire distribution of that estimate in a tiny model.

## Fix the order of events

Start in state s0=0. Choose a0 in {0,1}, receive r0=1 when a0=s0 and zero otherwise, then transition to s1. The probability of s1=1 is alpha after a0=1 and 1-alpha after a0=0. Choose a1 in state s1, receive r1=1 when a1=s1, and terminate. Return is G=r0+gamma*r1. There are eight paths coded 4*a0+2*s1+a1. At alpha=1 some paths have probability zero.

Let p=sigmoid(theta0) and q=sigmoid(theta1) be action-one probabilities in states zero and one. Transitions and rewards are known and independent of theta. This last assumption matters: differentiating a parameter-dependent environment would require additional terms.

## An independent expression for the value

The probability of visiting state one is w=1-alpha+p*(2*alpha-1). Conditional on the terminal state, expected reward is 1-p or q. Consequently,

$$
J=1-p+\gamma[(1-w)(1-p)+wq].
$$

Both the choice at state zero and the induced state visitation change with p. Differentiating gives

$$
\nabla_\theta J=\left(p(1-p)\{-1+\gamma[-(1-w)+(2\alpha-1)(q-1+p)]\},\ \gamma wq(1-q)\right).
$$

Ignoring the derivative of w would omit the effect of the first action on later opportunities. The implementation compares this expression with a separate path expectation, so agreement does not rely on differentiating the same enumeration twice.

## Scores, reward-to-go and baselines

A Bernoulli logit has score a-p in state zero and a-q in state one. Put each score in its corresponding parameter coordinate, calling the resulting vectors z0 and z1. Because transition factors do not depend on theta, the trajectory score is z0+z1. Differentiating the finite probability sum yields E[G(z0+z1)]=gradient J.

The past reward r0 has zero expected product with z1 conditional on the history before a1. We may remove it from that score term. Moreover, E[z_t|state and history before action]=0. Any baseline independent of the current action therefore preserves the expectation:

$$
g=(G-b_0)z_0+\gamma(r_1-b_1(s_1))z_1,\qquad E[g]=\nabla_\theta J.
$$

Here b0=lambda*J and b1(s)=lambda*V1(s), where V1(0)=1-p and V1(1)=q. The baseline and return are held fixed inside this actor score estimator. Backpropagating through a learned baseline as part of the actor surrogate introduces additional derivatives and is not the identity just proved. A critic may be trained separately; an arbitrary critic does not automatically satisfy stronger actor-critic convergence assumptions.

## A hand calculation about variance

At p=q=1/2, alpha=4/5 and gamma=1, the expected return is one and the gradient is (-3/8,1/8). The full-return estimator has trace covariance 139/160. Reward-to-go with baseline scale lambda has

$$
\operatorname{tr}\operatorname{Cov}(g)=\frac5{16}\lambda^2-\frac{59}{80}\lambda+\frac{83}{160}.
$$

Scale zero gives 83/160; scale one gives 3/32. Completing the square gives the minimum at 59/50, with variance 669/8000. Thus an exact state-value baseline reduces variance here but does not minimize it. This trace metric uses the displayed logit coordinates; changing parameterization can change its numerical value and optimal baseline.

## Read the visualization

Blue circles represent path estimates, with area proportional to path probability. Equal estimates may overlap. An outline identifies the selected positive-mass path; the orange point is the directly computed gradient. A zero-probability selected path has no distribution marker and its sample readout is unavailable. The lower blue curve varies baseline scale; the orange horizontal curve is full-return variance. Neither curve is a training trajectory.

## Research extension: an unbiased gradient does not certify a finite update

For a finite-horizon MDP with $V_H^\pi=0$, define $A_t^\pi=Q_t^\pi-V_t^\pi$. Telescoping the old value function along a **new-policy** trajectory yields

$$J(\pi')-J(\pi)=E_{\pi'}\sum_{t=0}^{H-1}\gamma^t A_t^\pi(s_t,a_t).$$

Replacing new-policy state occupancy by old occupancy gives a surrogate, not this identity. Consider a separate two-stage MDP: at the start, safe terminates with zero reward; go reaches a state where good pays +1 and bad -1. Old probabilities are go=1/100, good=9/10; new probabilities are go=1/2, good=1/10. The change in visitation overwhelms the positive old-occupancy surrogate.

| Quantity | Exact value |
|---|---:|
| Old expected reward | 1/125 |
| Old-occupancy advantage surrogate | 47/125 |
| Actual reward improvement | -51/125 |

```python
from fractions import Fraction as F
p,q=F(1,100),F(9,10); newp,newq=F(1,2),F(1,10)
old=p*(2*q-1); new=newp*(2*newq-1)
first=(newp-p)*(2*q-1)
terminal=2*(newq-q)
surrogate=first+p*terminal
identity=first+newp*terminal
assert old==F(1,125) and surrogate==F(47,125)
assert identity==new-old==F(-51,125)
print(old,surrogate,identity)
```

The native nonnegative-reward model remains unchanged. Baselines and reward-to-go can preserve gradient means and reduce variance; these properties alone do not prove monotonic improvement under an arbitrary finite parameter step. Controlling occupancy change requires additional assumptions or a suitable bound.

## Exercise and solution

Exercise: set gamma=0. Which gradient coordinate must vanish? Can changing the terminal logit still change the expected immediate reward?

Solution: J=1-p, so the second coordinate is zero and the first is -p*(1-p). The second score is multiplied by gamma in the reward-to-go estimator. The terminal policy cannot affect an already received reward. The full-return estimator can still have sample noise from that future score, although its expected contribution is zero.

Exercise: does a smaller variance at lambda=1.18 establish global convergence of policy optimization? Solution: no. The calculation concerns an unbiased estimator distribution at fixed parameters. Step sizes, changing distributions, optimization geometry and approximation errors require separate analysis.

## Research connection

The general policy-gradient framework is developed by [Sutton, McAllester, Singh and Mansour, NeurIPS 1999](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf). Its function-approximation results have specific compatibility assumptions. The finite MDP, variance polynomial and exercises above are worked derivations for this textbook, not claims of general convergence or an implementation of a trained actor-critic.
