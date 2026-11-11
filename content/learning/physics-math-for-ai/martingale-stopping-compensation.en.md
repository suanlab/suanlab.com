## A martingale is a conditional-mean statement

A process M_n is a martingale relative to its information filtration when it is integrable, adapted and E[M_(n+1)|F_n]=M_n. It is not defined by a path looking flat, unpredictable or oscillatory. One realization can move strongly in either direction while the conditional-mean relation remains exact.

Use a random walk with increments plus one with probability p and minus one otherwise. Its predictable drift is d=2p-1. The raw walk is a martingale only when that drift vanishes. Subtracting the accumulated predictable drift gives

$$
M_n=S_n-dn,\qquad E[\Delta M_n\mid\mathcal F_n]=0.
$$

The same compensation idea appears in the Poisson martingale N(t)-lambda t. The experiment uses the walk so all bounded stopped expectations can be calculated through a finite probability recursion rather than estimated from one simulation.

## Stop the clock as well as the state

Let tau be the first hit of either barrier plus or minus B, capped by N. This is a bounded stopping time. At a displayed intermediate limit n, the stopped process is M_(min(tau,n)). Its compensation uses the stopped time, not the wall-clock index after hitting a barrier.

$$
M_{\tau\wedge n}=S_{\tau\wedge n}-d(\tau\wedge n).
$$

Continuing to subtract dn while the state remains frozen would create artificial drift. The upper diagram shows an actual stopped raw and compensated path. The main diagram instead shows exact expected values, which need not resemble that single path.

## Verify bounded optional stopping explicitly

The model tracks probabilities of both position and stopping time. Live states branch with probabilities p and 1-p; already-stopped states preserve their stopped clock. Summing their compensated values gives zero expectation up to floating-point rounding. Boundedness supplies the integrability control required here; it is not a license to apply optional stopping at every unbounded random time.

At p=0.75, B=2 and N=4, the probability of hitting at time two is p²+(1-p)²=0.625. Otherwise the stopped time is four. Thus E[tau]=2.75 and E[S_tau]=0.5 times 2.75=1.375, while E[M_tau]=0.

## Why an almost surely finite stopping time may still fail

For a fair walk starting at zero, stop at either +1 or −B. This stopping time has finite expectation B, and the stopped position is bounded. Solving the affine harmonic boundary problem gives p=Pr(exit at +1)=B/(B+1). Therefore E[Sτ]=p−B(1−p)=0. Optional stopping is justified here; the finite expectation also follows by solving the quadratic mean-exit-time recursion.

Now let B increase. The fair walk eventually hits +1 almost surely, but at that unbounded hitting time its position equals 1, so E[Sτ]=1. The rare lower exit has probability 1/(B+1) and magnitude B. Its expectation contribution tends to −1 rather than disappearing. The bounded stopped positions in this sequence are not uniformly integrable, so expectation cannot be passed through the limit.

$$
E[|S_{\tau_B}|\,\mathbf1_{\{|S_{\tau_B}|>K\}}]
\ge B/(B+1)\quad(B>\max(K,1)).
$$

| B | Upper-exit probability | Lower-exit expectation contribution |
|---|---:|---:|
| 1 | 1/2 | −1/2 |
| 3 | 3/4 | −3/4 |
| 9 | 9/10 | −9/10 |

```python
from fractions import Fraction as F
for B in (1, 3, 9):
    upper = F(B, B+1)
    lower_contribution = -B*(1-upper)
    assert upper+lower_contribution == 0
    # Harmonic probability and expected time for -B <= x <= 1.
    probability = lambda x: F(x+B, B+1)
    duration = lambda x: (x+B)*(1-x)
    for x in range(-B+1, 1):
        assert probability(x)==(probability(x-1)+probability(x+1))/2
        assert duration(x)==1+F(duration(x-1)+duration(x+1),2)
    assert duration(0)==B
    print(B, upper, lower_contribution, duration(0))
```

The rational harmonic calculation exposes the missing tail control. The live two-sided bounded-horizon recursion remains valid. A stopping rule selected by observed performance cannot be justified by an unconditional mean alone; the information, integrability, and stopping hypotheses must all be checked.

## Exercise: change the seed, not the theorem

Compare two seeds in that biased example. Their sampled stopped states can differ, but the exact probability recursion and its expectations are unchanged. Next use B=1: every path stops after the first step. At displayed time eight, correct compensation still subtracts only d once. Calculate the incorrect value produced by subtracting eight d instead, and explain why it does not contradict the martingale theorem.

Further reading: [Lawler's martingale and stopping-time notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The finite recursion and enumerated example are independently implemented.
