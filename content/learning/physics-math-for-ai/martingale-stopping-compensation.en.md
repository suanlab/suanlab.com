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

## Exercise: change the seed, not the theorem

Compare two seeds in that biased example. Their sampled stopped states can differ, but the exact probability recursion and its expectations are unchanged. Next use B=1: every path stops after the first step. At displayed time eight, correct compensation still subtracts only d once. Calculate the incorrect value produced by subtracting eight d instead, and explain why it does not contradict the martingale theorem.

Further reading: [Lawler's martingale and stopping-time notes](https://www.math.uchicago.edu/~lawler/finbook.pdf). The finite recursion and enumerated example are independently implemented.
