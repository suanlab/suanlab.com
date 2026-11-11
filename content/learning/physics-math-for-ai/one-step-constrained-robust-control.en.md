## A small cost does not establish safety

Imagine a controller steering a position toward zero while limiting actuator effort. A prediction that ignores wind may stay within a wall, yet the actual position can cross that wall. Robust control asks for a decision that respects the constraint for every disturbance in a specified set. This differs from minimizing an expected cost or allowing a small probability of failure. The uncertainty set is part of the mathematical claim: an unmodeled larger disturbance is not covered by the guarantee.

This chapter makes one decision. It extends the earlier input-constrained LQR/MPC example by adding uncertainty and a hard constraint on the next state. It also shows when no admissible decision exists, rather than displaying a clipped input as a supposed safe optimum.

## Specify who acts first

The controller observes a scalar state x, chooses u, and then a disturbance w occurs. The dynamics and constraints are

$$
x^+=ax+u+w,\qquad |u|\le U,\quad |w|\le d,\quad |x^+|\le L.
$$

The same chosen u must work for every allowed w. Choosing a different u after observing w would solve a different problem with more information. The coefficient a and state x are known. All quantities use a consistent nondimensional scale, and q is a nonnegative state penalty. There is no assumed probability distribution on w.

## Turn infinitely many constraints into an interval

Write c=ax. For fixed u, successors fill [c+u-d,c+u+d]. Their maximum absolute magnitude is |c+u|+d: an endpoint aligned with the center attains the triangle-inequality bound. Consequently safety requires d<=L and

$$
u\in\mathcal F=[-U,U]\cap[-L+d-c,\ L-d-c].
$$

Let lo and hi be the intersection endpoints. A feasible input exists exactly when d<=L and lo<=hi. If lo=hi there is a single feasible input, not an empty set. If d>L even a perfectly centered successor interval is too wide. If actuator authority is too small, the center cannot be moved far enough. Increasing uncertainty tightens the feasible interval; increasing actuator authority can expand it but cannot fix d>L.

## Derive the min-max cost

The worst-case objective is input effort plus weighted squared successor:

$$
F(u)=\max_{|w|\le d}\{u^2+q(c+u+w)^2\}
=u^2+q(|c+u|+d)^2.
$$

It is strictly convex because the input square supplies positive curvature. Set z=c+u. Away from z=0, differentiate the corresponding quadratic. At z=0 use its interval of subgradients. The unconstrained optimum is

$$
z_0=\operatorname{sign}(c)\max\left(\frac{|c|-qd}{1+q},0\right),\qquad u_0=z_0-c.
$$

If the feasible interval is nonempty, the constrained optimum is min(hi,max(lo,u0)). This projection works here because a scalar convex objective decreases toward its unconstrained minimizer and increases afterward. It is not a rule that arbitrary multidimensional constrained problems can be solved by clipping an unconstrained optimum. Infeasibility leaves the input, cost and successor readouts unavailable.

## Research extension: compute the set that stays safe forever

One-step feasibility asks whether one input protects the next state. Recursive safety instead uses $\operatorname{Pre}(K)=\{x:\exists u\ \forall w,\ f(x,u,w)\in K\}$ and $K_{j+1}=K_0\cap\operatorname{Pre}(K_j)$. These sets describe states safe for increasing horizons; under the usual compactness and continuity conditions their intersection is a controlled invariant kernel.

For the auxiliary scalar model $x^+=2x+u+w$, $|u|\le1/2$, $|w|\le1/10$, $K_0=[-1,1]$, symmetry gives radii $r_{j+1}=\min(1,(r_j+1/2-1/10)/2)$ while $r_j\ge1/10$. They decrease to $r_*=2/5$. Saturating $u=-2x$ to the actuator interval preserves this limiting set. Outside it, the adversary can increase the outward excess until safety fails.

| Horizon | Safe radius |
|---|---:|
| 1 | 7/10 |
| 2 | 11/20 |
| Infinite | 2/5 |

![Nested safe-set radii converge from one to two fifths; one-step safety at radius seven tenths is visibly wider than the invariant kernel.](/assets/learning/control-invariant-kernel-en.svg)

```python
from fractions import Fraction as F
r=F(1); radii=[r]
for j in range(10):
    r=min(F(1),(r+F(1,2)-F(1,10))/2); radii.append(r)
assert radii[1]==F(7,10) and radii[2]==F(11,20)
limit=F(2,5)
for x in [-limit,-limit/2,F(0),limit/2,limit]:
    u=max(F(-1,2),min(F(1,2),-2*x))
    assert abs(2*x+u)+F(1,10)<=limit
assert 2*F(3,5)-F(1,2)+F(1,10)==F(4,5)
assert 2*F(4,5)-F(1,2)+F(1,10)>1
print(radii[:4],limit)
```

At $x=0.6$, one-step feasibility holds, but even the strongest inward input permits successor 0.8, from which one-step feasibility fails. The native lab remains a one-step optimizer. Its unconstrained input now uses a factored expression to preserve tiny penalties; `expandedUnconstrained` records the cancellation-prone subtraction for comparison.

## Work an exact example

Choose a=x=q=U=1, d=1/4 and L=1/2. The robust interval is [-1,-3/4]. The unconstrained robust input is -5/8, outside that interval, so u*=-3/4. Its successor interval is [0,1/2] and its worst cost is

$$
F(-3/4)=9/16+(1/4+1/4)^2=13/16.
$$

A nominal controller uses d=0 both in its model cost and safety constraint. It chooses -1/2 with nominal cost 1/2. Under the actual d=1/4, however, its successor interval is [1/4,3/4], its safety margin is -1/4 and its actual worst cost is 13/16. Equal numerical costs do not make an unsafe action acceptable. Comparing nominal cost 1/2 directly with robust cost 13/16 also compares different objectives.

For d=3/4, U=1/2 and L=1, the robust intersection is [-1/2,-3/4], which is empty. The nominal input still exists, illustrating a failure caused by uncertainty and limited control authority rather than a solver malfunction.

## Read the three rows and cost plot

The top row is an input axis; its blue segment is the feasible robust interval. The selected robust input is a point and the nominal input has an orange outline. The next two rows share a successor-state axis. Blue and orange intervals include all actual disturbances for their respective inputs; central dots mean w=0, not expected values. Shading between the safety walls identifies [-L,L]. An infeasible solution has a message instead of an invented successor interval.

The lower blue curve is F(u); orange uses the different nominal objective u²+q(c+u)². Each dot minimizes its own objective over its own admissible interval. Curves span the entire actuator range, including unsafe inputs. Read the native safe-input segment and the explicit safety margins before interpreting a low point as usable.

## Exercises and solutions

Exercise: set q=0. Does the optimizer always select zero? Solution: it minimizes u² over the safe interval. Zero is optimal only if it is feasible; with L=d and c=1, safety forces u=-1 if the actuator permits it.

Exercise: with c=1, q=4, d=1/2, U=L=2, find the robust input. Solution: qd=2 exceeds |c|, so z0=0 and u=-1, with worst cost two. The absolute-value kink is handled by a subgradient containing zero, not an ordinary derivative at the kink.

Exercise: does a safe next state guarantee a safe input exists at the following step? Solution: no. The present test does not establish invariance of the whole safe set. Repeated feasibility and stability need additional horizon, policy and terminal-set arguments.

## Beyond a one-step example

[Rawlings, Mayne and Diehl's Model Predictive Control](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-6th-printing.pdf), Chapter 3, develops robust invariance and recursive feasibility with explicit assumptions. Those broader results are further study; the interval, optimizer and examples above are direct finite-dimensional derivations. A constrained MDP with an expected resource budget or a chance constraint also has different semantics from this hard bound for every disturbance.
