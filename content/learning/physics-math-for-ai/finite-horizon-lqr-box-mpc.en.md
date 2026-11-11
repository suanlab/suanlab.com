## Choose the behavior before choosing the controller

Suppose a measured state is a deviation from a target. A large input can remove that deviation quickly, but an actuator has a limit and using it has a cost. Optimal control makes this compromise explicit. A learned dynamics model does not choose the compromise for us: the objective, constraints and time convention remain modeling decisions.

This lab uses the discrete scalar system x_(t+1)=a x_t+u_t, with input coefficient one and a fully observed, noise-free state. One step is the chosen sampling interval. Neither a nor the input should be read as a continuous-time derivative. The model is deliberately small enough that every optimization can be checked independently.

## Specify the finite-horizon problem

Fix the current state x0 and a positive integer planning horizon H. The decision variables are the H inputs, with cost

$$
J=\sum_{t=0}^{H-1}(q x_t^2+r u_t^2)+p x_H^2,
\qquad x_{t+1}=a x_t+u_t.
$$

Here q and p are nonnegative and r is strictly positive. The q*x0² term is constant during the current optimization, but belongs in the reported cost. The terminal state receives p, rather than both q and p. These details matter when checking numerical answers.

A finite-horizon controller asks what is best over these H steps under this terminal valuation. It is not automatically the infinite-horizon optimum. When q=p=0, minimizing input cost chooses zero input even if the open-loop system grows. An objective that ignores the target cannot be expected to stabilize it.

## Derive LQR by completing one square

Without input constraints, let the optimal remaining cost be V_t(x)=P_t*x². Begin at P_H=p. If the next value coefficient is P_(t+1), the one-step minimization is

$$
q x^2+r u^2+P_{t+1}(a x+u)^2.
$$

The coefficient of u² is r+P_(t+1)>0. Completing the square gives

$$
K_t=\frac{aP_{t+1}}{r+P_{t+1}},\qquad u_t^*=-K_t x_t,
$$

$$
P_t=q+\frac{a^2 r P_{t+1}}{r+P_{t+1}}.
$$

This backward recursion is the discrete finite-horizon Riccati equation for this particular input coefficient. The gain generally changes with time because the number of remaining stages changes. More precisely, the expression being minimized is P_t*x²+(r+P_(t+1))*(u+K_t*x)². Summing these identities proves the optimal value P0*x0² and identifies the penalty for deviating from the feedback along a trajectory.

## A bounded actuator changes the optimization

Now impose |u_t|<=U at every planned time. Eliminating the state gives

$$
x_t=a^t x_0+\sum_{j=0}^{t-1}a^{t-1-j}u_j.
$$

Write the resulting cost as c+2*fᵀu+uᵀM*u. The matrix M is r*I plus positive semidefinite state-cost contributions, so it is positive definite. There is exactly one optimum in the compact input box. No state limits or terminal equality are imposed here, and the zero input sequence is always feasible. Feasibility therefore says nothing about reaching zero or remaining in a desired state region.

The lab enumerates the small box's faces: each input is free, at its lower bound or at its upper bound. For each face it solves the free positive-definite linear system, keeps feasible candidates and chooses the least cost. At H<=5 this is at most 3^5 faces. This explicit teaching solver is not intended to replace scalable sparse QP software.

The reported projected KKT residual is the largest component of u-proj_box(u-gradient J). At the solution it is approximately zero. A lower-bound input can have a positive derivative, an upper-bound input a negative derivative, and a free input a zero derivative. Strict convexity makes these conditions sufficient for the global optimum. Small numerical residuals are useful evidence for this computed plan; they are not a closed-loop stability proof.

## Clipping and planning ahead are different

A simple alternative clips the unconstrained feedback into the actuator range. For a one-step problem, this is optimal because it projects the minimum of a one-dimensional convex quadratic. Over several steps, a future saturated actuator can alter what should be done now.

The third guided example uses a=0.8,q=0,r=1,p=3,H=5,x0=2,U=0.2. The unconstrained first input is about -0.095432, already inside the bound. Yet the constrained optimum starts near -0.103150 because later planned inputs reach -0.2. Clipping the shrinking-horizon LQR sequence costs about 0.157044, versus 0.154381 for the joint constrained plan. The difference compares identical initial states, horizons and terminal costs. It does not establish that one controller wins under every objective or every model.

## Replan, execute one input, then observe again

Model predictive control solves a plan, applies only its first input, observes the next state and solves again with the same horizon H. The state–input diagram shows the **initial** constrained and clipped plans in time order. Its dots mark their first actions; its connecting lines are not a complete feedback law.

The lower time plot shows **executed receding-horizon** trajectories. Both comparators use the first Riccati gain K0 again at each new state; the clipped comparator additionally respects U. The initial clipped-plan cost instead uses K_t over its shrinking initial horizon. These are deliberately separate comparisons. The table lists actual applied inputs and each new MPC plan's value; these successive values are not costs over one common remaining terminal time.

No disturbance or model mismatch is added in this lab. Replanning is a feedback procedure, but its existence alone does not guarantee robustness or stability. Terminal invariant sets, suitable terminal costs and local feedback conditions can support such guarantees in appropriately formulated MPC. They are not assumed for the arbitrary parameters here.

## Research extension: why MPC needs a terminal certificate

An input-feasible plan today need not remain feasible tomorrow when state or terminal constraints are added. Suppose a terminal set $X_f$ is invariant under an admissible feedback $\kappa$, and $V_f(f(x,\kappa(x)))-V_f(x)\le-\ell(x,\kappa(x))$ there. Shift the optimal plan by one step and append $\kappa$ at its terminal state. This is a feasible candidate at the next state, so optimality gives $V_N(x^+)\le V_N(x)-\ell(x,u_0^*)$. Positive stage cost and suitable bounds on $V_N$ then yield stability for the exact nominal model.

For the auxiliary integrator $x^+=x+u$, take $\ell=x^2+u^2$, $\kappa=-x/2$, $X_f=[-1,1]$, $|u|\le1/2$, and $V_f=5x^2/3$. The terminal decrease is exactly $-5x^2/4=-\ell(x,\kappa(x))$.

| Initial state | Feasible two-input candidate | Terminal state |
|---|---|---:|
| 1 | (-1/2, -1/2) | 0 |
| 3/2 | (-1/2, -1/2) | 1/2 |
| 2 | (-1/2, -1/2) | 1 |

```python
from fractions import Fraction as F
def cost(x,us):
    J=F(0)
    for u in us:
        J+=x*x+u*u; x+=u
    return J+F(5,3)*x*x,x
for x in [F(1),F(3,2),F(2)]:
    us=[F(-1,2),F(-1,2)]
    J,end=cost(x,us)
    tail=-end/2
    shifted,_=cost(x+us[0],[us[1],tail])
    assert abs(end)<=1 and abs(tail)<=F(1,2)
    assert shifted==J-(x*x+us[0]**2)
    print(x,end,shifted)
```

The code verifies feasible candidates, not optimality of those initial plans. The native arbitrary terminal weight and input-only MPC do not acquire this certificate. Its Riccati recursion divides before multiplying tiny penalties; `expandedP` and `expandedK` retain the former arithmetic for comparison, without a numerical floor.

## Worked exercise and solution

Set a=q=r=p=1,H=1,x0=1. Then J=1+u²+(1+u)²=1.5+2*(u+0.5)². With U=2, the optimum is u=-0.5, the next state is 0.5 and the cost is 1.5. With U=0.2, the minimum moves to u=-0.2, the next state is 0.8 and the cost is 1.68. The derivative there is positive, consistent with a lower-bound optimum.

As a second exercise set H=1,p=0 and q=1. The chosen action is zero because the only state cost is the current fixed x0². The controller cannot value a state after its horizon unless a cost or constraint expresses that intention. Increasing the horizon and selecting a meaningful terminal treatment change the problem rather than merely improving a solver.

## Connection to AI research

Learned world models can supply the prediction inside MPC; differentiable planning can expose gradients through selected optimization problems. Both require attention to active constraints, model error, observation uncertainty and solver accuracy. LQR offers an exactly checkable reference around a local linear model. The present scalar examples establish the distinction between an unconstrained policy, a clipped heuristic and a jointly constrained plan before larger learned-control systems are introduced.

## Primary references

Russ Tedrake, [Linear Quadratic Regulators](https://underactuated.mit.edu/lqr.html) and [Trajectory Optimization](https://underactuated.mit.edu/trajopt.html), provide the broader control and optimization setting. The scalar discrete derivation, bounded example, face-enumeration solver and independent numerical checks here are original teaching constructions.
