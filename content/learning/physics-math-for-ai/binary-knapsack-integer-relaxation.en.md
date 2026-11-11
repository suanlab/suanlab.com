## Why a fraction may not be a decision

Suppose two projects require indivisible approvals. A continuous optimizer might recommend funding one whole project and one third of another, but that recommendation is unavailable when each project must be accepted or rejected. Integer programming represents the indivisibility explicitly. A relaxation helps reason about the original problem by enlarging its feasible set; it does not automatically turn fractional recommendations into executable decisions.

The interactive model has only four binary choices so we can inspect every candidate. This small enumeration is a transparent reference calculation, not a claim that large integer programs are easy. Later chapters develop combinatorial structure and formal search separately.

## State the model before relaxing it

Item zero has weight two and item one has weight three. Their nonnegative values are v0 and v1. The decision variables x0,x1 indicate whether each item is selected. In at-most mode,

$$
\max\ v_0x_0+v_1x_1\quad\text{subject to}\quad 2x_0+3x_1\le C,\quad x_0,x_1\in\{0,1\}.
$$

Exact-load mode replaces only the load inequality by equality. It asks for a different task: use exactly C units of weight. The four binary loads are 0,3,2,5 for codes 0,1,2,3 respectively, where code=2*x0+x1. At-most mode always permits the empty choice. Exact mode has an integer solution only when C is one of 0,2,3,5.

The LP relaxation retains the objective and load constraint but allows x0,x1 in [0,1]. Its feasible set is the unit square intersected with a half-plane or a line. For C between zero and five, this continuous set is nonempty in both modes.

## Prove the direction of the bound

Every feasible binary point is a feasible continuous point. Therefore, whenever an integer optimum exists,

$$
z_{\mathrm{IP}}\le z_{\mathrm{LP}}.
$$

The inequality points this way because we maximize over a larger set. For minimization a relaxation gives a lower bound instead. A feasible integer candidate supplies a lower bound on the maximization optimum; an infeasible rounded candidate supplies no such certificate.

The additive gap is z_LP-z_IP. This unit also reports z_LP/z_IP when the integer optimum is strictly positive. A zero denominator makes that ratio unavailable even if both values are zero. If the integer problem is infeasible, its optimum, gap and ratio are all unavailable. Assigning zero would incorrectly confuse absence of any feasible decision with a feasible decision of zero value.

## Find a continuous optimum without hiding the geometry

A linear objective over a nonempty compact polygon has a maximizing vertex; a segment or singleton is included as a degenerate case. Enumerate square corners satisfying the constraint and intersections of the load boundary with square edges. Evaluate the objective at each resulting vertex. All comparisons use rational arithmetic on the canonical decimal strings of the inputs. Floating displays are rounded, while feasibility and ties are not decided by adding an epsilon.

A selected optimizer need not be unique. The model chooses the lexicographically first maximizing vertex, but also highlights all maximizing vertices. When value densities v0/2 and v1/3 are equal and positive, a whole feasible load-boundary segment can be optimal. With both values zero, every feasible point is optimal.

For this particular one-constraint model, fractional knapsack gives a separate explanation: allocate capacity to the larger value-per-weight density first. Exchanging a small amount of load from a lower-density item to a higher-density unsaturated item improves value. This does not justify a density rule for arbitrary integer programs or multiple resource constraints.

## Work the upper bound and rounding examples

Set values to (3,4) and C=3 in at-most mode. Item zero has density 3/2, exceeding 4/3. The LP selects (1,1/3), with

$$
z_{\mathrm{LP}}=3+4/3=13/3.
$$

The feasible binary points are (0,0), (1,0) and (0,1). Their values are 0,3,4, so the integer optimum is (0,1), with value four and additive gap 1/3. The selected query point can be feasible without being optimal.

At C=4, the LP selects (1,2/3), value 17/3. Independently rounding each coordinate to its nearest binary value gives (1,1), weight five, which violates capacity four. The display uses half-up rounding at one half and checks the result against the original constraint. Rounding is a proposal requiring validation, not an exact solver or an approximation guarantee.

Switch that C=4 problem to exact-load mode. The LP remains feasible, but no binary load equals four. The integer marker disappears and its value stays unavailable. The continuous optimum cannot prove that an integer solution exists.

## Read the diagrams and capacity curves

The native diagram uses equal axis scales. Blue shading or a blue segment is the relaxed feasible set. Blue circles select an LP optimum, orange dots select an integer optimum and orange rings mark every integer optimum. Crosses are infeasible binary points. A thin ring selects the query binary point; a dashed square shows the rounded LP point. Overlapping markers can represent the same coordinates with different meanings.

The lower horizontal axis varies C, not optimization iterations. Blue is the LP value. Orange is the integer value: right-open horizontal segments plus closed dots in at-most mode, and only isolated feasible loads in exact mode. Missing exact-load capacities are not joined by fictitious integer values. The current LP value also has a blue marker. Very small positive exact fractions can round to zero in a floating display; the exact fraction in the model statement preserves that distinction. A ratio beyond the display range is marked separately from a zero denominator.

## Research extension: certify the relaxation without trusting its solver

For the at-most model, introduce a resource price $\lambda\ge0$ and upper-bound prices $\beta_i\ge0$ satisfying $w_i\lambda+\beta_i\ge v_i$. For every feasible relaxed point,

$$v^Tx\le\lambda w^Tx+\beta^Tx\le C\lambda+\sum_i\beta_i.$$

This is weak duality, proved by multiplying inequalities rather than by trusting an optimizer. The dual minimizes that final expression. If a feasible primal point reaches the dual bound, both are optimal. A learned proposal for either side becomes a certificate only after its feasibility is checked.

For values $(3,4)$ and weights $(2,3)$, the examples below exhibit matching primal and dual values. At capacity three or four, take $\lambda=4/3$ and $\beta=(1/3,0)$. At capacity one, take $\lambda=3/2$, $\beta=(0,0)$.

| Capacity | Primal LP point | Certified LP value | Integer optimum |
|---|---|---:|---:|
| 1 | (1/2, 0) | 3/2 | 0 |
| 3 | (1, 1/3) | 13/3 | 4 |
| 4 | (1, 2/3) | 17/3 | 4 |

```python
from fractions import Fraction as F
from itertools import product
values=[F(3),F(4)]; weights=[F(2),F(3)]
for C,x,price,beta in [(1,[F(1,2),F(0)],F(3,2),[F(0),F(0)]),(3,[F(1),F(1,3)],F(4,3),[F(1,3),F(0)]),(4,[F(1),F(2,3)],F(4,3),[F(1,3),F(0)])]:
    assert sum(w*z for w,z in zip(weights,x))<=C
    assert all(w*price+b>=v for w,b,v in zip(weights,beta,values))
    primal=sum(v*z for v,z in zip(values,x)); dual=C*price+sum(beta)
    assert primal==dual
    integer=max(sum(v*z for v,z in zip(values,y)) for y in product([0,1],repeat=2) if sum(w*z for w,z in zip(weights,y))<=C)
    assert integer<=dual
    print(C,primal,integer)
```

Complementary slackness explains the prices: a positive resource price requires a tight resource constraint; positive $\beta_i$ requires $x_i=1$; an item strictly below its reduced-value threshold stays unselected. This certifies the LP, not an integral solution. In exact-load mode the resource multiplier is unrestricted in sign; importing the nonnegative-price dual unchanged would describe a different problem.

## Exercises and solutions

Exercise: with values (3,4), C=1 and at-most mode, compute the values and decide whether the multiplicative gap is defined.

Solution: only the empty binary choice fits, so z_IP=0. The LP chooses (1/2,0), giving z_LP=3/2 and additive gap 3/2. The ratio is undefined because its denominator is zero. This is different from integer infeasibility: the empty choice exists.

Exercise: choose values (2,3), C=3. Is the selected LP optimum unique? Solution: the objective equals the load. Every feasible point on the load-three boundary segment is optimal. A deterministic selected vertex is a display convention, not a uniqueness theorem. The integer point (0,1) attains the same value even though other LP optimizers are fractional.

Exercise: does equality of optimal values imply that every relaxed optimizer is integral? Solution: no, as that equal-density example demonstrates. It establishes an integral optimizer exists here, not that the entire optimal face contains only binary points.

## Further study

[James B. Orlin's MIT integer-programming reference guide](https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/86d472ffa3f1c341c586cb26ba1093c1_MIT15_053S13_iprefguide.pdf) develops binary subset-selection formulations and the role of relaxation bounds. The two-item calculations, equality-mode counterexample and exact comparisons above are worked examples for this textbook. Branch-and-bound, approximation guarantees and differentiable relaxations require additional reasoning in later units.
