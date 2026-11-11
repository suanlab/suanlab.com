## Universal approximation is an existence statement

A typical result concerns a continuous target on a compact domain: with appropriate activation, biases and sufficient width, a finite network can achieve any prescribed uniform accuracy. Continuous nonpolynomial activations such as ReLU support standard results when their other hypotheses hold.

$$
\forall f\in C(K),\ \forall\varepsilon>0,\ \exists g_\theta:\|f-g_\theta\|_\infty<\varepsilon.
$$

One fixed width is not promised to work for every target and tolerance. Neither practical size nor a successful training algorithm follows merely from existence.

## Assemble a ReLU approximation

Split [−1,1] into n intervals and interpolate x². Initial value is one, initial slope s₀=−2+2/n, and each internal knot tⱼ=−1+2j/n increases slope by 4/n.

$$
g_n(x)=1+s_0(x+1)+\sum_{j=1}^{n-1}\frac4n\max(0,x-t_j).
$$

This uses an affine skip plus n−1 ReLU hinges. Four intervals require three hinges and give error 1/16; eight give 1/64. The interval length is 2/n and quadratic interpolation error is h²/4. This construction illustrates one target, not the entire general theorem.

## Why linear activations fail

Sums and compositions of affine maps remain affine. The best uniform affine approximation of x² on [−1,1] is constant 1/2:

$$
\inf_{a,b}\sup_{|x|\le1}|x^2-(a+bx)|=1/2.
$$

Symmetry of the endpoints and the center establishes the lower bound. The lab's linear mode shows that best baseline. The n=1 ReLU interpolation construction instead gives constant one and error one. There is no contradiction: a specified construction need not be optimal in its family.

## Separate research questions

Representability, compact representation, finite-sample generalization and efficient optimization are distinct. Depth-efficiency claims also need a target class and cost model. A small successful width experiment cannot settle all of them.

## Prove the one-dimensional continuous-target result constructively

Take any continuous f on [0,1] and knots 0=t₀<⋯<t_n=1 of maximum spacing h. On each interval, the interpolant is a convex combination of endpoint values. Its difference from f(x) is therefore bounded by ω_f(h). Choose h small enough to make this less than ε. If s_j is the j-th secant slope, the entire interpolant is

$$
I_hf(x)=f(0)+s_0x+\sum_{j=1}^{n-1}(s_j-s_{j-1})\operatorname{ReLU}(x-t_j),
\qquad\|I_hf-f\|_\infty\le\omega_f(h).
$$

To verify the representation, start with the first slope and observe that every hinge changes it by exactly the next slope difference. Values agree at zero and continuity matches adjacent intervals. The affine skip can itself be represented with ReLU(x)−ReLU(−x) and an output constant. This proves density for a single input interval with biases and variable width; the general multidimensional theorem needs its own argument.

For an L-Lipschitz f, the endpoint convex combination improves the interpolation bound to Lh/2: its weighted distances sum to 2(x−a)(b−x)/(b−a)≤h/2. No optimization algorithm was used to find these coefficients; the construction assumes access to the chosen function values.

## Knot placement can matter more than a larger uniform width

The separate target f(x)=|x−1/3| has a corner. It is exactly 2ReLU(x−1/3)−x+1/3, but its uniform-grid interpolants need not improve monotonically when their grids are not nested. Both functions are affine away from knots and the corner, so their maximum difference occurs at one of those finitely many points.

| Intervals n | Exact uniform-grid error | Reason |
|---|---|---|
| 2 | 2/9 | Corner inside a cell |
| 3 | 0 | Corner is a knot |
| 4 | 1/9 | Corner inside a cell again |
| 6 | 0 | Corner is a knot |

The code checks this different target exactly. It adds no corner-location control to the x² lab. A ReLU network with every bias removed, including the output bias, satisfies g(0)=0 and cannot uniformly approximate the constant one on an interval containing zero. Nonlinearity alone does not preserve every version of the theorem when architecture assumptions change.

**Exercise:** Does the n=3 to n=4 increase refute density? **Explanation:** No. Density asserts the existence of some sufficiently accurate networks, not monotone improvement for every prescribed sequence of coefficient rules.

```python
from fractions import Fraction as F

corner=F(1,3)
def target(x): return abs(x-corner)
for n in [2,3,4,6]:
    knots=[F(j,n) for j in range(n+1)]
    values=[target(x) for x in knots]
    slopes=[n*(values[j+1]-values[j]) for j in range(n)]
    def network(x):
        return values[0]+slopes[0]*x+sum(
            (slopes[j]-slopes[j-1])*max(F(0),x-knots[j])
            for j in range(1,n))
    j=min(n-1,int(corner*n))
    lo,hi=knots[j:j+2]
    interpolated=values[j]+(corner-lo)*slopes[j]
    error=interpolated-target(corner)
    assert network(corner)==interpolated
    # On each side of the corner both functions are affine;
    # their maximum absolute difference is attained at a knot or the corner.
    assert max(abs(network(x)-target(x)) for x in knots+[corner])==error
    assert error<=F(1,2*n)
    assert 2*max(F(0),corner-corner)-corner+corner==target(corner)
    print(n,str(error),float(error))
```

## Exercises with explanations

1. Does this finite ReLU spline equal x² everywhere? **No.** It is piecewise linear and approaches it as width grows.
2. Can biases be removed without rechecking assumptions? **No.** Even control of knot locations changes.
3. How large n ensures error below ε here? **Choose n>1/√ε**, specifically for this construction and target.

Further reading: [Leshno et al., Nonpolynomial Activation and Approximation](https://doi.org/10.1016/S0893-6080(05)80131-5).
