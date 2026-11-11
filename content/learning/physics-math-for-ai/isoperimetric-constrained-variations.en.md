## A global integral constraint restricts directions

Minimize derivative energy J[y]=∫₀¹(y′)²/2 dt with y(0)=y(1)=0 and prescribed signed area ∫₀¹y dt=A. Negative A is allowed; it represents a curve below the baseline rather than a negative probability.

A candidate curve can satisfy the endpoints while violating area. Likewise, a direction can vanish at the endpoints while changing area. Feasibility of the current curve and admissibility of a variation are separate tests.

## Use a multiplier with an explicit sign

Define L[y,λ]=J[y]+λ(∫y−A). Integration by parts gives

$$
\delta L[y;h]=\int_0^1(-y''+\lambda)h\,dt,
\qquad -y''+\lambda=0.
$$

The endpoint conditions make y=−λt(1−t)/2. Since ∫₀¹t(1−t)dt=1/6, the area condition yields

$$
y^*(t)=6A\,t(1-t),\qquad \lambda^*=-12A,\qquad J[y^*]=6A^2.
$$

Changing the sign convention for L would change the multiplier's sign without changing the minimizing curve. Equality multipliers are not restricted to be nonnegative.

## An exact minimum, not just a stationary parabola

For a feasible difference h=y−y*, endpoints and area both vanish. The cross term is ∫y*′h′=−∫y*″h=12A∫h=0. Therefore J[y]−J[y*]=∫(h′)²/2≥0. Fixed endpoints eliminate a nonzero constant difference, giving uniqueness.

This proof uses feasibility. A lower energy at a curve with the wrong area is not a counterexample to the constrained optimum. The lab withholds a feasible objective-excess certificate for such candidates.

## Read the candidate and direction controls separately

The candidate is y=6At(1−t)+b sin 2πt+v sin πt. The second sine mode integrates to zero, while the first integrates to 2/π. Thus the candidate area residual is 2v/π and feasibility requires v=0.

The direction is h=sin 2πt+c sin πt, with area variation 2c/π. It is admissible exactly when c=0. At a constrained optimum, δJ must vanish on admissible directions; it need not vanish on arbitrary directions. The multiplier balances the latter through δL=δJ+λδG.

## Why a multiplier exists here, and when it can fail

Let G(h)=∫h on H₀¹. The direction w(t)=6t(1−t) satisfies G(w)=1, so the constraint differential maps onto the scalar constraint space. At a constrained minimizer, the differential DJ vanishes on ker G: both positive and negative small feasible steps are available. Decompose any h as h−G(h)w+G(h)w. The first term lies in ker G, proving DJ[h]=DJ[w]G(h). Thus λ=−DJ[w] makes DJ+λG zero on every direction. This proves the multiplier relation rather than assuming it.

The value function is V(A)=6A². Its derivative is −λ*=12A: the negative multiplier is the marginal energy cost of increasing the required area under our sign convention. A finite area change has an additional quadratic cost:

$$
V(A+\varepsilon)-V(A)=12A\varepsilon+6\varepsilon^2.
$$

For a counterexample to an unconditional multiplier rule, constrain ∫y²=0 and minimize −∫y over H₀¹. The only feasible function is zero, hence it is a minimizer. But the constraint derivative there is zero in every direction, whereas the objective derivative is −∫h and is not zero. No scalar multiplier can cancel it. Vanishing linearized constraint does not identify genuine feasible directions for this singular constraint.

| A | V(A) | λ* | V′(A) |
|---|---:|---:|---:|
| −1/2 | 3/2 | 6 | −6 |
| 0 | 0 | 0 | 0 |
| 1/2 | 3/2 | −6 | 6 |

```python
from fractions import Fraction as F
V = lambda a: 6*a*a
eps = F(1, 10)
for a in (F(-1, 2), F(0), F(1, 2)):
    multiplier = -12*a
    change = V(a+eps)-V(a)
    assert change == -multiplier*eps+6*eps*eps
    print(a, V(a), multiplier, -multiplier, "finite change:", change)
# h(t)=t(1-t): integral h=1/6, integral h^2=1/30.
objective_derivative, constraint_derivative = -F(1, 6), F(0)
assert objective_derivative != 0 and constraint_derivative == 0
print("singular constraint:", objective_derivative, constraint_derivative)
```

The rational code checks the value derivative and a direction in the singular example. It is separate from the lab's area-preserving sine family. For multiple or nonlinear constraints, state the derivative's rank or surjectivity condition before invoking a multiplier theorem.

## Exercise and explained answer

Choose A=0.5,b=v=c=0. The curve is 3t(1−t), energy 1.5 and multiplier −6. Set b=0.2: area stays 0.5, but objective excess becomes 0.04π²≈0.394784. The variation along sin 2πt is 0.4π², so this oscillating feasible curve is not stationary.

Restore b=v=0 and set c=1. The direction is no longer area-preserving. Its objective variation is 12/π, canceled by λδG=−12/π. Explain why a nonzero δJ in this inadmissible direction does not contradict constrained optimality. Integral constraints of this type appear in mass conservation, normalized density estimation and variational physical models, with domains and signs adapted to each application.

Further reading: [MIT, Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
