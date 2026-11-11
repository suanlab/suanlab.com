## Solve a relation locally

The circle x²+y²=1 is not globally one function y=f(x), since many x values have two corresponding y values. Near (0.6,0.8), however, its upper branch is differentiable:

$$
F(x,y)=x^2+y^2-1,\qquad F_x=2x,\quad F_y=2y.
$$

For C¹ F with F(x₀,y₀)=0 and Fᵧ≠0, the implicit function theorem supplies a locally unique differentiable y(x). Locality matters; it does not select a globally unique branch around the entire circle.

## Derive implicit differentiation

Differentiate F(x,y(x))=0 using the chain rule:

$$
y'(x)=-\frac{F_x}{F_y}=-\frac xy.
$$

At the reference this is −0.75. At (1,0), the denominator vanishes; the upper branch has no finite derivative. The lab does not replace a vertical tangent with zero slope.

For vector-valued y, Fᵧ is a square Jacobian and must be invertible. The derivative becomes −(Fᵧ)⁻¹Fₓ. Solving a linear system can be preferable to explicitly materializing an inverse.

## Inverse function theorem: a sufficient condition

A C¹ map f:Rⁿ→Rⁿ with invertible Jacobian at a point has a local C¹ inverse, with

$$
D(f^{-1})(f(x_0))=[Df(x_0)]^{-1}.
$$

Invertibility of the derivative is sufficient for this differentiable local inverse, not necessary for mere existence of an inverse. The cube x³ is globally one-to-one despite f′(0)=0. Its cube-root inverse exists but has no finite derivative at zero.

## Interpret singular cases carefully

At the circle endpoint, expressing y as a differentiable function of x fails, but expressing x in terms of y can succeed. Changing the dependent variable changes the relevant Jacobian block. The curve itself has not disappeared.

Failure of a theorem’s assumption is not automatically the negation of its conclusion; it removes the theorem’s guarantee and calls for separate analysis.

## Differentiate a fixed point and its loss

Let z∈Rᵈ and θ∈Rᵖ. Suppose f is C¹ on an open neighborhood of a selected solution z*=f(z*,θ), and I−J_z f is invertible there. Writing G=z−f and differentiating G=0 gives

$$
(I-J_z f)\frac{dz^*}{d\theta}=J_\theta f.
$$

Intuitively, changing θ directly changes f, while the changed state feeds back through J_z f. The linear system sums this feedback at the selected equilibrium. It supplies a local derivative; existence of a root and its numerical computation still need justification.

For scalar loss ℓ(z*,θ), use column gradients. Solve the adjoint system once:

$$
(I-J_z f)^T\lambda=\nabla_z\ell,\qquad
\nabla_\theta\ell=\partial_\theta\ell+(J_\theta f)^T\lambda.
$$

The direct partial derivative holds z fixed. The second term follows because multiplying the sensitivity equation by λᵀ replaces ∇_zℓᵀ dz*/dθ. Here λ has d entries and the resulting parameter gradient has p entries.

## Compare finite iterations with equilibrium sensitivity

Take z*=az*+b with fixed a≠1. Its solution is b/(1−a), and dz*/db=1/(1−a). Starting from z₀=0 independent of b, unfold zₜ₊₁=azₜ+b. Differentiating every step gives p₀=0 and pₜ₊₁=apₜ+1:

$$
z_T=b\sum_{j=0}^{T-1}a^j,\qquad
p_T=\frac{\partial z_T}{\partial b}=\frac{1-a^T}{1-a}.
$$

For a=1/2 and b=1, both the state and its sensitivity approach two:

| Iterations T | z_T and p_T | Absolute error from 2 |
|---|---|---|
| 1 | 1 | 1 |
| 4 | 1.875 | 0.125 |
| 16 | 1.999969482421875 | 0.000030517578125 |

The error is exactly 2·(1/2)ᵀ. For |a|<1 the geometric tail vanishes. For a=6/5, however, the unique differentiable fixed point is −5 at b=1 and its sensitivity is −5, while iteration from zero diverges. Invertibility of 1−a does not establish convergence of this particular solver. At a=1, b=0 gives infinitely many solutions and b≠0 gives none.

The residual r(z)=z−az−b satisfies

$$
|z-z^*|=\frac{|r(z)|}{|1-a|}.
$$

When a is near one, a small residual can accompany substantial state error and a large parameter sensitivity. For a=19/20 the amplification is 20; for a=1/2 it is two. A solver tolerance should be interpreted together with conditioning.

## Reproduce the scalar comparison in Python

The following standard-library code uses exact fractions for the recurrence, closed form, residual identity and derivative check. Its four slopes include oscillatory convergence, slow convergence and divergence. The interactive plot above remains the circle/inverse-function lab; this is a separate runnable research example.

```python
from fractions import Fraction as Q

for a in [Q(1, 2), Q(-4, 5), Q(19, 20), Q(6, 5)]:
    b = Q(1)
    exact = b / (1 - a)
    sensitivity = 1 / (1 - a)
    for steps in [1, 4, 16]:
        z, p = Q(0), Q(0)
        for _ in range(steps):
            z, p = a * z + b, a * p + 1
        assert p == (1 - a**steps) / (1 - a)
        assert z == b * p
        residual = z - a * z - b
        assert abs(z - exact) == abs(residual) / abs(1 - a)
        h = Q(1, 1000)
        difference = ((b + h) / (1 - a) - (b - h) / (1 - a)) / (2 * h)
        assert difference == sensitivity
        print(a, steps, float(z), float(p), float(sensitivity))
```

Research exercise: choose a target state error, derive the required residual tolerance for each a, and compare that stopping rule with a fixed iteration budget. Record the finite-solver derivative p_T separately from the equilibrium derivative. In a nonlinear system, using an approximate root also introduces error into the implicit linear system.

Source connection: [Deep Equilibrium Models, v2](https://arxiv.org/abs/1909.01377v2), §3.1, Theorem 1, Eq. (11) and Appendix A, uses implicit differentiation for equilibria. The scalar comparison and conditioning calculations here are independently constructed; the paper’s training experiments are outside this reproduction.

## AI connections and exercises

Implicit differentiation of stationary conditions G(θ,z)=0 can give solution sensitivity to data z. Check existence, solution selection, Jacobian invertibility and stability. Unrolling an iterative solver and implicitly differentiating its limiting condition have different computational and mathematical requirements.

Mode zero uses the upper circle branch; mode one compares a cube with its inverse. Axes have equal unit length. Undefined derivatives are distinct from missing function values.

1. At (0,1), the circle slope is **zero**.
2. At x=1 for the cube, the inverse derivative is **one third**.
3. An invertible Jacobian guarantees **a local inverse, not global invertibility**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
