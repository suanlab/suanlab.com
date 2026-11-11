## From a differential equation to executable steps

A differential equation specifies an instantaneous rate, whereas a computer stores finitely many states. For cooling-like decay y′=-λy, the exact solution is y(t)=y₀ exp(-λt). Different choices of intermediate slopes produce different numerical trajectories even when they reach the same final physical time T. This experiment evaluates actual stages and repeats actual updates; it does not draw an assumed error curve.

The horizontal axis is physical time. Blue connects computed grid states, and orange is the exact exponential. A straight connector between two numerical states is a visual guide, not a claim that the continuous solution is linear there. Changing N changes h=T/N while preserving the comparison horizon.

## Euler and Runge–Kutta slopes

Forward Euler freezes the initial slope. Explicit midpoint first predicts a half-step state and uses the slope there:

$$
y_{n+1}=y_n+h f(y_n),\qquad
k_1=f(y_n),\quad k_2=f(y_n+hk_1/2),\quad y_{n+1}=y_n+hk_2.
$$

Classical RK4 additionally evaluates k₃=f(yₙ+hk₂/2) and k₄=f(yₙ+hk₃), then adds h(k₁+2k₂+2k₃+k₄)/6. These are slopes at different predicted states, not repeated measurements at one point. Under appropriate smoothness and stability assumptions their global orders are one, two and four. Count both stage evaluations and steps when comparing computational cost.

## Solve the implicit equation

Backward Euler evaluates the slope at the unknown next state. For this linear decay equation the solve is algebraic:

$$
y_{n+1}+h\lambda y_{n+1}=y_n,\qquad
y_{n+1}=\frac{y_n}{1+h\lambda},\qquad
R_E(z)=1+z,\quad R_{BE}(z)=\frac1{1-z}.
$$

For a general nonlinear vector field, an implicit step requires a separate nonlinear solve. The current experiment does not claim to implement that general solver. For negative real z=-λh, backward Euler damps perturbations, but its multiplier is still different from exp(z). Stability describes amplification of perturbations; accuracy describes distance from the desired solution.

## An implicit solve has its own error budget

Suppose an approximate backward-Euler state satisfies (1+λh)ŷₙ₊₁−ŷₙ=rₙ instead of zero. Compare it with the exact solution of the discrete backward-Euler equations, not yet with the continuous exponential. With R=1/(1+λh), the algebraic error obeys eₙ₊₁=R(eₙ+rₙ). For λ>0, |rₙ|≤ρ and e₀=0,

$$
|e_N|\le\frac{\rho}{\lambda h}(1-R^N).
$$

Thus a fixed equation-residual tolerance can accumulate like ρ/h at fixed physical T=Nh. To keep backward Euler's global first order, a sufficient per-step algebraic budget is ρ=O(h²). If the residual is defined after dividing the equation by h, its tolerance convention changes by that same factor.

For a differentiable nonlinear field with m≥0 and f′≤−m on the interval containing two candidate roots, the mean-value theorem gives a local state-error bound |r|/(1+hm). This requires the actual implicit equation and a derivative bound; an iteration count alone supplies neither. The live linear-decay step solves its equation algebraically. The inexact solves below are auxiliary examples.

| N at T=λ=1 | h | Residual budget ρ |
|---|---:|---:|
| 4 | 1/4 | 1/16 |
| 8 | 1/8 | 1/64 |
| 16 | 1/16 | 1/256 |

```python
from fractions import Fraction as F
for N in (4,8,16):
    h=F(1,N);R=1/(1+h);rho=h*h
    exact=inexact=F(1)
    for _ in range(N):
        exact=R*exact
        inexact=R*(inexact+rho)
    error=inexact-exact
    bound=rho/h*(1-R**N)
    assert error==bound
    print(N,h,rho,float(error))
```

The exact fractions attain the bound by choosing every residual positive. The live terminal error also uses expm1 near zero to compare a represented state with the analytic decay change. Two displayed states can round to the same number while the continuum error remains positive; this diagnostic is distinct from an ideal exact-arithmetic truncation error.

## Worked example and exercise

Set λ=T=y₀=1 and N=2. Euler gives 1→1/2→1/4, whereas the exact terminal value is exp(-1)≈0.367879. Its absolute error is approximately 0.117879. Backward Euler gives 1→2/3→4/9, with absolute error approximately 0.076565. Notice that the two errors have opposite signs before taking absolute values.

Exercise: compute the RK4 one-step multiplier for h=1/2. Solution: 1-h+h²/2-h³/6+h⁴/24=0.6067708333. After two steps, square this number. For a general nonlinear f, evaluate the stages rather than substituting this linear test-equation polynomial. The interactive implementation evaluates those stages directly.

## Connection to AI computation

A Neural ODE can share the same continuous vector field while producing different losses and gradients under different numerical solvers. Record the solver, step settings, observation horizon and evaluation cost. A finite sequence of steps defines a particular computation graph, which must be distinguished from the continuous flow. The next chapter separates one-step defects from accumulated terminal error.

Reference: [official SciPy IVP integrator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). This fixed-step teaching model is separate from the package's adaptive algorithms.
