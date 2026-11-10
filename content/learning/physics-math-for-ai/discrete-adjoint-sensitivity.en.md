## Decide which function is being differentiated

A loss defined by the continuous exact solution differs from a loss defined by N actual numerical steps. A discrete adjoint applies the chain rule to the latter computation graph. A difference from the continuous gradient is therefore not automatically an implementation bug. First compare independent derivatives of the same discrete function.

Use logistic dynamics y′=θy(1-y), y(0)=0.2, integrated by forward Euler. Define J=(y_N-target)²/2 from the computed terminal state. Blue is the Euler trajectory and orange is the continuous exact solution. Saved state, forward sensitivity and reverse adjoint appear in the same time-indexed table.

## Derive the forward tangent

$$
F(y,\theta)=y+h\theta y(1-y),\quad
F_y=1+h\theta(1-2y),\quad F_\theta=hy(1-y).
$$

The initial state does not depend on θ, so s₀=0. Apply s_next=F_y s+F_theta at each step, then multiply s_N by y_N-target. Propagating separate tangents for many parameters can be expensive. The single parameter here makes forward and reverse derivatives easy to inspect independently.

## Derive the reverse adjoint

$$
a_N=y_N-\mathrm{target},\qquad
a_n=F_y(y_n,\theta)a_{n+1},\qquad
\frac{dJ}{d\theta}=\sum_{n=0}^{N-1}a_{n+1}F_\theta(y_n,\theta).
$$

Each parameter contribution uses a_next rather than a_n. Since θ is shared across every step, all contributions must be added. The implementation reads saved forward states backward. Larger simulations must account for storage, checkpointing and recomputation; reverse differentiation does not make those costs disappear.

## Check the discrete objective independently

The centered finite difference reruns the same Euler integration at θ+ε and θ-ε. The selected ε=10^-5 is suitable for this small smooth example, not a universal optimal constant. Large ε causes truncation error; very small ε can cause cancellation. A finite-difference mismatch needs to be interpreted with those competing errors in mind.

The continuous exact solution is y(t)=1/(1+4 exp(-θt)), giving dy(T)/dθ=T y(T)(1-y(T)). Its objective gradient is also reported, but it differentiates a different finite-h function. Checking convergence toward that continuous derivative is distinct from checking that forward tangent and reverse adjoint agree on the discrete graph.

## Worked example and exercise

At θ=0, all Euler states remain 0.2. Every step has F_y=1 and F_theta=0.16h, so s_N=0.16T. All reverse adjoints equal 0.2-target. Both methods therefore give gradient (0.2-target)0.16T.

Exercise: set T=1 and target=0.8. Predict and calculate the gradient. Solution: it is -0.6×0.16=-0.096. Increasing θ slightly raises the logistic state toward the target, reducing the loss, consistent with the negative sign. In this special θ=0 case, the derivative is identical for every selected step count.

## Gradient validation in AI research

For differentiable simulators, Neural ODEs and optimal control, record exactly which forward discretization is differentiated. Adaptive step selection, events and unconverged implicit solves introduce additional issues. This experiment covers a smooth scalar fixed-step graph; it does not claim to implement a complete continuous-adjoint solver or checkpointing system.

Reference: [official SciPy IVP definition and solver settings](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
