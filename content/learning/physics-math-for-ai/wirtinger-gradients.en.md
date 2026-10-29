## A real loss with complex parameters

A complex parameter z=x+iy does not require a holomorphic learning objective. The real loss |z|²=x²+y² is nonholomorphic but smooth on R² and readily optimized.

Wirtinger operators are combinations of real derivatives:

$$
\partial_z=\frac12(\partial_x-i\partial_y),\qquad
\partial_{\bar z}=\frac12(\partial_x+i\partial_y).
$$

Treating z and its conjugate formally as independent is a computational rule derived from these definitions, not a doubling of the underlying real degrees of freedom.

## Recover the real gradient

For L=|z−c|²,

$$
\frac{\partial L}{\partial z}=\overline{z-c},\qquad
\frac{\partial L}{\partial\bar z}=z-c.
$$

Encoding the Euclidean real gradient as Lₓ+iLᵧ gives 2∂L/∂z̄. The factor two follows from the stated half-factor convention. Check a software library’s convention rather than transferring names alone.

## Why this gives descent

The real differential is

$$
dL=2\operatorname{Re}\left[\left(\frac{\partial L}{\partial\bar z}\right)^*dz\right].
$$

Choosing dz=−2η∂L/∂z̄ makes the first-order change nonpositive. Using the other Wirtinger derivative without accounting for conjugation can instead move away from the target.

## Solve the quadratic update exactly

The lab applies z_next=z−2η(z−c), so

$$
e_{\text{next}}=(1-2\eta)e,\qquad
L_{\text{next}}=(1-2\eta)^2L.
$$

For 0<η<1 the error contracts. At η=1 it changes sign without shrinking; above one, nonzero errors grow. At zero there is no update; at one half, one step reaches the minimizer. This range belongs to this explicit quadratic, not every complex objective.

At z=1+i,c=0,η=0.25, the next point is 0.5+0.5i and loss falls from two to one half. Shifting c changes the displacement used by the gradient.

## Read the implementation

Blue is six actual parameter updates; orange is the minimizer. This is a parameter-plane trajectory, not a loss-height surface. Loss values are separate readouts.

Independent real/imaginary finite differences can validate a complex autodiff convention. The engine here evaluates analytic expressions and does not claim to run a specific autodiff library. Nonsmooth magnitudes and branch operations need additional care.

## Exercises and answers

1. η=1/2 **reaches c in one step**.
2. η=1 **preserves nonzero loss while oscillating**.
3. A real-valued learning objective **need not be holomorphic**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
