## Two different questions about internal state

Can an input move the system to a desired state? Can output measurements distinguish different initial states? Controllability and observability answer these questions separately. A well-actuated system may have an invisible state direction; a fully observed system may have a direction its actuator cannot reach.

The lab uses a continuous-time, stable, two-state system with known inputs:

$$
\dot x=Ax+Bu,\quad y=Cx,\qquad
A=\begin{pmatrix}-1&0\\0&-2\end{pmatrix},\quad
B=\begin{pmatrix}1\\b\end{pmatrix},\quad C=(1,c).
$$

The distinct decay rates matter. Structural rank conclusions here are specific to these matrices, not rules saying that two nonzero coefficients always make every system controllable.

## Rank tests and their meaning

The controllability matrix has columns B and AB; its determinant is -b. The observability matrix has rows C and CA; its determinant is -c. Consequently, the ranks are two when the respective channel is nonzero and one when it is zero.

For a known common input, the difference between two outputs depends only on their initial-state difference. Starting from unit directions gives y1(t)=exp(-t) and y2(t)=c exp(-2t). At c=0, the second direction is invisible. When c is nonzero, the two distinct exponentials can be distinguished from a sufficient observation interval, even though one instantaneous scalar measurement cannot recover two coordinates.

## Integrate energy over a finite horizon

Define g_r(T)=(1-exp(-rT))/r. The reachability Gramian is

$$
W_c(T)=\begin{pmatrix}g_2&b g_3\\b g_3&b^2g_4\end{pmatrix},
\qquad W_o(T)=\begin{pmatrix}g_2&c g_3\\c g_3&c^2g_4\end{pmatrix}.
$$

These are the integrals of exp(At)BBᵀexp(Aᵀt) and exp(Aᵀt)CᵀCexp(At), respectively. For a zero initial state, the minimum squared L2 input energy required to reach a target is

$$
E_{min}=x_{target}^{T}W_c^{-1}x_{target}
$$

when the Gramian is invertible. A minimizing input is

$$
u_*(s)=B^T e^{A^T(T-s)}W_c^{-1}x_{target}.
$$

Any other feasible input adds a component v that leaves the final state unchanged. Its inner product with u* is zero, so its energy is E_min plus the nonnegative integral of v². Input components unused by the target can only waste energy.

The blue native contour is W_c^(1/2) applied to the unit circle: it bounds states attainable with input energy at most one. It is not an observability confidence ellipse or a state-safety boundary.

## Singular channels need a support check

At b=0, only the first coordinate is reachable. If target x2 is nonzero, no finite input energy reaches it; the readout is unavailable. If target x2=0, minimum energy is x1²/g2. This restricted solution is computed directly rather than adding an artificial ridge to invert a singular matrix.

A small nonzero b still gives structural rank two, but a poorly conditioned Gramian and high control energy. The code uses a scaled-coordinate energy calculation to avoid unnecessary division by b² during matrix inversion. A finite mathematical energy that exceeds machine representation also remains unavailable, separately from the structural reachability flag.

## Research extension: separate coordinates from weak dynamics

A small Gramian eigenvalue depends on state units. Under $z=Tx$, $W_c'=TW_cT^\top$ but $W_o'=T^{-\top}W_oT^{-1}$. Therefore $W_c'W_o'=T(W_cW_o)T^{-1}$: the square roots of its eigenvalues, Hankel singular values, are coordinate invariant. Rescaling position from meters to millimeters must not change this input-output diagnostic.

For an auxiliary infinite-horizon version with $b=c=s$, stability gives

$$W_c=W_o=W=\begin{pmatrix}1/2&s/3\\s/3&s^2/4\end{pmatrix},\qquad\det W=s^2/72.$$

Here the Hankel singular values are the positive eigenvalues of $W$, and the smaller is asymptotic to $s^2/36$ as $s\to0$. A nonzero channel can be structurally reachable yet practically weak. The existing finite-horizon controls remain a different experiment.

| $s$ | Product of Hankel singular values |
|---|---:|
| 1 | 1/72 |
| 1/10 | 1/7200 |
| 1/100 | 1/720000 |

```python
from fractions import Fraction as F
from math import sqrt
for s in [F(1),F(1,10),F(1,100)]:
    a,b,d=F(1,2),s/3,s*s/4
    determinant=a*d-b*b
    assert determinant==s*s/72
    # T=diag(2,1/2): transformed controllability/observability Gramians.
    C=((4*a,b),(b,d/4)); O=((a/4,b),(b,4*d))
    product_trace=sum(C[i][j]*O[j][i] for i in range(2) for j in range(2))
    assert product_trace==a*a+2*b*b+d*d
    large=(float(a+d)+sqrt(float((a-d)**2+4*b*b)))/2
    small=float(determinant)/large
    print(s,determinant,large,small)
```

**Research limit:** discarding a physical coordinate is not automatically balanced truncation. Neither a small singular value nor structural rank alone certifies closed-loop stability after model reduction.

## Exercise and explained solution

Hold T fixed and target (0,1). Compare b=1 with b=0.1, keeping c=1.

**Solution.** In scaled coordinates the target is (0,1/b), while the base Gramian is independent of b. Its quadratic energy therefore grows as 1/b²: reducing b by ten multiplies minimum energy by one hundred. Structural controllability rank remains two in both cases, and observability does not change. At b=0, the target becomes unreachable instead of merely expensive.

Now set b=1,c=0. The actuator can reach both directions, but the orange output curve is zero for the second initial direction. Being controllable does not make that direction observable.

## Connection to learned models and sensors

A recurrent model can contain hidden coordinates that data do not distinguish, while a robot can have states that its current actuators cannot independently manipulate. Rank is a structural question; finite-horizon Gramian eigenvalues add information about energy and numerical sensitivity. Noise, unknown inputs, nonlinear dynamics and limited measurement times require additional analysis. The next filter chapter estimates state uncertainty probabilistically rather than turning observability into an automatic accuracy guarantee.

## Primary reference

MIT, [Dynamic Systems and Control, 6.241J](https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/). The diagonal system, energy ellipse and channel comparisons here are original examples.
