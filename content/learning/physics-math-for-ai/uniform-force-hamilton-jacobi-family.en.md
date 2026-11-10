## A scalar function can organize an entire family of motions

Suppose a particle moves along a line under a constant force. Newton's equation is easy to integrate. That makes this a useful place to learn a harder idea: a scalar function of position, time and a parameter can encode both momentum and the labels of all trajectories.

Imagine contour lines on a map. Knowing the height alone does not tell you how to move; the slope matters. In Hamilton-Jacobi theory the relevant slope is a momentum, not a velocity. Another derivative identifies which trajectory you are following. A proposed map must satisfy a differential equation before its labels can be treated as constants of motion.

We will derive the scalar function, recover a trajectory from it, and deliberately damage it. The damaged function still generates a canonical transformation. This counterexample separates two conditions that are often confused.

## State the mechanical problem and its units

Use position $q$ in m, momentum $p$ in kg m/s and time $t$ in s. Mass $m>0$ is constant, and the external force $F$ is independent of position and time. Choose

$$
H(q,p)=\frac{p^2}{2m}-Fq,\qquad
L(q,\dot q)=\frac m2\dot q^2+Fq.
$$

The potential is $-Fq$, so a positive force accelerates toward increasing $q$. The Hamiltonian equations are $\dot q=p/m$ and $\dot p=F$. Both need an initial condition. A solution of the position equation alone is insufficient if the displayed momentum fails the first Hamilton equation.

The generating function $S$ has action units Js. Consequently $S_q$ has momentum units, while $S_t$ has energy units. Never compare an action directly with energy or interpret $S_q$ as a speed without dividing by mass.

## Derive a complete integral of the equation

The Hamilton-Jacobi equation for this problem is

$$
S_t+\frac{(S_q)^2}{2m}-Fq=0.
$$

Try $S=(\alpha+Ft)q+f(\alpha,t)$. Its position derivative is $p=\alpha+Ft$, as expected from integrating the force. Substitution gives

$$
f_t=-\frac{(\alpha+Ft)^2}{2m}.
$$

Integrating with respect to time gives the family

$$
S_0(q,\alpha,t)=(\alpha+Ft)q
-\frac{\alpha^2t}{2m}-\frac{\alpha Ft^2}{2m}
-\frac{F^2t^3}{6m}+\frac{c\alpha^2}{2}.
$$

The last term is independent of position and time, so it does not affect this PDE. Here $\alpha$ is a momentum parameter and $c$ has units s/kg. The subscript on $S_0$ denotes the valid family, not evaluation at time zero.

## Explain what complete means locally

For one configuration coordinate, a complete integral has one independent momentum parameter and a nonsingular mixed derivative on the chart:

$$
\frac{\partial^2S_0}{\partial q\partial\alpha}=1.
$$

This derivative lets us invert the momentum relation for $\alpha$ and use the parameter derivative as its conjugate coordinate. An additive action constant does not count as the required independent momentum parameter.

The word complete does not mean that one scalar value contains a trajectory without further data. The pair of constants still has to be supplied. Nor does this example prove that every Hamiltonian admits one global, single-valued complete integral. Later oscillator charts will fail at specific projections even when the phase flow remains smooth.

## Recover the initial conditions from two constants

Introduce $\beta=S_{0,\alpha}$, with length units:

$$
\beta=q-\frac{\alpha t}{m}-\frac{Ft^2}{2m}+c\alpha.
$$

Solving this equation gives

$$
q(t)=q_0+\frac{\alpha t}{m}+\frac{Ft^2}{2m},\qquad
p(t)=\alpha+Ft,\qquad q_0=\beta-c\alpha.
$$

Thus $\alpha=p(0)$, but $\beta=q(0)$ only when $c=0$. Differentiate the reconstructed functions to check both Hamilton equations. The initial position and initial momentum are recovered separately; a free additive constant in the action would not replace either one.

For $m=1$, $F=1$, $\alpha=1$, $\beta=c=0$ in their SI units, the state at $t=1$ is $q=3/2$ m and $p=2$ kg m/s. Its energy is $1/2$ J.

## Evaluate a field query without confusing it with a trajectory

At fixed $\alpha$ and time, $S_0$ is a function of every position. It describes a family of trajectories rather than only the selected trajectory with one value of $\beta$.

The model therefore has a query-position offset. Adding $1$ m to the selected position changes $S_{0,\alpha}$ by $1$ m. It does not alter the reference trajectory. With the valid family, the displaced point can lie on another physical characteristic with a different initial position.

In the numerical example, the displaced position is $5/2$ m, momentum remains $2$ kg m/s and energy becomes $-1/2$ J. This is a different characteristic under the same force, rather than an energy-conservation error on the original one. Read the selected-path membership flag before comparing those energies.

## Damage the PDE while preserving the canonical map

Consider the deliberately modified candidate

$$
S_\lambda=S_0+\frac\lambda2tq^2,\qquad [\lambda]=\mathrm{N/m}.
$$

The mechanical Hamiltonian is unchanged: $\lambda$ is a candidate-function defect, not a new spring force. Its derivatives give

$$
p_\lambda=\alpha+Ft+\lambda tq,\qquad
\beta=S_{\lambda,\alpha}=S_{0,\alpha}.
$$

The mixed derivative remains one. The transformation from constants $(\beta,\alpha)$ to the displayed phase state has Jacobian

$$
M=\begin{pmatrix}
1&t/m-c\\
\lambda t&1+\lambda t(t/m-c)
\end{pmatrix},\qquad \det M=1.
$$

The off-diagonal entries have units s/kg and kg/s. This is a canonical map in one degree of freedom, including when $\lambda\ne0$. It need not turn the actual Hamiltonian into zero.

## Compute the global defect rather than trusting one query

Direct differentiation yields

$$
\mathcal R=S_{\lambda,t}+H(q,S_{\lambda,q})
=\left(\frac\lambda2+\frac{\lambda^2t^2}{2m}\right)q^2
+\frac{\lambda t(\alpha+Ft)}m q.
$$

Every coefficient must vanish as a function of independent position and time. For the displayed family that occurs exactly when $\lambda=0$. The coefficient units differ across the basis $q^2,t^2q^2,tq,t^2q$; do not sum their raw magnitudes into one physical error norm.

At $q=0$ the residual vanishes for every parameter choice. One passing query therefore proves neither a global PDE solution nor a physical selected motion. Even a whole zero trajectory can remain physical in an invalid candidate family when $q_0=\alpha=F=0$.

## Check what happens if the new labels are frozen incorrectly

Freezing $\alpha$ and $\beta$ in the defective family retains the reference position but assigns the wrong momentum. The two Hamilton residuals are

$$
\dot q-\frac{p_\lambda}{m}=-\frac{\lambda tq}{m},\qquad
\dot p_\lambda-F=\lambda\left(q+t\dot q\right).
$$

Use $m=1$, $F=0$, $\alpha=1$, $\beta=c=0$, $\lambda=1$ and $t=1$. The reference position is $1$ m, frozen-label momentum is $2$ kg m/s, and the residuals are $-1$ m/s and $2$ N. Moving the separate field query to $q=0$ makes its PDE residual zero while these selected-path residuals remain nonzero.

For a general generating transformation the transformed Hamiltonian is $K=\mathcal R$ expressed in the new variables. Its correct equations are $\dot\beta=K_\alpha$ and $\dot\alpha=-K_\beta$. A state-dependent nonzero $K$ cannot simply be ignored.

## Relate generating values to the actual action integral

Along the physical reference trajectory, integration of the declared Lagrangian gives

$$
I(t)=\left(\frac{\alpha^2}{2m}+Fq_0\right)t
+\frac{\alpha Ft^2}{m}+\frac{F^2t^3}{3m}.
$$

For the valid family,

$$
S_0(q(t),\alpha,t)-S_0(q_0,\alpha,0)=I(t).
$$

Subtract the initial value. In the first example $I(1)=11/6$ Js. With $c=1$ s/kg and compensating $\beta=1$ m, the same physical path has initial generating value $1/2$ Js and final value $7/3$ Js. The difference is still $11/6$ Js.

For the damaged candidate the difference from the physical integral is $\lambda t q(t)^2/2$. The model reports that defect without pretending that the damaged momentum is the physical one.

## Understand relabeling without changing the mechanical model

Adding $c\alpha^2/2$ changes the conjugate label by $c\alpha$. This is a freedom in how the family is labeled. It is not the same operation as adding a total time derivative of a position function to the mechanical Lagrangian.

To compare the same physical initial state after changing $c$, adjust $\beta$ so that $\beta-c\alpha$ stays fixed. Holding the label $\beta$ fixed instead changes the physical initial position. Compare actual initial conditions before calling two curves equivalent.

This is useful when comparing learned generating functions: a difference in their scalar values can represent a harmless label convention, while a position-dependent derivative error can change the inferred motion. Those cases need different diagnostics.

## Read the curves and limiting cases

Time curves cover $0\le t\le2$ s. Field curves hold query time fixed and sample positions around the reference position; their horizontal coordinate is actual position, not time or the offset code. Moving only the query offset leaves both collections unchanged.

Action differences have units Js, energies J, velocity residuals m/s and force residuals N. They belong on separate axes. Exact query values and polynomial coefficients use reduced rational numbers. Visual curves are samples of the analytic model rather than results from an ODE solver.

Setting $F=0$ gives free motion. Setting $\alpha=0$ need not give rest when the force is nonzero. At $t=0$ a defective candidate can pass both selected Hamilton residuals and fail immediately afterward. None of these cases requires an epsilon denominator.

## Apply the lesson to AI research carefully

A scalar neural model can be differentiated into a momentum field. That construction alone does not enforce Hamilton-Jacobi, the intended initial conditions, or physical characteristic dynamics. Test those requirements separately.

A useful experiment records the Hamiltonian, units, parameter domain, initial-state labels, mixed-Hessian invertibility and PDE residual over independent states. It also checks both Hamilton equations on reconstructed trajectories and compares action increments with independently integrated Lagrangians.

A small sampled residual does not prove a global theorem. A canonical architecture does not by itself make the transformed Hamiltonian vanish. This finite polynomial experiment illustrates these distinctions exactly; it does not establish convergence of neural PDE training, solve arbitrary control problems, or supply a global theory beyond this family.

## Work through six checks with explanations

### Exercise 1

For the first valid example, compute the state, energy and action increment at one second.

**Solution:** $q=1+1/2=3/2$ m and $p=1+1=2$ kg m/s. Then $H=2-3/2=1/2$ J. The integrated action is $1/2+1+1/3=11/6$ Js. Check the endpoint generating difference rather than treating energy as action.

### Exercise 2

Change $c$ from zero to $1$ s/kg while retaining the same $q_0=0$ and $\alpha=1$ kg m/s. What must happen to $\beta$?

**Solution:** Set $\beta=c\alpha=1$ m. Initial and final generating values both increase by $1/2$ Js. Their difference, the position trajectory and the physical momentum remain unchanged.

### Exercise 3

Why does a zero residual at $q=0$ fail to certify the defective candidate?

**Solution:** Every term of $\mathcal R$ contains $q$. Away from this slice the $q^2$ and $q$ coefficients need not vanish. The mixed Hessian is nevertheless one, so checking canonicality would also miss the PDE defect.

### Exercise 4

For the damaged free example at one second, compare the reference momentum with the frozen-label momentum.

**Solution:** The reference momentum is $1$ kg m/s, while the damaged derivative gives $2$ kg m/s at the reference position. Reference velocity is $1$ m/s, so the first Hamilton residual is $-1$ m/s. The damaged momentum changes at $2$ N even though the physical force is zero.

### Exercise 5

Can an invalid family contain a physical trajectory with a constant selected charge?

**Solution:** Yes. When $q_0=\alpha=F=0$, the selected position and momentum remain zero even for nonzero $\lambda$. A stationary selected trajectory does not eliminate the independent-state PDE coefficients of the family.

### Exercise 6

Explain why a displaced valid field query can have a different energy from the selected trajectory.

**Solution:** Its derivative label $\beta$ changes with the displacement. It represents a different initial position. Under constant force the potential $-Fq$ then changes the energy even though momentum at the shared time is equal. Energy conservation applies along each physical characteristic, not across arbitrary members of the family.

## Sources and notation

The canonical generating-function and Hamilton-Jacobi framework follows [Tong, Classical Dynamics, sections 4.4.3 and 4.7](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4) and [Sussman and Wisdom, Structure and Interpretation of Classical Mechanics, section 6.1](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter006.html). This lesson uses its own $S_0,S_\lambda,\alpha,\beta$ conventions. The constant-force family, defect, label compensation and exact examples are derived here and checked independently by symbolic differentiation and action integration.
