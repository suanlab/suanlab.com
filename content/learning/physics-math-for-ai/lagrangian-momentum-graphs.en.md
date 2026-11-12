## Study a surface in phase space

A phase trajectory is one-dimensional. A graph assigning a momentum to every configuration point is a different object: with two position coordinates, its graph is a two-dimensional surface in a four-dimensional phase space. The word Lagrangian describes a geometric property of that surface. It does not mean that a physical Lagrangian function has been evaluated along a trajectory.

This lesson compares linear momentum fields, their tangent planes and the action accumulated around a configuration-space loop. The query parameter follows a constructed rectangle, not physical time. No equation of motion or numerical ODE integrator is introduced.

## Keep the ambient form nondegenerate

Use canonical coordinates (q1,q2,p1,p2), positions in meters and momenta in kg m/s. The ambient symplectic form is

$$
\omega=dq_1\wedge dp_1+dq_2\wedge dp_2.
$$

Its rank is four at every point, including the Cartesian origin. A subspace is isotropic when omega evaluates to zero on every pair of tangent vectors within that subspace. A Lagrangian subspace is isotropic and has half the ambient dimension. In this four-dimensional example its dimension must be two.

A vanishing restriction does not make the ambient form degenerate. Nondegeneracy asks about pairing a vector with every ambient direction, not only with directions lying on the selected surface. The distinction is similar to restricting a function before differentiating in the previous angular-momentum lesson: the space on which an operation is performed matters.

## Embed a momentum field as a graph

Let

$$
p=Aq,\qquad
A=\begin{pmatrix}a&b+c\\b-c&d\end{pmatrix}.
$$

Each coefficient has units kg/s. The embedding is i(q)=(q,Aq). Its tangent matrix is

$$
E=\begin{pmatrix}I_2\\A\end{pmatrix}.
$$

The top identity block guarantees rank two even if A has rank zero or one. In particular p=0 is a perfectly valid two-dimensional graph. Invertibility of the position-to-momentum map is not required.

This map should not be confused with a Legendre velocity-to-momentum relation. Here A differentiates momentum with respect to position and has units kg/s; a Cartesian mass coefficient differentiating momentum with respect to velocity has units kg. The earlier positive velocity-Hessian condition addresses a different inverse problem.

## Pull back the form to configuration space

For configuration variations u and v, graph tangents are Eu and Ev. Direct substitution gives

$$
\omega(Eu,Ev)=u^T(A-A^T)v.
$$

Thus the matrix of the pulled-back form is

$$
E^TJE=A-A^T=
\begin{pmatrix}0&2c\\-2c&0\end{pmatrix}.
$$

The graph is isotropic exactly when c=0. Because its dimension is already two, that is also exactly its Lagrangian condition. This is a statement about the full linear graph, not about whether one selected position happens to give a small residual.

The pullback coefficient has units kg/s; evaluating it on two position variations with units m gives an action in Js. A zero coefficient is compared exactly in the controlled rational family. No numerical threshold decides whether the graph is Lagrangian.

## Construct a generating action from the symmetric part

Decompose A into its symmetric part As and antisymmetric part. The quadratic scalar

$$
S(q)=\frac12q^TA_sq
=\frac12(aq_1^2+2bq_1q_2+dq_2^2)
$$

always exists and has action units Js. Its gradient is As q. The full momentum field differs by

$$
p-\nabla S=c\begin{pmatrix}q_2\\-q_1\end{pmatrix}.
$$

Only when c=0 does this S generate the entire momentum field. Writing down a plausible scalar quadratic for a nonsymmetric A does not establish that p is its gradient.

At q=0 the difference is zero even for c nonzero. A pointwise fit at the origin therefore cannot identify the full integrability condition. The graph's pullback form and a finite loop provide independent structural tests away from that one point.

## Distinguish the one-form and the two-form signs

The canonical one-form is theta=p dot dq. On the graph, its exterior derivative is

$$
d(i^*\theta)=
(\partial_{q_1}p_2-\partial_{q_2}p_1)dq_1\wedge dq_2
=-2c\,dq_1\wedge dq_2.
$$

Our convention is omega=−d theta, so the pullback of omega has the opposite coefficient, +2c. The two signs agree once the conventions and orientation are fixed.

On a counterclockwise rectangle of width w>0 and height h>0 in the q1-q2 plane,

$$
\oint p\cdot dq=-2cwh,
\qquad
\int i^*\omega=2cwh.
$$

Both integrals have units Js. This is an action integral, not the mechanical work integral of force dot displacement, whose unit would be J. Momentum arrows must not be silently interpreted as displacement arrows on the same axis without a declared display scale.

## Work a graph that is Lagrangian but not convex

Choose a=1 kg/s, d=−1 kg/s, b=c=0. Then p=(q1,−q2) in the stated units, and

$$
S=(q_1^2-q_2^2)/2.
$$

The quadratic has one positive and one negative curvature direction. It is not convex, yet A is symmetric, the graph has dimension two, and its pulled-back symplectic form is zero. It is therefore Lagrangian.

Taking both diagonal coefficients negative gives a concave generating action and another Lagrangian graph. Taking A=0 gives the zero section, also Lagrangian, although the position-to-momentum map is singular. These examples separate symmetry, convexity, rank of A and rank of the graph embedding.

This is not a claim that an associated physical system is stable or unstable. We have specified no physical Hamiltonian for evolution. A generating action's curvature and an energy stability criterion are different questions.

## Work a nonsymmetric field with zero pointwise error

Choose a=d=b=0 and c=1 kg/s. Then

$$
p=(q_2,-q_1),\qquad S=0.
$$

At the origin p=grad S=0, but the graph's pullback form has nonzero coefficient 2 kg/s. On the unit square, the counterclockwise action integral is −2 Js.

One can check the sign edge by edge. Along the lower edge q2=0, the contribution is zero. Along the right edge p2=−1 kg m/s and dq2 is positive, giving −1 Js. Along the upper edge p1=1 kg m/s while dq1 is negative, giving another −1 Js. The left edge contributes zero.

A nonzero loop action reveals path dependence. Rotating the loop orientation reverses the sign. Translating the rectangle leaves the closed integral unchanged for this linear field, although the partial edge integrals and selected momentum values can change.

## Inspect partial integrals as well as closure

For c=0, the line integral from the base point to any selected boundary point equals S(q)−S(q0). This holds on each edge and at intermediate points, not only after completing the loop. The cumulative action and potential-change curves therefore coincide throughout the displayed construction.

For c nonzero, the symmetric action still contributes its endpoint difference, but the antisymmetric part adds a path-dependent term. Returning to the initial configuration makes the potential change zero while the accumulated action remains −2cwh.

The parameter runs from zero to four, one unit per rectangle edge. Its numerical value has no unit of seconds. It must not be used to infer physical velocity, force, power, trajectory uniqueness or energy conservation. The model's extra straight line through the base point is another geometric subset, not an alternative physical orbit.

## A one-dimensional isotropic curve is not enough

Choose any nonzero configuration direction v and follow q=q0+lambda v. Its phase-space tangent is Ev. Because omega is alternating,

$$
\omega(Ev,Ev)=0.
$$

The one-dimensional line is isotropic for every A, even a nonsymmetric one. Its dimension is one, whereas a Lagrangian subspace in this ambient space must have dimension two. It is therefore not Lagrangian.

This explains why testing only tangent pairs that are parallel can never detect the antisymmetric defect. Two independent directions are needed to probe a two-form. A successful one-direction training trace cannot establish that an entire learned momentum graph admits a generating action.

## Separate local gradients from global exactness

For this linear field on all of R², a symmetric A gives the explicit globally defined quadratic S. More generally, a graph of a one-form is Lagrangian when that one-form is closed. On a sufficiently small contractible chart, closedness gives a local potential. A global single-valued potential requires more information about the domain.

For example, on a configuration circle use angle theta and constant angular momentum ell. The one-form ell dtheta is closed and its graph is Lagrangian in the cotangent cylinder. Around the circle its integral is 2pi ell, so nonzero ell does not admit a global single-valued action function of the angle. Here ell has angular-momentum units Js, unlike a Cartesian momentum in kg m/s.

The rectangle experiment tests the linear Cartesian family only. It does not numerically certify this global topological example, classify arbitrary Lagrangian submanifolds, or solve the Hamilton-Jacobi equation. The example identifies why a future extension must track domains and topology rather than treating every closed form as globally exact.

## Read exact structure and numerical drawings separately

Coefficients and configuration controls are half-integers; widths and heights are positive half-integers. The model uses exact rational matrix products, ranks and polynomial edge integrals. It reports a numeric spectrum of the symmetric part for intuition, but its convexity classification comes from exact algebra.

The graph always has dimension two. Its ambient form stays nondegenerate while the restriction has rank zero or two. Phase projections pair one position with its corresponding momentum; a configuration-space rectangle instead has two length axes. Neither picture alone embeds the whole four-dimensional graph.

A separate floating-point loop calculation is compared with the exact action. Its discrepancy is a numerical display diagnostic, not the criterion for Lagrangian structure. No tiny radius, artificial regularization or residual overwrite is used at the origin or at singular A.

## Research extension: a closed momentum field without a global action

Imagine carrying an angle counter around a hole. After one turn you return to the same position, but the counter has advanced by a full turn. A local derivative cannot detect that discrepancy. The following nonlinear companion example makes the earlier local/global distinction calculable. It is separate from the constant-matrix graph controlled in the native lab.

### Specify the domain before differentiating

Let position have units of length, and let the constant $\kappa$ have units of action, J s. The missing origin is part of the model, rather than a plotting inconvenience.

$$
Q=\mathbb R^2\setminus\{0\},\qquad
p_\kappa(x,y)=\frac{\kappa}{x^2+y^2}(-y,x),\qquad
\beta=p_\kappa\cdot dq.
$$

Differentiating each component gives a symmetric Jacobian. Since the identity block makes $E$ rank two everywhere in $Q$, the two-dimensional graph is Lagrangian in four-dimensional phase space.

$$
Dp_\kappa=\frac{\kappa}{(x^2+y^2)^2}
\begin{pmatrix}2xy&y^2-x^2\\y^2-x^2&-2xy\end{pmatrix},\qquad
E^TJE=Dp_\kappa-Dp_\kappa^T=0,\quad E=\begin{pmatrix}I\\Dp_\kappa\end{pmatrix}.
$$

### Count winding instead of subtracting principal angles

Choose a continuous angular branch on a simply connected chart. The local action exists there, but its value changes by $2\pi\kappa$ after a counterclockwise turn about the excluded point.

$$
S_{\rm local}=\kappa\theta,\qquad dS_{\rm local}=\beta,\qquad
\oint_C\beta=2\pi\kappa w(C),\quad w(C)\in\mathbb Z.
$$

For $\kappa\ne0$, a single-valued global $S$ would give zero on every closed-loop integral, contradicting this period. A small loop centered at $(2,0)$ with radius $1/2$ is contractible in $Q$ and has zero period. Reversing orientation reverses the sign; winding twice doubles it. For $\kappa=0$, the zero field is globally exact. A principal `atan2` endpoint difference silently discards winding.

![Loops and their momentum periods on the punctured plane](/assets/learning/analytical-topology-period.en.svg)

| Encircling radius | Period / (J s), $\kappa=1$ J s | Contractible loop / (J s) |
|---|---|---|
| 0.5 | $2\pi$ | 0 |
| 1 | $2\pi$ | 0 |
| 2 | $2\pi$ | 0 |

The last column uses the same small off-center loop in every row. The loop period is independent of the enclosing radius, even though momentum magnitude varies as $1/r$.

### A Lagrangian graph still needs a Hamiltonian

Symmetry of $Dp$ alone does not establish stationary Hamilton–Jacobi dynamics. The free-particle Hamiltonian varies across this graph. For $m>0$, adding the following explicit attractive inverse-square potential makes the local action solve the stationary equation with energy zero:

$$
H_{\rm free}(q,p_\kappa(q))=\frac{\kappa^2}{2mr^2},\qquad
U(q)=-\frac{\kappa^2}{2mr^2},\qquad
H(q,p_\kappa(q))=0,\quad r>0.
$$

The velocity is tangent to circles, and differentiating the field along that velocity reproduces the force:

$$
\dot q=\frac{p_\kappa}{m},\qquad
Dp_\kappa\dot q=-\frac{\kappa^2 q}{mr^4}=-\nabla U(q).
$$

Thus the graph is invariant under this specified Hamiltonian flow while still lacking a global scalar action for nonzero $\kappa$. This proves neither stability nor continuation through the origin. Position alone, a local curl penalty, or a Hamiltonian residual cannot certify global exactness. An AI model using one global scalar potential must state the topological sectors it excludes.

### Execute and explain

This standard-library cell integrates $p(q)\cdot q^\prime$ directly. It checks two loop classes, three radii and reversed double winding. Quadrature is a numerical check; the angular differential above supplies the proof.

```python
from math import cos, sin, pi

def period(center, radius, kappa, winding, steps=1024):
    total = 0.0
    for j in range(steps):
        angle = winding * 2 * pi * j / steps
        c, s = cos(angle), sin(angle)
        x, y = center[0] + radius*c, center[1] + radius*s
        r2 = x*x + y*y
        if r2 == 0:
            raise ValueError("The origin is excluded")
        px, py = -kappa*y/r2, kappa*x/r2
        dx, dy = -radius*winding*s, radius*winding*c
        total += px*dx + py*dy
    return total * 2*pi / steps

for radius in [.5, 1, 2]:
    encircling = period([0, 0], radius, 1, 1)
    contractible = period([2, 0], .5, 1, 1)
    assert abs(encircling - 2*pi) < 3e-12
    assert abs(contractible) < 3e-12
    print(radius, round(encircling, 6), round(contractible, 6))
assert abs(period([0, 0], 1, 1, -2) + 4*pi) < 3e-12
```

**Research questions with answers**

- Can zero curl certify a global action? **Answer:** Only with suitable domain assumptions; the nonzero period here is a counterexample.
- Does the free-particle Hamiltonian preserve this graph? **Answer:** For nonzero $\kappa$, it is not constant on the connected graph, and its straight trajectories do not follow these circles.
- Can you shrink the enclosing loop to a point in $Q$? **Answer:** No. Such a contraction would have to cross the excluded origin.

## Exercises with explanations

1. Compute E^TJE for a general two-by-two A. **Solution:** Multiplying the blocks gives A−A^T. For the controlled coefficients its off-diagonal entry is 2c, so c=0 is the exact condition.
2. Does singular A prevent a Lagrangian graph? **Solution:** No. The identity position block keeps the embedding rank two. A=0 produces the Lagrangian zero section.
3. Is a symmetric indefinite A Lagrangian? **Solution:** Its graph is Lagrangian because the antisymmetric part vanishes. Indefiniteness concerns the generating quadratic's curvature, not this two-form restriction.
4. Evaluate the unit-square action for p=(q2,−q1). **Solution:** The two nonzero edge contributions are −1 Js each, giving −2 Js. The pulled-back omega flux is +2 Js under the stated convention.
5. Why can a selected line pass the isotropy test for a failed graph? **Solution:** A two-form vanishes on any repeated tangent vector. The line lacks the independent second direction and the required half-dimensionality.
6. Does a Lagrangian graph always have a global scalar generating action? **Solution:** Not on an arbitrary domain. Closedness gives local potentials, while the circle example has a nonzero period and obstructs a global single-valued one.

## Research connection and sources

A learned momentum field should be tested for its full derivative symmetry and loop action, rather than only pointwise agreement with a proposed potential. Convexity may be an additional modeling choice, but it is not the definition of a Lagrangian graph. Record the ambient space, graph dimension, units and domain before connecting the result to later Hamilton-Jacobi or variational work.

For geometric definitions and graphs of closed one-forms, see Ana Cannas da Silva's [Symplectic Geometry, section 2.1](https://arxiv.org/html/math/0505366v1) and the author-hosted [Lectures on Symplectic Geometry, section 3.2](https://people.math.ethz.ch/~acannas/Papers/lsg.pdf). The explicit coefficient family, rectangle controls and action comparisons here are derived directly and checked independently.
