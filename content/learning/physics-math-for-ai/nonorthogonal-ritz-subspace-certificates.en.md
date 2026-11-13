## 1. Approximate the space before searching inside it

A trial space replaces an enormous or infinite state space by combinations of a manageable set of vectors. Rayleigh–Ritz then finds stationary energy directions inside that space. The vectors spanning it need not be orthogonal. This flexibility is useful for localized functions, learned features, Krylov vectors and physically motivated states, but it makes the inner-product metric visible in the coordinates.

Imagine a plane embedded in a three-dimensional physical space. Redrawing the grid lines on the plane changes coordinates, while tilting or enlarging the plane changes the admissible states. An algorithm can improve the first task without fixing an error caused by the second. We will keep that distinction visible through exact matrices, a hand example and residuals pointing outside the plane.

| Change | Physical meaning | Appropriate comparison |
| --- | --- | --- |
| Invertible basis change within one space | A different coordinate grid | Same states, projector and Ritz values |
| Adding independent trial directions | A larger admissible space | Nested-space energy bounds |
| Changing to a space without inclusion | A different approximation | No automatic monotonic ordering |

The assumptions for the numerical chapter are a finite Hermitian H and a general complex matrix B with independent columns. Energy units are inherited from H. No time coordinate or dynamical trajectory is supplied.

## 2. The Gram matrix is the coordinate metric

Collect the k trial vectors as the columns of an n by k matrix B. A coefficient vector c represents the physical state Bc. Its norm is not generally the Euclidean norm of c, because columns may have different lengths and nonzero mutual overlaps.

$$
|\psi\rangle=Bc,\qquad
S=B^\dagger B,\qquad
\|\psi\|^2=c^\dagger Sc,\qquad
c\ne0\ \Longrightarrow\ c^\dagger Sc>0
\quad\text{when }\operatorname{rank}B=k.
$$

The proof is direct: the quadratic form is the squared norm of Bc, and independent columns make Bc nonzero for nonzero c. Thus S is Hermitian positive definite. If two columns become exactly dependent, S becomes singular and these coordinates cease to describe a k-dimensional space. A nearly dependent basis is different: it can still have full exact rank but be numerically difficult.

For a single vector b=(1,0,1), the coefficient c=one gives squared physical norm two, not one. The Gram matrix in that one-dimensional coordinate system is simply the number two. This small example explains why the metric is part of the problem rather than an optional normalization decoration.

## 3. Stationarity produces a generalized eigenvalue problem

Restrict the energy expectation to the trial space. The projected energy matrix A is Hermitian. Differentiating its quotient with respect to all complex coefficient directions gives the generalized stationary equation.

$$
A=B^\dagger HB,\qquad
E(c)=\frac{c^\dagger Ac}{c^\dagger Sc},\qquad
Ac=ESc,\qquad
C=S^{-1}A,\qquad SC=C^\dagger S.
$$

C has real eigenvalues because it is self-adjoint in the S inner product. It is generally not Hermitian in the ordinary Euclidean inner product. Passing C directly to a numerical routine that assumes an ordinary Hermitian matrix is therefore an unjustified shortcut.

One valid numerical route uses S=LL dagger with L nonsingular: transform A to L inverse A L inverse dagger, solve that Hermitian problem and transform coordinates back. The independent reference calculation uses this route. The exact engine instead constructs the real characteristic polynomial of C and isolates its roots. Different computational routes make the comparison stronger than calling the same eigensolver twice.

## 4. Project onto the physical trial space

The orthogonal projector onto the column space of B includes the inverse Gram matrix. Without that factor, B times its adjoint would generally rescale or mix vectors instead of projecting them.

$$
P_V=BS^{-1}B^\dagger,\qquad
P_V^\dagger=P_V,\qquad
P_V^2=P_V,\qquad
P_VB=B,\qquad
\operatorname{tr}P_V=k.
$$

For idempotence, insert B dagger B=S into the middle of the product. For Hermiticity, use the Hermiticity of S inverse. The projector depends on the physical space, not on a particular independent coordinate grid. It is an effective object for comparing two descriptions that look quite different as arrays.

Within the space, the compressed Hamiltonian maps a state to PV H times that state. Its physical spectral projectors sum to PV, not to the identity of the larger ambient space. A repeated Ritz value belongs to a whole spectral subspace. Choosing a numerical basis inside that subspace would add a nonunique convention; the prepared engine reports the projector instead.

## 5. Inclusion and min–max give the useful bounds

List ambient eigenvalues lambda in increasing order, including multiplicities, and do the same for the k Ritz values mu. Restricting allowed directions cannot lower the minimum below the ambient ground energy. More generally, the min–max principle compares each ordered value.

$$
\lambda_j\le\mu_j\le\lambda_{j+n-k},\qquad j=0,\ldots,k-1,\qquad
V\subseteq W\ \Longrightarrow\ \min_{\psi\in W\setminus\{0\}}E(\psi)
\le\min_{\psi\in V\setminus\{0\}}E(\psi).
$$

For the lower ordered bound, the minimization over candidate subspaces inside V has fewer choices than the ambient min–max problem. For the upper bound, intersect V with the ambient low-energy span through index j+n-k. The dimension inequality guarantees at least j+one common directions. Every vector in that intersection has quotient at most the indicated ambient eigenvalue, giving an admissible comparison subspace.

The minimum decreases when an independent trial direction is actually added. Merely increasing a parameter count or choosing a different larger-dimensional space does not establish inclusion. These statements also require one fixed H, one physical inner product and consistent constraints. Changing a boundary condition or symmetry sector changes the comparison problem.

## 6. A Ritz residual can point outside the trial space

For any selected state in V, decompose its full eigenvalue residual into components inside and outside V. The two parts are orthogonal. Coefficient stationarity means the inside part vanishes, while an ambient eigenstate requires both parts to vanish.

$$
r=(H-E)\psi,\qquad
r_V=P_Vr,\qquad r_\perp=(I-P_V)r,\qquad
B^\dagger r=Ac-ESc,\qquad
\operatorname{Var}(H)=\frac{\|r_V\|^2+\|r_\perp\|^2}{\|\psi\|^2}.
$$

If c solves the generalized eigenvalue equation, B dagger r is zero. That is the Galerkin orthogonality condition: the residual is invisible to all trial-space test directions. It does not say that the residual is small in the full space. A small coefficient gradient can therefore coexist with a substantial physical approximation error.

Also distinguish the compressed second moment, the squared norm of PV H psi divided by the state norm, from the full second moment involving H squared. Compression can discard the part of H psi that leaves V. A zero variance for the compressed problem is not a zero energy variance for H. Draw the state plane and its normal direction before interpreting either number.

## 7. Work through a nonorthogonal plane

Use H=diag(0,1,3), with trial vectors b0=(1,0,1) and b1=(0,1,1). Their span is the plane whose third coordinate equals the sum of the first two. The exact ambient ground vector (1,0,0) is absent from that plane.

$$
B=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\qquad
S=\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad
A=\begin{pmatrix}3&3\\3&4\end{pmatrix},\qquad
\det(A-\mu S)=3\mu^2-8\mu+3.
$$

The matrix entries can be checked with dot products. The first energy entry is three because H sends b0 to (0,0,3); the off-diagonal entry is also three. Keeping the Gram denominator is essential. Solving the quadratic gives the two Ritz energies.

$$
\mu_\pm=\frac{4\pm\sqrt7}{3},\qquad
\mu_-\approx0.45141623,\qquad
\mu_+\approx2.21525044,\qquad
0\le\mu_-\le1\le\mu_+\le3.
$$

With b0 alone, the only quotient is three halves. Adding b1 lowers the minimum to mu minus but cannot reach zero, since the ambient ground direction is still unavailable. Adding a third independent vector such as (0,0,1) spans the whole space and permits the exact minimum zero. This is a concrete nested sequence with a geometric explanation.

## 8. Follow the optimized state and its remaining error

For either Ritz root, choose coefficient c0=one and solve for c1=rho from the first row of the generalized equation. The smaller root yields the minimizing direction in the plane. In this small example, the vector can be derived analytically even though the general engine does not return numerical Ritz eigenvectors.

$$
\rho=\frac{2\mu-3}{3-\mu}=-\frac{\mu}{1-\mu},\qquad
\psi=\begin{pmatrix}1\\\rho\\1+\rho\end{pmatrix},\qquad
(H-\mu)\psi=\mu\begin{pmatrix}-1\\-1\\1\end{pmatrix},\qquad
\operatorname{Var}(H)=\frac{3\mu^2}{2(1+\rho+\rho^2)}.
$$

The residual direction is orthogonal to both trial vectors. Inside the plane, no first-order direction can lower the energy at this Ritz state. Outside the plane, the nonzero residual reveals missing directions. This makes a useful paired visualization: one arrow for the projected residual and a separate arrow for the outside residual, with both measured in the same energy convention.

The residual norm is useful for deciding whether to enlarge a space, but a convergence-rate claim needs more assumptions. The ambient ground energy is known here, so the variational excess can be checked directly. In a research problem, an unknown ambient energy cannot be replaced by the current trial minimum and then declared to have zero approximation error.

## 9. Change coordinates without changing the state

Let M be an invertible complex matrix and change both the basis and the coefficients. The Gram and energy matrices transform by congruence, not by ordinary similarity. Their combination S inverse A transforms by similarity, so the Ritz values are unchanged.

$$
B'=BM,\quad c'=M^{-1}c,\quad
S'=M^\dagger SM,\quad A'=M^\dagger AM,\quad
B'c'=Bc,\quad S'^{-1}A'=M^{-1}(S^{-1}A)M.
$$

For the hand plane, use a scale and a complex shear. The new coordinate grid is oblique and has different lengths, while the physical plane and every state represented with transformed coefficients stay fixed.

$$
M=\begin{pmatrix}2&1+i\\0&1\end{pmatrix},\qquad
S'=\begin{pmatrix}8&6+4i\\6-4i&8\end{pmatrix},\qquad
A'=\begin{pmatrix}12&12+6i\\12-6i&16\end{pmatrix}.
$$

| Quantity for the state b0 | Original coordinates | Changed coordinates |
| --- | --- | --- |
| Coefficient vector | (1,0) | (1/2,0) |
| Euclidean coefficient squared norm | 1 | 1/4 |
| Physical squared norm and energy | 2 and 3/2 | 2 and 3/2 |

The determinant of the new Gram matrix is twelve instead of three. The generalized pencil determinant acquires a factor four, leaving its roots unchanged. Renormalizing coefficients by their bare Euclidean length after this basis change would generally change the physical state. Normalize with the metric when a normalized state is required.

## 10. Exact independence and good conditioning are different

Consider the real two-by-two basis with rows (M,M-1) and (M-1,M-2). Its determinant is minus one for every M, so it never loses rank. At large M, its columns are nevertheless almost parallel, making the Gram metric difficult for ordinary finite precision.

$$
\det B=-1,\qquad \det S=1,\qquad
T=\operatorname{tr}S=4M^2-8M+6,\qquad
\lambda_{\min}(S)=\frac{2}{T+\sqrt{T^2-4}},\qquad
\kappa_2(S)\sim T^2.
$$

At M=one million the Gram condition number is about sixteen times ten to the power twenty-four. The smaller eigenvalue remains positive. A fixed floating tolerance could discard a mathematically independent direction and thereby change the trial space. If a research algorithm intentionally truncates small singular values, document the criterion, the resulting space and its effect on observables.

The Gram condition number is the square of the two-norm condition number of B. It is a property of the coordinate representation, not a direct ground-state error. Orthogonalization can improve coordinates without adding any new physical direction. Conversely, adding a useful physical direction can improve approximation while making a poorly chosen basis more ill-conditioned.

## 11. Extending to continuous operators needs another layer of assumptions

In a differential-operator problem, trial functions must belong to the appropriate operator or quadratic-form domain and obey the intended boundary conditions. Matrix entries must represent the same physical inner product and form. A conforming finite-dimensional restriction of a suitable self-adjoint, lower-bounded problem inherits the variational logic. Existence of ordered discrete eigenvalues and convergence of a sequence need additional spectral assumptions.

This is the link to finite elements: an energy or stiffness matrix and an overlap or mass matrix form a generalized eigenproblem. Inexact quadrature, inconsistent boundary enforcement or a nonconforming space can break the simple upper-bound interpretation. A finely drawn matrix visualization does not check those modeling assumptions automatically.

A dense sequence of admissible trial spaces can support convergence results under suitable conditions, but one finite residual plot supplies no universal rate. Nonlinear ansatz families, Hartree–Fock states and tensor-network constraints introduce further optimization structure. Their stationary conditions should not be relabeled as unrestricted linear Ritz equations merely because both methods use the word variational.

## 12. Connect the metric to AI and research practice

Krylov and subspace eigensolvers create a space from vectors related by repeated Hamiltonian actions. Davidson-type corrections and residual enrichment use information about missing directions. Neural features or previously learned states can also form a trial basis. In each case, separate the choice of physical space from its numerical coordinates and from the optimizer or eigensolver used inside it.

When the basis depends on an external parameter, derivatives of the overlap matrix matter. For a differentiable isolated Ritz branch with coefficient normalization in the S metric, differentiating the generalized equation gives a metric-aware energy derivative.

$$
Ac=ESc,\qquad c^\dagger Sc=1,\qquad
\frac{dE}{dq}=c^\dagger\left(\frac{dA}{dq}-E\frac{dS}{dq}\right)c.
$$

Terms involving the coefficient derivative cancel using stationarity and normalization. Omitting the derivative of S is generally wrong when the coordinate basis moves. At a degeneracy, derivatives must be resolved in the degenerate subspace instead of assigning one formula to an arbitrary chosen vector. The supplied fixed-input experiment does not compute parameter derivatives of a moving basis; this is a research connection to derive and check separately.

For a concrete check, take one moving column B(q)=(1,q) and H=diag(0,2). Then S=1+q squared and A=2q squared. At q=one, the correct energy derivative is one; omitting the overlap derivative gives two. This simple discrepancy makes the missing metric term observable.

A reproducible study should record basis construction, rank decisions, conditioning, H and overlap conventions, energy and full residuals, and changes under space enlargement. If a learned surrogate predicts energies, test physical states and independent matrix expectations rather than validating it only against the same surrogate's outputs.

For the subspace/Galerkin viewpoint, compare [MIT numerical-methods lecture notes](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-7/); the positive-metric generalized eigenproblem is discussed in [Cornell matrix-computation notes](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-11-03.html). The concrete plane, coordinate transformation and checks here are derived explicitly.

## Research extension: the missing-space block, Schur correction and residual enrichment

### Separate coordinates from the omitted physical sector

The original Gram metric supplies an orthonormal frame Q for the same trial space. Complete it with an orthonormal exterior frame R. Both sectors are nonempty here. The symbol AV below is an orthonormal compression, distinct from the original B†HB coordinate matrix:

$$
Q=BS^{-1/2},\quad Q^\dagger Q=I,\quad [Q\ R]\text{ unitary},\qquad
 \widehat H=[Q\ R]^\dagger H[Q\ R]=\begin{pmatrix}A_V&C^\dagger\\C&D\end{pmatrix},\quad
 A_V=Q^\dagger HQ,\ C=R^\dagger HQ,\ D=R^\dagger HR.
$$

The coupling C records Hamiltonian action that crosses between sectors. Orthogonalizing a basis fixes its coordinate metric; it does not remove this physical coupling. If an eigenvector has sector coordinates (x,y), its second block equation can be solved only away from the exterior spectrum:

$$
E\notin\operatorname{spec}(D):\quad
 y=-(D-EI)^{-1}Cx,\qquad
 \left[A_V-C^\dagger(D-EI)^{-1}C\right]x=Ex.
$$

Substitution gives an energy-dependent effective matrix on the retained sector. This is finite block elimination, the basic [Feshbach–Schur mechanism](https://arxiv.org/pdf/2105.02058). It is not permission to invert a singular D−EI or to replace E by a convenient trial value without an error analysis.

If the whole exterior block has a verified lower bound β, then below it:

$$
D\succeq\beta I,\quad E<\beta:\qquad
 0\preceq C^\dagger(D-EI)^{-1}C\preceq\frac{C^\dagger C}{\beta-E}.
$$

The correction is positive semidefinite and is subtracted from AV. The inverse bound follows by diagonalizing D. This explains why restricting a low state to the retained sector tends to raise its energy. The energy dependence and the exterior separation control whether a perturbative picture is reliable.

### A two-by-two comparison bounds the missing-space energy error

Suppose μ is the lowest retained Ritz energy, β bounds every exterior direction, and γ bounds the operator norm of C. For arbitrary sector components:

$$
\mu=\lambda_{\min}(A_V),\quad D\succeq\beta I,\quad\|C\|_2\le\gamma,\quad\beta>\mu,\qquad
 \langle(x,y),\widehat H(x,y)\rangle\ge
 \mu\|x\|^2+\beta\|y\|^2-2\gamma\|x\|\|y\|.
$$

The cross term is bounded by Cauchy–Schwarz. Minimizing the resulting real two-by-two quadratic form over the two component lengths gives:

$$
L=\frac{\mu+\beta-\sqrt{(\beta-\mu)^2+4\gamma^2}}2\le\lambda_0(H)\le\mu,\quad g=\beta-\mu>0,\qquad
 0\le\mu-\lambda_0(H)\le\frac{2\gamma^2}{g+\sqrt{g^2+4\gamma^2}}\le\frac{\gamma^2}{g}.
$$

The upper bound uses a retained trial vector. The stable expression for the error avoids subtracting nearly equal square roots when γ is small. Its final γ²/g bound is simpler but can be much looser. Verifying an exterior lower bound may itself be difficult; a small residual for one retained vector does not supply that bound. All energies and γ use the same energy unit.

A one-dimensional retained sector makes the comparison exact:

$$
H_{\Delta,t}=\begin{pmatrix}0&t\\t&\Delta\end{pmatrix},\quad\Delta>0,\quad t\ge0,\quad
 V=\operatorname{span}\{e_0\},\quad\mu=0,\quad\gamma=t,\quad\beta=\Delta,\qquad
 \delta=\mu-\lambda_0=\frac{\sqrt{\Delta^2+4t^2}-\Delta}{2},\quad
 w_{\rm outside}=\frac12\left(1-\frac{\Delta}{\sqrt{\Delta^2+4t^2}}\right).
$$

For t much smaller than Δ, energy error is approximately t²/Δ while exterior population is approximately t²/Δ². Energy error and vector leakage have different units and different gap dependence. Approaching Δ=0 invalidates treating t/Δ as a small parameter; the exact two-level solution remains the appropriate comparison. The general error certificate above was stated with β>μ.

### Repair coordinates, then decide whether to enlarge the space

Different orthonormal frames of the same space differ by a unitary. The physical quantities used in the block bound are unchanged. Even a raw state rescaling cannot change its normalized residual:

$$
Q'=QU,\quad U^\dagger U=I:\qquad
 \lambda_{\min}(Q'^\dagger HQ')=\mu,\quad
 \|R^\dagger HQ'\|_2=\|CU\|_2=\|C\|_2,\qquad
 \psi=\alpha e_0\ (\alpha\ne0):\quad\frac{\|(H_{\Delta,t}-\mu)\psi\|}{\|\psi\|}=t.
$$

Thus making coordinate coefficients tiny does not repair the missing-space coupling. Conversely, improving a poorly conditioned Gram matrix can stabilize a solve while leaving the physical approximation error fixed.

There is a direct enrichment experiment. Choose a normalized minimizing Ritz vector and its nonzero full residual. Galerkin orthogonality places that residual outside the existing trial space:

$$
H\psi=\mu\psi+r,\quad\|\psi\|=1,\quad r\perp V,\quad r\ne0,\quad
 \zeta=r/\|r\|,\quad\nu=\zeta^\dagger H\zeta,\quad V_{\rm new}=V+\operatorname{span}\{\zeta\},\qquad
 \mu_{\rm new}\le\frac{\mu+\nu-\sqrt{(\nu-\mu)^2+4\|r\|^2}}2<\mu.
$$

In the two-dimensional span of ψ and ζ the compressed matrix has diagonal entries μ,ν and off-diagonal entry ‖r‖. Its lower eigenvalue is strictly below μ. The enlarged space contains this pair, which proves the bound. If r=0 no new direction can be defined this way; an invariant retained space can still miss a lower exterior eigenstate. This is one step of improvement under its hypotheses, not a general convergence-rate theorem.

For Δ=1, use t=u/(1−u²) with0≤u<1. The root √(1+4t²)=(1+u²)/(1−u²) is rational, so the exact comparison can be reproduced without floating eigenvalue decisions:


```python
from fractions import Fraction as Q
for u in [Q(0), Q(1, 3), Q(1, 2), Q(3, 4)]:
    coupling = u/(1-u*u)  # Delta = 1
    root = (1+u*u)/(1-u*u)
    assert root*root == 1+4*coupling*coupling
    error = (root-1)/2
    outside = (1-1/root)/2
    assert error <= coupling*coupling
    print(u, coupling, error, coupling*coupling, outside)
```

| u | coupling t | exact energy error | looser t squared bound | exterior population |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1/3 | 3/8 | 1/8 | 9/64 | 1/10 |
| 1/2 | 2/3 | 1/3 | 4/9 | 1/5 |
| 3/4 | 12/7 | 9/7 | 144/49 | 9/25 |

![The exact two-sector energy correction and exterior population depend on coupling relative to the exterior separation; a quadratic energy estimate becomes loose as that ratio grows.](/assets/learning/quantum-variational-research-complement-schur-enrichment.en.svg)

The table is exact and the curves show the stated two-level model. The current reader retains its general nonorthogonal basis and residual diagnostics. It has no new exterior-bound input, automatic enrichment iteration or energy-dependent Schur solver.

#### Research check 1: a small norm in the wrong sector

Let H=diag(0,2), but retain only e1. The Ritz energy is2 and its full residual is zero, yet the omitted ground energy is0. Here C=0 and the exterior lower value0 is not above μ. The assumed exterior ordering fails, so the ground certificate cannot label this state exact ground merely from its residual.

#### Research check 2: one residual direction solves the two-level example

For H with Δ=1 and t=3/8, the retained e0 has residual(0,3/8). Adding its direction gives the complete two-dimensional space. The energy minimum decreases from0 to−1/8, the exact ground energy, and the exterior weight is1/10. This exactness is special to completing this two-dimensional model.

#### Research check 3: coordinate repair versus model enrichment

Apply an invertible complex shear to a larger retained basis and transform coefficients consistently. The orthogonal projector, retained Ritz minimum and norm of the physical coupling remain unchanged. Adding an independent exterior residual changes the projector and admissible states. Only the latter addresses this missing-direction approximation error; neither action alone proves a claimed rate for a general learned trial family.

## 13. Exercises with explanations

**1. Why the denominator matters.** Use only b0=(1,0,1) with H=diag(0,1,3). What are the energy matrix, Gram matrix and energy?

**Explanation.** A is the one-by-one matrix three and S is the one-by-one matrix two. Their quotient is three halves. Treating the coefficient c=one as a normalized physical vector would give three, the wrong energy. A nonorthogonal basis changes normalization even when there is only one column.

**2. Find the two Ritz values.** Expand the determinant of A minus mu S in the hand plane and compare with the ambient spectrum.

**Explanation.** Multiply (3-2mu)(4-2mu) and subtract (3-mu) squared. The result is 3mu squared minus 8mu plus 3. Its roots are (4 plus or minus square root seven)/3. They lie in the two interlacing intervals [0,1] and [1,3], consistent with the ambient eigenvalues zero, one and three.

**3. Stationary without an ambient eigenstate.** Let H=diag(0,2,4), use b0=(1,0,1), b1=(0,1,0) and coefficients (1,1). Compute both residual parts.

**Explanation.** The state is (1,1,1), its squared norm is three and its energy is two. The full residual is (-2,0,2), perpendicular to both trial columns. Hence the projected residual is zero, while the outside variance is eight thirds. Here A equals two S, so the whole two-dimensional trial space is one degenerate Ritz space. Its states need not be ambient eigenstates.

**4. Undo a complex coordinate shear.** Use the M in section nine and original coefficients (1,0). What changes physically?

**Explanation.** The transformed coefficients are (one half,zero). Multiplication by the new first column reproduces the original state exactly. The bare coefficient squared norm changes from one to one quarter, but the physical squared norm stays two and energy stays three halves. Both generalized characteristic polynomials have the same roots after removing their nonzero determinant factor.

**5. Nearly parallel but independent.** Evaluate the determinant and Gram determinant of the basis in section ten. Should a fixed tolerance declare rank one?

**Explanation.** M(M-2) minus (M-1) squared is minus one. The Gram determinant is its squared magnitude, one. The columns are exactly independent, even at one million. Their Gram eigenvalues are very unequal, which is a conditioning diagnosis. A chosen numerical rank truncation changes the approximation and must be justified and reported as such.

**6. More dimensions without inclusion.** With H=diag(0,1,3), compare V spanned by the ambient ground vector with W spanned by the second and third coordinate vectors. Must the minimum decrease?

**Explanation.** V has dimension one and minimum zero; W has dimension two and minimum one. W does not contain V, so there is no contradiction with nested-space monotonicity. A larger dimension alone is not the missing hypothesis. Draw the line outside the plane to make the failure of inclusion explicit.

## 14. Read the prepared experiment and its limits

The numerical family accepts every dimension and column count with one through four ambient dimensions and independent columns. It uses sixty-three owned bounded integer fields, exact complex-rational Gram and residual algebra, 4096-bit square-free/Sturm roots, 2048-bit directed work and outward 512-bit observations. Rank is not chosen by a floating threshold. Probabilities are intersected with their proven range; basis and state vectors are not silently corrected.

Compare the hand plane, a full space and an ill-conditioned coordinate grid. Predict whether the selected state is Ritz stationary and whether it is an ambient eigenstate before revealing the residual decomposition. Read the full-residual Frobenius statistic as a diagnostic for a spectral subspace, not for an unreported individual eigenvector. Nested spaces in this experiment are the actual column prefixes; the built-in coordinate comparison uses a nonzero scale and complex shear, while the freely supplied basis itself is general.

Use the fixed comparisons, guided predictions, saved controls and quizzes to connect the metric calculation with approximation quality. Real three-dimensional examples provide a camera-controlled space view and two orthographic projections; complex cases retain complete component diagrams and matrices. The entry point is **Book → Interactive Book**. Continuous convergence theory, many-body approximation and research depth require further chapters and independent evidence; this finite experiment does not complete them.
