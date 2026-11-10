## 1. Why a phase needs a reference

Imagine measuring a direction with a clock face at every vertex of a network. Turning one clock face changes the numerical reading at that vertex. Comparing two readings requires knowing how their clock faces are aligned. A connection supplies that comparison rule.

A complex number $\psi_i$ is a convenient pair of real components. Its magnitude can represent an amplitude, while its phase is expressed relative to a local internal frame. Changing that frame is different from rotating the represented quantity relative to fixed surroundings. The distinction becomes precise only after we state which variables transform and which observations define the model.

Review [complex numbers](/book/online/physics-math-for-ai/real-complex-numbers/) and the preceding field-action chapters as needed. This chapter is a static, finite, nondimensional graph experiment. It does not provide an electromagnetic time evolution or a continuum approximation.

## 2. Declare the variables and the direction convention

At each site $i$, place $\psi_i\in\mathbb C$. For an oriented stored edge $(i,j)$, assign a unit complex link $U_{ij}$. Our convention is that $U_{ij}$ converts components expressed in the **$j$ frame into the $i$ frame**. Consequently,

$$
D_{ij}=\psi_i-U_{ij}\psi_j
$$

compares quantities in one frame. Traversing that edge backwards uses $U_{ij}^*=U_{ij}^{-1}$. The stored edge direction labels how to select a factor in a path product; it does not mean that $U_{ij}$ transports an $i$-frame vector into the $j$ frame.

Parallel edges are separate comparisons. A self-loop is also permitted: its difference is $(1-U_{ii})\psi_i$, which need not vanish. An isolated vertex remains a site even when it has no edge.

## 3. Change both the components and the comparison rule

Choose $g_i\in U(1)$ independently at every site. Define

$$
\psi'_i=g_i\psi_i,\qquad
U'_{ij}=g_iU_{ij}g_j^*.
$$

Substitution gives

$$
D'_{ij}=g_i\psi_i-g_iU_{ij}g_j^*g_j\psi_j
=g_iD_{ij}.
$$

The difference is **covariant**: its components change predictably. Its squared magnitude is **invariant**, since $|g_i|=1$. Covariance does not mean that every displayed complex number stays unchanged.

Rotating $\psi$ while holding $U$ fixed performs a different operation. Its energy can change, and an accidental equality of two energies does not establish a gauge transformation.

## 4. Global and local do not determine physical meaning alone

A uniform choice $g_i=g$ is called global; independent site choices are local. Here we explicitly declare all these internal frame changes to be coordinate redundancy. Thus the uniform subgroup is also redundant in this model.

Other models can have a physical global symmetry or restrict the allowed boundary transformations. An external phase reference can make a relative phase observable. Deciding what is redundancy requires specifying the observable algebra, external references and allowed transformations, rather than applying the slogan “global is physical, local is gauge.” Boundary anchors and gauge fixing receive a separate treatment later.

## 5. Specify the energy before interpreting a number

For positive edge weights $w_e$, real $a$ and nonnegative $b,\kappa$, use

$$
E=\sum_{e=(i,j)}w_e|\psi_i-U_e\psi_j|^2
+a\sum_i|\psi_i|^2+\frac b2\sum_i|\psi_i|^4
+\kappa\sum_{C\in\mathcal W}(1-\operatorname{Re}W_C).
$$

The selected list $\mathcal W$ contains closed walks. Repeating a walk repeats its contribution; choosing a different list changes the energy functional. Each term is invariant under the coordinated transformation.

The parameter $a$ may be negative. With $b=0$, a negative quadratic direction can make the energy unbounded below as the amplitude grows. Gauge invariance alone does not imply stability. All values here are nondimensional; neither a physical lattice spacing nor an energy unit has been supplied.

## 6. Work through a triangle without a calculator

Store edges $0\to1$, $1\to2$ and $2\to0$, all with $U=1$, weight one. Set $\psi=(1,1,1)$, $a=b=0$, $\kappa=1$, and select the forward triangle walk. Every difference is zero and the loop product is one, so $E=0$.

Choose $g=(1,i,-1)$. Then

$$
\psi'=(1,i,-1),\qquad (U'_{01},U'_{12},U'_{20})=(-i,-i,-1).
$$

For the first edge, $1-(-i)i=0$; the other two differences also vanish. The new loop product is $(-i)(-i)(-1)=1$, so the coordinated energy is still zero.

If only matter is changed, the original unit links give squared differences $|1-i|^2=2$, $|i+1|^2=2$ and $|-1-1|^2=4$. The total is eight. The contrast concerns two explicitly different operations, not a failure of invariance.

## 7. Closed products cancel intermediate frames

For a walk $i_0\to i_1\to\cdots\to i_m$, multiply the appropriate forward links or inverse links in that order. With our convention this product compares the final frame with the initial frame. Its transformation is

$$
P' =g_{i_0}P g_{i_m}^*.
$$

When $i_m=i_0$, the U(1) factors commute and $W'_C=W_C$. Reversing the entire walk conjugates the product. Traversing one edge and immediately returning gives $UU^*=1$, even if other cycles carry nontrivial holonomy.

An open product alone is not invariant. An endpoint-dressed quantity $\psi_{i_0}^*P\psi_{i_m}$ is invariant because the endpoint phases cancel. The one-edge version is $\psi_i^*U_{ij}\psi_j$.

## 8. The real part loses information

Change the triangle's first link to $i$, leaving the other links and all matter equal to one. The hopping contribution is two. The loop product is $i$, giving penalty one and total energy three.

Replacing that first link by $-i$ again gives hopping two and penalty one. Yet the full loop products differ. Keeping only $\operatorname{Re}W$ or only the energy cannot distinguish these connections. A visualization should therefore show both real and imaginary components, together with the selected traversal direction.

If the matter is zero and the only selected walk immediately backtracks, its product is one for either connection. That measurement misses the triangle flux entirely. A correct calculation can still be an incomplete observation.

## 9. Count cycles without overclaiming completeness

For a graph with $N$ vertices, $E_g$ edges and $C$ connected components, the cycle rank is

$$
\beta_1=E_g-N+C.
$$

Self-loops count as edges and parallel edges can form a cycle. A graph with two parallel links between two sites and one isolated site has $E_g=2$, $N=3$, $C=2$, hence rank one.

The number of selected closed walks is not generally the cycle rank. Walks can repeat, backtrack or fail to cover independent cycles. This chapter does not claim that its selected products form a complete invariant set. Even a complete set for connections alone would not classify a joint matter-and-connection configuration: local magnitudes and suitable relative observables would still matter.

## 10. Derive the Hermitian quadratic operator

Let the row of $B_U$ associated with $(i,j)$ act as $(B_U\psi)_e=\psi_i-U_e\psi_j$. Then

$$
H_U=B_U^\dagger W B_U+aI,\qquad
E_{\mathrm{quadratic}}=\psi^\dagger H_U\psi.
$$

Here $W$ is the diagonal matrix of positive edge weights. A self-loop contributes $w|1-U|^2$ to its site's diagonal; adding two independent diagonal entries would be incorrect because both row coefficients occupy the same column.

For $G=\operatorname{diag}(g_i)$, direct substitution gives

$$
H_{U'}=G H_U G^\dagger.
$$

Thus the operator's entries change but its spectrum is unchanged by unitary similarity. Trace and squared Frobenius norm provide useful checks, although agreement of those two numbers alone does not prove similarity. The experiment calculates the full matrix covariance residual rather than inferring it from those summaries.

## 11. Keep the gradient convention explicit

For $\psi_i=x_i+iy_i$, the displayed gradient is the Wirtinger derivative

$$
\frac{\partial E}{\partial\psi_i^*}
=\frac12\left(\frac{\partial E}{\partial x_i}
+i\frac{\partial E}{\partial y_i}\right)
=(H_U\psi)_i+b|\psi_i|^2\psi_i.
$$

The loop term has no matter dependence. The factor one-half is essential: this complex gradient is not the unscaled pair of real derivatives. Under a coordinated frame change it transforms by $g_i$, just like the matter components. This is a useful condition when designing an equivariant update rule, but no optimizer or time integrator is specified here.

## 12. Understand what exact rational phases cover

The calculator uses

$$
z(t)=\frac{1-t^2+2it}{1+t^2},\qquad t\in\mathbb Q,
$$

plus a separate representation of $-1$. Expanding the numerator shows $|z(t)|^2=1$ exactly. For example, $t=1/2$ gives $3/5+4i/5$.

A bounded rational input chart samples the circle; it does not enumerate every real phase. Products can have much longer rational representations than their inputs. Exact fractions avoid phase-wrapping and floating-point norm errors for the declared finite experiment, without proving anything about continuum limits or measurement uncertainty.

## 13. Predict and compare the prepared experiments

1. **Flat triangle:** predict zero coordinated energy and matter-only energy eight, then inspect each edge.
2. **Positive/negative flux:** compare the full complex loop products before looking at the common energy three.
3. **Reverse walk:** reverse every step and the order of the steps; check conjugation, not equality in general.
4. **Open chain:** observe zero cycle rank alongside nonzero matter hopping energy. No loop does not mean no relative observation.
5. **Disconnected triangles:** use one common phase per component. Links remain unchanged even when the frame is not uniform across the entire graph.
6. **Self-loop and parallel links:** compare the matrix entries and component counts with the definitions above.
7. **Rational frames:** inspect exact covariance residuals with nontrivial complex matter and unequal weights.

These preparations and the numerical foundation are authored. Integration into the interactive reader follows independent verification and the teaching-interface checks; this draft does not imply that the controls are already published.

## 14. Connect the construction to AI carefully

A graph model can store features in local frames and transport neighboring features before aggregation. The identity $D'_{ij}=g_iD_{ij}$ shows the kind of transformation law an equivariant layer should respect. A loss built from $|D|^2$ can then ignore arbitrary frame choices.

A research test should transform both features and transport data, compare the predicted transformation of outputs, and include a deliberately incorrect matter-only transformation. It should also test disconnected graphs, self-loops, reversed paths and incomplete measurements. Passing this U(1) experiment does not establish equivariance for rotations in three dimensions, non-Abelian groups or arbitrary network architectures.

## 15. Questions with explained answers

**Why is $\psi_i-\psi_j$ generally unsuitable?** Its two components acquire different phases. Without a connection, they cannot be combined into a single covariant difference under independent frame changes.

**Can every link be set to one around a triangle with product $i$?** No. A coordinated change preserves that product, whereas three unit links would have product one. A later chapter constructs tree gauge fixing and its remaining freedom.

**Does equal total energy imply gauge equivalence?** No. The $+i$ and $-i$ examples have the same total energy but different invariant complex loop products.

**Does zero matter remove all observables?** No. It removes the dressed matter quantities, while closed connection products can still differ.

**Why can an open chain have nonzero energy?** Its matter values can disagree after transport. Absence of cycles removes closed-loop degrees of freedom of the connection; it does not force those matter comparisons to vanish.

**Where is Gauss's law?** It is not part of this static energy experiment. Canonical electric variables, charges, constraints and an allowed boundary gauge group must be introduced before making that claim.

## 16. Source and limits

The link-transformation, Wilson-product and dressed-matter framework is discussed in David Tong's [Gauge Theory, Chapter 4, §4.2.1](https://davidtong.org/pdfs/teaching/gauge-theory/gauge4.pdf). That source treats lattice SU(N) theory. The arbitrary finite U(1) graph, rational chart, exact examples and numerical checks here are a separately derived teaching construction.

No graph embedding, cell areas, physical field strength, Haar integration, quantum state, canonical constraint, time evolution or continuum convergence is supplied. Next come gauge orbits, tree fixing and boundary anchors, followed by canonical constraints. Completion of this chapter alone does not complete the symmetry-and-gauge topic.
