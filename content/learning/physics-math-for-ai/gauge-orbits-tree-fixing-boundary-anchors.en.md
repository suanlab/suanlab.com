## 1. Equivalent descriptions are more than equal energies

Two coordinate descriptions may look different while representing the same state. To decide this, we need an allowed transformation that converts **every** relevant variable in one description into the corresponding variable in the other. Equal energy is much weaker: many unrelated states can share one energy value.

Review [local frames, U(1) connections and Wilson loops](/book/online/physics-math-for-ai/graph-u1-local-frames-wilson-loops/). Here we compare two independently specified connections and matter configurations on the same finite graph. We construct a transformation when it exists, and identify a failed invariant when it does not. There is still no time evolution or Gauss constraint.

## 2. Define the action and its orbit

For each oriented stored edge $e=(i,j)$, the unit complex number $U_{ij}$ converts $j$-frame components into $i$-frame components. A site transformation acts by

$$
\psi_i\longmapsto g_i\psi_i,\qquad
U_{ij}\longmapsto g_iU_{ij}g_j^*,\qquad |g_i|=1.
$$

The **orbit** of $(U,\psi)$ is the set of all configurations obtainable by the declared group. Unless anchors are specified, this group is $U(1)^N$. Two configurations are equivalent when their orbits agree, or equivalently when some group element maps one to the other.

The group is continuous. Rational Cayley parameters are an exact input representation for this experiment, not a declaration that the gauge group is a finite grid. A valid computed transformation can have longer fractions than the input chart permits.

## 3. A spanning forest supplies unique comparison paths

In a connected graph, a spanning tree includes all vertices and contains no cycles. For $C$ connected components, choose one tree per component: a spanning forest with $N-C$ edges. An isolated vertex is a one-vertex tree.

Choose one root in each tree. There is a unique simple tree path from its root to any vertex. Self-loops cannot be tree edges, and selecting both parallel links between the same two vertices would create a cycle. Such edges remain in the graph; they are not discarded from the model.

For a tree path, multiply stored links when traversing forward and conjugate links when traversing backward. The root and forest must be the same when we compare the resulting coordinates of two configurations.

## 4. Construct the fixing phases instead of guessing them

Set $h_r=1$ at each root. For a tree edge traversed from parent $p$ to child $q$, define

$$
h_q=\begin{cases}
h_pU_{pq},&\text{stored forward},\\
h_pU_{qp}^*,&\text{stored backward}.
\end{cases}
$$

Thus $h_i$ is the product along the root-to-$i$ path. Use these phases to define

$$
Q_{ij}=h_iU_{ij}h_j^*,\qquad z_i=h_i\psi_i.
$$

Every selected tree link becomes one. For a forward edge, $h_j=h_iU_{ij}$, so $Q_{ij}=h_iU_{ij}(h_iU_{ij})^*=1$. This is a change of representation: matter must also be transformed. Deleting a link without transforming the remaining data would be a different operation.

## 5. Non-tree links carry fundamental closed products

For a non-tree edge $(i,j)$, form the walk from root to $i$, then traverse that edge, then return from $j$ to the root along the tree. Its product is exactly

$$
W_e=h_iU_{ij}h_j^*=Q_{ij}.
$$

There are $\beta_1=E-N+C$ non-tree edges. They provide independent circle coordinates for the connection quotient in this finite U(1) setting. Unlike an arbitrary selected-walk list, these walks come from a complete spanning-forest construction.

The count includes self-loops and parallel edges. The product is complex: replacing it by its real part can lose information. A graph need not be embedded in space for these path products to make sense; calling them products does not assign a physical area or field strength.

## 6. Prove completeness for connections

Let $h^L,h^R$ be the fixing phases for two connections $U,V$ using the same forest and roots. If their non-tree coordinates agree, their full fixed connections agree, since tree coordinates are all one. Define

$$
g_i=(h_i^R)^*h_i^L.
$$

Then

$$
g_iU_{ij}g_j^*
=(h_i^R)^*Q^L_{ij}h_j^R
=(h_i^R)^*Q^R_{ij}h_j^R=V_{ij}.
$$

This constructs an exact witness. Conversely, a gauge transformation preserves every closed U(1) product, so equivalent connections have the same non-tree coordinates. Both directions are needed for a completeness claim.

The witness is not unique. Multiplying every $g_i$ in one connected component by the same phase leaves the connection transformation unchanged.

## 7. A tree condition leaves a componentwise phase

A transformation preserving all unit tree links must satisfy $g_i=g_j$ at the ends of each tree edge. Therefore it is constant on each connected component. Tree fixing removes $N-C$ local phase directions, leaving $C$ common phases.

For U(1), these common phases leave all links unchanged because scalar multiplication commutes. They can still rotate the matter values. Setting $h_r=1$ was a convention used to construct one representative; it did not prove that the original theory had a physically fixed root frame.

Changing only the root of a fixed U(1) tree multiplies every $h_i$ in that component by a common factor. The fixed links and matter Gram entries remain the same. Changing the tree can relocate or recombine the non-tree coordinates, so raw coordinate arrays from different tree choices should not be compared blindly.

## 8. Complete the comparison when matter is present

The fixed matter vectors satisfy joint equivalence precisely when, in each component, there exists one unit phase $\lambda$ such that

$$
z_i^R=\lambda z_i^L\quad\text{for every site in that component},
$$

in addition to connection equivalence. The full witness becomes $g_i=(h_i^R)^*\lambda h_i^L$.

A convenient invariant is the componentwise Gram matrix

$$
M_{ij}=(z_i)^*z_j
=\psi_i^*h_i^*h_j\psi_j.
$$

This is a matter observation dressed by the unique tree path between its endpoints. Entries between disconnected components are not included: there is no graph path relating their independent root frames.

## 9. Why the Gram criterion is sufficient

If $z^L$ is not zero, choose a pivot $p$ with $z_p^L\ne0$. Equal Gram diagonals give $|z_p^R|=|z_p^L|$, so

$$
\lambda=z_p^R/z_p^L
$$

has unit magnitude. Equal entries $M_{pj}$ then imply $z_j^R=\lambda z_j^L$ for every $j$. If the entire left vector is zero, its Gram diagonal vanishes everywhere; equality forces the right vector to be zero as well, and any common phase works.

The comparison uses a ratio of two known pivots. Fixing one arbitrary pivot to a positive real value is a different gauge section and can require a square root: the magnitude of $1+i$ is $\sqrt2$. That additional section is not implemented here.

Equal densities alone are insufficient. With unit triangle links, $(1,1,1)$ and $(1,i,1)$ have equal densities but different off-diagonal Gram entries. A transformation preserving those links must be common across the triangle and cannot produce that relative phase pattern.

## 10. Stabilizers describe remaining non-identifiability

A stabilizer consists of transformations leaving a configuration unchanged. For connections alone, the U(1) stabilizer has one common phase per connected component, hence dimension $C$.

For a joint configuration, a nonzero matter value in a component forces that common phase to one. Only components with matter zero at every site retain a continuous stabilizer. If there are $C_z$ such components, the joint orbit dimension is $N-C_z$.

For the flat triangle, nonzero matter gives orbit dimension three. Zero matter gives a one-dimensional stabilizer and orbit dimension two. These are dimensions of a kinematic group action, not counts of canonical physical degrees of freedom. They also warn against claiming a single globally smooth coordinate chart: zero matter changes the stabilizer, and its phase is undefined.

## 11. Anchors change the allowed group

An anchor is a site whose frame is held fixed relative to an external reference. For an anchor set $A$, declare

$$
\mathcal G_A=\{g\in U(1)^N:g_a=1\text{ for every }a\in A\}.
$$

This is a different equivalence relation, not merely a different convenient root. A path product beginning and ending at anchors is invariant even if the path is open, because its endpoint transformation factors are one.

The experiment uses anchors only to specify this subgroup. It does not identify an arbitrary boundary condition with an anchor, introduce boundary charges, or implement a canonical boundary generator. Those statements require a further model. In particular, imposing $\psi_a=0$ does not fix its frame: every unit phase preserves zero. An anchor is an additional restriction on allowed transformations.

## 12. When a full tree can still be fixed

The root-normalized phases $h$ are allowed under anchors only if every $h_a=1$. A more general tree-fixing transformation is $\lambda h_i$, with one common $\lambda$ per component. It is allowed exactly when all anchor values $h_a$ within each component are equal; then choose $\lambda=h_a^*$.

A single anchor always permits this adjustment, even when it is away from the computational root. Multiple anchors can obstruct it. On a chain with links $i$ and $1$, the endpoint-to-endpoint product is $i$. Holding both endpoint frames fixed prevents setting both links to one, because that would change the invariant product to one.

If the chain links are $i$ and $-i$, the endpoint product is one. Full-tree fixing is then compatible with the two endpoint anchors. This is a condition on the actual connection, not just on the number of anchors.

## 13. An anchor-compatible partial forest

To construct a representative for every anchored orbit, keep a maximal subset of the declared tree edges that never connects two anchors. The implemented deterministic rule scans the supplied forest order and skips an edge that would merge two already anchored subtrees. Each resulting subtree in an anchored original component has exactly one anchor and uses it as its root. An original component without anchors keeps its declared root.

Let $C_0$ be the number of original components without anchors and let $|A|$ count anchored sites. The retained forest has

$$
N-|A|-C_0
$$

edges. Its fixing phases equal one at every anchor. The remaining link coordinates number

$$
E-N+|A|+C_0
=\beta_1+|A|-C_A,
$$

where $C_A=C-C_0$ counts components containing anchors. A remaining link represents either a closed product or an open path between anchor roots. Both are invariant under the declared subgroup.

## 14. Work through the anchored triangle

Take the earlier pair

$$
U=(1,1,1),\quad\psi=(1,1,1),\qquad
V=(-i,-i,-1),\quad\phi=(1,i,-1).
$$

With tree edges $0\to1$ and $1\to2$, root zero gives $h^L=(1,1,1)$ and $h^R=(1,-i,-1)$. Both full fixed connections and both full fixed matter vectors are identical. Thus the pair lies on one full joint orbit.

Now anchor sites zero and one. Scanning the same tree skips edge $0\to1$ and retains $1\to2$. The right fixing phases become $(1,1,-i)$. Its two remaining coordinates are $Q_{01}=-i$ and $Q_{20}=i$, whereas the left coordinates are both one. Their product still gives the same closed triangle product, but the individual anchor-to-anchor observations differ. No allowed anchored transformation relates the connections.

If only site one is anchored, the connections are still equivalent, but the joint configurations are not: after allowed fixing, the right matter vector is $(i,i,i)$ and the left is $(1,1,1)$. The common matter phase is now relative to a fixed reference.

## 15. Complete anchored joint comparison

Use the same anchor-compatible forest on both configurations. Equality of every remaining link coordinate is necessary and sufficient for anchored connection equivalence. The witness $(h_i^R)^*h_i^L$ automatically equals one at the anchors.

In a component containing an anchor, no residual common phase remains, so compare the fixed matter values themselves. In a component without anchors, use the earlier Gram criterion and permit one common phase. This combination is complete for the declared finite U(1) joint model.

The anchored gauge group has dimension $N-|A|$. Connections retain a stabilizer of dimension $C_0$. Joint configurations retain only the zero-matter components that also have no anchor. Again, these counts concern gauge orbits; they are not a substitute for Gauss constraints or phase-space reduction.

## 16. Predict, compare and inspect the witness

The prepared examples include the same-orbit triangle, one or two anchors, an anchor away from the root, opposite fluxes, magnitude and relative-phase mismatches, zero matter, open chains, disconnected components, self-loops, parallel edges, isolated sites, alternate tree choices and rational data.

For each pair, predict four separate decisions: connection equivalence and joint equivalence, each under the full group and under anchors. Then inspect the fixing paths, remaining coordinates and actual witness. A connection witness may have nonzero matter residuals; it was not claiming joint equivalence. A full-group witness may violate anchors; only an anchored witness must satisfy them.

Exact fractions are the numerical authority. The current manuscript and numerical foundation are prepared; response validation, Workers, teaching visualizations and reader registration follow separately.

## 17. Research lessons and explained questions

For an AI model, gauge-dependent coordinates can create non-identifiability: several parameter settings describe the same observations. An invariant representation must state whether it retains enough information for the task. Complete connection invariants need not be complete joint invariants, and adding external references changes which variations are redundant. A phase chosen from a zero or tiny matter pivot is undefined or poorly conditioned; exact arithmetic does not remove that modeling issue.

**Does tree fixing remove all U(1) redundancy?** No. A common phase per unanchored component survives and can rotate matter.

**Do equal loop products and equal densities prove joint equivalence?** No. Relative matter phases appear in off-diagonal tree-dressed Gram entries.

**Can a loop-free graph retain anchored connection information?** Yes. An open path between fixed endpoint frames can be invariant and nontrivial.

**Does root-normalized fixing fail whenever the root is not an anchor?** It may fail the anchor condition as written, while a common rescaling still gives an allowed full-tree fixing. Check all anchor transports.

**Why is a witness important?** It proves sufficiency by directly mapping all specified variables. A scalar score or an unverified coordinate count cannot do that.

**Does a zero Dirichlet matter value anchor the frame?** No. Every phase preserves zero; the condition $g_a=1$ must be justified separately by the declared external reference.

**What if all matter vanishes?** Connection information remains; joint stabilizers grow on unanchored components. Never divide by a zero pivot.

## 18. Sources and limits

The group-valued link framework is reviewed in David Tong's [Gauge Theory, Chapter 4, §4.2.1](https://davidtong.org/pdfs/teaching/gauge-theory/gauge4.pdf). Burbano and Bauer's [graph gauge-theory paper, §III.1 and Appendix C.3.2](https://arxiv.org/html/2409.13812v2) describes maximal-tree constructions and remaining root transformations in an SU(2) setting.

The disconnected U(1) multigraph, independently specified matter pairs, anchor subgroup, partial-forest rule and completeness arguments above are a separately derived teaching model. We do not transfer these scalar classification rules to non-Abelian traces, implement a quantum Hilbert space, specify dynamics or claim a continuum limit. The next unit introduces canonical electric variables and Gauss constraints before assessing the full symmetry-and-gauge topic.
