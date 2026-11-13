## A larger object can keep a large boundary

Imagine making a box larger in two different ways. You can enlarge its length, width and height together, or keep it thin while stretching only its broad face. Both procedures add many sites at a fixed lattice spacing. They do not make the exposed boundary equally unimportant. A thin sheet can retain a large boundary compared with its volume even when its total number of sites is enormous.

An interaction introduces another distinction. A short-range bond crosses a cut only near the joining face. In a completely connected system, every site on one side interacts with every site on the other. Dividing the coupling by the total number of sites can keep an energy scale proportional to size without eliminating those cross interactions.

This chapter separates geometry, energy scaling and cutting. You need sums, products and the identity for the square of a sum. [Campa, Dauxois and Ruffo, section II.A](https://arxiv.org/pdf/0907.0323) provides background on extensivity versus additivity. Our open rectangles, excluded self pairs, exact finite corrections and growth paths are specified below. We calculate deterministic microstate energies; an equilibrium free-energy theorem would require additional work.

## Name the geometry and the quantities held fixed

Join two rectangular lattice blocks along their first coordinate. Their lengths in that direction are a and b. Each of the remaining d−1 directions has width w. The dimension d is one, two or three. Boundary conditions are open: there are no wraparound bonds. A vertical bar in a drawing denotes the cut between blocks, not an extra layer of sites.

Keep the lattice spacing a_lat fixed. A site occupies the same reference volume throughout the comparison, so physical volume is proportional to site count. Holding density fixed does not require every linear dimension to grow. That missing condition is precisely what the thin-sheet comparison will expose.

Each block is uniformly filled with a fixed classical spin sA or sB, each equal to +1 or −1. These are selected configurations, not thermally sampled averages. J is nonnegative and has a chosen common energy unit E0. The calculator represents J/E0 by an integer numerator and positive integer denominator.

| Symbol | Meaning | Convention |
| --- | --- | --- |
| a, b | Longitudinal lengths of the two blocks | Positive site counts |
| w | Each transverse width | One in dimension one |
| NA, NB, N | Block and total site counts | N=NA+NB |
| BA, BB | Internal nearest-neighbor bonds | Each unordered bond once |
| C | Bonds or pairs crossing the cut | Depends on the interaction graph |
| sA, sB | Uniform spins inside each block | Selected values ±1 |
| J | Coupling strength | Energy unit E0 |
| λ | Spatial growth factor | Not a time step |

## Count the sites and internal bonds

The geometry gives the site counts directly:

$$
N_A=aw^{d-1},\qquad N_B=bw^{d-1},\qquad N=(a+b)w^{d-1}.
$$

Within block A, a−1 longitudinal links occur for each transverse position. For each transverse axis, w−1 links occur along that axis while all other coordinates are fixed. Adding these directions gives the internal bond count. The same reasoning applies to block B:

$$
B_A=(a-1)w^{d-1}+(d-1)a(w-1)w^{d-2},\qquad
B_B=(b-1)w^{d-1}+(d-1)b(w-1)w^{d-2}.
$$

For d=1 the transverse term is absent and w=1. Do not try to interpret a negative exponent from an unused transverse term as an extra geometric factor.

Here is a two-dimensional preparation with a=b=2 and w=3:

```text
+ + | - -
+ + | - -
+ + | - -
```

There are six sites in each block. In the left block, the three rows contain three horizontal bonds, and its two columns contain four vertical bonds. Thus BA=7, and BB is also seven. One nearest-neighbor bond crosses the bar in each row. Drawing all twelve vertices first prevents counting the interface as a new column or silently adding periodic bonds.

## Separate the joining face from the exposed surface

The interface contains one nearest-neighbor bond at every transverse position. The outside has two end faces and two faces for each transverse direction:

$$
C_{AB}^{\rm nn}=w^{d-1},\qquad
F_{\rm out}=2w^{d-1}+2(d-1)(a+b)w^{d-2},\qquad
\frac{F_{\rm out}}N=\frac2{a+b}+\frac{2(d-1)}w.
$$

Again, the second contribution is absent in dimension one. External faces and cross bonds answer different questions. The interface is internal to the joined object and is not counted in Fout. Corners do not create extra bonds in either formula.

For the twelve-site example the interface has three bonds, while the exposed face count is fourteen. Therefore interface per site is1/4 and exposed faces per site is7/6. A face count per site can exceed one; it is not a probability. These geometric quantities carry no temperature, entropy or sampling interpretation.

## Choose a sequence of larger systems

A thermodynamic-limit question must say how the finite systems change. In uniform dilation, the base lengths and width are fixed while the spatial factor λ grows:

$$
a=\lambda a_0,\qquad b=\lambda b_0,\qquad w=\begin{cases}1,&d=1,\\\lambda w_0,&d\ge2,\end{cases}\qquad
N=\lambda^d(a_0+b_0)w_0^{d-1}.
$$

For d=1 only the longitudinal lengths grow; the unused width remains one. In this family, both interface per site and exposed faces per site decrease as1/λ. For these rectangular lattices, this is the boundary-negligibility captured by the van Hove geometry label.

The fixed-thickness path is different:

$$
a=a_0,\qquad b=b_0,\qquad w=\lambda w_0,\qquad
N=\lambda^{d-1}(a_0+b_0)w_0^{d-1},\qquad d\ge2.
$$

The interface per site remains1/(a0+b0), and exposed faces per site approach2/(a0+b0). Density is still fixed and N still diverges. This does not make the path a boundary-negligible bulk limit. It also does not imply that every energy density on the path fails to converge. We will obtain a finite, thickness-dependent limit explicitly.

## Fix the microstate before comparing energies

Nearest-neighbor interactions connect only adjacent lattice sites. Internal spins have product one, whereas a cross bond has product s=sA sB. With each unordered bond counted once:

$$
H_{\rm nn}=-J(B_A+B_B+s_As_BC_{AB}^{\rm nn}),\qquad
H_A=-JB_A,\quad H_B=-JB_B.
$$

J favors alignment, so a parallel bond contributes −J and an antiparallel bond contributes +J. Reversing every spin changes the total spin's sign but leaves every bond product, and hence every energy, unchanged.

The model has no bath, temperature or trajectory. Comparing Hamiltonians after deleting a set of bonds is an energy bookkeeping operation at a fixed configuration. It is not automatically a measured heat transfer, a reversible work protocol or a surface free energy. In particular, a domain-wall energy at a selected spin configuration is not the finite-temperature free energy of an interface.

## A local cut has an explicit size

Deleting the cross bonds while keeping the same J inside each block gives the exact cut difference:

$$
\Delta H_{\rm nn}=H_{\rm nn}-H_A-H_B=-JsC_{AB}^{\rm nn},\qquad
\frac{\Delta H_{\rm nn}}N=-\frac{Js}{a+b},\quad s=s_As_B.
$$

Here the ledger difference is joined energy minus restricted energy. The energy change after deleting bonds at the same configuration has the opposite sign. A cut removes favorable bonds for parallel blocks and unfavorable bonds for antiparallel blocks. The magnitude per site, J/(a+b), is often the more useful quantity for deciding whether this interaction becomes negligible at a macroscopic split.

Under uniform dilation its magnitude vanishes. Under fixed-thickness growth it remains J/(a0+b0) when J>0. Both statements follow from the exact finite identity; neither requires fitting a power law to a few plotted points. The corresponding zero-coupling statement is also exact: J=0 gives no energetic cut cost even when the geometric surface fraction remains nonzero.

## Count all-to-all pairs without self interactions

Now change the interaction graph. Every unordered pair of distinct sites interacts, including pairs far from the joining face. Internal and cross pair counts are:

$$
P_A=\frac{N_A(N_A-1)}2,\qquad
P_B=\frac{N_B(N_B-1)}2,\qquad
C_{AB}^{\rm all}=N_AN_B.
$$

The factor one half removes double counting within a block. A cross pair already has a unique endpoint in each block, so it is not divided by two. If each block contains one site, PA=PB=0 but the joined system still has one cross pair.

Let S be the sum of all N spins. Expanding its square includes N self terms, each equal to one. Removing those terms gives an independent way to check the pair-energy calculation:

$$
S=s_AN_A+s_BN_B,\qquad
\sum_{i<j}\sigma_i\sigma_j=\frac{S^2-N}2,\qquad
H_{\rm raw}=-\frac J2(S^2-N).
$$

The subtraction of N is not optional in this finite model. Including self terms lowers the raw Hamiltonian by JN/2 and the global Kac Hamiltonian by J/2. At fixed N these are configuration-independent shifts, but their size dependence differs. Keep that convention fixed when comparing references and when cutting blocks.

## A linear energy scale is not a thermodynamic theorem

For the complete Kac graph, every pair has coupling J/N rather than J:

$$
H_{\rm Kac}=-\frac JN\sum_{i<j}\sigma_i\sigma_j=\frac{H_{\rm raw}}N.
$$

An informative energy scale is the largest absolute energy over all spin configurations on the fixed graph. It is not restricted to the two uniform-block configurations currently selected for the illustration. The triangle inequality bounds it by the sum of bond magnitudes, and the all-aligned configuration attains that bound:

$$
\|H_{\rm nn}\|_\infty=J(B_A+B_B+C_{AB}^{\rm nn}),\qquad
\|H_{\rm raw}\|_\infty=\frac{JN(N-1)}2,\qquad
\|H_{\rm Kac}\|_\infty=\frac{J(N-1)}2.
$$

For J>0 the nearest-neighbor and Kac energy norms grow linearly in N along the admitted paths. The unscaled complete-graph norm grows quadratically. These statements distinguish an extensive energy scale from a superextensive one. At J=0 every norm is exactly zero.

A controlled energy norm is useful, but it does not itself establish convergence of an equilibrium free-energy density, independence of boundary conditions or equivalence of statistical ensembles. Those questions require a specified statistical preparation and their own hypotheses. This chapter does not replace such proofs with an energy plot.

## Kac normalization leaves a macroscopic cross term

Restrict the Kac Hamiltonian to the two blocks while keeping the original denominator N. Its cross energy per site is:

$$
\frac{H_{AB}^{\rm Kac}}N=-Js\frac{N_AN_B}{N^2}
=-Js\frac{a_0b_0}{(a_0+b_0)^2}.
$$

Both growth paths preserve the two block fractions. Consequently, the magnitude above is independent of λ and remains nonzero for J>0. Equal blocks give magnitude J/4. Extensivity of the Kac energy scale has not eliminated the interaction between macroscopic pieces.

This is the operational distinction behind the additivity discussion. You can make the total energy scale linear while retaining a cut interaction of order N. Increasing system size alone does not turn a complete graph into a local graph. Drawing fewer visible lines in a dense diagram would hide pairs, not remove them from the Hamiltonian.

## Renormalizing the pieces is a second operation

A tempting comparison assigns coupling J/NA to block A and J/NB to block B after splitting. For the uniformly aligned spins inside each piece, that produces:

$$
H_A^{\rm sep}=-\frac{J(N_A-1)}2,\qquad
H_B^{\rm sep}=-\frac{J(N_B-1)}2,\qquad
H_A^{\rm sep}+H_B^{\rm sep}=-\frac{J(N-2)}2.
$$

This is a well-defined comparison, but it changes internal couplings as well as removing cross pairs. Let T denote the restricted internal sum minus the separately normalized internal sum:

$$
T=(H_A^{\rm res}+H_B^{\rm res})-(H_A^{\rm sep}+H_B^{\rm sep})
=J\frac{N_AN_B}N-\frac J2.
$$

The full difference includes both this counterterm and the original cross interaction:

$$
H_{\rm Kac}-H_A^{\rm sep}-H_B^{\rm sep}
=T-Js\frac{N_AN_B}N
=J(1-s)\frac{N_AN_B}N-\frac J2.
$$

For parallel blocks it is merely −J/2, even though the original cross interaction per site remains nonzero. The small difference contains a cancellation between two operations. For antiparallel blocks it instead contains the macroscopic term2JNA NB/N. A correct ledger names the normalization used for every internal energy.

## Work through the finite ledger and the growth paths

Take d=2, a=b=2, w=3, J=1E0, sA=+1 and sB=−1. All entries in the following ledger use the original coupling of the corresponding joined model. Energies are in E0; energies per site are in E0 per site.

| Interaction | Restricted A | Restricted B | Cross | Joined energy | Energy per site | Maximum absolute energy |
| --- | --- | --- | --- | --- | --- | --- |
| Nearest neighbor | −7 | −7 | 3 | −11 | −11/12 | 17 |
| Complete, unscaled | −15 | −15 | 36 | 6 | 1/2 | 66 |
| Complete, global J/N | −5/4 | −5/4 | 3 | 1/2 | 1/24 | 11/2 |

Separately normalized blocks have sum −5. Their normalization counterterm is5/2, and adding the Kac cross energy3 gives a full-minus-separate difference11/2. Notice that a positive total energy in one selected configuration is compatible with a ferromagnetic coupling: the many unfavorable cross pairs can outweigh favorable internal ones.

Next grow the same base geometry. The local energy density and cross magnitude expose the difference between paths:

| Growth path | λ | N | Nearest-neighbor H/N | Local cross magnitude per site |
| --- | --- | --- | --- | --- |
| Uniform | 1 | 12 | −11/12 | 1/4 |
| Uniform | 2 | 48 | −35/24 | 1/8 |
| Uniform | 4 | 192 | −83/48 | 1/16 |
| Fixed thickness | 1 | 12 | −11/12 | 1/4 |
| Fixed thickness | 2 | 24 | −13/12 | 1/4 |
| Fixed thickness | 4 | 48 | −7/6 | 1/4 |

These are exact fractions, not fitted extrapolations. A plot should label its horizontal coordinate as spatial growth or site count and retain which path produced each point. Equal λ values do not mean equal N across the two paths.

## Derive finite corrections before taking the limit

Combining the local bond counts gives a formula valid at every admitted finite size:

$$
\frac{H_{\rm nn}}N=-J\left[d-\frac{2-s}{a+b}-\frac{d-1}w\right].
$$

Write L=a0+b0. In the two paths this becomes:

$$
\begin{aligned}
\text{uniform:}\quad &\frac{H_{\rm nn}}N=-Jd+\frac J\lambda\left[\frac{2-s}L+\frac{d-1}{w_0}\right],\\
\text{fixed thickness:}\quad &\frac{H_{\rm nn}}N=-J\left[d-\frac{2-s}L\right]+\frac{J(d-1)}{w_0\lambda}.
\end{aligned}
$$

The uniform limit is −Jd; the fixed-thickness limit retains L and the spin orientation across the interface. A finite limit has not disappeared in the second case. Its dependence on a retained thin direction is exactly what a boundary-independent bulk claim would need to address.

For the complete graph, the spatial mean of the selected configuration stays fixed along either path. It is not a spontaneous magnetization selected from an equilibrium ensemble:

$$
m_{\rm cfg}=\frac{s_Aa_0+s_Bb_0}{a_0+b_0},\qquad
\frac{H_{\rm raw}}N=-\frac{JN m_{\rm cfg}^2}2+\frac J2,\qquad
\frac{H_{\rm Kac}}N=-\frac{Jm_{\rm cfg}^2}2+\frac J{2N}.
$$

Balanced antiparallel blocks have m_cfg=0. Their raw energy per site is exactly J/2 at every size, while the raw norm per site diverges. Their Kac energy per site tends to zero while its cross magnitude per site stays J/4. If m_cfg is nonzero and J>0, the raw energy per site instead tends to negative infinity. This is an analytic path limit, not a floating-point overflow of a finite energy.

## Read the arithmetic and the scope correctly

The calculator accepts integer base lengths, width and growth factor from1 to1,000,000. Its coupling numerator is0 through1,000,000 and its denominator is1 through1,000,000. A one-dimensional width other than one, a fixed-thickness one-dimensional path, fractional lengths and a zero denominator are rejected. These are computational contracts, not bounds on physical system sizes.

Geometry counts are exact decimal strings because products can exceed the integer precision of binary64. Finite energy observations carry equal rational endpoints. Rounded numbers and approximate logarithms are display projections, while analytic infinities have separate signed tags. Never infer exact equality from matching short decimals. The large-number example below is not floating-point underflow: it illustrates integer precision and small displayed corrections within the finite model's range.

The size history contains powers of two through the requested factor and includes the exact requested endpoint. Its last row is the current geometry. There is no random trajectory, thermal relaxation or measured animation period. A visual comparison should keep the active path, graph, coupling normalization and selected configuration visible rather than blending their meanings into one smooth curve.

## Connect growth paths and cuts to the experiment

Keep the default twelve-site preparation with opposite block spins. Three nearest-neighbor bonds cross the interface, giving cut magnitude per site 3/12=1/4. Distinguish enlarging a sheet in every direction from spreading it sideways at fixed thickness. A large N alone does not tell you that the boundary became negligible.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Compare ‘Fourfold growth in every direction’ with ‘Fixed thickness, fourfold transverse growth’ | N is192 versus48; local cross magnitude per site is1/16 versus1/4 | Does the same spatial factor four imply the same site count or boundary fraction? |
| Switch the interaction graph between nearest neighbor, unscaled complete and global Kac | Selected-configuration energy, worst energy and cross term | How does controlling the overall energy scale differ from removing interactions across the cut? |
| Read the Kac separate-normalization ledger for parallel spins | Cross energy, normalization counterterm and full-minus-separate difference | Does a small final difference imply that the original cross term was small? |

‘One spatial-size step’ doubles the requested scale, capping the final step at the allowed maximum. ‘Choose spatial growth record’ selects an already computed row without changing the maximum requested scale. Read exact pair counts rather than counting schematic complete-graph lines, and retain the common energy unit E0. Size scanning compares spatial systems at the same preparation; it is not thermal relaxation or a time trajectory.

## Research extension: control a thermal free energy without claiming full independence

The previous ledger fixed a spin configuration. We now add a statistical preparation: all binary spin configurations are allowed, their counting measure is fixed, and the inverse temperature β is positive and finite. This lets us prove a free-energy bound from the cut energy. It also exposes a useful limitation: negligible free-energy change per site does not imply that the full probability laws become close.

### Sum a uniform energy bound over every configuration

Write the joined Hamiltonian as the two restricted Hamiltonians plus their interaction V. Keep the original couplings inside both blocks. Let Δ bound the absolute interaction for every configuration. For C nearest-neighbor cross bonds of magnitude J, Δ=JC is valid.

$$
H(\sigma_A,\sigma_B)=H_A(\sigma_A)+H_B(\sigma_B)+V(\sigma_A,\sigma_B),\qquad |V|\leq\Delta,\qquad Z_0=Z_AZ_B.
$$

Multiplying each uncoupled Boltzmann weight by exp(−βV) changes it by a factor between exp(−βΔ) and exp(βΔ). Positivity lets us sum these bounds before taking a logarithm:

$$
e^{-\beta\Delta}Z_AZ_B\leq Z\leq e^{\beta\Delta}Z_AZ_B,\qquad |\ln Z-\ln Z_A-\ln Z_B|\leq\beta\Delta.
$$

With F=−β⁻¹ln Z, the total free-energy difference is bounded in energy units. For a boundary-negligible growth path, the corresponding difference per site vanishes at fixed J and temperature.

$$
|F-F_A-F_B|\leq\Delta,\qquad \frac{|F-F_A-F_B|}{N}\leq\frac{JC}{N}\longrightarrow0\quad(C/N\to0).
$$

This is a sufficient boundary-control estimate, not a complete existence theorem for every thermodynamic limit. Geometry and interactions still need their hypotheses. For the global Kac graph the analogous interaction bound is J N_A N_B/N, which stays order N at a fixed macroscopic split. The resulting free-energy bound does not vanish per site; that fact alone also does not prove a nonzero actual free-energy difference at every temperature. An upper bound is not an equality.

### Solve one open chain exactly

Take a zero-field open nearest-neighbor Ising chain of N sites with K=βJ≥0, and cut one bond into blocks A,B. Summing the final spin successively gives a factor 2 cosh K for each bond. The initial spin has two choices, so

$$
Z_L=2(2\cosh K)^{L-1},\qquad \frac{Z_N}{Z_AZ_B}=\cosh K,\qquad \beta(F-F_A-F_B)=-\ln\cosh K.
$$

There is one cross bond, so the generic dimensionless bound is K. The exact magnitude ln cosh K is smaller. At fixed K, its per-site contribution vanishes as 1/N. This full thermal sum is a different calculation from the earlier uniform-block microstate energy.

Let P be the joined Gibbs law and P₀ the product law of the two cut chains. Each boundary spin is fair by spin-flip symmetry. Under P₀ the two boundary spins are independent, giving the exact likelihood ratio

$$
\frac{P(\sigma_A,\sigma_B)}{P_0(\sigma_A,\sigma_B)}=\frac{e^{K\sigma_a\sigma_{a+1}}}{\cosh K},\qquad \mathbb E_P[\sigma_a\sigma_{a+1}]=\tanh K.
$$

### Distinguish a density limit from a global probability metric

Integrating out either block multiplies the other block's original weight by the same constant cosh K. Thus its marginal law is unchanged. Nevertheless the joined blocks remain dependent. Their mutual information equals the relative entropy to P₀, and their total-variation separation is

$$
D_{\mathrm{KL}}(P\Vert P_0)=I_P(\sigma_A;\sigma_B)=K\tanh K-\ln\cosh K,\qquad \|P-P_0\|_{\mathrm{TV}}=\frac12\tanh K.
$$

Both stay strictly positive at fixed K>0 as N grows. To verify the TV expression, group configurations by whether the boundary spins agree. P₀ gives each group probability one-half; P shifts their probabilities to (1±tanh K)/2. More remote internal configurations contribute no extra likelihood-ratio factors.

This does not prevent deep interior observations from decorrelating. For spins at depths d_A,d_B from the two sides of the cut, the unique open-chain path has d_A+d_B+1 bonds. Summing its transition factors gives

$$
\operatorname{Cov}_P(\sigma_{a-d_A},\sigma_{a+1+d_B})=(\tanh K)^{d_A+d_B+1}.
$$

For finite K, this tends to zero as those depths grow, while the interface correlation remains tanh K. Local bulk behavior, mutual information of entire blocks and free-energy density answer different questions. Even the relative entropy per site tends to zero in this example without total variation tending to zero.

The exact open-chain pressure also has a directly verified limit:

$$
\frac{\ln Z_N}{N}=\ln(2\cosh K)-\frac{\ln\cosh K}{N}\longrightarrow\ln(2\cosh K).
$$

![One-bond free-energy error per site shrinking with size while global block dependence remains and deep-interior covariance decays.](/assets/learning/finite-size-research-boundary-free-energy-information.en.svg)

The table uses K=1 and equal blocks. It lists |βΔF|/N, the bound K/N, block mutual information in nats, TV, and the covariance between two interior sites with d_A=d_B=N/4−1. The final distance is N/2−1 bonds. These sizes are a separate Python example, not extra temperature controls in the energy-only reader.

| N | abs(βΔF)/N | Bound K/N | Block MI | TV | Bulk covariance |
| --- | --- | --- | --- | --- | --- |
| 8 | 5.422260e-02 | 1.250000e-01 | 0.327813 | 0.380797 | 4.417442e-01 |
| 32 | 1.355565e-02 | 3.125000e-02 | 0.327813 | 0.380797 | 1.682109e-02 |
| 128 | 3.388913e-03 | 7.812500e-03 | 0.327813 | 0.380797 | 3.536608e-08 |
| 512 | 8.472282e-04 | 1.953125e-03 | 0.327813 | 0.380797 | 6.910647e-31 |

```python
from math import tanh, log, cosh

k = 1.0  # Dimensionless beta*J in the separate thermal extension.
t = tanh(k)
for n in (8,32,128,512):
    free_energy_density = log(cosh(k))/n
    bound = k/n
    mutual_information = k*t-log(cosh(k))
    total_variation = t/2
    distance = n//2-1
    bulk_covariance = t**distance
    print(n, f'{free_energy_density:.6e}', f'{bound:.6e}',
          f'{mutual_information:.6f}', f'{total_variation:.6f}',
          f'{bulk_covariance:.6e}')
```

### Research checks and limits

#### What changed when we introduced a free energy?

**Research explanation.** We specified a positive temperature and summed over every allowed binary configuration. The fixed selected configurations in the original energy ledger do not supply that ensemble by themselves. The new bound needs a uniform cut-energy bound over the whole summed state space.

#### Does a small free-energy density difference prove independence of blocks?

**Research explanation.** No. At fixed K>0 the open-chain example has vanishing free-energy change per site but constant positive TV and mutual information. A macroscopic block includes its boundary spin even when most of its sites lie far away.

#### Does a nonvanishing upper bound prove a nonvanishing actual difference?

**Research explanation.** No. The bound can be loose. A Kac interaction of order N prevents this particular vanishing-bound argument; deciding the actual thermal free energy requires an additional calculation, not reversal of the inequality.

[Tong's chain treatment](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf), section 5.3.1, supplies background on thermal spin sums. The open-boundary product, cut inequalities and probability-metric comparison above are derived here. No general ensemble-equivalence or arbitrary-shape limit theorem is being substituted for these explicit statements.

## Six explained exercises

### 1. Reconstruct the twelve-site ledger

For the basic antiparallel preparation, count internal and cross nearest-neighbor bonds, then reconstruct the Kac energy using its restricted pieces.

**Explanation.** Each block has three longitudinal bonds and four transverse bonds, so BA=BB=7. There are three cross bonds and the local energy is−7−7+3=−11E0. The complete internal counts are fifteen each and the cross count is thirty-six. Global normalization by N=12 gives restricted energies−5/4 each and cross energy3, hence joined Kac energy1/2E0. All three pieces use the same denominator. Counting the interface as three new sites would corrupt both N and the normalization.

### 2. Increase size along two paths

At λ=2, compare N and the local cut energy per site for uniform dilation and fixed-thickness growth. Does the existence of both finite energy-density limits make the paths equivalent?

**Explanation.** Uniform growth produces a=b=4,w=6,N=48 and cross magnitude6/48=1/8E0 per site. Fixed thickness retains a=b=2 with w=6,N=24 and cross magnitude6/24=1/4. Their local energy densities are−35/24 and−13/12 respectively. Taking λ to infinity yields−2 and−5/4E0 per site. Both limits exist, but they correspond to different geometric sequences. Existence alone does not make the surface negligible or the limiting values identical.

### 3. Explain the small parallel-block difference

Use the basic geometry but set sB=+1. The full Kac energy minus the two separately normalized energies is−1/2E0. Does that establish a negligible original interaction between the blocks?

**Explanation.** No. The original cross energy is−3E0 and its magnitude per site is1/4. The restricted internal sum is−5/2, whereas the separately normalized sum is−5. Their counterterm is+5/2, which cancels most of−3 and leaves−1/2. The joined energy is−11/2. The small final difference describes a cut plus a change of internal couplings, not the original cut alone. Under uniform enlargement the same cancellation persists while the cross magnitude per site stays1/4.

### 4. Test a misleading finite configuration

Balanced antiparallel blocks have raw H/N=1/2E0. Now take the same base width but a0=2,b0=3. Determine raw H/N at λ=1 and λ=2, and distinguish this selected path from the raw energy norm.

**Explanation.** The spatial mean is−1/5 and N=15λ². The identity gives Hraw/N=1/2−N/50. At λ=1 it is1/5; at λ=2 it is−7/10E0 per site. A positive finite value therefore does not imply a positive or finite limit. The selected unequal-block path tends to negative infinity. The balanced path instead has exact cancellation and stays at1/2, yet in both cases the worst absolute energy over all configurations is N(N−1)/2E0. The norm's quadratic growth is a property of the Hamiltonian, not of the displayed configuration alone.

### 5. Keep self-pair conventions and zero coupling explicit

For d=1,a=b=w=1 with antiparallel spins, find the raw and Kac energies. Then explain what J=0 changes in a fixed-thickness two-dimensional example.

**Explanation.** Two single-site pieces contain no internal pair. There is one cross pair, giving raw energy J and global Kac energy J/2. Using N−1 or including self pairs would define another finite model. At J=0 every energy, cut cost and worst energy norm is exactly zero. Nevertheless, a fixed-thickness rectangle still has a nonvanishing exposed-face fraction. The energetic classification changes; the geometry does not become a van Hove path merely because coupling is absent.

### 6. Read a huge count and a tiny correction

Choose d=3,a0=b0=w0=10^6,λ=10^6, antiparallel spins and J=10^−6E0. Find N, the local and Kac cross magnitudes per site, and the balanced Kac energy density.

**Explanation.** Each actual length or width is10^12, so N=2×10^36. The local cross magnitude per site is J/(2×10^12)=5×10^−19E0. The Kac cross magnitude is J/4=2.5×10^−7E0 per site. Its balanced joined energy density is J/(2N)=2.5×10^−43E0 per site. The latter is small but nonzero and remains within the binary64 exponent range. Store N exactly as an integer string and read the rational energy endpoints. A short decimal display can hide the correction; it cannot turn it into an exact zero or prove that the macroscopic cross interaction vanished.

## Use a visual comparison to ask a research question

Begin a visual experiment with the twelve-site grid and ask the learner to count the three cross bonds before revealing the energy ledger. Next keep the block colors and spin orientation fixed while switching between local and complete graphs. For the complete graph, show the exact pair count rather than suggesting that a few drawn lines represent all interactions. Plot cut magnitude per site and worst energy per site separately; include both growth paths and the Kac normalization label.

The same distinction is useful when discussing normalized dense interactions in AI: controlling an aggregate scale does not by itself remove cross-block influence. This energy calculation is not a theorem about attention, a training algorithm, a sampler or computational speed. A research claim needs its own variables, assumptions and notion of a limit.

To turn the present energetic comparison into equilibrium statistical mechanics, specify the full allowed configuration space, its measure and temperature, define a partition function, then justify the required convergence and boundary control. If derivatives or responses are needed, prove the relevant interchange of operations rather than differentiating an attractive limiting curve without a condition. The next families examine fluctuation concentration and finite partition-function rounding; the current reader lets you manipulate the growth path and inspect the exact ledger. Continue to those families through Book → Interactive Book.
