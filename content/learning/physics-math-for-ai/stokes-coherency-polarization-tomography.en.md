## A measured beam need not have one deterministic ellipse

A detector may average light prepared in several polarization states. Its measurements can be stable even though no single Jones vector describes every realization. Drawing one deterministic ellipse for such a beam can therefore convey a false physical picture.

We will average quadratic observations, organize them in a coherency matrix and reconstruct Stokes parameters from analyzer intensities. The aim is to distinguish a pure state, a partially polarized mixture, unpolarized light and no light. We will also separate the validity of one predicted state from the stronger conditions needed for a physically constructed optical transformation.

## Ensemble assumptions and normalized observations

Use the same right-handed transverse basis and $e^{-i\omega t}$ convention as in the Jones lesson. A finite ensemble contains vectors $\mathbf E_r$ with weights $w_r\ge0$ and positive total weight. Normalize $p_r=w_r/\sum_sw_s$.

This is an explicit preparation distribution or sampling model for a detector average. It does not specify every microscopic temporal fluctuation, spatial correlation or photon statistic. The same algebra also appears in other statistical theories, but a classical second-moment calculation alone does not establish a quantum-light model.

Normalized intensity is an ensemble average of squared complex field magnitudes. Converting it to SI irradiance still requires medium calibration and the real-field cycle factor. A zero-intensity observation and positive-intensity unpolarized light must remain distinct throughout normalization and inference.

## Build the coherency matrix before taking observations

Define

$$
C=\sum_rp_r\mathbf E_r\mathbf E_r^\dagger
=\begin{pmatrix}C_{xx}&C_{xy}\\C_{xy}^*&C_{yy}\end{pmatrix}.
$$

The dagger denotes conjugate transpose. For any analyzer vector $\mathbf a$,

$$
\mathbf a^\dagger C\mathbf a
=\sum_rp_r|\mathbf a^\dagger\mathbf E_r|^2\ge0.
$$

Thus $C$ is Hermitian and positive semidefinite by construction. Its trace is the total normalized intensity. The positivity follows from actual nonnegative realized observations, not from clipping a fitted matrix afterward.

For equally weighted horizontal fields $(1,0)$ and $(-1,0)$, the mean field vanishes but $C=\operatorname{diag}(1,0)$. Squaring the mean field incorrectly predicts darkness; averaging each realization's intensity gives1. Averaging amplitudes and averaging detector readings are different operations.

## Fix the Stokes signs and physical cone

With the stated convention, define

$$
S=(S_0,S_1,S_2,S_3)
=(C_{xx}+C_{yy},C_{xx}-C_{yy},2\operatorname{Re}C_{xy},-2\operatorname{Im}C_{xy}).
$$

The inverse relation is

$$
C=\frac12\begin{pmatrix}S_0+S_1&S_2-iS_3\\S_2+iS_3&S_0-S_1\end{pmatrix}.
$$

Its eigenvalues are $\lambda_\pm=(S_0\pm R)/2$, where $R=\sqrt{S_1^2+S_2^2+S_3^2}$. Physical data satisfy $S_0\ge0$ and $R\le S_0$. A vector outside this cone cannot be a positive coherency matrix, even if its plotted components appear individually reasonable.

For $(1,i)/\sqrt2$, $C_{xy}=-i/2$, so $S_3=+1$. Writing the matrix and harmonic convention explicitly avoids reversing circular-polarization signs when moving between references or software packages.

## Degree of polarization and the normalized ball

For $S_0>0$, the degree of polarization is

$$
P=\frac{R}{S_0},\qquad
P_L=\frac{\sqrt{S_1^2+S_2^2}}{S_0},\qquad
P_C=\frac{S_3}{S_0}.
$$

$P_C$ is signed under this convention, whereas $P$ is nonnegative. The normalized vector $(S_1,S_2,S_3)/S_0$ lies in the unit ball. A nonzero deterministic Jones state lies on its surface; partially polarized mixtures lie inside. An unpolarized state lies at the center.

If $S_0=0$, all physical Stokes components are zero and the normalized state and degrees are undefined. Assigning $P=0$ would confuse no signal with a valid unpolarized observation. Near a circular or nearly unpolarized state, some orientation angles are also ill-conditioned even though the unnormalized matrix remains meaningful.

## What a polarized-plus-unpolarized decomposition means

For $R>0$, let $\mathbf u$ be the principal unit eigenvector of $C$. Then

$$
C=\frac{S_0-R}{2}I_2+R\mathbf u\mathbf u^\dagger.
$$

The two intensities are $I_{\rm unpol}=S_0-R$ and $I_{\rm pol}=R$. This explains the fraction $P=I_{\rm pol}/S_0$. The principal direction is defined up to global phase when $R>0$; it disappears when the eigenvalues coincide.

This decomposition does not reveal a unique microscopic ensemble. Equal mixtures of horizontal/vertical and diagonal/antidiagonal Jones states both produce $C=I_2/2$. They have the same second-order polarization observations but need not describe identical temporal preparation processes. The effective polarized component is not the single actual electric vector of a partially polarized beam.

## Deterministic optical elements act on coherency matrices

For a deterministic Jones matrix $J$,

$$
C'=JCJ^\dagger.
$$

The formula follows by transforming every realization and then averaging its outer product. A unitary retarder preserves trace and eigenvalues, hence degree of polarization. A lossy projection generally preserves neither total intensity nor eigenvalue ratios.

An ideal horizontal analyzer sends $C=I_2/2$ to $\operatorname{diag}(1/2,0)$. Intensity falls from1 to1/2 while the transmitted degree rises from0 to1. The analyzer selects one component; it has not created power or transformed the entire incident beam into an equally intense pure state. Report unnormalized transmission and the conditional normalized output together.

## Construct depolarization as a physical mixture

One simple isotropic channel is

$$
\mathcal D_d(C)=dC+(1-d)\frac{\operatorname{Tr}C}{2}I_2,
\qquad 0\le d\le1.
$$

It preserves intensity and scales the polarization vector by $d$. It can be constructed as a nonnegative mixture of identity and three Pauli conjugations: identity weight $(1+3d)/4$, and each other weight $(1-d)/4$. This explicitly preserves positivity rather than merely fitting a shorter vector on a sphere.

Placement matters. In the implemented experiment this channel acts before the deterministic optical elements. Depolarizing a horizontal unit-intensity input completely and then filtering horizontally gives intensity1/2 and degree1. Filtering first and depolarizing afterward gives intensity1 and degree0. Depolarization and loss are distinct operations and their order is part of the model.

## From Jones transformations to Mueller matrices

Use the Hermitian basis $(\sigma_0,\sigma_1,\sigma_2,\sigma_3)=(I_2,Z,X,Y)$, so $C=\sum_bS_b\sigma_b/2$. Taking traces after transformation gives

$$
S'_a=\sum_bM_{ab}S_b,\qquad
M_{ab}=\tfrac12\operatorname{Tr}(\sigma_aJ\sigma_bJ^\dagger).
$$

The entries are real even though Jones matrices are complex. A Mueller matrix describes the specified quadratic observations; a global Jones phase is lost. For an ensemble of devices, average their Mueller transformations or their transformed coherency matrices. Averaging the Jones matrices first generally describes a different process.

For the channel above, the Mueller representation is $\operatorname{diag}(1,d,d,d)$. With depolarization first, the combined matrix is the deterministic Mueller matrix multiplied by this diagonal matrix on the right. Matrix order remains physically meaningful.

## Reconstruct Stokes data using six analyzers

Use unit analyzer vectors $H=(1,0)$, $V=(0,1)$, $D=(1,1)/\sqrt2$, $A=(1,-1)/\sqrt2$ and $C_\pm=(1,\pm i)/\sqrt2$. Their intensities are $I_a=\mathbf a^\dagger C\mathbf a$. Then

$$
S_0=I_H+I_V,\quad S_1=I_H-I_V,\quad
S_2=I_D-I_A,\quad S_3=I_{C+}-I_{C-}.
$$

The three complementary pairs should each sum to the same total intensity in this ideal calibrated model. Their disagreement in a real experiment can indicate drift, unequal detector response, noise or an inadequate preparation model.

The computation averages positive analyzer intensities of actual realizations, avoiding spurious negative powers from subtractive cancellation in expanded expressions near darkness. The reconstructed Stokes vector is kept as a measurement result; invalid noisy data should not be silently projected onto the physical cone and presented as an unchanged observation.

## A complete linear scan still misses circular information

For a linear analyzer at angle $\gamma$,

$$
I(\gamma)=\tfrac12[S_0+S_1\cos(2\gamma)+S_2\sin(2\gamma)].
$$

There is no $S_3$ term. Consequently $(1,0,0,0)$, $(1,0,0,1)$ and $(1,0,0,-1)$ all give the same constant1/2, although their degrees are0,1 and1. More angles or lower linear-detector noise cannot supply the missing circular observation.

The contrast of this scan identifies a linear-polarization fraction, not the full degree in general. Circular-sensitive analysis, often implemented through a calibrated wave plate and linear analyzer, supplies the missing component. This is an identifiability limitation of the observation map, not an optimization failure that a larger neural network can automatically overcome.

## Work through a partially polarized preparation

Mix horizontal and vertical unit Jones states with weights3:1. The normalized coherency matrix is $\operatorname{diag}(3/4,1/4)$, giving $S=(1,1/2,0,0)$ and $P=1/2$. Its eigenvalues are3/4 and1/4; no single deterministic Jones vector has this matrix.

Pass this ensemble through a quarter plate with axis $\theta=\pi/4$. Under the fixed convention, horizontal input becomes negative-$S_3$ circular and vertical input positive-$S_3$ circular. The output is $S=(1,0,0,-1/2)$, with the same intensity and degree as before.

The four linear analyzer readings are each1/2, while the positive- and negative-$S_3$ analyzer readings are1/4 and3/4. Linear measurements alone would miss the surviving polarization. A horizontal filter afterward gives intensity1/2 and a fully horizontal transmitted state, illustrating selection and changed normalized degree without increased power.

## Valid states are weaker evidence than physical maps

A fitted real $4\times4$ matrix can send tested Stokes vectors to valid outputs yet fail to represent a nonnegative mixture of deterministic Jones transformations. Even preservation of the entire Stokes cone is a weaker requirement in this model class.

A useful counterexample is transposition of coherency matrices. It maps a positive matrix to another positive matrix and corresponds to $M=\operatorname{diag}(1,1,1,-1)$ in the fixed convention. But its unnormalized Choi matrix

$$
\mathcal C_\Phi=\sum_{i,j}|i\rangle\langle j|\otimes\Phi(|i\rangle\langle j|)
$$

has eigenvalues $(-1,1,1,1)$. A map $C\mapsto\sum_rK_rCK_r^\dagger$ has a positive semidefinite Choi matrix, so transposition cannot have that random-Jones form. This is a statement about a fixed complex-linear Jones-ensemble model, not a claim about every possible active or phase-conjugating device class.

Intensity preservation, passivity and physical realizability are also different constraints. A lossy Jones map need not preserve trace. State-output tests alone do not certify a learned optical device.

## Research design, calibration and AI evaluation

Train a model to predict analyzer readings, coherency matrices or device maps only after defining which information is observed. Positive-cone constraints help avoid invalid states but do not fix missing $S_3$ data, unknown detector gains or an incorrect device model. A matrix that is valid on one input is weaker evidence than one that has a constructive physical channel representation.

Near an unpolarized state, noise in several fitted components can bias a norm-based degree estimate upward. Record uncertainty and calibration assumptions rather than interpreting every small positive estimate as intrinsic polarization. The present exact ensemble model does not simulate those experimental error distributions.

Evaluate unitary degree preservation, filtering with lost power, zero intensity, linear-only ambiguity and noncommuting operation order as distinct controls. Independent real-field integrals, eigenvalue degrees and six-state Mueller reconstruction check separate parts of the calculation. Successful model tests still do not replace actual visual and interaction review of a future reader interface.

## Explained exercises

### Exercise 1 — Mixture or coherent superposition?

Compare an equal incoherent mixture of $H,V$ with the deterministic state $(H+V)/\sqrt2$.

**Solution:** Both have total intensity1. The mixture has $C=I_2/2$, while the superposition has off-diagonal entries1/2 and is pure diagonal polarization. A diagonal analyzer transmits1/2 from the mixture and1 from the coherent superposition. Averaging realized observations cannot be replaced by adding amplitudes.

### Exercise 2 — Check a proposed Stokes vector

Is $(1,0.8,0.8,0)$ a physical state?

**Solution:** No. Its polarization magnitude is $\sqrt{1.28}>1$, giving a negative eigenvalue $(1-\sqrt{1.28})/2$. An appropriate analyzer would have negative predicted intensity. Reject or explicitly model a noisy inference problem rather than silently accepting the vector as physical.

### Exercise 3 — Compute a degree from a preparation

Prepare $H,V$ with weights3:1. Find the degree and polarized fraction.

**Solution:** After weight normalization, $C=\operatorname{diag}(3/4,1/4)$ and $R=1/2$. Degree and polarized intensity fraction are both1/2. A positive-intensity unpolarized contribution remains; this is not one deterministic ellipse of a smaller amplitude.

### Exercise 4 — Identify a missing observation

A perfect linear-analyzer scan is constant1/2. Does it establish that the beam is unpolarized?

**Solution:** No. Pure circular polarization gives the same scan. A circular-sensitive analyzer pair distinguishes $S_3=0$ from $S_3=\pm1$. Infinitely many accurate samples of a measurement insensitive to $S_3$ cannot uniquely determine it.

### Exercise 5 — Increase degree without creating power

What does a horizontal analyzer do to $C=I_2/2$?

**Solution:** It produces $\operatorname{diag}(1/2,0)$. Output intensity is1/2 and degree is1. The transmitted subensemble is pure because the orthogonal component was removed. Reporting only its normalized state would hide the reduced total transmission.

### Exercise 6 — Valid outputs or a valid device?

A learned map flips only $S_3$. It preserves the Stokes cone. Is this sufficient to certify a random-Jones realization?

**Solution:** No. In the fixed basis it represents matrix transposition, whose Choi eigenvalues include−1. Nonnegative Jones mixtures require a positive semidefinite Choi matrix. Constructive device constraints are stronger than checking individually valid outputs.

## Sources and reproducible evidence

[NIST StokesVector](https://pages.nist.gov/SCATMECH/docs/stokesvector.htm) and [conventions](https://pages.nist.gov/SCATMECH/docs/convent.htm) specify the polarization signs used here. [NIST MuellerMatrix](https://pages.nist.gov/SCATMECH/docs/muellermatrix.htm) distinguishes cone-preserving validity from the stronger Jones-mixture physicality criterion. [TU Delft Interactive Optics](https://interactivetextbooks.tudelft.nl/interactive-optics/content/Chap4_Polarisation/Polarization_2022_01Clean.html) provides the optical-element context. The ensemble examples and algebra above are derived explicitly for this lesson.

`stokesPolarizationExperiment` uses positive finite preparations, a constructed Pauli channel and deterministic optical elements. Raw Stokes input validation rejects vectors outside the cone. Floating-point corrections to normalized degree occur only within a stated roundoff tolerance after coherency positivity validation; the unnormalized input matrix is retained. The independent generator integrates real analyzer fields, infers Mueller columns from six physical input states and computes eigenvalue degrees. These references support the model and manuscript, not an unperformed homepage integration or an installed NIST-library verification.
