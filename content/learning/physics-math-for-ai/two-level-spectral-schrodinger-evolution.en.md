## 1. A state can move without being measured

The previous chapters associated a preparation with a ray and a measurement with projectors. Now we introduce a physical clock. We ask what happens between preparation and a later measurement when an isolated system evolves under a fixed Hamiltonian. Reading a plot at several times means comparing separately prepared experiments; it does not insert repeated measurements into one trajectory.

Our state lives in C². Two coordinates can label internal alternatives, such as two levels, rather than two locations in ordinary space. The raw nonzero vector z specifies the initial normalized representative ψ(0)=z/||z||. A Hermitian matrix H specifies the generator of time evolution. Its eigenvalues have energy units. The matrix is not a transition-probability table, and its entries need not be positive.

## 2. From a differential equation to a propagator

The Schrödinger equation relates a state to its time derivative. A propagator U maps the initial vector to the vector at a later time. For a constant finite Hermitian H, the matrix exponential solves that initial-value problem.

$$
i\hbar\frac{d\psi}{dt}=H\psi,\qquad
\psi(t)=U(t)\psi(0),\qquad U(t)=\exp(-itH/\hbar).
$$

The minus sign follows by dividing by iℏ. To check the solution, differentiate its exponential series: H commutes with every power of itself, so the derivative is (−iH/ℏ)U. At t=0 the exponential is I. This is a linear evolution of amplitudes. It is not the gradient flow dψ/dt=−Hψ used to reduce a quadratic objective.

Think of an ordinary real rotation: a point changes direction while retaining length. Quantum evolution generalizes that geometry to a complex inner-product space. Probabilities can change in a chosen measurement basis even while the vector length remains fixed.

## 3. Why norm and inner products are preserved

Hermiticity gives U†=exp(+itH/ℏ). Multiplying the exponentials is justified here because their generators commute. This gives unitarity and hence preservation of every inner product, not only the initial norm.

$$
U(t)^\dagger U(t)=I,\qquad
\langle U(t)\phi,U(t)\psi\rangle=\langle\phi,\psi\rangle,\qquad
U(s)U(t)=U(s+t),\quad U(-t)=U(t)^\dagger.
$$

The addition law assumes the same constant H throughout. It cannot be transferred without qualification to intervals with different Hamiltonians. Norm preservation is a mathematical prediction to inspect numerically. Dividing a computed output by its current norm would hide whether the propagation actually preserved it.

## 4. An energy basis turns evolution into phase rotation

Construct a rank-one projector P from a nonzero raw axis v, put Q=I−P and select real center a and signed gap b. The resulting family has an explicit spectral decomposition and an explicit propagator.

$$
P=\frac{vv^\dagger}{v^\dagger v},\quad Q=I-P,\quad K=2P-I,\quad
H=aI+bK=(a+b)P+(a-b)Q,
$$

$$
U(t)=e^{-i(a+b)t/\hbar}P+e^{-i(a-b)t/\hbar}Q
=e^{-iat/\hbar}\bigl(\cos\theta\,I-i\sin\theta\,K\bigr),\qquad \theta=bt/\hbar.
$$

Since P and Q are orthogonal, the two projected parts keep their lengths while rotating in complex phase. A negative b exchanges which energy is lower; it does not exchange the attached projectors. When b=0 the Hamiltonian is scalar and has one distinct energy. We do not display two different measurement outcomes with identical labels.

A linked visualization should distinguish the two complex phase arrows from the two probability bars. Arrows rotate; energy-measurement bars remain fixed. That distinction is the central intuition of spectral propagation.

## 5. Put units back before comparing speeds

The exponent must be dimensionless: energy times time divided by action. The calculation declares an energy unit E0 and a time unit T0. If H=E0 H̃, t=T0 t̃ and ℏ=E0T0 ℏ̃, the numerical equation uses H̃, t̃ and ℏ̃. Setting the numerical ℏ̃ to 1 chooses compatible units; it does not say that the physical constant has no units.

With b=E0 and ℏ=E0T0, a duration T0 gives θ=1 radian. Halving ℏ while keeping those declared numerical energy and time values fixed doubles θ. It is a comparison between model parameters, not an experimental knob that changes nature's constant. Changing units consistently leaves all dimensionless phases and predicted probabilities unchanged.

## 6. A common energy shift and a relative phase differ

Replacing H by H+cI multiplies every component of the evolved state by exp(−ict/ℏ). Born probabilities and the ray are unchanged. By contrast, two energy components with different energies accumulate a relative phase −2bt/ℏ. That phase can change measurements which mix the energy eigenspaces.

For H=Z=diag(1,−1), ℏ=1 and initial +X=(1,1)/√2, the following formulas make the distinction concrete.

$$
\psi(t)=\frac{1}{\sqrt2}\begin{pmatrix}e^{-it}\\e^{it}\end{pmatrix},\qquad
p_Z(+1)=p_Z(-1)=\tfrac12,\qquad
|\langle\psi(0),\psi(t)\rangle|^2=\cos^2t.
$$

The Z histogram is fixed but the ray generally moves. At t=π/2 the state is orthogonal to the initial +X ray, up to an overall phase. At t=π it returns to that ray with the representative multiplied by −1. The exact π times are analytic landmarks; bounded rational time inputs need not hit them exactly. A rounded fidelity of 1 near a return is not a proof of equality.

## 7. Conserved energy does not freeze every statistic

Because H commutes with U, each energy projector commutes with U. Its probability, the mean energy and the second moment are constant. In this family the variance has a particularly transparent form, with w=||Pψ(0)||².

$$
\langle H\rangle=a+b(2w-1),\qquad
\langle H^2\rangle-\langle H\rangle^2=4b^2w(1-w).
$$

For b nonzero, zero variance means the initial state lies in one energy eigenspace, so only a common phase changes. The ray is stationary. If b=0 every ray is stationary. A nonzero variance in the two-level family means relative phase evolution is possible; a selected measurement can still be insensitive to it.

| Initial state and H, with ℏ=1 | Energy statistics | Evolution of the ray |
| --- | --- | --- |
| First basis state and Z | Mean 1, variance 0 | Stationary; representative gains exp(−it) |
| +X and Z | Mean 0, variance 1 | Generally moves; Z probabilities remain 1/2 |
| +X and Z+3I | Mean 3, variance 1 | Same ray trajectory as under Z |
| Any normalized state and 2I | Mean 2, variance 0 | Stationary for every preparation |

## 8. Forward Euler is not a unitary step

Applying the simplest ODE update gives one step ψₖ₊₁=(I−iΔtH/ℏ)ψₖ. On an energy eigencomponent E its multiplier is 1−iEΔt/ℏ. Its modulus exceeds 1 whenever EΔt is nonzero. After n steps on one fixed grid, the squared norm follows an exact algebraic identity.

$$
\|\psi_n^{\mathrm{Euler}}\|^2
=\sum_j w_j\left(1+\frac{E_j^2\Delta t^2}{\hbar^2}\right)^n,\qquad
\Delta t=T/n.
$$

For H=Z, initial +X, ℏ=1 and T=1, one step gives squared norm 2. Two steps give 25/16. Four steps give (17/16)⁴. These outputs approach the exact norm 1 as the fixed-duration grid is refined, but each finite nonzero step expands the norm. With two steps the vector is (3/4−i,3/4+i)/√2; normalizing it afterward would change the experiment being diagnosed.

Euler growth is a discretization error, not physical gain, dissipation or a new Born distribution. Even a common energy shift can change Euler's numerical error while leaving the exact ray dynamics unchanged. Squared unnormalized components are not a probability distribution when their sum is greater than 1.

## 9. A guided reading of the registered experiment

First choose +X under Z and change the final physical time. Predict which bars stay constant before comparing amplitudes and fidelity. Then choose the first energy eigenstate: the complex representative moves, but the ray and every measurement distribution remain fixed. Finally add a scalar center and check which quantities shift and which remain unchanged.

For the numerical comparison, hold H and final time fixed while changing only the Euler step count. The sampling grid merely selects times to display; it is not an extra sequence of measurements. The Euler count must be divisible by the sampling count so every displayed Euler value belongs to the same integration grid. Comparing different step sizes at different final times would confound two effects.

The Interactive Book screen labels time, energy, phase, norm and probability separately. Motion is a way to inspect computed samples, not evidence that a state follows a classical path through two spatial dots. Pause and single-step controls leave the same numerical result available without animation.

## 10. What the arithmetic can certify

The input uses eighteen bounded integer fields for raw coordinates, rational parameters and counts. Hamiltonians, energy weights and moments, physical time points and the Euler recurrence are calculated with exact complex rationals. The evolving exponential generally is not rational. Its amplitudes and probabilities use directed 768-bit work arithmetic and outward 512-bit reported intervals.

Trigonometric enclosures use reduced-angle Taylor series with an explicit remainder and repeated doubling. A Number drawing coordinate is not the enclosure itself. A residual interval containing zero supports consistency with an identity; it is not the same as a structurally exact zero. A tiny unresolved interval can retain null as its public value. No post-evolution norm correction or invented epsilon turns that uncertainty into certainty.

Independent symbolic spectral data, high-precision general matrix exponentials and separately integrated ODEs check this controlled family. Numerical ODE agreement is a comparison with a stated tolerance, not a theorem about its global integration error. Large Euler growth remains a magnitude or exact fraction rather than being relabelled a probability.

## Research extension: ray speed, orthogonality and actual repeated measurements

### The energy spread sets a geometric speed

A rotating representative is not enough to measure physical change: an energy shift changes its overall phase. Apply the perpendicular projection used in the ray chapter to the Schrödinger derivative:

$$
v=(I-|\psi\rangle\langle\psi|)\dot\psi=-\frac i\hbar(H-\langle H\rangle)\psi,
 \qquad \|v\|^2=\frac{\sigma_H^2}{\hbar^2}.
$$

Here σ_H is the energy standard deviation, not the mean energy. It has energy units, so σ_H/ℏ has inverse-time units. The neighboring-fidelity expansion is:

$$
F(t,t+dt)=1-\frac{\sigma_H^2(t)}{\hbar^2}dt^2+o(dt^2),\qquad
 \sigma_{H+cI}=\sigma_H.
$$

For differentiable finite evolution the ray angle from the initial preparation cannot exceed the path length:

$$
d(t)=\arccos|\langle\psi(0)|\psi(t)\rangle|\le
 \int_0^t\frac{\sigma_H(s)}\hbar\,ds,\qquad 0\le d(t)\le\pi/2.
$$

One way to derive the inequality is to choose the instantaneous representative so its overlap with the initial state is real and nonnegative. Decompose the initial state into its component along the present state and an orthogonal unit direction. Cauchy–Schwarz bounds the derivative of that overlap by the perpendicular speed times the square root of one minus the squared overlap. Differentiating arccos cancels that square root. Integrate, using continuity at the endpoints where that intermediate expression is singular.

Our angle convention is arccos of the overlap modulus, with orthogonal rays at π/2. The distance and speed in [Anandan and Aharonov's original geometric treatment](https://doi.org/10.1103/PhysRevLett.65.1697) are twice this normalization. A factor-of-two convention must change both together; it cannot change the physical time bound.

### A lower time bound need not be attainable

For a constant Hamiltonian, σ_H is conserved. Because cos² decreases on [0,π/2], the path-length result gives:

$$
F(0,t)\ge\cos^2(\sigma_Ht/\hbar)\quad
 (0\le\sigma_Ht/\hbar\le\pi/2),\qquad
 t_\perp\ge\frac{\pi\hbar}{2\sigma_H}\quad(\sigma_H>0).
$$

The fidelity bound is restricted to the stated interval; extending cos² periodically would create false later-time lower bounds. If σ_H=0, the ray is stationary and division by σ_H is unavailable. If σ_H>0, the orthogonality inequality is a necessary condition, not a guarantee that an orthogonal state will ever be reached.

The chapter's exact two-level family shows the difference. With w the initial P weight:

$$
\langle\psi(0)|\psi(t)\rangle=e^{-iat/\hbar}
 \bigl(we^{-ibt/\hbar}+(1-w)e^{ibt/\hbar}\bigr),\qquad w=\|P\psi(0)\|^2.
$$

Taking the squared modulus and using the conserved variance gives:

$$
F(0,t)=1-4w(1-w)\sin^2(bt/\hbar),\quad
 \sigma_H=2|b|\sqrt{w(1-w)},\quad
 F_{\min}=(2w-1)^2\quad(b\ne0).
$$

A balanced preparation follows a shortest arc until its first orthogonal point and saturates the bound. An unbalanced one never becomes orthogonal:

$$
w=\tfrac12,\ b\ne0:\quad t_\perp=\frac{\pi\hbar}{2|b|},\qquad
 w\ne\tfrac12:\quad F(0,t)>0\ \text{for all }t\quad(b\ne0).
$$

For example w=1/4 gives minimum fidelity 1/4 even though its positive energy spread supplies a finite lower bound on a hypothetical orthogonality time. The inequality did not promise a reachable endpoint. Adding aI changes none of these ray statements.

### Display samples do not perform measurements

Now deliberately change the experiment. Start from +X under bZ, fix total time T and perform an ideal yes/no projection onto that same initial ray after each interval T/N. Keep only the history in which every answer is yes. Each successful projection restores the initial ray, so the conditional next-yes probability is the same each time:

$$
P_N=\left[\cos^2\!\left(\frac{bT}{N\hbar}\right)\right]^N\longrightarrow1,\qquad
 \log P_N=-\frac{b^2T^2}{N\hbar^2}+O(N^{-3})\quad(N\to\infty).
$$

The logarithmic expansion holds for fixed finite bT/ℏ and sufficiently large N. It follows from log cos²x=−x²+O(x⁴). This is the ideal projective quantum Zeno limit. It assumes instantaneous, perfectly specified projections; it establishes no finite-bandwidth apparatus design or resource cost.

The probability is for the entire all-yes history. It is not the state conditioned on success, which is already the initial ray, nor the probability distribution after discarding all measurement records. Without these projections the survival is cos²(bT/ℏ), independent of the number of plotted samples. [Preskill §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) supplies the unitary and conditional-projection rules used in this direct calculation.

For b=ℏ=T=1, run this standalone illustration. Values are rounded double-precision results, not the reader's directed interval certificates. No repeated-measurement control has been added to the existing unitary lab.

```python
from math import cos
for n in [1, 2, 4, 16]:
    step = 1/n  # b = hbar = T = 1
    single = cos(step)**2
    all_yes = single**n
    print(n, f"{step:.6f}", f"{single:.6f}",
          f"{all_yes:.6f}", f"{cos(1)**2:.6f}")
```

| N | step | single yes | all yes | unmeasured F |
|---|---|---|---|---|
| 1 | 1.000000 | 0.291927 | 0.291927 | 0.291927 |
| 2 | 0.500000 | 0.770151 | 0.593133 | 0.291927 |
| 4 | 0.250000 | 0.938791 | 0.776741 | 0.291927 |
| 16 | 0.062500 | 0.996099 | 0.939375 | 0.291927 |

![The two-level survival curve is compared with its geometric lower bound; a second panel distinguishes unmeasured survival from the all-yes probability of actual repeated projections.](/assets/learning/quantum-unitary-research-ray-speed-zeno.en.svg)

The second panel changes the measurement protocol, rather than the display resolution. The first panel applies its lower bound only before its allowed angle reaches π/2.

#### Research check 1: a large mean with no motion

An energy eigenstate under H=100I+Z can have mean energy101 and zero standard deviation. Its ray speed is zero. The mean energy alone does not set this geometric speed, and subtracting 100I does not change the ray.

#### Research check 2: a finite bound without orthogonality

Take w=1/4 and |b|=ℏ=1. The spread is √3/2, but minimum fidelity is 1/4. The orthogonality bound is a necessary inequality for endpoints that exist, not a certificate of controllability or reachability.

#### Research check 3: increase samples but do not measure

Plot N=1 or N=16 samples of the unmeasured experiment at the same final T=1. The endpoint fidelity stays cos²1. Obtaining the rising all-yes values requires the additional projectors and records; animation frames alone do not insert them.

## 11. Exercises with explanations

**1. Why does a positive energy produce exp(−iEt/ℏ), rather than exp(−Et/ℏ)?**

**Explanation.** Dividing the Schrödinger equation by iℏ gives a derivative multiplier −iE/ℏ. It rotates phase. The real negative multiplier would produce exponential decay and generally would not preserve norm.

**2. Under Z, does a fixed Z histogram imply a stationary +X state?**

**Explanation.** No. The two probabilities remain 1/2, while the overlap with the initial state has squared modulus cos²t. A basis that mixes the energy components can detect their changing relative phase.

**3. What happens to mean and variance when 3I is added to Z?**

**Explanation.** Every energy and its mean increase by 3. Differences from the mean and the variance are unchanged. The exact representative gains a common phase; its ray trajectory does not change.

**4. What are the Euler norm and energy at two steps in the worked example?**

**Explanation.** The squared norm is 25/16, not 1. The direct quadratic form with Z is still 0 by symmetry, but that coincidence does not make the update unitary. A single apparently conserved number is insufficient to validate an integrator.

**5. If b=0, do we need to select one special initial eigenvector?**

**Explanation.** No. H=aI acts by the same energy on every vector. Every initial ray is stationary and the distinct energy measurement has one outcome a.

**6. Does U(−t) represent a second positive pulse in the same direction?**

**Explanation.** No. It is the inverse propagator with the sign of the action reversed. For the same constant H, U(−t)U(t)=I. The next chapter distinguishes this from changing the order of two positive-duration pulses.

## 12. Connections and next questions

For AI research, this example connects spectral functions, complex embeddings, norm-preserving layers and numerical ODE stability. A layer with a unitary matrix preserves all pairwise inner products, a stronger property than normalizing individual outputs. Real-time Schrödinger evolution also differs from imaginary-time methods used to emphasize low-energy components; the latter require their own interpretation and normalization analysis.

The next experiment changes H between intervals. Infinite-dimensional operator domains, wavefunction PDE boundary conditions, open systems and measurement back-action remain separate topics. This finite model does not prove their existence or stability. Open this textbook through **Book → Interactive Book**.

For background, [Preskill §2.1, Axiom 4](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) states closed-system unitary dynamics. [NIST DLMF §4.19](https://dlmf.nist.gov/4.19) supplies the trigonometric series. The spectral-family derivations, Euler calculations and interval contract above are developed for this textbook.
