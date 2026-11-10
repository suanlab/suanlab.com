## Ask where a packet carries actual energy

A Gaussian envelope can show how a peak moves, but a normalized squared field is not automatically physical energy. We now use masses connected by linear springs, where kinetic energy, elastic energy and power have explicit mechanical definitions. This gives a concrete setting in which group motion can be compared with transport rather than inferred from appearance.

The chain is periodic. A packet crossing the right edge reappears at the left because these positions are neighbors, not because a boundary absorbed and recreated it. We will compare individual particle motion, an analytic envelope, assigned mechanical energy and link power. Their centers and shapes need not coincide.

## Mechanical variables and supported assumptions

Let the chain have length $L$, $N$ particles, spacing $a=L/N$, line density $\rho$ and tension parameter $\mathcal T$. Assign each particle mass $m=\rho a$ and each link stiffness $\kappa=\mathcal T/a$. Their units are kg and N/m. Displacements $q_j$ are in m and velocities $v_j$ in m/s.

Assume a lossless nearest-neighbor harmonic model with identical particles and periodic links. For a transverse-string interpretation, the linear relation also needs small relative displacements and slopes. The harmonic equations remain a mathematical model at larger configured amplitudes, but an animation does not validate their physical small-slope approximation.

We use32 or64 particles and strictly positive Fourier modes below Nyquist. The zero translation mode and the Nyquist endpoint are excluded deliberately. No disorder, damping, external driving or nonlinear spring law is included. The model reports the all-phase bound $\sum_n k_n A_n$ for displacement slope. A bound near or above1 warns against an unchecked small-slope string interpretation; it is not a universal material-error certificate.

## Derive the lattice dispersion relation

Newton's equation is

$$
m\ddot q_j=\kappa(q_{j+1}-2q_j+q_{j-1}).
$$

Substitute $q_j\propto e^{i(kja-\omega t)}$ to obtain

$$
\omega^2=\frac{4\kappa}{m}\sin^2\frac{ka}{2},\qquad
\omega(k)=\frac{2c}{a}\sin\frac{ka}{2},\quad
c=\sqrt{\mathcal T/\rho},\quad 0<ka<\pi.
$$

The positive sine branch follows from the chosen positive-wave-number band. Extending the formula outside that band without absolute values or a branch convention would misidentify positive frequencies. At small $ka$, $\omega\approx ck$ and the nondispersive continuum relation emerges. Close to the lattice cutoff, neighboring sites are poorly represented by a smooth continuum wave.

## Phase, group and curvature are different derivatives

Within the stated band,

$$
v_p(k)=\frac{\omega(k)}{k},\qquad
v_g(k)=c\cos\frac{ka}{2},\qquad
\omega''(k)=-\frac{ca}{2}\sin\frac{ka}{2}.
$$

The phase speed exceeds group speed for positive resolved modes. The curvature is negative, so a narrow packet has an envelope chirp and spreading associated with that sign. Group speed decreases toward the cutoff even though the oscillation frequency remains nonzero.

A configured carrier mode can be noninteger: it specifies the center of spectral weights, not an extra unresolved physical mode. Actual allowed wave numbers remain $k_n=2\pi n/L$ for integer $n$. Evaluating a smooth derivative at the carrier is a reference value, while propagation uses the actual discrete spectrum.

## Build a packet and declare its normalization

Choose coefficients

$$
w_n=e^{-(n-n_c)^2/(2s_n^2)},\qquad
A_n=A_{\max}\frac{w_n}{\sum_rw_r},\qquad
U_j(t)=\sum_nA_n e^{i[k_n(ja-x_0)-\omega_nt+\phi_0]}.
$$

The sum contains $n=1,\ldots,N/2-1$. The physical displacement is $q_j=\operatorname{Re}U_j$. The initial parameter $A_{\max}$ bounds $|U_j|$ by the triangle inequality; it is not a promise that one actual lattice site attains that value. An off-grid center and carrier phase can lower the observed real peak.

The Gaussian mode weights are truncated to the resolved positive band. They are not the exact Fourier transform of an unbounded real-line Gaussian. At wide bandwidth or near a cutoff, truncation and periodic images matter. Changing bandwidth at fixed peak bound changes energy; this preparation does not secretly renormalize every experiment to equal energy.

## Evolve real displacement and velocity consistently

For each term with phase $\theta_{nj}=k_n(ja-x_0)-\omega_nt+\phi_0$,

$$
q_j=\sum_nA_n\cos\theta_{nj},\qquad
v_j=\sum_n\omega_nA_n\sin\theta_{nj}.
$$

The positive sign in velocity comes from differentiating $\cos(kx-\omega t)$. Choosing zero initial velocity instead would prepare counterpropagating components rather than the same right-moving packet. Initial displacement alone is insufficient to specify the intended experiment.

Evolution is an exact finite spectral sum for the harmonic equations. No time step accumulates integration drift. Nevertheless, independent matrix-exponential checks remain useful: exact formulas can still contain a wrong sign, normalization or initial phase. Only returned particle sites are physical masses; a smooth interpolating curve is a visualization of the resolved modes.

## Give mechanical energy its own definition

Kinetic energy at site $j$ and spring energy on link $j\to j+1$ are

$$
K_j=\tfrac12mv_j^2,\qquad
V_j=\tfrac12\kappa(q_{j+1}-q_j)^2.
$$

Assign half of each neighboring link to a site:

$$
e_j=K_j+\tfrac12(V_{j-1}+V_j),\qquad
E=\sum_je_j=\sum_j(K_j+V_j).
$$

These quantities are in J. Another local assignment can produce different-looking local density, but the chosen assignment and its associated power must be used consistently. Counting every spring's energy twice would break the total even if the plotted packet looked plausible.

The analytic quantity $a\sum_j|U_j|^2$ has units m³ because $U$ is a displacement. It is separately conserved by the chosen spectral evolution, but it is not mechanical energy. High-frequency components cost more physical energy for the same displacement amplitude.

## Derive local power balance

For the symmetric site-energy assignment, define positive rightward link power

$$
J_j=-\frac{\kappa}{2}(q_{j+1}-q_j)(v_{j+1}+v_j).
$$

Differentiate $e_j$ using Newton's equation and the link deformation rate. The terms reorganize into

$$
\dot e_j=J_{j-1}-J_j.
$$

This is a local statement: incoming power minus outgoing power changes assigned site energy. Summing over the periodic chain cancels every link and gives $\dot E=0$. Link power is in W, and $a\sum J_j/E$ therefore has units m/s.

Interference can produce local negative power in a multi-mode packet. Do not force every arrow positive merely because all prepared spectral branches have positive group velocity. The global transport identity below concerns the complete spatial sum with its specified energy, not every instantaneous local arrow.

## Modal energy explains the weighting

Orthogonality over the resolved periodic sites gives each prepared traveling mode energy

$$
E_n=\frac{mN}{2}\omega_n^2A_n^2,\qquad E=\sum_nE_n.
$$

The factor differs from a standing wave with the same displacement coefficient because the traveling preparation includes a specific nonzero velocity. Both kinetic and elastic terms contribute. The zero and Nyquist exclusions also matter to the orthogonality argument.

A useful negative control is to compare squared-amplitude weights with energy weights. The former omit $\omega_n^2$ and generally predict a different mean velocity. Weighting by an attractive-looking envelope instead of actual mechanical energy can therefore give a systematically biased transport estimate.

## An exact global energy-transport identity

For this finite, positive-mode preparation,

$$
v_E=\frac{a\sum_jJ_j}{E}
=\frac{\sum_nE_n v_g(k_n)}{\sum_nE_n}.
$$

The equality follows by summing the real link power over all sites: cross terms between distinct resolved modes cancel. For a nonzero packet, $0<v_E<c$. The identity is exact for this model and preparation, rather than a general theorem that every group's peak must move at an energy speed.

For a sufficiently narrow spectrum, $v_g(k_n)$ changes little and $v_E\approx v_g(k_c)$. A broad packet can differ substantially from the carrier derivative. In complete darkness, both numerator and denominator are zero; transport velocity is undefined. It is not a meaningful zero-speed measurement.

## Work through one physical preparation

Choose $L=8$ m, $N=32$, $\rho=0.8$ kg/m, $\mathcal T=1.7$ N, $A_{\max}=0.08$ m, $n_c=4$, bandwidth $s_n=2$, $x_0=2.96$ m and $\phi_0=0.4$. Then $a=0.25$ m, $m=0.2$ kg and $\kappa=6.8$ N/m. The continuum reference speed is approximately1.4577 m/s.

The independent physical-equation calculation gives total energy approximately0.06790960 J and energy transport1.28638809 m/s. The carrier group speed is about1.3468 m/s, illustrating the effect of finite bandwidth and energy weighting. The conserved analytic norm is approximately0.00776644 m³; its different dimensions already rule out interpreting it as the same energy.

At the initial time both circular centers are2.96 m for this symmetric preparation, with resultant strengths about0.9305 for analytic intensity and0.9231 for assigned energy. Coinciding initial centers do not imply identical subsequent distributions or equal definitions of their speed.

## Centers on a circle need a concentration diagnostic

A linear average of sites near0 and $L$ incorrectly places their center near $L/2$. For nonnegative weights $w_j$, use the first circular moment

$$
M=\frac{\sum_jw_je^{i2\pi x_j/L}}{\sum_jw_j},\qquad
x_c=\frac{L}{2\pi}\arg M\pmod L.
$$

Report $|M|$ along with the angle. A uniform distribution has $M=0$, so its center angle is undefined. Equal opposite lobes can also have zero first moment; undefined center does not imply uniformity. A tiny moment makes the angle highly sensitive to perturbations; the implementation rejects a resultant below $10^{-10}$ rather than displaying an arbitrary center. Nonzero small values remain weakly conditioned measurements.

Use analytic intensity and assigned mechanical energy as separate weight choices. Neither circular center derivative is automatically $a\sum J/E$: differentiation of a circular moment includes position-dependent phase weights. Unwrapping through an undefined center can manufacture jumps or apparent superfast motion.

## Narrow-band limits and convergence experiments

Near a carrier $k_c$, the lattice relation can be expanded with the phase, group and curvature derived above. This connects to the preceding Gaussian envelope lesson. The approximation is useful only when the resolved spectrum is localized and accumulated higher-order phase errors remain small. Wide spectra, truncation near the cutoff and wrapping around the domain can each invalidate a simple Gaussian description.

When changing $N$ at fixed $L,\rho,\mathcal T$, update $a$, particle mass and link stiffness together. Holding mass and stiffness fixed would define a different material rather than a refinement of the same continuum reference. To test convergence, hold the physical initial spectrum or preparation conditions fixed and state which is being compared.

A finite grid is not automatically an error. Here it defines the actual harmonic lattice. Distinguish errors in approximating that lattice's evolution from physical differences between a lattice and an ideal continuum string.

## Research checks and AI prediction targets

An AI model can predict future displacements, local energy or link power from the initial state and material parameters. Require consistency between its outputs: a small displacement error does not guarantee correct energy, because energy is quadratic and includes spatial differences and velocities. Evaluate total conservation and local incoming-minus-outgoing balance separately.

Use zero amplitude, narrow bandwidth, a translated center and a wide-band carrier near the cutoff as distinct controls. Evaluate unseen material ratios and bandwidth regimes. Adjacent samples of one trajectory do not establish generalization to a different dispersion law.

Reference generation uses the exponential of the real physical evolution matrix and reconstructs the analytic field through an independent FFT. Comparing that result with the finite spectral model checks more than re-evaluating its own trigonometric expression. A verified mathematical foundation still needs a separately tested reader interface before it counts as an integrated homepage topic.

## Explained exercises

### Exercise 1 — Refine the same material

Double $N$ at fixed $L,\rho,\mathcal T$. What happens to mass and stiffness?

**Solution:** Spacing halves, so $m=\rho a$ halves and $\kappa=\mathcal T/a$ doubles. The reference $c=\sqrt{\mathcal T/\rho}$ stays fixed. Keeping both mass and stiffness unchanged would change the material represented by the refinement.

### Exercise 2 — The sign of the initial velocity

For $q=A\cos(kx-\omega t)$, find velocity at $t=0$.

**Solution:** Differentiation gives $v=\omega A\sin(kx-\omega t)$, hence $v(x,0)=\omega A\sin(kx)$. Setting velocity to zero prepares a different superposition, so it cannot be used as the same traveling-packet initial state.

### Exercise 3 — Choose the correct mean speed

Two modes have energies1 J and3 J and group velocities1.4 and0.6 m/s. Find $v_E$.

**Solution:** The energy-weighted mean is $(1\cdot1.4+3\cdot0.6)/4=0.8$ m/s. An unweighted mean would give1 m/s and ignore the physical distribution of transported energy. Amplitude weighting also requires justification and is generally different.

### Exercise 4 — Less visible does not mean less energy

At fixed amplitude bound, does broadening the mode spectrum keep energy constant?

**Solution:** No. Normalization fixes $\sum A_n$, whereas energy is proportional to $\sum\omega_n^2A_n^2$. Both amplitude allocation and frequency weights change. Measure the prepared energy before comparing transport rather than assuming the same normalization preserves every quantity.

### Exercise 5 — A center at a periodic edge

Equal weights occur only at0 and6 m on a ring of length8 m. Where is their circular center?

**Solution:** Their phases are0 and $3\pi/2$, whose mean has angle $-\pi/4$. The center is7 m modulo8, with resultant $1/\sqrt2$. A linear average would incorrectly put it at3 m, away from the short arc between the two sites.

### Exercise 6 — Undefined center and transport

Can a traveling mode have an undefined circular energy center but nonzero transport?

**Solution:** Yes. A pure sinusoidal mode has assigned energy with a constant part and a doubled-mode component, so its first circular moment can vanish. This need not mean spatially uniform energy. Link power can still have a nonzero spatial sum. Center tracking cannot replace the physical flux definition.

## Sources and reproducible evidence

[MIT 8.03SC lecture13](https://www.ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/432d23acf0bce355921a8b9a4eefb0b6_MIT8_03SCF16_Lec13.pdf) introduces dispersive waves; [Harvard Matthew Schwartz lecture11](https://schwartz.scholars.harvard.edu/sites/g/files/omnuum7046/files/schwartz/files/lecture11-wavepackets.pdf) develops wave-packet expansions. The specified finite-chain normalization, energy assignment, local power and global transport identity are derived here from the stated mechanical equations.

`latticeDispersivePacket` in `src/lib/learning-dispersion.ts` computes real site data and the distinct observables. `scripts/learning/generate-dispersion-references.py` uses SciPy matrix exponentials and FFT reconstruction without product imports. Tests compare actual outputs and check force, local balance, conservation, translation and undefined observables. These checks support this defined model; they do not establish every real material approximation or certify an untested visualization.
