## 1. A result distribution does not specify a measurement process

A detector reports either a minus or a plus sign. Suppose two laboratories agree on the probability of each sign for every input state. Must the retained systems also be identical? No: one laboratory can apply feedback after recording a result, while the other does not. Their records can agree and their subsequent experiments can disagree.

The useful conceptual picture is **input state → instrument branch → classical result and retained quantum state → optional conditioning or averaging → subsequent measurement**. A POVM specifies the probabilities of the classical result. An instrument supplies the state changes as well. We will hold one POVM fixed and compare three instruments so that these two roles remain visible.

Our examples use dimensionless Pauli observables and pointer values. This lets us derive exact formulas without confusing a change of physical units with an improvement in accuracy. The aim is to understand a specified model, rather than to claim one error definition captures every operational measurement task.

## 2. Construct a tunable binary POVM

Write the qubit state in Bloch form, choose unit vectors n,b and define A=n·σ, B=b·σ. The number κ controls the strength of information about A in the two outcomes s=−1,+1.

$$
\rho=\frac{I+r\cdot\sigma}{2},\quad |r|\le1,\quad
A=n\cdot\sigma,\quad B=b\cdot\sigma,\quad |n|=|b|=1,\qquad
E_s=\frac{I+s\kappa A}{2},\quad 0\le\kappa\le1,\quad
p_s=\operatorname{Tr}(\rho E_s)=\frac{1+s\kappa m}{2},\quad m=n\cdot r.
$$

Since A has eigenvalues ±1, each effect has eigenvalues (1±κ)/2. They are nonnegative exactly over the stated interval, and E−+E+=I. At κ=1 these effects are projectors. At κ=0 each equals I/2, so the record is a fair coin independent of the input. A fair-coin record does not determine whether the system was left intact.

To obtain n from an arbitrary nonzero complex vector z, form P=zz†/(z†z) and A=2P−I. A nonzero change of representative z→az leaves A unchanged. The experiment uses this construction for both directions, avoiding a requirement to type irrational normalized coordinates.

## 3. Kraus operators specify branches and conditional states

An outcome may contain several unresolved Kraus labels α. Each branch is completely positive, and summing the effects over outcomes gives the trace-preserving condition.

$$
\mathcal I_s(\rho)=\sum_\alpha K_{s\alpha}\rho K_{s\alpha}^\dagger=\tau_s,\qquad
\sum_\alpha K_{s\alpha}^\dagger K_{s\alpha}=E_s,\qquad
\rho_{\mid s}=\frac{\tau_s}{p_s}\ (p_s>0),\qquad
\Psi(\rho)=\sum_s\tau_s.
$$

The unnormalized branch has trace ps. Keeping it is useful because its entries remain linear in the input state. Conditioning divides by that trace and is nonlinear. Ignoring a recorded outcome gives Ψ(ρ), the nonselective output; it is not a weighted sum of unnormalized branches multiplied by ps again.

If ps=0, positivity implies τs=0, but no conditional density is defined. The experiment returns null for its conditional density, Bloch vector, purity and subsequent conditional probabilities. Adding a numerical epsilon would manufacture a state for an event that cannot happen under the model.

## 4. Compare three instruments with the same effects

The positive-square-root, or Lüders, choice uses one operator per outcome. Outcome-dependent unitary feedback changes the output while preserving K†K. A measure-and-prepare instrument instead prepares the A eigenstate matching the recorded sign.

$$
K_s^{L}=\sqrt{E_s},\qquad K_s^{F}=U_s\sqrt{E_s},\qquad
K_{s t}^{P}=\sqrt{\frac{1+st\kappa}{2}}\,|n_s\rangle\langle n_t|,
\quad s,t\in\{-1,+1\},\qquad
\mathcal I_s^{P}(\rho)=p_s|n_s\rangle\langle n_s|.
$$

Summing (KstP)†KstP over t gives Es. Thus all three instruments give the same ps for every ρ. Their conditional states generally differ. Feedback uses U=(aI−i(vxX+vyY+vzZ))/√(a²+|v|²), where the four real inputs need not be normalized. Negating all four changes only a global phase and leaves the channel unchanged.

A subsequent sharp A test is certain to equal s after the prepare instrument. That does not make an unsharp detector repeat its own noisy record with certainty: another application of the original POVM to |n_s⟩ gives the same sign with probability (1+κ)/2. The experiment labels sharp-A repeatability separately from any repeated unsharp measurement.

## 5. Derive the Bloch update and picture the channel

Let h=√(1−κ²) and split r into its parallel part mn and perpendicular part. Write τs=(ps I+ws·σ)/2. Expanding √Es in the A eigenbasis gives the following unnormalized branch and nonselective maps.

$$
w_s^{L}=\frac{h(r-mn)+(m+s\kappa)n}{2},\quad
r_L'=h(r-mn)+mn,\qquad
w_s^{F}=R_sw_s^{L},\quad r_F'=\sum_sR_sw_s^{L},\qquad
w_s^{P}=s p_s n,\quad r_P'=\kappa mn.
$$

For ps>0 the conditional Bloch vector is ws/ps. At κ=0, the Lüders channel is the identity, while the prepare channel sends every state to I/2. At κ=1, both become complete A-basis dephasing, provided the feedback is absent. These endpoint checks test the formulas as well as their interpretation.

Visualize a Bloch ball being compressed perpendicular to n under the Lüders channel. Its axis is retained and transverse components shrink by h. The prepare channel retains only an attenuated axial component. Feedback can rotate different branches differently, translating the image ellipsoid so it need not be centered at the origin. Conditioning describes selected branches inside the ball; averaging gives the channel output.

| Strength κ, input +Y, A=Z | Lüders output Bloch vector | Prepare output | Lüders RMS disturbance squared for X |
| --- | --- | --- | --- |
| 0 | (0,1,0) | I/2 | 0 |
| 3/5 | (0,4/5,0) | I/2 | 2/5 |
| 1 | (0,0,0) | I/2 | 2 |

The record probabilities in every row are 1/2,1/2 for both instruments. A record-only comparison would miss the differences in the retained systems.

## 6. Successive outcomes are a joint law for an ordered experiment

After recording s, measure B sharply with projector Qt=(I+tB)/2. The joint law refers to this specified order and instrument. It is a valid nonnegative distribution even when A and B do not commute.

$$
p(s,t)=\operatorname{Tr}(Q_t\tau_s)=\frac{p_s+t\,b\cdot w_s}{2},\qquad
\sum_t p(s,t)=p_s,\qquad
p(t\mid s)=\frac{p(s,t)}{p_s}\quad(p_s>0).
$$

Its B marginal is computed from Ψ(ρ), not generally from ρ. The first recorded sign is an unsharp meter result, not automatically the original sharp A value. Changing the instrument while retaining Es can change this joint table; reversing the experimental order can also change it.

This does not contradict the absence of an apparatus-independent joint distribution for incompatible initial observables. An ordered process includes additional physical choices. Its distribution is about that process, rather than values assumed to coexist before the first interaction.

## 7. Distinguish readout spread, intrinsic noise and RMS error

Assign pointer value ts to sign s, where the scale t is one for the raw meter and 1/κ for the calibrated meter. The first two operator moments are O1=tκA and O2=t²I. Intrinsic POVM noise is N=O2−O1²; bias is O1−A.

$$
\mathbb E[M]=t\kappa m,\quad
\operatorname{Var}(M)=t^2(1-\kappa^2m^2),\quad
N=t^2(1-\kappa^2)I,\qquad
\epsilon_t(A)^2=\operatorname{Tr}\rho\,[O_2-O_1A-AO_1+A^2]
=t^2+1-2t\kappa.
$$

Here the error also equals the expectation of N+(O1−A)². Thus intrinsic noise and bias both contribute; the outcome spread has an additional dependence on the prepared state. A meter can be unbiased in its mean for every input while still noisy on individual trials.

For κ>0, calibration makes O1=A. Its cost is εcal²=(1−κ²)/κ². At κ=0 calibration is undefined and returns null. Dividing by an epsilon would claim information that the fair-coin record does not contain.

| κ=3/5, input +Z, A=Z | Pointer values | Readout variance | RMS error squared relative to A |
| --- | --- | --- | --- |
| Ideal sharp measurement | ±1 | 0 | 0 |
| Raw unsharp meter | ±1 | 16/25 | 4/5 |
| Calibrated unsharp meter | ±5/3 | 16/9 | 16/9 |

The input variance of A is zero in all rows. The last row's unbiased mean does not recover a sharp trial-by-trial measurement. Calibration rescales both information and randomness.

## 8. Define the RMS disturbance through an isometry

Attach orthogonal record labels and define V=ΣKₛα⊗|s,α⟩. Kraus completeness gives V†V=I. A pointer acts on the record register; B acts on the retained system. The following squared norms define the device quantities used here, with mixed states represented by √ρ.

$$
\epsilon_t(A)^2=\sum_{s,\alpha}\|(tsK_{s\alpha}-K_{s\alpha}A)\sqrt\rho\|_{\mathrm{HS}}^2,\qquad
\eta(B)^2=\sum_{s,\alpha}\|(BK_{s\alpha}-K_{s\alpha}B)\sqrt\rho\|_{\mathrm{HS}}^2.
$$

These expressions need no assignment of simultaneous initial A and pointer values. Expanding them gives the earlier error formula and an observable-based channel expression for disturbance. If Ψ*(B)=tB I+d·σ, where tB is a scalar coefficient unrelated to the meter scale, Pauli B²=I gives η²=2−2d·b−2tB(r·b).

For the Lüders channel, η(B)²=2(1−h)[1−(n·b)²]. It vanishes for B=A and can stay positive for a fixed input density. In particular, sharp Z dephasing leaves I/2 unchanged but gives η(X)=√2. “This density did not change” and “this observable has zero RMS disturbance” are different claims.

## 9. Compare state distance and distribution change separately

For qubits the trace distance has a simple Bloch expression. The total variation of a subsequent binary B measurement is bounded by it.

$$
D(\rho,\Psi(\rho))=\frac{|r'-r|}{2},\qquad
\operatorname{TV}_B=\frac{|b\cdot(r'-r)|}{2}\le D(\rho,\Psi(\rho)).
$$

At the unchanged I/2 example both quantities are zero despite positive RMS disturbance of X. Conversely, some channel changes are invisible to one selected B histogram while visible to another. None of these metrics should be silently substituted into a theorem formulated for η.

A sharp Z record followed by X feedback in both branches is another useful check. On input +Z, the plus record is certain, but feedback sends the retained state to −Z. The input/output trace distance is one, a following sharp Z result disagrees with the recorded sign, and η(Z)²=4. Projective effects alone did not force a repeatable state update.

## 10. Use the correct error-disturbance inequality

Define C=|Tr(ρ[A,B])|/2=|(n×b)·r|. Robertson concerns σAσB, while the device quantities defined above obey Ozawa's corrected relation. A product-only replacement need not hold.

$$
\sigma_A\sigma_B\ge C,\qquad
\epsilon(A)\eta(B)+\epsilon(A)\sigma_B+\sigma_A\eta(B)\ge C.
$$

One route to the second inequality embeds the isometry in a unitary dilation. The output pointer and output B commute because they act on different factors. Expand that commutator using their deviations from input A and B. The three resulting commutator expectations are bounded by Cauchy–Schwarz, with standard deviations of the deviations no larger than their RMS norms. This produces the three terms; discarding two of them is not justified in general.

Take input +Y, A=Z, B=X and a sharp Lüders measurement. Then C=1, σA=σB=1, ε=0 and η=√2. The naive product εη is zero, while the corrected left side is √2≥1. Preparation uncertainty is also satisfied. This example distinguishes three equations using one state and one device rather than appealing to an unspecified disturbance slogan.

## 11. Audit channel structure and numerical limits

For a channel, form the unnormalized Choi matrix J=Σij Ψ(|i⟩⟨j|)⊗|i⟩⟨j| in output⊗input order. It is positive with trace two; tracing over the output gives I. Its rank equals the minimum Kraus count, which can be smaller than the number of supplied operators. At κ=0 the identity Lüders channel has rank one while the completely depolarizing prepare channel has rank four, despite identical effects.

The experiment accepts 22 owned integers of magnitude at most10⁶: a rational Bloch vector, two raw complex direction vectors, a rational κ and two feedback quaternions. It verifies the Bloch ball and nonzero directions exactly. Channel entries, conditional states, squared RMS values and Choi rank use Q(√(1−κ²)); rational roots are folded into rational numbers. Nonzero algebraic signs are decided before plotting.

Display square roots and explicit normalized Kraus entries use 768-bit work and outward 512-bit reporting. Opposite-sign cancellation is rationalized before interval evaluation. An unresolved interval is reported as unresolved rather than clipped to a presumed theorem. The calculator has no general multiqubit or arbitrary POVM input. Naimark dilation explains why POVMs can be realized with larger systems, but this example does not simulate every such realization.

## 12. Exercises, explanations and connections to AI

**1. Does κ=0 imply no disturbance?**

**Explanation.** It implies no information in the outcome probabilities. The Lüders realization is the identity, but the prepare realization outputs I/2 for every input. Effects alone do not determine state changes.

**2. Why is there no calibrated meter at κ=0?**

**Explanation.** Its scale would be 1/κ. An input-independent coin cannot become an unbiased A measurement through finite rescaling, so the result is null rather than a large invented value.

**3. Can an unbiased meter have nonzero RMS error?**

**Explanation.** Yes. At κ=3/5 calibration makes O1=A, but intrinsic noise remains16/9. Mean calibration and trial-level accuracy answer different questions.

**4. If a channel leaves I/2 unchanged, must η(X) vanish?**

**Explanation.** No. Sharp Z dephasing gives η(X)²=2 while trace distance and every change of that state's output histograms vanish. The RMS observable definition uses the channel action beyond one density comparison.

**5. Does preparing |n_s⟩ make an unsharp record repeat with certainty?**

**Explanation.** A sharp A measurement repeats s certainly. The original unsharp POVM repeats it only with probability (1+κ)/2. At κ=0 that probability is1/2.

**6. Which relation survives ε=0 in the sharp +Y example?**

**Explanation.** Robertson and the corrected three-term relation both survive. The naive εη≥C fails because its left side is0 and C=1; the corrected left side is√2.

In AI, matching a model's output distribution does not establish equivalence of its hidden state transitions. Calibration of means does not guarantee low sample-level error, and intervention order matters. These are structural analogies; a generic stochastic neural network is not thereby a quantum instrument. Further study includes general instruments, quantum channels, estimation, feedback control and operational error definitions. These research questions remain separate from the finite experiment's completion.

Open the textbook through **Book → Interactive Book**. For background see [Preskill §3.2.4](https://www.preskill.caltech.edu/ph229/notes/chap3.pdf); the RMS definitions and corrected relation are described in [Ozawa's measurement-operator paper](https://arxiv.org/abs/quant-ph/0510083). The three-device comparison, derivations and worked inputs here are developed for this textbook.
