## Compare a spontaneous fluctuation with a controlled response

Imagine a small particle held by a spring in a fluid. Even without an applied force it fluctuates. You can record how a displacement at one time correlates with a later displacement. In a second experiment, apply a known force and measure the change in the average position across many repetitions. A fluctuation–dissipation relation connects these two experiments under specified assumptions.

The fluctuating position of one particle is not the average response curve. A photograph of a stationary cloud is not a record of its time correlations. Before comparing graphs, specify the force, its time dependence, the measured coordinate, the initial distribution and the time ordering of the correlation.

We first derive an equilibrium relation and then add a circulating drive that preserves the same Gaussian cloud. This makes it possible to see exactly which equilibrium shortcut fails and how stationary-current information corrects it.

## Introduce an explicit thermal model

The earlier probability-current chapter did not infer temperature from arbitrary damping and diffusion coefficients. Here we introduce additional physical assumptions: a positive bath thermal energy theta=kBT, scalar mobility mu, spring stiffness kappa and the Einstein relation D=mu theta. Both coordinates are positions and are even under time reversal. The model is

$$
dX_t=[AX_t+\mu f(t)]\,dt+\sqrt{2\mu\theta}\,dW_t,
\quad A=-\gamma I+\omega R,\quad\gamma=\mu\kappa,
\quad R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

Mobility has units m/(N·s), stiffness N/m, theta J, force f N, and gamma and omega s⁻¹. Diffusion has units m²/s. For omega=0 the force is the gradient force of the potential kappa|x|²/2. For omega≠0 an additional nonconservative force(omega/mu)Rx drives circulation. Positive omega is counterclockwise in an ordinary Cartesian coordinate plot.

These are assumptions of this new thermal model, rather than consequences of a stationary histogram. We do not retroactively attach a bath to every earlier Markov or Gaussian example. Matrix entries in the tables use indices0 and1 for the first and second position coordinates.

## Establish the unperturbed stationary law

Before the additional force is applied, initialize a zero-mean Gaussian independently of future Brownian increments. Its variance per coordinate is v=theta/kappa. The covariance equation verifies stationarity:

$$
\Sigma=vI,\quad v=\frac{\theta}{\kappa},\quad D=\mu\theta,
\qquad A\Sigma+\Sigma A^T+2DI=0.
$$

The rotation cancels from the covariance equation. Thus changing omega leaves this Gaussian law unchanged while changing how positions at different times are related. At equilibrium the same Gaussian follows from the Boltzmann weight of the harmonic potential. In the driven case, the same density does not establish detailed balance.

The correlation below is measured in the unperturbed stationary process. Applied deterministic forces change the mean. Covariance about that changing mean remains vI in this affine model, but a raw second moment of the forced process also contains the mean outer product. Keep centered covariance and raw moments distinct.

## Distinguish an impulse from a step force

Use a script-like letter for the response kernel, reserving R for rotation. Causality means forces applied later cannot affect an earlier observation:

$$
\delta\langle X(t)\rangle=\int_0^t\mathcal R(t-s)f(s)\,ds,
\qquad \mathcal R(t)=\mu e^{At}\ (t>0),
\qquad C(t)=\mathbb E[X_tX_0^T]=v e^{At}.
$$

The kernel is zero at negative lags. Its value at zero is interpreted as the right limit mu I. A force impulse has an integrated strength I_f measured in N·s; it produces mean displacement ℛ(t)I_f. A constant force F switched on at zero has units N and produces mean displacement chi(t)F, where chi is the time integral of the kernel:

$$
\chi(t)=\int_0^t\mathcal R(s)\,ds,
\qquad m_{\rm step}(t)=\chi(t)F,
\qquad m_{\rm impulse}(t)=\mathcal R(t)I_f.
$$

The kernel has units m/(N·s), while chi has units m/N. A unit force and a unit force impulse are different inputs despite having the same numerical coefficient. At time zero the step mean is zero, while an ideal impulse has the immediate right-limit mean mu I_f. This is the ideal impulse of the overdamped equation, not a finite-inertia acceleration calculation.

## Derive the equilibrium relation and its sign

For omega=0, the matrix exponential is exp(−gamma t)I. Differentiate the positive-lag correlation and use v gamma=mu theta:

$$
C(t)=v e^{-\gamma t}I,
\qquad -\frac{C'(t)}{\theta}
=\frac{\gamma v}{\theta}e^{-\gamma t}I
=\mu e^{-\gamma t}I=\mathcal R(t).
$$

The minus sign belongs to differentiation with respect to increasing lag t. Differentiation with respect to the earlier time in a stationary two-time correlation has the opposite sign. A formula copied without its time convention can therefore predict the wrong sign.

Integrating gives the equilibrium step susceptibility:

$$
\chi(t)=\frac{1-e^{-\gamma t}}{\kappa}I
=\frac{C(0)-C(t)}{\theta}.
$$

With coefficients mu=kappa=theta=1 and a force along the first coordinate, at t=1 the kernel is exp(−1)≈0.367879 times I, while the step mean for F=(1,0) is approximately(0.632121,0). The impulse response decays; the step mean approaches its displaced stationary value. They should not be plotted as if they were the same response.

## Change one physical control at a time

The variance, relaxation rate and mean response depend on different combinations of coefficients. The following equilibrium comparisons use the same observation time t=1 and F=(1,0).

| Preparation | Coordinate variance v | Relaxation gamma | First step-mean component |
| --- | --- | --- | --- |
| mu=kappa=theta=1 |1 |1s⁻¹ |0.632121m |
| mu=2, kappa=theta=1 |1 |2s⁻¹ |0.864665m |
| kappa=2, mu=theta=1 |1/2 |2s⁻¹ |0.432332m |
| theta=3, mu=kappa=1 |3 |1s⁻¹ |0.632121m |

Increasing mobility changes the clock without changing the equilibrium width or static displacement F/kappa. Increasing stiffness narrows the cloud and reduces that static displacement. Increasing thermal energy broadens spontaneous fluctuations but does not change the deterministic mean response when mu, kappa and omega stay fixed. Dividing the correlation derivative by theta is essential to recover the same kernel in the hotter example.

These statements hold for the specified linear thermal model. They are not a general claim that temperature cannot change a real material’s mobility or stiffness.

## Add rotation while preserving the Gaussian cloud

For nonzero omega, the propagator rotates as it decays. With the covariance order fixed as E[X_t X_0 transpose],

$$
C(t)=v e^{-\gamma t}
\begin{pmatrix}\cos\omega t&-\sin\omega t\\\sin\omega t&\cos\omega t\end{pmatrix},
\qquad\mathcal R(t)=\mu e^{-\gamma t}
\begin{pmatrix}\cos\omega t&-\sin\omega t\\\sin\omega t&\cos\omega t\end{pmatrix}.
$$

Off-diagonal entries have signs. They are correlations or response coefficients, not probabilities, so negative entries must not be clipped. Reversing omega reverses the transverse response. Transposing C would also change those signs, which is why its time order must be visible.

At mu=kappa=theta=omega=1 and t=1, the actual first response column is approximately(0.198766,0.309560). The naive equilibrium expression−C'/theta gives(0.508326,0.110794). Both the magnitude and direction are wrong even though the stationary cloud has exactly the same width as the equilibrium reference.

## Identify the current correction

The stationary probability-flow velocity is v_s(x)=omega R x. It is a transport field J/p, not the ordinary instantaneous velocity of a Brownian path. Correlate its value at the earlier time with the later position:

$$
K(t)=\mathbb E[X_t v_s(X_0)^T]
=C(t)(\omega R)^T=-\omega C(t)R.
$$

Using C'=AC and gamma=mu kappa gives

$$
\mathcal R(t)=-\frac{C'(t)+K(t)}{\theta},
\qquad -\frac{C'(t)}{\theta}-\mathcal R(t)
=-\frac{\omega}{\kappa}R e^{At}.
$$

This equation names the missing information in the equilibrium shortcut. The naive prediction is−C'/theta; the correction is−K/theta. The drift decay and circulating current enter separately. In equilibrium K=0 and the familiar relation returns.

An especially simple check is t=0+. The true0,1 response is zero. With unit coefficients and omega=1, the naive0,1 response is1 and the current correction is−1. Testing only the diagonal at zero would miss the failure. A nonzero transverse response at exactly zero lag would incorrectly turn a force along the second coordinate directly into an instantaneous first-coordinate displacement in this scalar-mobility model.

## Connect the response to a density score

For an additive force, the perturbation of the density generator is−mu f·gradient. Dividing its action on the stationary density by that density produces a conjugate observable:

$$
B_j(x)=-\mu\partial_j\log\rho(x)=\frac{\mu}{v}x_j,
\qquad\mathcal R_{ij}(t)=\mathbb E[X_i(t)B_j(X_0)]
=\frac{\mu}{v}C_{ij}(t).
$$

This is a spatial density score combined with a specified perturbation operator. It is not simply the derivative of a parameterized stationary density with respect to the applied force. The two concepts can be related under additional derivations but should not be identified by their shared word “score.”

For score-based AI, the lesson is useful but specific: knowing a one-time score does not identify an entire generator or every possible response. Here we also know how force enters and use the measured two-time correlation C. A different mobility or rotational drive can change the response while leaving the one-time Gaussian score unchanged.

## Integrate the driven kernel before predicting a step experiment

The damped matrix A is invertible. Integrating the kernel yields

$$
\chi(t)=\mu A^{-1}(e^{At}-I),
\qquad\chi(\infty)=-\mu A^{-1},
\qquad m_{\rm step}(t)=\chi(t)F.
$$

The naive equilibrium integral(C(0)−C(t))/theta generally differs. At the unit driven example with F=(1,0), the true t=1 step mean is approximately(0.555397,0.245837). The naive mean is approximately(0.801234,−0.309560). The long-time true mean is(1/2,1/2), rather than(1,0), because the background nonconservative force also acts on a displaced cloud.

This affine model has an exactly linear deterministic mean equation. Its formula is therefore exact for the chosen finite force. General response theory usually concerns a derivative at zero perturbation and requires separate control of nonlinear corrections. Do not export this exact finite-force statement to an arbitrary learned dynamical model.

## Treat an effective thermal-energy ratio as a diagnostic

One might force the equilibrium formula to fit a component by defining

$$
\theta^{\rm eff}_{ij}(t)=-\frac{C'_{ij}(t)}{\mathcal R_{ij}(t)},
\qquad
\theta^{\rm eff}_{\rm diag}=\theta\left(1+\frac{\omega}{\gamma}\tan\omega t\right),
\quad
\theta^{\rm eff}_{\rm off}=\theta\left(1-\frac{\omega}{\gamma}\cot\omega t\right).
$$

These expressions apply only when the relevant response denominator is nonzero. At zero lag the off-diagonal response is zero, so the ratio is undefined, including the equilibrium0/0 case. An infinite-looking plot limit is not a defined value at that time.

For the unit driven case at t=1, the diagonal ratio is approximately2.557408J and the off-diagonal ratio0.357907J, although the bath thermal energy is1J. At t=1/4 the off-diagonal ratio is approximately−2.916317J. This negative diagnostic is not a negative bath temperature. It shows that one scalar temperature does not repair an inappropriate equilibrium formula for all times and components.

Near a zero response component, small measurement errors can produce enormous ratio errors. The rational angle355/226 is close to, but not exactly, pi/2. Its diagonal response at t=1 is small and negative, not an exact structural zero. Preserve that distinction before interpreting a large ratio.

## Preserve small values and explain what is exact

The calculations use exact rational coefficients and directed enclosures for exponentials and trigonometric functions. A tiny nonzero linear response can be unresolved at the working precision without becoming physically zero. Its decay logarithm and, when the trigonometric sign is resolved, its separate absolute-response logarithm remain useful:

$$
\log|\mathcal R_{ij}(t)|
=\log\mu-\gamma t+\log|g_{ij}(\omega t)|,
\qquad g_{ij}\in\{\cos,\sin,-\sin\}.
$$

The logarithm uses the kernel’s unit m/(N·s) as its reference. A genuinely zero sine component has no finite logarithm. When computing the component diagnostic, the common strictly positive exponential can be canceled analytically before division. Consequently that ratio can remain available even when a direct linear-response display is unresolved.

Corrected and score response matrices use the exact algebraic identity of this family. A displayed identity residual of zero certifies that derivation, not an exact subtraction of noisy empirical curves or a proof about general data. Plot coordinates and rounded numbers are presentation aids; inspect the enclosures when signs or cancellations matter.

## Plan controlled comparisons and research extensions

The22 preparations compare equilibrium coefficient changes, both rotation directions, longitudinal and transverse forcing, independent impulse and step directions, zero lag, short lags, dense time records, extreme damping and near-zero response components. A faster-clock preparation preserves the lag phase by shortening the observation time; it preserves the corresponding correlation while rescaling the impulse kernel. Equal physical time and equal phase are different comparisons.

Use separate plots for unperturbed correlation, force-response kernel and integrated mean displacement, with units on every axis. Display actual response, naive prediction and current correction without clipping signed components. A control that changes the selected time record should not regenerate a random path or relabel an ensemble curve as a trajectory.

The [Seifert review](https://arxiv.org/pdf/1205.4176), Sections8.2–8.3, motivates distinguishing equilibrium and stationary nonequilibrium response and identifying the conjugate observable. The two-dimensional formulas and examples here are direct calculations, rather than a reproduction of the paper’s one-dimensional ring experiment.

Further work includes nonlinear observables, nonstationary initial conditions, general perturbation operators, frequency-domain response and estimating derivatives from finite data. The separate linear-response and entropy-production topics remain ahead. This first solvable family does not exhaust fluctuation–dissipation theory or the research curriculum.

## Research extension: response spectra, noise and probe work

### Specify a Fourier convention before comparing curves

An impulse response describes time lags. A sinusoidal experiment asks how the mean moves at a selected frequency after the switching transient has decayed. Keep the chapter's thermal assumptions mu,kappa,theta>0, gamma=mu kappa, D=mu theta and a stationary unforced covariance vI. Allow signed omega. Drive one coordinate with a deterministic sinusoidal force. With the exp(+i Omega t) input convention, the retarded transform is

$$
f(t)=\operatorname{Re}(F e^{i\Omega t}),\qquad \widehat\chi(\Omega)=\int_0^\infty\mathcal R(t)e^{-i\Omega t}\,dt=\mu(i\Omega I-A)^{-1}=\mu\frac{(\gamma+i\Omega)I+\omega R}{(\gamma+i\Omega)^2+\omega^2}.
$$

Omega is angular frequency in rad/s, not cycles/s. The susceptibility has units m/N. Its imaginary part carries a convention-dependent sign: exp(+i Omega t) means a passive equilibrium response has a negative imaginary part at positive frequency. The formula follows by integrating exp[(A−i Omega I)t]; positive gamma makes the boundary at infinity vanish. It describes the long-running periodic mean, not a force switched on without a transient. This frequency calculation is a standalone extension, not a new control in the current impulse/step reader.

### A noise spectrum uses both positive and negative lags

Define the two-sided spectral density matrix, keeping the existing covariance order E[X_t X_0 transpose]:

$$
S(\Omega)=\int_{-\infty}^{\infty}C(t)e^{-i\Omega t}\,dt,\qquad C(-t)=C(t)^T\quad(t>0).
$$

For negative lag, use the transpose relation, not exp(At) with a negative t. The latter would grow rather than represent the stationary covariance. Fourier transformation of the linear SDE, or separate integration on each half-line, gives

$$
H(\Omega)=(i\Omega I-A)^{-1},\qquad S(\Omega)=2D H(\Omega)H(\Omega)^\dagger,\qquad S=S^\dagger\succeq0.
$$

This spectrum describes the unforced stationary process, or centered fluctuations after subtracting the periodic forced mean. The dagger means complex conjugate transpose. S has units m²·s and is Hermitian positive semidefinite, even though an off-diagonal component may be imaginary. Using the chapter's coordinate indices 0 and 1 gives the explicit forms

$$
\mathcal D_\Omega=(\gamma^2+\omega^2-\Omega^2)^2+4\gamma^2\Omega^2,\quad S_{00}=\frac D{\gamma^2+(\Omega-\omega)^2}+\frac D{\gamma^2+(\Omega+\omega)^2},\quad S_{01}=\frac{4iD\omega\Omega}{\mathcal D_\Omega}.
$$

The first component is a sum of two Lorentzian peaks associated with opposite circular polarizations. The sign of the imaginary cross spectrum depends on rotation direction and the stated time order. An individual off-diagonal value is not a scalar power that must be real and nonnegative.

### Derive the equilibrium relation, then test its driven use

From the susceptibility matrix, direct division yields

$$
\operatorname{Im}\widehat\chi_{00}(\Omega)=\frac{\mu\Omega(\omega^2-\gamma^2-\Omega^2)}{\mathcal D_\Omega}.
$$

At zero rotation, substitution of D=mu theta recovers the spectral equilibrium fluctuation–dissipation relation:

$$
\omega=0:\qquad S_{00}(\Omega)=-\frac{2\theta}{\Omega}\operatorname{Im}\widehat\chi_{00}(\Omega)\quad(\Omega>0).
$$

The factor two belongs to this two-sided spectrum and the chosen retarded transform. A one-sided convention needs its own normalization. The [Seifert review, Section 8.2](https://arxiv.org/pdf/1205.4176) distinguishes equilibrium response identities from stationary driven ones. The Fourier formulas here are derived from this chapter's linear thermal model.

For nonzero rotation, the same equilibrium replacement leaves a positive defect:

$$
S_{00}(\Omega)+\frac{2\theta}{\Omega}\operatorname{Im}\widehat\chi_{00}(\Omega)=\frac{4D\omega^2}{\mathcal D_\Omega}\quad(\Omega>0).
$$

Thus unchanged stationary variance does not restore the equilibrium frequency relation. For |omega|>gamma and sufficiently small positive Omega, the naive expression −2theta Im(chi00)/Omega is negative. The actual noise spectrum remains positive. This failure is a response-model mismatch; it is not a negative noise power or a negative bath temperature. At Omega=0 use a limit if needed, rather than dividing by zero.

### A driven response can return work to a probe

Apply force f0(t)=Re[F0 exp(i Omega t)] along coordinate 0, leaving the other probe component zero. Differentiate its periodic mean and average the product of force and mean displacement rate over a period:

$$
\overline P_{\rm probe}=\overline{f_0(t)\dot m_0(t)}=-\frac{|F_0|^2\Omega}{2}\operatorname{Im}\widehat\chi_{00}(\Omega),\qquad \overline P_{\rm probe}<0\iff\omega^2>\gamma^2+\Omega^2\quad(\Omega>0,\ F_0\ne0).
$$

This is the mean work rate of a smooth deterministic probe. Computing it from f0 times the mean derivative does not assign an ordinary velocity to Brownian sample paths. In equilibrium it is nonnegative. In the driven model it can be negative: the probe receives work while the nonconservative circulating force remains powered. This is only the probe's contribution, not the total work of all forces or a total heat balance.

For mu=kappa=theta=1 and omega=2/s, the sign changes at Omega=sqrt(3)/s. The bath is still specified by theta=1J. Inferring a universal temperature from the driven imaginary susceptibility would mix the response of an active steady state with an equilibrium identity.

![Driven OU noise remains positive while an equilibrium shortcut becomes negative; the probe work changes sign at a finite frequency.](/assets/learning/response-research-thermal-response-spectrum.en.svg)

The upper panel compares the actual coordinate spectrum with the equilibrium shortcut for the driven coefficients above. The lower panel shows probe work rate divided by force amplitude squared; negative values mean work returned to that probe. The external rotational drive is not included in that plotted contribution.

```python
for frequency in [0.25, 0.5, 1.0, 2.0]:
    mobility, stiffness, thermal, omega = 1.0, 1.0, 1.0, 2.0
    gamma = mobility * stiffness
    diffusion = mobility * thermal
    z = gamma + 1j * frequency
    chi00 = mobility * z / (z*z + omega*omega)
    spectrum = diffusion / (gamma**2 + (frequency - omega)**2)
    spectrum += diffusion / (gamma**2 + (frequency + omega)**2)
    naive = -2 * thermal * chi00.imag / frequency
    probe_power_per_force_squared = -frequency * chi00.imag / 2
    print(f"{frequency:.2f} {chi00.real:.6f} {chi00.imag:.6f} "
          f"{spectrum:.6f} {naive:.6f} "
          f"{probe_power_per_force_squared:.6f}")
```

Rows list Omega, real and imaginary chi00 in m/N, actual and naive S00 in m²·s, and probe power per squared force in m/(N·s). These are floating population values, not noisy measurements or directed-rounding certificates.

| Omega [1/s] | Re(chi00) | Im(chi00) | S00 | naive S00 | P/F² |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.205551 | 0.029818 | 0.411102 | -0.238541 | -0.003727 |
| 0.50 | 0.222812 | 0.058355 | 0.445623 | -0.233422 | -0.014589 |
| 1.00 | 0.300000 | 0.100000 | 0.600000 | -0.200000 | -0.050000 |
| 2.00 | 0.529412 | -0.117647 | 1.058824 | 0.117647 | 0.117647 |

### Integrating the spectrum recovers the same cloud

Circular polarization diagonalizes the spectrum, giving positive eigenvalues. Integration of each Lorentzian checks the normalization:

$$
\lambda_\pm(S)=\frac{2D}{\gamma^2+(\Omega\mp\omega)^2}>0,\qquad \frac1{2\pi}\int_{-\infty}^{\infty}S_{00}(\Omega)\,d\Omega=\frac D\gamma=v.
$$

Rotation redistributes spectral weight without changing its total integral. A histogram therefore misses both shifted frequency content and probe work. Resolving these from finite records additionally requires a sampling interval, observation duration, window convention and uncertainty analysis. The continuous spectrum is not automatically the spectrum of an aliased finite data record.

### Research self-checks

#### R1. Why does the equilibrium formula have a minus sign?

At omega=0, calculate Im(chi00) for positive Omega and recover S00.

**Research explanation.** Chi00=mu/(gamma+i Omega), so its imaginary part is−mu Omega/(gamma²+Omega²). Multiplication by−2theta/Omega gives2D/(gamma²+Omega²), the two-sided spectrum. Changing the input convention changes the response phase convention and must be tracked consistently.

#### R2. Can a negative shortcut certify negative noise?

Use gamma=D=mu=theta=1, omega=2/s and Omega=1/s. Compare the signs of the true spectrum and the shortcut.

**Research explanation.** Chi00=(1+i)/(4+2i)=0.3+0.1i in m/N. The true spectrum is1/2+1/10=0.6m²·s, while the shortcut is−0.2m²·s. Their defect is0.8m²·s. The actual covariance spectrum stays positive; the equilibrium substitution is inappropriate for this driven state.

#### R3. Where can the returned work come from?

For the same case with |F0|=1N, compute the probe work rate and explain why the stationary histogram is insufficient to interpret it.

**Research explanation.** The rate is−(1)(0.1)/2=−0.05W. The externally maintained rotational force can supply work to the probe. The Gaussian width remains v=1m² but does not describe that energy supply. This calculation does not give the total heat or an engine efficiency without including the other force contributions.

## Exercises with explanations

### 1. Recover the equilibrium kernel

Starting with C(t)=v exp(−gamma t)I, derive the force kernel and check its units.

**Explanation.**−C'/theta=(gamma v/theta)exp(−gamma t)I. Since v=theta/kappa and gamma=mu kappa, the prefactor is mu. Correlation derivative has units m²/s; division by theta in J=N·m gives m/(N·s), the units required before integrating a force over time.

### 2. Separate a step from an impulse at zero

For F=(1,0)N and impulse integral I_f=(0,1)N·s with mu=1, what are the right-limit mean changes?

**Explanation.** The step susceptibility starts at zero, so the step mean is(0,0)m. The impulse kernel starts at mu I, giving the immediate impulse mean(0,1)m. The equal numerical amplitudes have different units and act in different directions.

### 3. Catch a transverse sign error

For unit coefficients and omega=1, find the0,1 entries of the true kernel, naive prediction and current correction at zero lag.

**Explanation.** The true entry is0. C'01(0)=−1, so−C'01/theta=1. K01(0)=1, so−K01/theta=−1. Their sum is0. Using C transpose or omitting the current term breaks this check.

### 4. Explain a hotter cloud with the same mean response

Triple theta while holding mu, kappa, omega and the applied force fixed. What changes?

**Explanation.** D and v triple, so unperturbed correlations triple. A and the deterministic mean equation do not change, hence the impulse kernel and step mean remain the same. Dividing correlation-based expressions by the new theta restores the proper response. Real materials whose mobility or stiffness changes with temperature require a different comparison.

### 5. Reject a universal effective temperature

At t=1 the unit driven example gives diagonal diagnostic2.557408J and off-diagonal diagnostic0.357907J. Can one replace the bath theta by either number everywhere?

**Explanation.** No. The ratios depend on component and lag. The bath thermal energy was specified as1J. A negative off-diagonal ratio at another lag does not make the bath temperature negative. At a zero response denominator the ratio is undefined rather than a measured temperature.

### 6. Distinguish a model identity from data verification

The corrected response residual is exactly zero, while a measured response contains noise. What does the zero prove, and how should an unresolved tiny response be handled?

**Explanation.** The zero follows from the stated OU generator, thermal relation and algebraic simplification. It does not prove that measured data follow that model. Keep an unresolved linear response separate from a structural zero, inspect its logarithm and enclosure, and report uncertainty in any data-derived correlation derivative or ratio.
