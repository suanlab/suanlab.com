# First waves topic: forcing, damping and resonance

2026-11-05. Two model functions and four bilingual manuscripts are now authored; see [foundation report](waves-oscillations-foundation.md). Both units are now registered and connected to native figures, controls, guides and quizzes; current bilingual browser, accessibility, interaction and full English verification have passed; see [integration report](waves-oscillations-authoring-status.md). The model/manuscript milestone remains separately recorded. The original topic `조화진동·감쇠·공진` has passed introductory verification. Coupled oscillations and normal modes are next; see [their source audit](coupled-modes-source-audit.md).

## Existing material and the actual gap

`damped-oscillator-work-power-heat` solves a **constant** drive, shifted equilibrium and signed work/heat. Its manuscript explicitly excludes periodic forcing. The existing oscillator integrators, stationary-action and Hamilton-Jacobi chapters answer different questions. New content must derive sinusoidal forcing, initial-condition transients, frequency response and resonant energy transfer.

## Primary references actually consulted

- [MIT 18.03SC frequency response text, pp.1–2](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/4f6ba441e3d09f9e87b78dedb89fb1dc_MIT18_03SCF11_s19_2text.pdf): gain, phase and the conditional displacement-amplitude maximum. Its gain peak moves below the natural frequency and disappears at sufficiently strong damping.
- [MIT 8.03SC lecture3, pp.1–5](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/d07bffb5150376ad8bc942566f327f1d_MIT8_03SCF16_Lec3.pdf): damped homogeneous regimes, sinusoidal particular solution and initial-condition-dependent transient. Check equations directly rather than copying imperfect PDF text extraction; distinguish natural, damped and driving frequencies.
- [MIT 8.03SC chapter2, sections2.2–2.3](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/782069da3820fc514c10c26ae0c15b01_MIT8_03SCF16_Text_Ch2.pdf): complex forcing convention, real force times real velocity, absorbed power and half-power width. The chapter uses exp(-i omega t); any product convention using exp(+i omega t) must change the imaginary sign consistently.

## Derivations to implement and independently verify

For `m xddot+c xdot+k x=F0 cos(Omega t)`, positive m,k and nonnegative c, derive `D=(k-m Omega²)²+c² Omega²`, displacement amplitude `F0/sqrt(D)` and lag `atan2(c Omega,k-m Omega²)`. The positive interior displacement peak exists only at `Omega_x²=k/m-c²/(2m²)>0`. This is distinct from both the damped free frequency and the absorbed-power peak.

From the real periodic response derive mean power `F0² c Omega²/(2D)`. For c>0 its maximum is at `sqrt(k/m)`. Half-power roots are `(sqrt(c²+4mk) ± c)/(2m)`, so their separation is `c/m`. These statements should be checked symbolically and against independent quadrature; a finite startup window is not a periodic steady-state average.

At c=0 and matched frequency there is no bounded sinusoidal particular response, while the finite-time zero-initial-data solution is `F0 t sin(omega0 t)/(2m omega0)`. Near this case, use an analytically continued sinc representation or an independent augmented-system solution to avoid cancellation. A mathematical limit is different from clamping invalid input. DC has no finite cycle period. Zero forcing, zero response and undefined measured phase require distinct labels.

Candidate pair: `sinusoidal-forcing-transient-response` for actual initial-value evolution and beats/resonant buildup; `oscillator-frequency-response-power-resonance` for complex susceptibility, phase, displacement versus power peaks and width. Both identifiers now have detailed KO/EN manuscripts with theory, worked examples and six explained exercises, plus independently checked models. Native visualizations, controls, registry integration and browser verification remain pending. Preserve the current442-unit registry and existing theme/static export. Do not use coupled normal modes to prematurely credit the next topic.
