## Ask which response is resonating

“The resonance frequency” is incomplete unless the observable is specified. The displacement amplitude, free damped oscillation frequency and absorbed-power maximum generally occur at different frequencies. This distinction matters when identifying a sensor, designing a filter or validating a learned physical model.

We study $m\ddot x+c\dot x+kx=F_0\cos(\Omega t)$ with $m,k>0$, $c\ge0$ and $F_0\ge0$. Read the transient-response chapter first: a periodic particular solution does not describe arbitrary initial motion immediately after switching on. Here a frequency curve compares separate experiments with the same force amplitude and different drive frequencies.

## Keep angular frequency and physical units visible

Mass is in kg, damping in N s/m, stiffness in N/m, force in N, displacement in m and angular frequency in rad/s. Frequency in cycles per second is $f=\Omega/(2\pi)$. A bandwidth in rad/s must also be divided by $2\pi$ before reporting it in Hz.

Force amplitude is held fixed during the frequency sweep. Fixing displacement, acceleration or supplied power instead would define a different experiment and can change its response curve. We use viscous damping and an ideal force source; actuator limits and nonlinear saturation are outside this model.

## Derive the complex susceptibility

Represent physical quantities by real parts using $e^{+i\Omega t}$. Substituting $x_p=\operatorname{Re}(F_0\chi e^{i\Omega t})$ gives

$$
\chi(\Omega)=\frac{1}{k-m\Omega^2+ic\Omega}
=\frac{k-m\Omega^2-ic\Omega}{D},\qquad
D=(k-m\Omega^2)^2+c^2\Omega^2.
$$

Susceptibility has units m/N. Its imaginary part is nonpositive under this convention. Using $e^{-i\Omega t}$ conjugates the complex expression; the physical motion and its positive lag do not change when the convention is changed consistently.

## Gain and phase describe different aspects

For a nonsingular transfer function,

$$
G=|\chi|=D^{-1/2},\qquad A=F_0G,\qquad
\phi=\operatorname{atan2}(c\Omega,k-m\Omega^2),\qquad
x_p=A\cos(\Omega t-\phi).
$$

Below the natural frequency, the spring term dominates and displacement approximately follows force. Far above it, inertia dominates, the gain decreases as $1/(m\Omega^2)$ and displacement is nearly opposite to force. At $\Omega=\omega_0=\sqrt{k/m}$ with $c>0$, lag is $\pi/2$.

If $F_0=0$, there is no driven displacement whose phase can be measured. The intrinsic transfer function can still be defined. The model therefore keeps intrinsic lag separate from measured response phase and uses an unavailable value for the latter at zero input or DC. At DC, the intrinsic value denotes the limiting transfer phase.

## Locate the displacement-amplitude peak

At fixed nonzero force amplitude, maximizing $A$ minimizes $D$. Differentiate:

$$
\frac{dD}{d\Omega}=2\Omega\left(2m^2\Omega^2+c^2-2mk\right).
$$

For $c>0$, a positive interior maximum exists only when $c^2<2mk$:

$$
\Omega_x=\sqrt{\frac{k}{m}-\frac{c^2}{2m^2}}.
$$

At or beyond the boundary $c^2=2mk$, the maximum over nonnegative frequencies is at DC. This is different from the underdamped condition $c^2<4mk$. There is an interval where free motion oscillates but the displacement frequency curve has no positive interior peak.

At $c=0$ the transfer is singular at $\omega_0$. Calling it a finite amplitude maximum would hide the absence of a bounded periodic response.

## Compute power using real force and real velocity

Velocity is $v_p=A\Omega\sin(\phi-\Omega t)$. Multiply it by the real force, then average over $T=2\pi/\Omega$ for $\Omega>0$:

$$
\langle P\rangle=\frac{1}{T}\int_0^T F_0\cos(\Omega t)v_p(t)\,dt
=\frac{F_0A\Omega}{2}\sin\phi
=\frac{F_0^2c\Omega^2}{2D}.
$$

The factor $1/2$ comes from averaging a product of sinusoids. Multiplying two complex signals without the appropriate real-part/conjugation rule can produce a wrong sign or miss this factor. The independently computed mean heat rate is $c\langle v_p^2\rangle=cA^2\Omega^2/2$, which agrees with the mean input power.

## Why the power maximum occurs at the natural frequency

For $c>0$, $F_0>0$ and $\Omega>0$, rewrite the mean power as

$$
\langle P\rangle=\frac{F_0^2c}{2\{(k/\Omega-m\Omega)^2+c^2\}}.
$$

The square in the denominator vanishes at $\Omega=\omega_0$. Hence the peak is $F_0^2/(2c)$ W. The velocity amplitude also peaks there. This statement is exact for this force-driven linear model, including strong damping; it is not the displacement-peak formula.

At this frequency, velocity is in phase with force. Large displacement alone does not guarantee large energy transfer: its velocity and timing relative to force matter.

## Derive the exact half-power frequencies

Half of the maximum power requires $|k/\Omega-m\Omega|=c$. The two positive roots are

$$
\Omega_- =\frac{\sqrt{c^2+4mk}-c}{2m},\qquad
\Omega_+ =\frac{\sqrt{c^2+4mk}+c}{2m}.
$$

Therefore $\Delta\Omega=\Omega_+-\Omega_-=c/m$ exactly. Their product is $k/m=\omega_0^2$, so the natural frequency is their geometric mean. Their arithmetic midpoint is generally larger than $\omega_0$; the half-power frequencies are not generally symmetric around it on a linear frequency axis.

The implementation evaluates the lower root as $2k/(\sqrt{c^2+4mk}+c)$ to avoid subtracting close numbers. The formula concerns half power for a fixed force amplitude. It is not an exact bandwidth formula for every possible displacement or transfer observable.

## Quality factor and cycle energy

For $c>0$, define $Q=m\omega_0/c=\omega_0/\Delta\Omega$. This parameter remains defined for strong damping even when an interior displacement peak disappears. At zero damping the model returns no finite quality-factor value instead of serializing infinity as a number.

For $\Omega>0$, the mean stored mechanical energy is

$$
\langle E\rangle=\frac{m\Omega^2+k}{4}A^2,
\qquad E_{\rm heat,cycle}=\frac{2\pi}{\Omega}\langle P\rangle.
$$

At resonance, $2\pi\langle E\rangle/E_{\rm heat,cycle}=Q$. Away from resonance, that energy ratio is generally a different quantity. Do not replace the definition with an identity that holds only at the resonant drive.

## DC, no damping and zero input need different labels

At $\Omega=0$, the particular response is the static displacement $F_0/k$. There is no finite cycle, so period, heat per cycle and cycle-averaged stored energy are unavailable. Its actual static energy is $F_0^2/(2k)$; using the sinusoidal half-square average would give the wrong DC value. The asymptotic power is zero.

At $c=0$ away from resonance, a periodic particular solution exists and its mean input power is zero, but initial free motion does not decay. At exact resonance with nonzero force there is no bounded periodic response. If force is also zero, the driven amplitude is zero while the intrinsic transfer is still singular. Arbitrary initial free oscillation can remain present in that zero-input experiment. The cycle preview can select the identically zero particular solution without assigning a finite intrinsic gain. With zero input, power is zero at every frequency, so a distinct half-power bandwidth is unavailable; the intrinsic quality-factor parameter remains defined for positive damping.

## Worked example: three different frequencies

Take $m=k=c=F_0=1$ in SI units. The natural frequency and power peak are 1 rad/s. The free damped frequency is $\sqrt{3}/2\approx0.866$ rad/s, and the displacement peak is $1/\sqrt2\approx0.707$ rad/s.

At the displacement peak, $A=2/\sqrt3\approx1.155$ m and mean power is $1/3$ W. At the power peak, $A=1$ m and mean power is $1/2$ W. A smaller displacement can therefore correspond to greater mean energy transfer. The half-power frequencies are $(\sqrt5-1)/2\approx0.618$ and $(\sqrt5+1)/2\approx1.618$ rad/s, with width 1 rad/s.

## Worked example: oscillatory decay without a displacement peak

Let $m=1$, $k=2$, $c=2$, $F_0=1$ in SI units. Since $c^2=4<4mk=8$, the homogeneous motion is underdamped with $\omega_d=1$ rad/s. However, $c^2=2mk$, so the displacement gain is maximal at DC and has no positive interior peak. The power curve still peaks at $\sqrt2$ rad/s.

This example distinguishes a statement about motion after releasing the mass from a statement about comparing many driven experiments.

## Plan a frequency-response visualization

The interactive view provides separate gain and mean-power panels with the same horizontal frequency axis. Select the displacement-versus-power example, then compare the peak markers with the table. Mark $\omega_d$, $\Omega_x$ when available, $\omega_0$ and the two half-power frequencies. Display unavailable markers as text rather than moving them to zero. A phase panel should show radians and the chosen complex convention.

The selected frequency represents one experiment in this family. A curve joining frequencies is not the time trajectory of a single mass. Pair the frequency view with an actual time trace when discussing startup. Preserve the existing site colors and distinguish curves by labels as well as color.

## Research extension: positive absorption has a conserved spectral area

Peak height and bandwidth describe only part of a response curve. A narrow peak can become arbitrarily tall without acquiring infinite total area. Here we prove a full positive-frequency integral for this ideal viscously damped oscillator, then ask what a finite measurement band actually captures. This is a derived supplement to the existing gain and power laboratory, not a broadband measurement feature.

### Fix the sign and physical observable



$$
\chi_c(\Omega)=\frac1{k-m\Omega^2+ic\Omega},\quad m,k,c>0,\quad \Omega\ge0,\qquad
\mathcal A_c(\Omega)=\Omega[-\operatorname{Im}\chi_c(\Omega)]=\frac{c\Omega^2}{(k-m\Omega^2)^2+c^2\Omega^2}.
$$

The convention remains exp(+i Omega t), so the imaginary part of susceptibility is negative at positive frequency. Its negative sign is not active energy generation. Multiplying by frequency converts it to the nonnegative absorption weight:



$$
\langle P\rangle=\frac{F_0^2}{2}\mathcal A_c(\Omega)\ge0,\qquad
[\chi_c]=\mathrm{s^2/kg},\quad [\mathcal A_c]=\mathrm{s/kg},\quad \left[\int\mathcal A_c\,d\Omega\right]=\mathrm{kg^{-1}}.
$$

This is periodic steady-state input power; it is not a claim that instantaneous startup power is always positive. The real-power averaging and its convention follow the preceding derivation and [MIT 8.03SC chapter 2](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/782069da3820fc514c10c26ae0c15b01_MIT8_03SCF16_Text_Ch2.pdf). The integral proof below is supplied explicitly for the present scalar model.

### Derive the area instead of estimating a peak rectangle



$$
\int_0^\infty\mathcal A_c(\Omega)\,d\Omega=\frac\pi{2m},\qquad
\int_0^\infty\langle P\rangle\,d\Omega=\frac{\pi F_0^2}{4m}\quad(F_0\text{ fixed across experiments}).
$$

The integral is independent of positive damping and stiffness, including critical and overdamped cases. It integrates separate periodic experiments at equal force amplitude. It is not heat accumulated during a time sweep. To prove it, put s=Omega/omega0 and gamma=c/(m omega0)>0:



$$
\omega_0=\sqrt{k/m},\quad \gamma=\frac{c}{m\omega_0},\qquad
\int_0^\infty\mathcal A_c\,d\Omega=\frac{c}{m^2\omega_0}J_2,\quad
J_j=\int_0^\infty\frac{s^j\,ds}{s^4+(\gamma^2-2)s^2+1}\quad(j=0,2).
$$

The positive quartic denominator makes both integrals finite. Substitute s=1/r in J0 to obtain J0=J2. Add them, then use u=s−1/s, whose derivative is 1+1/s²:



$$
2J_2=\int_0^\infty\frac{1+s^2}{s^4+(\gamma^2-2)s^2+1}\,ds
=\int_{-\infty}^\infty\frac{du}{u^2+\gamma^2}=\frac\pi\gamma.
$$

The prefactor now cancels gamma and gives pi/(2m). No small-damping approximation or symmetric half-power assumption was used. Changing mass changes the area; changing damping redistributes it.

### A vanishing pointwise response can keep finite spectral weight



$$
\Omega\ne\omega_0:\quad \mathcal A_c(\Omega)\longrightarrow0,\qquad
\mathcal A_c(\Omega)\,d\Omega\ \rightharpoonup\ \frac\pi{2m}\delta_{\omega_0}\quad(c\downarrow0)\text{ on }(0,\infty).
$$

The last statement means convergence after integration against a continuous compactly supported test function. Away from any neighborhood of omega0, the integral tends to zero: on a bounded region the denominator stays away from zero, and the far tail is bounded as below. Total mass stays pi/(2m). Inside a shrinking neighborhood the peak becomes tall and narrow. This proves the stated weak limit; it is not the ordinary value of a bounded periodic solution at c=0 and exact resonance. There is no damping-independent integrable dominating curve, so interchanging pointwise limit and integration is invalid.

### Report the actual observation band



$$
C_c(a,b)=\frac{2m}{\pi}\int_a^b\mathcal A_c(\Omega)\,d\Omega,\quad 0\le a<b,\qquad 0<C_c(a,b)<1.
$$

The table fixes m=k=1 in SI units. It compares a narrow band around 1 rad/s with the native frequency range 0–8 rad/s. Values are numerical fractions of the proved ideal full area, not independent measurements of that full area. Composite Simpson refinement is checked separately.

| c (N s/m) | fraction in 0.8–1.2 rad/s | fraction in 0–8 rad/s | ideal full fraction |
|---|---|---|---|
| 0.05 | 0.920064 | 0.995979 | 1 |
| 0.2 | 0.702791 | 0.983920 | 1 |
| 1 | 0.241925 | 0.920008 | 1 |
| 4 | 0.063445 | 0.702273 | 1 |

```python
from math import pi

# m=k=1 in SI units. Fractions of the proved full positive-frequency area.
def integrate(f, left, right, n=20000):
    h = (right-left)/n
    return h/3*(f(left)+f(right)+sum(
        (4 if i % 2 else 2)*f(left+i*h) for i in range(1, n)))

for c in [.05, .2, 1, 4]:
    weight = lambda w: (2/pi)*c*w*w/((1-w*w)**2+c*c*w*w)
    narrow = integrate(weight, .8, 1.2)
    wide = integrate(weight, 0, 8)
    print(c, round(narrow, 6), round(wide, 6), 1)
    assert 0 < narrow < wide < 1
```



$$
W\ge\sqrt2\,\omega_0:\qquad 0\le\int_W^\infty\mathcal A_c(\Omega)\,d\Omega\le\frac{4c}{m^2W}.
$$

For that tail bound, k≤m Omega²/2 implies |k−m Omega²|≥m Omega²/2, so the integrand is at most 4c/(m² Omega²). The bound is conservative and can exceed the whole area. Finite-band agreement alone cannot certify an infinite-band sum rule or exclude unobserved modes. A real actuator or sensor may cease to obey this ideal viscous single-mode model at high frequency.

![Normalized absorption becomes narrower and taller as damping decreases, while its positive-frequency area remains one; finite-band cumulative fractions stay below one.](/assets/learning/waves-resonance-spectral-area.en.svg)

### Research questions with explanations

**Does a negative imaginary susceptibility violate passivity?** **Answer:** Not under exp(+i Omega t). Positive-frequency mean input power uses −Omega Im chi and is nonnegative; changing the Fourier convention also changes the imaginary sign.

**Can a model pass every sampled off-resonance value yet miss the area?** **Answer:** Yes. A progressively narrower peak can lie between samples while pointwise values away from resonance vanish. Resolve the peak, report integration refinement and bound omitted tails.

**Does this prove a sum rule for every learned sensor?** **Answer:** No. The proof assumes one linear viscous oscillator over all positive frequencies with calibrated physical susceptibility. Finite data, hidden modes, nonlinearity and measurement scaling require separate identification and physical assumptions.

## Six exercises with explanations

1. **Choose the observable:** For $m=k=c=1$, which frequency maximizes displacement? **Answer:** $1/\sqrt2$ rad/s. The value 1 rad/s maximizes mean absorbed power instead.
2. **Phase quadrant:** At $m=k=c=1$, $\Omega=2$, calculate $\chi$. **Answer:** $\chi=(-3-2i)/13$ m/N. Lag is $\operatorname{atan2}(2,-3)>\pi/2$; a negative one-argument arctangent would choose the wrong quadrant.
3. **Convert bandwidth:** If $c/m=1$ s⁻¹, what is the half-power width in Hz? **Answer:** $1/(2\pi)\approx0.159$ Hz. Do not report 1 Hz from a width of 1 rad/s.
4. **Test half power:** For $m=k=c=F_0=1$, evaluate the power at either exact half-power root. **Answer:** $1/4$ W, half the peak $1/2$ W. Their separation is 1 rad/s and their product is 1 (rad/s)².
5. **DC energy:** With $F_0=k=1$, find the static stored energy. **Answer:** $1/2$ J. The formula $kA^2/4$ assumes a nonzero-frequency sinusoidal average and would incorrectly give $1/4$ J here.
6. **Zero force:** Does a zero measured driven amplitude mean zero intrinsic gain? **Answer:** No. A finite intrinsic gain can multiply zero force, and free motion from initial conditions can still be nonzero. At undamped resonance, the transfer is singular even when input is zero.

## Connections to identification and learned dynamics

In a system-identification experiment, record the controlled input and measured output. A network trained only on displacement peaks could confuse damping with an observation scale or miss the phase needed for power prediction. Validate complex response, finite-time startup and energy transfer separately.

The independent fixtures average real force times real velocity by SciPy quadrature and compare the result with independently averaged dissipation. Analytic peak identities, limiting cases and initial-value tests complement that numerical agreement. A successful oscillator benchmark does not by itself establish performance on nonlinear, coupled or noisy systems.

## Sources and continuation

The conditional displacement peak follows [MIT 18.03SC frequency response, pp. 1–2](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/4f6ba441e3d09f9e87b78dedb89fb1dc_MIT18_03SCF11_s19_2text.pdf). For real power averaging and half-power width, see [MIT 8.03SC chapter 2, sections 2.2–2.3](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/782069da3820fc514c10c26ae0c15b01_MIT8_03SCF16_Text_Ch2.pdf). The complex convention is stated explicitly above. Next come coupled oscillations and normal modes, where several coordinates and mode participation replace this single response channel.
