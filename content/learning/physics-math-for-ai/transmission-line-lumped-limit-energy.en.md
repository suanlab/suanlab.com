## Does a terminal prediction determine internal energy?

Suppose a neural model predicts voltage and current at the entrance of a cable. Has it also learned the energy inside the cable? A transmission line stores electric and magnetic energy throughout space. A small circuit fitted to one terminal response need not reproduce that internal state. We will compare three questions for the same lossless line: its input response, its spatial fields, and its stored energy. Keeping these targets separate makes effective modeling more useful and makes its limitations easier to test.

## Assumptions and prerequisites

Take constant positive inductance L in H/m and capacitance C in F/m, length ℓ, an open right end, and a prescribed real input-voltage phasor Vin. Use exp(+iωt). Resistance, radiation, material dispersion, and switch-on transients lie outside this family. In a strictly lossless system a switch-on transient does not automatically decay. We are selecting a sinusoidal boundary-value solution, not asserting that every initial condition evolves toward it. Review complex phasors, derivatives, capacitor energy, and inductor energy before continuing.

## From a short segment to the telegrapher equations

A segment dx has series inductance L dx and shunt capacitance C dx. Positive current points to the right. The voltage drop along the segment drives its inductance, while the current lost along it charges its capacitance. Divide these balances by dx and take the limit. Drawing the incoming and outgoing currents helps more than memorizing the signs. Combining the two first-order relations gives a wave equation. These relations describe the distributed model that the lumped approximation will later compress.

$$
\partial_x V=-L\partial_t I,\qquad \partial_x I=-C\partial_t V,\qquad \partial_x^2 V=LC\partial_t^2 V.
$$

## Speed, impedance, and electrical length

Wave speed and characteristic impedance encode different information. The product LC sets travel time; the ratio L/C sets the voltage-to-current ratio of a traveling wave. Electrical length θ is the phase accumulated in one traversal. A cable is electrically short when θ is small at the frequency of interest, not merely when its length in metres looks small. Holding θ fixed while changing physical length changes the frequency. Read normalized phase and SI frequency as separate quantities.

$$
c=(LC)^{-1/2},\quad Z=\sqrt{L/C},\quad k=\omega/c,\quad \theta=k\ell=\omega\ell/c.
$$

## The open boundary and spatial solution

At an open end I(ℓ)=0. A cosine measured back from that end satisfies this condition once current is obtained from the telegrapher equations. Normalizing V(0) to Vin introduces a division by cosθ. Away from an incompatible resonance this is the distributed solution, rather than a lumped approximation. Its voltage phasor is real and its current phasor imaginary. Their instantaneous maxima therefore occur at different times. Here u=x/ℓ is a position fraction, so its range is zero to one.

$$
\widehat V(x)=V_{in}\frac{\cos[\theta(1-u)]}{\cos\theta},\qquad \widehat I(x)=i\frac{V_{in}}Z\frac{\sin[\theta(1-u)]}{\cos\theta},\quad u=x/\ell.
$$

## Input admittance at the terminal

Dividing input current by input voltage gives admittance Y. For this lossless family its real part vanishes, so it consumes no cycle-averaged power. Denote the imaginary part by B, or susceptance. For small θ, tanθ is approximately θ and the terminal looks like a capacitor Cℓ. This statement concerns a low-frequency port relation. It does not say that voltage is exactly uniform inside the line or that magnetic energy is exactly zero. An effective model must state which observation it approximates.

$$
Y_{in}=iB=i\frac{\tan\theta}{Z},\qquad Y_{lumped}=i\omega C\ell=i\frac{\theta}{Z}.
$$

## Reading a small correction without losing it

The relative correction is tanθ/θ−1. At θ=10⁻¹² it is about 3.33×10⁻²⁵, but a floating-point ratio can round to one before subtraction. Evaluate the small correction directly with a series. This preserves information in a specified analytic family; it does not recover information lost from measured data. If a nonzero result falls outside representable range, report it as unavailable rather than an exact zero. A displayed zero and a proved identity are different claims.

$$
\frac{\tan\theta}{\theta}-1=\frac{\theta^2}{3}+\frac{2\theta^4}{15}+\frac{17\theta^6}{315}+O(\theta^8).
$$

## Instantaneous power and local energy conservation

A lossless line can still exchange instantaneous power with its source. Energy enters during part of a cycle and returns during another part. Recover real time-domain voltage and current from the phasors, then multiply the telegrapher equations to obtain local energy conservation. At the open end current and power vanish. The conservation law belongs to this distributed line model. It does not by itself reconstruct every electromagnetic field surrounding a physical cable, its connectors, and its source.

$$
u(x,t)=\tfrac12CV(x,t)^2+\tfrac12LI(x,t)^2,\qquad \partial_t u+\partial_x(VI)=0.
$$

## Integrating cycle-averaged energy in space

For positive frequency the cycle average of a squared cosine is one half. Consequently phasor-amplitude energy densities use a factor of one quarter. Compute electric and magnetic contributions at each point and integrate along the line. Near resonance the internal standing wave can store a large energy even with input voltage held fixed. Comparing the spatial densities, rather than only one scalar total, makes it easier to see what a terminal approximation keeps and what it discards.

$$
\overline u_E=\tfrac14 C|\widehat V|^2,\quad \overline u_M=\tfrac14 L|\widehat I|^2,\quad \overline U=\int_0^\ell(\overline u_E+\overline u_M)dx=\frac{C\ell V_{in}^2}{4\cos^2\theta}.
$$

## Separating electric and magnetic storage

Integrating cosine squared and sine squared separates the two storage terms. Define sinc z as sin z/z, with value one at zero by continuity. Magnetic storage becomes small at low frequency but is generally nonzero. The difference 1−sinc can itself suffer cancellation, so a small-argument series is useful here too. An accurately representable total energy does not guarantee that its much smaller magnetic contribution is representable. Keep the availability of individual quantities separate.

$$
\overline U_E=\frac{\overline U}{2}[1+\operatorname{sinc}(2\theta)],\qquad \overline U_M=\frac{\overline U}{2}[1-\operatorname{sinc}(2\theta)].
$$

## The trap in fitting capacitance at one frequency

Dividing B by ω gives a fitted capacitance that reproduces input current at that frequency. Treating that number as a frequency-independent capacitor gives the wrong line energy. Above a resonance Cfit can even be negative. This is an inductive terminal response written in capacitor notation, not a newly discovered positive passive electrostatic capacitor. A reduced model needs a frequency domain and physical realizability conditions. Excellent fit at a single frequency is not enough to identify its internal storage interpretation.

$$
C_{fit}(\omega)=\frac{B(\omega)}\omega=C\ell\frac{\tan\theta}{\theta},\qquad \overline U_{fit,\,nondispersive}=\tfrac14 C_{fit}V_{in}^2.
$$

## What the frequency derivative adds

For this lossless open line, one quarter of input amplitude squared times the frequency derivative of B equals the actual average energy. The value at one frequency is insufficient; its neighborhood carries additional information. The relation below follows directly for this specified boundary problem with fixed input amplitude. Do not apply it unchanged to arbitrary lossy devices, multiple ports, or dispersive materials. Estimating the derivative from data also introduces a frequency-step choice and measurement noise.

$$
\frac{dB}{d\omega}=C\ell\sec^2\theta,\qquad \overline U=\frac{V_{in}^2}{4}\frac{dB}{d\omega}.
$$

## A worked one-metre example

Let L=250 nH/m, C=100 pF/m, ℓ=1 m, and Vin=1 V. Then c=2×10⁸ m/s and Z=50 Ω. At θ=0.2, ω=4×10⁷ rad/s, or about 6.37 MHz. The low-frequency lumped capacitor is 100 pF; the single-frequency fit is about 101.355 pF. Actual average energy is about 26.027 pJ, whereas interpreting the fit as nondispersive gives about 25.339 pJ. Both are plausible small numbers, but they answer different questions. Compare terminal and energy errors separately.

## Distinguishing exact resonance from nearby input

At θ=(2m+1)π/2, the open-end condition forces the input voltage to vanish. A nonzero prescribed Vin is incompatible with that sinusoidal boundary problem. With zero Vin the standing-wave amplitude is free, so the solution is nonunique. A nearby finite binary input is different from this exact symbolic condition. The model accepts these as distinct requests. It does not convert an arbitrary tolerance band into exact resonance. Large amplification near the pole indicates sensitivity, not a uniquely selected finite resonant state.

$$
\cos\theta=0:\quad V_{in}\ne0\Rightarrow\text{incompatible},\qquad V_{in}=0\Rightarrow\text{nonunique}.
$$

## DC and the low-frequency average are different limits

At exact DC the line has constant voltage Vin, zero current, and energy CℓVin²/2. Averaging over complete cycles at positive frequency and then taking ω toward zero gives CℓVin²/4 instead. Even a very slow cosine traverses its whole positive and negative range over a complete cycle. The order of taking the observation interval and frequency limits matters. A cycle average is not defined at exact DC, so the model reports static energy and the positive-frequency limiting average separately.

$$
U_{DC}=\tfrac12C\ell V_{in}^2,\qquad \lim_{\omega\to0^+}\overline U=\tfrac14C\ell V_{in}^2.
$$

## Questions to carry into AI research

When learning a reduced model, specify the observation, frequency band, boundary conditions, and hidden state. A model fitted to V/I at one frequency has a different validation scope from one predicting broadband slopes and spatial storage. Agreement between formulas from the same family is not independent certification. Compare reverse ODE integration, spatial quadrature, frequency differences, and high-precision references separately. [Haus and Melcher §14.5](https://web.mit.edu/6.013_book/www/chapter14/14.5.html) supplies background on sinusoidal transmission lines and terminations. Adding loss or switch-on transients requires new assumptions and validation.

## Check your understanding with worked exercises

### Exercise 1

Compute speed and impedance for L=250 nH/m and C=100 pF/m.

**Answer:** LC=2.5×10⁻¹⁷ s²/m² gives c=2×10⁸ m/s. L/C=2500 Ω² gives Z=50 Ω. The product and ratio control different physical quantities.

### Exercise 2

For a one-metre line at θ=0.2, find the lumped capacitance and input susceptance.

**Answer:** Total capacitance is 100 pF. B=tan(0.2)/50≈0.00405420 S. The lumped value is θ/Z=0.004 S, leaving a relative correction of about 1.355%.

### Exercise 3

With L=C=ℓ=Vin=1 and θ=0.8, which two energy calculations should agree?

**Answer:** The spatial integral of electric plus magnetic energy agrees with (1/4)dB/dω, about 0.51504. Cfit/4 is about 0.32176 instead. Matching a terminal response does not automatically match storage.

### Exercise 4

At an explicitly exact θ=π/2, what changes between Vin=1 and Vin=0?

**Answer:** Vin=1 is incompatible with the open boundary. Vin=0 permits an arbitrary standing-wave amplitude and is nonunique. Neither case justifies inventing a unique finite energy or selecting zero energy without extra data.

### Exercise 5

If direct tanθ/θ−1 is zero at θ=10⁻¹², is the lumped approximation exact?

**Answer:** No. The leading correction θ²/3 is about 3.33×10⁻²⁵. The ratio may have rounded to one. Evaluate the correction directly; if it is unrepresentable, mark it unavailable rather than an exact identity.

### Exercise 6

Compare DC energy and the low-frequency cycle-average limit for a 100 pF line at one-volt amplitude.

**Answer:** Exact DC stores 50 pJ; the positive-frequency cycle-average limit is 25 pJ. The latter averages a complete cosine cycle however slow it is. The observation and frequency limits are taken in different orders.
