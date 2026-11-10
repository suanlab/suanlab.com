## What does coil voltage observe?

Suppose a small coil produces a measured voltage in a magnetic field. Can that number immediately be labelled as the value of B at one point? An ideal induction coil observes flux change and the effects of wire motion. Its finite area, spatial field variation, trajectory, and initial information all matter when inferring a field. We start from area and line integrals rather than calling every voltage a magnetic-field measurement. This will distinguish a local field value, a spatial average, a time derivative, and an identifiable model parameter.

## Declare the observation model and assumptions

Use a circular material loop of radius R, normal plus z, and counterclockwise traversal viewed from above. Treat N turns as identical ideal loops. Loading, backreaction, resistance, distributed capacitance, and the readout circuit are omitted. The loop translates in the xy plane while preserving its shape and normal. Rotation, deformation, and changing sliding contacts require separate treatment. Review gradients and curls, Faraday law, Lorentz force, and matrix rank. Declaring the probe assumptions is necessary before comparing its output with a measured voltage from a real sensor.

## Build a smooth spatially varying field

Let B point along z and vary quadratically with x and y. B0 has units of tesla and g has units of tesla per square metre. Since the field is independent of z, its divergence vanishes. Use quadratic time polynomials so that first and second time derivatives are explicit. This is a smooth local field family requiring sources and boundaries, not a finite-energy field over all space. The spatial variation is simple enough for exact disk averages, while still exposing the error made by replacing an extended probe with its centre point.

$$
B_z=B_0(t)+g(t)(x^2+y^2),\quad B_0=b_0+b_1t+\tfrac12b_2t^2,\quad g=g_0+g_1t+\tfrac12g_2t^2.
$$

## Faraday law and the required sources

Choose the azimuthal electric field to satisfy Faraday law. A Cartesian expression remains regular on the axis, avoiding division by r. Ampere law then requires a distributed current. Using vacuum coefficients does not turn this configuration into source-free radiation. Compute J from the actual E and B rather than omitting it because the probe is passive. The passive-probe approximation concerns the sensor contribution; it does not erase the external sources maintaining the prescribed field. This distinction connects the measurement model back to local Maxwell consistency.

$$
E_\phi=-\tfrac12B_0^{\prime}r-\tfrac14g^{\prime}r^3,\quad \rho=0,\quad J_\phi=-\frac{2gr}{\mu}+\epsilon\left(\tfrac12B_0^{\prime\prime}r+\tfrac14g^{\prime\prime}r^3\right).
$$

## Voltage as an oriented force circulation

For a fixed loop use electric circulation. For a translating material loop use the circulation of E plus v cross B, with the declared orientation. When that same material loop moves smoothly, the total emf agrees with minus the flux-linkage rate. This rule should not be transferred indiscriminately to paths whose material changes through sliding contacts. N multiplies both voltage and flux linkage. Separating the electric and motional terms makes the sign check concrete: each comes from a different term of force per unit charge, even though they contribute to the same terminal observable.

$$
\mathcal E=N\oint_{\Gamma(t)}(\mathbf E+\mathbf v\times\mathbf B)\cdot d\boldsymbol\ell=-\frac{d(N\Phi)}{dt},\qquad \Phi=\int_{S(t)}\mathbf B\cdot\mathbf n\,dA.
$$

## The spatial average over a finite disk

Write each point in the disk as its centre plus a displacement. Symmetry makes the mean displacement zero; its mean squared radius is R squared over two. The disk-average B therefore differs from the centre value whenever curvature is nonzero. This difference is a genuine consequence of finite spatial support, not numerical integration error. A smaller probe reduces the correction but can also reduce voltage gain and worsen sensitivity to voltage noise. Area averaging and measurement uncertainty should therefore be assessed together rather than assuming the smallest possible probe is always best.

$$
\langle x^2+y^2\rangle_S=|\mathbf o|^2+R^2/2,\qquad \Phi=\pi R^2\left[B_0+g\left(|\mathbf o|^2+R^2/2\right)\right].
$$

## What a fixed coil observes over time

For a stationary coil the centre and radius are constant. The voltage observes a combination of B0-prime and g-prime. Adding static B0 or static g leaves that fixed-coil voltage unchanged. Integrating emf therefore cannot uniquely recover the absolute field without initial flux or another observation. A known initial flux changes the inference problem; it is additional information rather than a property supplied automatically by the voltage trace. In an AI dataset, label whether the targets are field coefficients, their rates, or flux with a supplied integration constant.

$$
\mathcal E_{\rm fixed}=-N\pi R^2\left[B_0^{\prime}+g^{\prime}\left(|\mathbf o|^2+R^2/2\right)\right].
$$

## The term omitted by the point approximation

Using area times the time derivative of the centre B misses the finite-area curvature term. Record this correction directly as well as comparing rounded full and point voltages. A direct formula can preserve a small correction that is lost against a larger background, but it remains a formula within the same family. Also separate the observed quotient flux/area from its analytic limiting value. At zero area the quotient is undefined, whereas the continuous limiting average tends to centre B. A limit is not a fabricated measurement from a collapsed probe.

$$
\mathcal E_{\rm point}=-N\pi R^2(B_0^{\prime}+g^{\prime}|\mathbf o|^2),\qquad \mathcal E_{\rm fixed}-\mathcal E_{\rm point}=-\tfrac12N\pi R^4g^{\prime}.
$$

## Worked example: a stationary finite coil

Take a one-turn coil of radius one metre centred at the origin. Let B0-prime be 0.3 T/s and g-prime be 0.2 T/(m²s). The fixed-coil emf is minus 0.4pi volts, while the point approximation is minus 0.3pi volts. Their difference is minus 0.1pi volts. The negative sign opposes increasing positive flux for the plus-z normal. These are transparent numbers for checking the local model, not a claim that an actual coil of this size has negligible loading and backreaction. The example also makes the units of curvature rate explicit.

$$
R=N=1,\quad \mathbf o=0,\quad B_0^{\prime}=0.3,\quad g^{\prime}=0.2:\qquad \mathcal E_{\rm fixed}=-0.4\pi,\quad \mathcal E_{\rm point}=-0.3\pi.
$$

## Translation adds a spatial-sampling term

When the centre translates with velocity v, its squared distance changes. The total flux derivative acquires the additional term2g o dot v, which matches the motional circulation. A static spatial curvature can therefore become visible through motion. A static uniform B0 remains invisible to this translating, nonrotating loop because its linked uniform flux does not change. Motion that rotates the normal is a different problem. This qualification is essential: observability depends on the declared sensor trajectory, not just on the name of the physical parameter.

$$
\mathbf o(t)=\mathbf o_0+\mathbf vt,\qquad \mathcal E=\mathcal E_{\rm transformer}+\mathcal E_{\rm motional},\qquad \mathcal E_{\rm motional}=-2N\pi R^2g\,\mathbf o\cdot\mathbf v.
$$

## Add the two voltages in a moving example

Keep R=N=1 and use instantaneous centre(1, 0)metres, velocity(0.5, 0)metres per second, and curvature0.4T/m². With the previous field rates, transformer emf is minus 0.6pi volts and motional emf is minus 0.4pi volts. Their sum is minus pi volts. Changing the constant uniform field offset leaves this sum unchanged. Evaluate the two line integrals independently to check the sign rather than adjusting it to match a preferred answer. This example connects trajectory information to what can and cannot be inferred from the measured voltage.

$$
\mathbf o=(1,0),\ \mathbf v=(0.5,0),\ g=0.4:\quad \mathcal E_{\rm transformer}=-0.6\pi,\quad \mathcal E_{\rm motional}=-0.4\pi,\quad \mathcal E=-\pi.
$$

## The observation matrix of two fixed coils

When area and turn count are nonzero and representable, normalize each voltage by its signed gain. Two fixed coils observe the rate pair B0-prime and g-prime. Equal spatial weights repeat the same combination, even if the coils occupy different positions. Conversely, different weights can identify the two rates within the declared exact model. Repeating a measurement may reduce random noise but does not create new independent spatial information. This is why counting sensors is not enough: inspect the observation matrix and state exactly which unknowns it is intended to recover.

$$
y_i=-\mathcal E_i/(N_iA_i),\quad s_i=|\mathbf o_i|^2+R_i^2/2,\qquad \begin{pmatrix}y_1\\y_2\end{pmatrix}=\begin{pmatrix}1&s_1\\1&s_2\end{pmatrix}\begin{pmatrix}B_0^{\prime}\\g^{\prime}\end{pmatrix}.
$$

## Uniqueness and stability are different

A nonzero weight contrast gives a unique slope, but a small contrast amplifies voltage errors. Conditioning also depends on column units, so introduce a length reference and scale the curvature-rate unknown accordingly. Full rank is therefore not the same as stable estimation. A mathematically distinct pair of weights can produce measurements whose informative difference is lost through rounding or noise. An exact-input geometry calculation can establish structural distinction, but it cannot recover information already absent from the recorded voltages. Report the condition estimate and measurement assumptions alongside the reconstructed rates.

$$
g^{\prime}=\frac{y_2-y_1}{s_2-s_1},\quad B_0^{\prime}=y_1-s_1g^{\prime},\qquad \widehat s_i=s_i/\ell^2,\quad \widehat g^{\prime}=g^{\prime}\ell^2.
$$

## Calculate conditional propagation of measurement errors

Assume known geometry and model, with normalized observation errors bounded by eta1 and eta2. Applying the triangle inequality gives the bounds below. Convert voltage error to normalized error by dividing by the corresponding turn-area gain first. These bounds describe the specified measurement uncertainty; they are not certified bounds for arithmetic error, incorrect geometry, loading, or model discrepancy. Reporting their scope prevents an uncertainty number from appearing more comprehensive than it is. Compare probes with different areas and weights to see how spatial resolution and voltage sensitivity interact.

$$
|\delta g^{\prime}|\le\frac{\eta_1+\eta_2}{|s_2-s_1|},\qquad |\delta B_0^{\prime}|\le\frac{|s_2|\eta_1+|s_1|\eta_2}{|s_2-s_1|}.
$$

## Underdetermination, inconsistency, and representation loss

Repeated rows with different normalized observations can be inconsistent with the stated error bounds rather than merely underdetermined. A zero-turn or zero-radius ideal probe predicts zero emf, so a nonzero reading calls for noise, coupling, or model-mismatch analysis. Positive radius lost in coordinate or area representation is a different case from an exactly collapsed probe. Exact binary-input geometry contrasts can distinguish small nonzero designs without labelling them zero by tolerance, while observations or the numerical solve can still be unavailable. Keep structural rank, data compatibility, representability, and estimation stability separate.

## Verify an AI sensor inverse problem

Compare actual electric and motional circulations with finite-area flux and the differentiated moving flux. Then check the inference stage using an independent linear solve and exact-input high-precision geometry references. Label voltage, averaged field, field rate, and initial flux distinctly in the training data. A small voltage loss cannot establish recovery of an unidentifiable static component. These numerical comparisons remain selected observations within the declared family. Background on fixed and moving induction paths is in [Feynman II §§17–1–17–2](https://www.feynmanlectures.caltech.edu/II_17.html); real readout circuits and sensor backreaction require additional modelling.

## Exercises with explanations

### Exercise 1

Does zero emf from a fixed coil imply B=0?

**Answer:** No. A nonzero constant flux also produces zero emf. Absolute field recovery needs additional information.

### Exercise 2

At the origin with R=N=1, B0-prime=0.3 and g-prime=0.2, find the emf.

**Answer:** It is minus 0.4pi volts. The point approximation of minus 0.3pi volts omits the finite-area term.

### Exercise 3

Can two fixed coils with equal R and equal centre distance identify both field rates?

**Answer:** Their weights match and rank is one. Different angular positions alone do not create independent rows in this axisymmetric family.

### Exercise 4

In the moving example, what is total emf and can a constant uniform B0 be identified?

**Answer:** Total emf is minus pi volts. A constant uniform B0 remains unobservable under this nonrotating translation.

### Exercise 5

Does a small nonzero weight contrast guarantee stable estimation?

**Answer:** Uniqueness can hold while noise and rounding are strongly amplified. Check column scaling, conditioning, and measurement-error propagation.

### Exercise 6

If an ideal zero-turn probe reports nonzero voltage with zero allowed error, how should it be classified?

**Answer:** The data are inconsistent with the stated model. Do not fill in arbitrary rates; revisit noise, coupling, and modelling assumptions.
