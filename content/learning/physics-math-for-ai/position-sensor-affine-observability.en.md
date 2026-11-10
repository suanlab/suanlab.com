## A sensor trace does not necessarily determine the physical state

A camera reports a cart's position at two times. Another instrument reports its velocity continuously. Which observation determines the cart's initial position and velocity? The answer depends on the measurement rule and its calibration, not simply on how many numbers were collected.

This lesson uses known one-dimensional constant-velocity dynamics. We derive the entire set of initial states compatible with the observations. Then we add an unknown constant position-sensor offset and show how a new ambiguity survives even when position measurements are available at arbitrarily many times.

These are continuous state families. A finite list of candidate states, a minimum-norm estimate or a visually convincing sensor trace is not a substitute for characterizing the full observation-equivalence set.

## Declare the dynamics and measurement rule

The physical trajectory is

$$
x(t)=x_0+v_0t.
$$

Here x0 is the initial laboratory position in meters, v0 is velocity in m/s and t is time in seconds. The dynamics are assumed known. We are estimating the state, rather than fitting a new force or acceleration model.

A position sensor may report z(t)=x(t)+b, where b is a constant offset in meters. A calibrated sensor has a known b that can be subtracted. An uncalibrated sensor makes b another unknown quantity. A velocity-only sensor supplies v0 directly but does not report absolute position.

The distinction between known and unknown calibration must be part of the model. Merely setting b to zero in code does not prove that the physical instrument's offset is zero.

## Recover state from calibrated positions at distinct times

Subtract the known offset from two position measurements to obtain y1 and y2. Their equations are

$$
\begin{pmatrix}1&t_1\\1&t_2\end{pmatrix}
\begin{pmatrix}x_0\\v_0\end{pmatrix}
=\begin{pmatrix}y_1\\y_2\end{pmatrix}.
$$

The determinant is t2-t1. If the times are distinct,

$$
v_0=\frac{y_2-y_1}{t_2-t_1},\qquad x_0=y_1-v_0t_1.
$$

For x0=1 m,v0=2 m/s,b=1 m, measurements at0 s and1 s are2 m and4 m. Subtracting the known1 m offset gives1 m and3 m, hence velocity2 m/s and initial position1 m.

This is unique recovery under the declared noiseless model. Close times can still make the velocity estimate sensitive to small position errors; exact uniqueness and numerical sensitivity remain separate questions.

## Repeated times leave an affine state family

If both times equal tstar, the same equation x0+tstar*v0=ystar is repeated. The complete family is

$$
(x_0,v_0)=(y_\star,0)+\lambda(-t_\star,1),\qquad\lambda\in\mathbb R.
$$

The units of this parameterization must be handled consistently: lambda has velocity units in this physical-coordinate form. The interface can instead use explicitly normalized state coordinates and a dimensionless family parameter.

At tstar=1 s with calibrated position3 m, both x0=1 m,v0=2 m/s and x0=3 m,v0=0 satisfy the observation. Their future positions differ. Duplicate observations at one time are not equivalent to observations at two distinct times.

A graph should show the observation time as well as the candidate trajectories. Agreement at the measured instant must not be presented as agreement over the full plotted interval.

## Velocity-only observations leave an unknown position origin

With a velocity-only instrument, the observation equation is y=v0. Every state of the form(x0,v0) with arbitrary x0 fits the same constant velocity trace.

Even continuous perfect observation of velocity does not determine the integration constant x0. Integrating velocity gives a displacement relative to an initial position; it does not supply that position by itself.

The complete state fiber has one free position direction. A program may select x0=0 as a representative, but that choice is not a measurement. An independent position anchor or known initial position would remove the ambiguity by adding information.

## Add an unknown position-sensor offset

When b is unknown, use the augmented unknown vector(x0,v0,b). Two position observations give

$$
\begin{pmatrix}1&t_1&1\\1&t_2&1\end{pmatrix}
\begin{pmatrix}x_0\\v_0\\b\end{pmatrix}
=\begin{pmatrix}z_1\\z_2\end{pmatrix}.
$$

The first and third columns are identical. For distinct times the matrix has rank2, but the augmented state has dimension3. The observations determine v0 and the sum x0+b, not the two position-like quantities separately.

For every real displacement delta,

$$
x_0\mapsto x_0-\delta,\qquad b\mapsto b+\delta
$$

leaves the complete sensor trace unchanged. The null direction is(-1,0,1) after compatible position normalization. Additional position times repeat this same column dependence, so they cannot by themselves remove the ambiguity.

## Work a full-trace ambiguity example

Let the true values be x0=1 m,v0=2 m/s,b=1 m. The sensor reports z(t)=2+2t meters. A second candidate with x0=2 m,v0=2 m/s,b=0 produces exactly the same sensor trace at every time.

The two physical laboratory trajectories differ by1 m. Their sensor offsets compensate for that difference. A sensor plot overlays while a physical-position plot separates. This is the reason to show both views.

The complete family is(x0,v0,b)=(2-delta,2,delta) in the stated units, with delta any real displacement. The true state is one member, not a quantity that the raw sensor trace uniquely reveals.

If only a single repeated position time is observed with unknown b, the rank is1 and two independent directions remain. One changes position against offset; another changes initial position against velocity while preserving the observed instant. A one-direction animation alone must not be labeled the entire two-dimensional family.

## Remove the offset ambiguity with an independent calibration

An independent measurement of b adds the row(0,0,1). With two distinct position times, the resulting3-by3 matrix has determinant t2-t1 and determines the full augmented state.

The word independent matters. A value inferred only by choosing one convenient member of the existing state family is not a new calibration measurement. The added equation must supply information absent from the position trace.

Calibration also does not repair every poor time design. If the two position times are equal, knowing b removes the offset direction but still leaves the initial-position/velocity family. The augmented system then has rank2 rather than3.

This is a useful example of separating two problems: a sensor reference can be unknown, and the observation times can be insufficient. Solving one does not automatically solve the other.

## Describe the entire fiber using a particular solution and nullspace

For a consistent linear observation system A*z=y, every compatible state has the form

$$
z=z_{\mathrm p}+N\lambda,
$$

where zp is a particular solution and the columns of N span the nullspace of A. Conversely, substituting this expression returns the same observations because A*N=0. This proves completeness when the basis contains every null direction.

Rational row reduction can select a particular solution by setting free variables to zero and then return a basis for all remaining directions. That particular point need not equal the physical truth. It is a coordinate representation of the full fiber, not a privileged estimate justified by the data.

A state component is uniquely determined precisely when every null direction has zero in that component. Thus velocity can be identifiable while initial position and offset remain individually ambiguous. A single all-or-nothing label would conceal this useful partial information.

## Normalize the state before drawing family geometry

Position, velocity and offset do not all have the same units. Choose positive reference length L and time T, and use the dimensionless state(x0/L,v0*T/L,b/L) and time t/T.

The illustrative interface uses L=1 m,T=1 s. Its normalized rows are(1,t/T,1), and its family controls are dimensionless coefficients of normalized null vectors. A finite slider window shows part of an unbounded real family; the algebraic description gives the complete set.

Physical-position plots use meters, while velocity-only sensor plots use m/s. Captions must switch with the measurement rule. A diagram should not combine those values on one unlabeled axis merely because they are both observations.

## Distinguish information ambiguity from coordinate choice

The laboratory coordinates in this example are fixed. Two possible laboratory positions produce the same instrument output because the instrument's offset is unknown. That is an information ambiguity under the measurement model.

It differs from explicitly changing the observer coordinates while consistently transforming a known physical state and its context. The two subjects are related through transformations, but an uncalibrated sensor does not certify one absolute laboratory state simply because another parameterization is convenient.

For learned representations, reconstructing sensor data accurately may identify only an equivalence class. If a downstream task requires absolute position relative to an obstacle or apparatus, it may need an independent anchor. The required information depends on the prediction or control task, not just reconstruction loss.

## Exercises with worked solutions

1. **Exercise:** Calibrated positions are1 m at0 s and3 m at1 s. Recover x0,v0. **Solution:** v0=2 m/s and x0=1 m.
2. **Exercise:** Both observations are3 m at1 s. Is the initial state unique? **Solution:** No. x0+v0=3 in the stated SI units leaves a whole affine line.
3. **Exercise:** A perfect velocity trace is2 m/s. Is x0 known? **Solution:** No. Any initial position with that velocity gives the same velocity observations.
4. **Exercise:** An uncalibrated position trace is z(t)=2+2t m. Compare(x0,v0,b)=(1,2,1) and(2,2,0). **Solution:** They give identical sensor traces but different physical positions.
5. **Exercise:** Do more position times remove a constant unknown offset? **Solution:** Not alone. The initial-position and offset columns remain identical for every time.
6. **Exercise:** Does an independent offset calibration guarantee complete state recovery with repeated position times? **Solution:** No. Distinct-time information is still required to separate initial position and velocity.

## Sources and further study

[Stanford EE263, Observability and State Estimation](https://ee263.stanford.edu/lectures/observ.pdf) gives background on linear observation maps and their unobservable nullspaces. The continuous constant-velocity sampling matrices, full affine fibers and sensor-offset/calibration examples here are direct derivations under the stated measurement models.
