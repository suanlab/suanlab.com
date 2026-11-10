## Two measurements can still be the same piece of information

Suppose an oscillator has unknown damping and stiffness. Measuring its acceleration twice sounds more informative than measuring it once. But if both measurements use the same relationship between displacement and velocity, they may repeat the same equation. The number of rows in a data table is not the number of independent parameter constraints.

This lesson uses controlled initial-state probes of one declared linear oscillator. Each probe resets displacement and velocity, then measures instantaneous acceleration under the same coefficients. The probes are distinct experiments, not points that we silently join into a single time trajectory. We will derive the complete parameter solution set and separate exact uniqueness, physical admissibility and sensitivity to measurement errors.

## Declare the coefficient model and units

Use a known scalar-mass normalization and zero imposed acceleration input:

$$
a=-\gamma v-\kappa x.
$$

Displacement x has units meters, velocity v has units m/s and acceleration a has units m/s squared. Damping gamma has units inverse seconds and stiffness per unit mass kappa has units inverse seconds squared. The mechanically admissible domain for this lesson is gamma>=0 and kappa>0.

These are the normalized coefficients. If an uncalibrated mass were also unknown, recovering gamma and kappa would not automatically identify mass, dimensional damping and dimensional stiffness separately. The force-mass lesson explains why an input scale can matter.

For a probe with prescribed x and v, the measured acceleration supplies gamma*v+kappa*x=-a. The unknown coefficients are shared across probes. Changing them between resets would define a different inverse problem.

## Assemble the two-probe equations

For two probes, write

$$
A\theta=y,\qquad
A=\begin{pmatrix}v_1&x_1\\v_2&x_2\end{pmatrix},\quad
\theta=\begin{pmatrix}\gamma\\\kappa\end{pmatrix},\quad
y=\begin{pmatrix}-a_1\\-a_2\end{pmatrix}.
$$

The determinant is D=v1*x2-v2*x1. Exact rational controls allow its zero or nonzero status to be evaluated without a floating tolerance. If D is nonzero, the algebraic solution is unique:

$$
\gamma=\frac{y_1x_2-y_2x_1}{D},\qquad
\kappa=\frac{v_1y_2-v_2y_1}{D}.
$$

That is a statement about this data matrix and declared linear model. It does not establish that all physical trajectories, unmodeled forces or real sensor errors have been explained.

## Work an independently excited example

Let the true coefficients be gamma=1/s and kappa=2/s squared. Probe1 uses v1=1 m/s,x1=0 and therefore a1=-1 m/s squared. Probe2 uses v2=0,x2=1 m and therefore a2=-2 m/s squared.

The matrix is the identity numerically in the stated SI reference scales. Its first row isolates damping and its second isolates stiffness. The recovered values are gamma=1 and kappa=2. No component can change while keeping both equations true.

This is more informative than simply taking two copies of the first probe. A duplicate row can improve averaging under a stated noise model, but it does not create a new exact direction of parameter information.

## Derive the complete rank-one family

Now use probes(v1,x1)=(1,1) and(v2,x2)=(2,2). With the same true coefficients, y=(3,6). The second row is twice the first, so only the equation gamma+kappa=3 remains.

The complete real solution set can be written

$$
(\gamma,\kappa)=(3,0)+t(-1,1),\qquad t\in\mathbb R.
$$

A program returning only the true pair(1,2) would hide other solutions. A least-norm fit or a regularizer can choose one member, but the observations themselves leave a whole affine line.

The physical domain further restricts the family: gamma=3-t>=0 and kappa=t>0 give 0<t<=3. The lower endpoint is open because zero stiffness is excluded; the upper endpoint is closed because zero damping is allowed. An accurate interface should retain these endpoint distinctions rather than replacing the family with a few sampled dots.

## Keep rank, consistency and the physical domain separate

If proportional rows have responses that are not in the same proportion, the equations are inconsistent. For example, the rows above cannot fit y=(3,6.01) exactly. Rank1 does not by itself imply that an affine solution line exists; the right-hand side matters.

If both probe rows are zero and both responses are zero, every real coefficient pair solves the equations. The physically admissible subset is the whole declared damping/stiffness domain. If a zero row has a nonzero response, no coefficient pair solves that row.

With rank2, every response vector has a unique algebraic fit, but the fit can have negative damping or nonpositive stiffness. A small controlled acceleration perturbation in a zero-damping example can already produce negative fitted damping. Report algebraic uniqueness and physical admissibility as separate flags.

A physically empty result need not mean that the underlying apparatus has impossible coefficients. It can indicate that exact fitting of noisy or misspecified observations is inappropriate. The lesson displays these distinctions without assigning a statistical confidence statement to a deterministic perturbation control.

## Make the observation geometry dimensionless

The two columns of A have different units. A raw determinant or Euclidean column norm should not be treated as an observer-independent numerical measure of experimental quality.

Choose explicit positive reference length L and time T. Define

$$
\bar A_i=\left(\frac{v_iT}{L},\frac{x_i}{L}\right),\qquad
\bar\theta=(\gamma T,\kappa T^2),\qquad
\bar y_i=\frac{y_iT^2}{L}.
$$

Then the dimensionless equation is Abar*thetabar=ybar. Changing positive reference scales does not change exact rank or the recovered physical coefficients. It can change numerical singular values and the condition number because it changes the chosen geometry and relative parameter scales.

The default references L=1 m and T=1 s make the familiar SI numbers appear directly. Their convenience should not hide the normalization assumption.

## Exact rank differs from conditioning

For the chosen dimensionless matrix and Euclidean norm, the two-norm condition number is sigmaMax/sigmaMin when rank is2. It describes worst-direction amplification in that specified linear system. A rank-deficient matrix has no finite inverse condition number; it should not be disguised as an ordinary large finite readout.

Compare the independent probes with rows(1,1) and(1,1.1). The latter determinant is0.1, so the fit is still exactly unique. But their directions are close, making some parameter combinations difficult to separate numerically.

With true coefficients(1,2), the noiseless response is(3,3.2). If the first measured acceleration increases by0.01, its response decreases to2.99 while the second remains3.2. The fit becomes gamma=0.89 and kappa=2.1. The same response perturbation in the independent identity design changes only gamma from1 to0.99. Exact uniqueness has survived in both designs, but their sensitivities differ.

## Distinguish numerical evidence from the exact certificate

The complete affine family and its intersection with gamma>=0,kappa>0 can be computed using rational elimination and exact inequalities. Singular values and the condition number are numerical geometry readouts. Their small residuals do not replace the exact rank or compatibility derivation.

Even a numerical formula deserves checking. Computing singular values by subtracting nearly equal expressions can lose accuracy when the two values coincide. Independent high-precision SVD references help catch such implementation errors. The implementation keeps numerical conditioning separate from the rational rank certificate.

Plots show visible segments of parameter constraints, not the whole infinite mathematical line. Row-vector diagrams use dimensionless coordinates and a stated normalization. A complete solution statement comes from the algebraic family and domain bounds, rather than from apparent intersections of sampled curves.

## Implications for learned physical parameters

A learning algorithm can achieve small equation error on a poorly excited set while its coefficients vary substantially. More duplicate observations do not necessarily remove a null direction. Priors and regularization may stabilize one answer without making it uniquely identified by the experiment.

A useful evaluation reports the feature design, exact or estimated rank, the physical coefficient domain, normalization and noise assumptions. When the aim is predicting complete motion, equation-level fitting also needs to be distinguished from simulation accuracy over time. Two instantaneous probes do not establish the correctness of an entire trajectory model.

## Exercises with worked solutions

1. **Exercise:** For rows(1,0),(0,1) and response(1,2), recover the coefficients. **Solution:** gamma=1 and kappa=2; each row isolates one coefficient.
2. **Exercise:** What complete family fits rows(1,1),(2,2) and response(3,6)? **Solution:** (gamma,kappa)=(3,0)+t(-1,1) for all real t.
3. **Exercise:** Which family members satisfy gamma>=0,kappa>0? **Solution:** 0<t<=3, with an open lower and closed upper endpoint.
4. **Exercise:** Does rank1 guarantee a solution for response(3,6.01)? **Solution:** No. Those proportional rows require proportional responses; this exact system is inconsistent.
5. **Exercise:** Can a unique rank2 fit be physically inadmissible? **Solution:** Yes. Noisy responses can produce negative damping or nonpositive stiffness even though the algebraic inverse exists.
6. **Exercise:** Does changing L or T add identifying information? **Solution:** No. It changes numerical representation and conditioning geometry, while exact rank and recovered physical coefficients remain unchanged.

## Sources and further study

[MIT Underactuated Robotics on system identification](https://underactuated.mit.edu/sysid.html) gives background on structured mechanical parameter fitting and data matrices. The reset-probe equations, complete families, physical interval and normalized conditioning examples above are direct derivations for this oscillator model.
