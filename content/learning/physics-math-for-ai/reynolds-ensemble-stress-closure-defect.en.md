## Averaging a product keeps information that averaging a velocity loses

Imagine photographing a channel repeatedly. At a particular height the mean horizontal velocity might be one metre per second, but the individual photographs differ. Whenever a parcel moves upward, does it also tend to move faster downstream? That relationship is a covariance. The mean velocity alone does not answer the question, yet the answer changes the transport of horizontal momentum across the channel.

This is the first obstacle in deriving an equation for a mean flow. Navier–Stokes contains a product of velocities. The average of that product is generally not the product of the averages. A new unknown appears even when density and molecular viscosity are constant. A closure model supplies information about that unknown using additional assumptions, data or equations.

We build a small ensemble whose actual velocities, derivatives, covariance and forcing can all be checked. It is a manufactured example for understanding Reynolds averaging. Its members are steady forced fields; they are not a direct numerical simulation of turbulent time evolution. This distinction lets us use the example honestly while still learning a central difficulty of turbulence modeling.

## State the averaging operation and its units

The channel height is H, the streamwise wavenumber is positive k and the streamwise period is L. The declared span width is W. Density rho is positive and dynamic viscosity mu is nonnegative. The flow is independent of the span coordinate and has no span velocity. W specifies an integration width in this homogeneous direction; it does not introduce additional no-slip side walls. Write xi for height divided by H. A phase alpha labels the ensemble member; it is not elapsed time.

$$
L=\frac{2\pi}{k},\qquad \xi=\frac yH,\qquad
\alpha_j=\frac{2\pi j}{8},\quad
\langle q\rangle=\frac18\sum_{j=0}^{7}q_j,\qquad
[R_{ij}]=\mathrm{m^2/s^2}.
$$

We use exactly eight equally weighted phases. The first and second phase moments of this single-frequency construction agree with uniform continuous-phase averages. Eight samples would not establish convergence for an arbitrary measured turbulent signal. Each height-dependent covariance below refers to this declared ensemble.

The covariance R contains velocity products without density. We call minus rho times its off-diagonal entry the Reynolds shear contribution to the mean stress. Other sources include density or a minus sign in the definition of “Reynolds stress”; compare definitions before comparing plots. Variable-density Favre averaging is a different operation and is outside this constant-density example.

## Construct a flow that respects incompressibility and the walls

A streamfunction makes the fluctuation divergence-free. Choose a bulk speed U-b and a dimensionless mean asymmetry c. The streamfunction amplitudes a and b have units of square metres per second. They are not velocity amplitudes: a physical y derivative is needed to obtain the horizontal velocity.

$$
\phi=\xi^2(1-\xi)^2,\quad \chi=\phi(2\xi-1),\quad
\psi=a\phi\sin(kx+\alpha)+b\chi\cos(kx+\alpha),
$$
$$
U=6U_b\xi(1-\xi)[1+c(2\xi-1)],\qquad
u=U+\psi_y,\quad v=-\psi_x,\quad
u_x+v_y=0.
$$

Both phi and chi, and their first y derivatives, vanish at either wall. Every ensemble member therefore has zero velocity there. The mean profile also satisfies the wall condition, and its height average is U-b. No penetration and no slip are properties of the actual fields, rather than boundary labels added to an unrelated animation.

Horizontal position and ensemble phase are entered as fractions of a cycle. An exact quarter-cycle input has an exact sine or cosine zero by convention. A small independently entered position offset is retained when combined with a quarter phase. An ordinary floating-point approximation to a general angle is still an approximation; the implementation does not claim symbolic trigonometry for all real numbers.

## Derive the covariance from the actual fluctuations

Let u-prime be the horizontal departure from U. The vertical mean vanishes. Averaging the actual phase products gives the following result, where subscripts y always mean physical derivatives, including the factors of inverse H.

$$
R_{uu}=\frac12(a^2\phi_y^2+b^2\chi_y^2),\qquad
R_{vv}=\frac{k^2}{2}(a^2\phi^2+b^2\chi^2),\qquad
R_{uv}=-\frac{abk}{H}\phi^2.
$$

The sign of R-uv records whether upward-moving parcels tend to carry positive or negative horizontal departures. Changing the sign of b reverses that relationship while leaving both diagonal variances unchanged. Increasing either variance does not determine the sign of the transport.

Because R is a covariance, its diagonal entries and determinant must be nonnegative. A stable Gram expression verifies this structural condition without subtracting two nearly equal covariance products.

$$
\det R=\frac{k^2}{4}
\left(a^2\phi\phi_y+b^2\chi\chi_y\right)^2\ge0.
$$

At the channel centre, phi-y and chi are exactly zero. The covariance loses rank there even when both velocity components fluctuate. Rank loss means perfect linear dependence of those two departures at that height; it does not mean that both variances vanish. A machine-generated covariance that violates positive semidefiniteness would not describe this ensemble.

## Retain the complete forcing before calling this a solution

For each steady member we set pressure to zero and manufacture a force per volume that satisfies both momentum equations. This force depends on the ensemble member. Averaging gives a streamwise force G and, generally, a nonzero vertical force.

$$
\boldsymbol f=\rho(\boldsymbol u\cdot\nabla)\boldsymbol u
-\mu\Delta\boldsymbol u,\qquad
G=\langle f_x\rangle=\rho(R_{uv})_y-\mu U_{yy},\qquad
\langle f_y\rangle=\rho(R_{vv})_y.
$$

Ignoring the vertical component would fail to make the chosen pressure-zero field a full steady Navier–Stokes solution. A force-free channel, or a channel driven only by a prescribed uniform pressure gradient, is a different problem. Our constructed force is what makes the exact field possible.

The mean streamwise stress is the molecular shear plus the covariance contribution. The mean momentum equation says that the force balances its negative gradient. A residual formed from these same expressions is a useful implementation check, but independent phase averages of the actual force provide stronger evidence.

## Locate the closure problem precisely

Suppose only U is supplied to a model. Keep U-b and c fixed while changing a or b. The mean profile stays the same; the covariance changes, and the force generally changes as well. Thus a mean profile by itself does not identify the ensemble covariance.

This example does not establish nonuniqueness for one fully specified forced initial-boundary-value problem. We changed part of that problem when we changed the manufactured force. Its lesson is about information discarded by averaging, not a theorem about Navier–Stokes uniqueness.

Writing an exact evolution equation for covariance does not automatically finish the task. Such equations introduce higher moments, pressure correlations and transport terms. Modeling those new quantities creates a hierarchy of closure questions. More equations can improve a model while still requiring assumptions.

## Test a constant nonnegative eddy-viscosity candidate

A common shear-only candidate relates covariance to mean shear through a kinematic eddy viscosity nu-m. It has units of square metres per second and is distinct from dynamic viscosity mu. Here we compare only the off-diagonal closure; we are not providing a complete three-dimensional isotropic stress model or a pressure correction.

$$
R_{uv,m}=-\nu_m U_y,\qquad
\nu_{\mathrm{inferred}}=-\frac{R_{uv}}{U_y}\quad(U_y\ne0),\qquad
\delta R=R_{uv}-R_{uv,m}=R_{uv}+\nu_m U_y.
$$

For a symmetric mean profile the shear changes sign at the centre, while R-uv can retain one sign. The inferred viscosity can therefore be positive on one side and negative on the other. At zero shear with nonzero covariance, no finite scalar viscosity can match the stress. If both vanish, the ratio is indeterminate: it does not select a unique value of zero.

Replacing covariance by the candidate also replaces the force required to keep the same mean profile. The force defect is therefore a physical consequence of the closure error, not just a difference between two fitted curves.

$$
\tau=\mu U_y-\rho R_{uv},\quad G=-\tau_y,\qquad
G_m=-(\mu+\rho\nu_m)U_{yy},\qquad
G-G_m=\rho(\delta R)_y.
$$

## Track mean energy without calling transfer dissipation

Multiply the mean equation by U, integrate across the channel and use the zero wall velocity. After multiplying by actual period and span width, every term below is a power in watts.

$$
LW\int_0^H GU\,dy
=LW\int_0^H\mu U_y^2\,dy
+LW\int_0^H P\,dy,\qquad
P=-\rho R_{uv}U_y.
$$

Positive P is loss from mean kinetic energy to the fluctuations under this convention. Negative P returns energy to the mean. Molecular mean loss is nonnegative. P is an exchange term and need not be nonnegative locally or globally. In a full fluctuation budget it appears with the opposite role; it is not molecular destruction of total kinetic energy.

The current calculation reports the mean budget and fluctuation kinetic energy, not a separately verified complete fluctuation forcing-work budget. Steadiness does not make a signed exchange vanish. The candidate with nonnegative nu-m predicts nonnegative mean-to-fluctuation transfer everywhere, so it cannot reproduce every signed exchange of this family.

For this polynomial profile the whole-cell mean molecular loss D and transfer P-total have simple exact forms. These are checked against separate physical-space integration of the actual profiles.

$$
D=LW\frac{\mu U_b^2}{H}\left(12+\frac{36c^2}{5}\right),\qquad
P_{\mathrm{total}}=LW\frac{4\rho abkU_bc}{1155H}.
$$

## Work through a numerical example

Take H = 2 m, k = 1.5 per metre, W = 1 m, rho = 1 kilogram per cubic metre, mu = 0.1 pascal seconds, U-b = 0.8 m/s, c = 0.25, a = 1 and b = 0.7 square metres per second. The period is approximately 4.18879 m.

Before multiplying by L times W, molecular mean loss is 0.3984 W per square metre, transfer is about 0.000363636 W per square metre and forcing work is about 0.398763636 W per square metre. The corresponding whole-cell values are approximately 1.6688 W, 0.0015232 W and 1.6703 W. Rounding the table should not be mistaken for an exact energy mismatch.

Now set c to zero. The integrated transfer vanishes by symmetry, but local positive and negative transfer remain. At the centre, R-uv is minus 0.00205078125 square metres per square second and the mean shear is zero. This is a particularly direct counterexample to treating the inferred scalar viscosity as an always finite material property.

## Relate the example to turbulence and AI research

The bulk Reynolds number compares inertial and molecular-viscous scales when mu is positive. It is not an automatic classifier for this forced ensemble.

$$
\mathrm{Re}_b=\frac{\rho |U_b|H}{\mu}\quad(\mu>0),\qquad
K'=\frac12\rho LW\int_0^H(R_{uu}+R_{vv})\,dy.
$$

Actual turbulence involves interacting scales and, often, a cascade whose direction and structure depend on dimension and physical conditions. This two-dimensional single-frequency family does not generate a three-dimensional inertial range, a turbulent spectrum or wall-law calibration. RANS predicts chosen averages; LES filters spatial structure and has different operator rules, developed in the next example.

For an AI closure, record what the input actually identifies. Training on mean velocity alone across changing forcing can merge physically different ensembles into the same input. Including forcing, geometry, wall distance and the averaging definition can reduce that ambiguity, but does not prove sufficiency for every flow. Check covariance realizability, momentum balance, signed energy transfer and generalization to held-out regimes. A fitted scalar viscosity should not silently fill undefined ratios with zero.

## Six exercises with explained answers

1. **Why does changing the sign of b leave the variances unchanged?** Each diagonal entry depends on amplitude squares. R-uv contains the product a times b, so its sign reverses. Variance and cross transport encode different information.
2. **Does zero mean vertical speed imply zero horizontal momentum transport?** No. Average v is zero, but the product of the horizontal departure and v can have nonzero average. Correlated departures survive averaging.
3. **What does a nonzero stress at zero shear imply for the scalar candidate?** Its predicted covariance is zero for every finite nu-m, so no finite value matches. A division-by-zero sentinel is not a learned physical coefficient.
4. **Why can total transfer vanish while the local curve remains nonzero?** For c = 0, covariance is symmetric about the centre and mean shear is antisymmetric. Their product has equal and opposite integrals on the two halves.
5. **What changes if span width doubles?** Whole-cell powers and energies double; local velocity, covariance, stress and inferred kinematic viscosity stay unchanged. This separates geometry from local constitutive interpretation.
6. **Does matching the mean budget prove a universal closure?** No. The integrated budget permits local errors to cancel. It also concerns one declared family and forcing distribution. Test local covariance, force gradients, singular points and different physical regimes as separate requirements.

## Sources and scope

The [INL/MOOSE Reynolds-averaging theory](https://mastodon.inl.gov/modules/navier_stokes/rans_theory.html) supplies the background on averaging and viscosity-based closure. [NASA CFPOST variable definitions](https://www.grc.nasa.gov/www/winddocs/cfpost/appc.html) help compare stress conventions and distinguish variable-density definitions. The streamfunction ensemble, its forcing and the worked budgets above are our own specialization. The source discussion is background, rather than an independent certification of this implementation.
