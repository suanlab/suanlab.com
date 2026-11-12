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

## Research extension: close the actual fluctuation forcing-work budget

The original lab reports the mean budget and explicitly leaves the complete fluctuation forcing-work budget unverified. This supplement closes that additional budget for the same manufactured fields. It does not replace the lab outputs or turn the ensemble into a turbulent simulation. Keep both velocity components and both components of the actual force. The force is per volume; no extra density factor is needed when it is dotted with velocity to form power density.

$$
\boldsymbol u=\boldsymbol U+\boldsymbol u^{\prime},\quad\boldsymbol U=(U,0),\quad\langle\boldsymbol u^{\prime}\rangle=0,\qquad\boldsymbol f=\langle\boldsymbol f\rangle+\boldsymbol f^{\prime}.
$$

Define three actual forcing powers over the periodic cell. The zero mean of the velocity departure makes the fluctuating-force and full-force expressions for fluctuation work equal. Cross terms disappear only after the declared ensemble average. A single member at one phase need not have zero cross work. Each listed integral includes the physical height, period and span width, giving watts rather than a local per-area quantity.

$$
F_m=LW\int_0^H U\langle f_x\rangle\,dy,\quad F_f=LW\int_0^H\langle\boldsymbol u^{\prime}\cdot\boldsymbol f^{\prime}\rangle\,dy=LW\int_0^H\langle\boldsymbol u^{\prime}\cdot\boldsymbol f\rangle\,dy,\quad F_{\rm total}=F_m+F_f.
$$

Use the gradient form of the integrated viscous power. Incompressibility, periodicity and zero wall velocity make it equivalent to the integral of twice dynamic viscosity times the squared strain-rate tensor. Those densities need not agree pointwise. Separating mean and departure gradients removes their averaged cross term. Dynamic viscosity mu already contains the mass dimension; multiplying these expressions by rho again would give the wrong units.

$$
D_m=\mu LW\int_0^H U_y^2\,dy,\qquad D_f=\mu LW\int_0^H\langle\|\nabla\boldsymbol u^{\prime}\|_F^2\rangle\,dy\ge0,\qquad D_{\rm total}=D_m+D_f.
$$

Dot each steady momentum equation with its own velocity and integrate. The advective work becomes a boundary flux; the no-slip walls and periodic faces remove it. Integrating viscous work by parts leaves negative dissipation. Pressure is zero in this manufactured family. The result is a total budget with no Reynolds transfer term: exchange between mean and fluctuations cannot create or destroy the combined kinetic energy.

$$
LW\int_0^H\langle\boldsymbol u\cdot\boldsymbol f\rangle\,dy=D_m+D_f,\qquad\frac{dK_{\rm total}}{dt}=F_{\rm total}-D_{\rm total}=0.
$$

The previously derived mean budget retains its signed transfer P. Subtract that budget from the independently integrated total budget to obtain the fluctuation budget. Positive P removes mean energy and supplies the fluctuations. It has opposite signs in the two equations. Steadiness means each complete right-hand side vanishes, not that its individual powers are zero. The same cancellation must hold in a model that reports both budgets.

$$
0=F_m-D_m-P,\qquad0=F_f-D_f+P,\qquad P=-\rho LW\int_0^H R_{uv}U_y\,dy=LW\frac{4\rho abkU_bc}{1155H}.
$$

Compute fluctuation dissipation from actual velocity gradients, rather than assigning it as a budget residual. Streamfunction differentiation produces two identical mixed-derivative contributions, one in each velocity component. Omitting either loses a factor of two in that part. All y derivatives below are physical derivatives. This is also a direct way to detect a model that forgot the height scale while differentiating a normalized coordinate.

$$
\langle\|\nabla\boldsymbol u^{\prime}\|_F^2\rangle=k^2(a^2\phi_y^2+b^2\chi_y^2)+\frac12(a^2\phi_{yy}^2+b^2\chi_{yy}^2)+\frac{k^4}{2}(a^2\phi^2+b^2\chi^2).
$$

The polynomial basis allows every height integral to be checked exactly. These are dimensionless integrals in xi, so changing back to y still contributes H and the derivative factors. Both rows have positive moments. They quantify different wall-normal shapes; equal streamfunction amplitudes a and b therefore do not generally give equal fluctuation dissipation.

$$
\int_0^1(\phi^2,\phi_{\xi}^2,\phi_{\xi\xi}^2)\,d\xi=\left(\frac1{630},\frac2{105},\frac45\right),\qquad\int_0^1(\chi^2,\chi_{\xi}^2,\chi_{\xi\xi}^2)\,d\xi=\left(\frac1{6930},\frac2{315},\frac47\right).
$$

Combine those moments to obtain the full fluctuation loss in watts. This expression contains squared amplitudes and is nonnegative for nonnegative dynamic viscosity. It is derived from the gradient integral independently of the transfer. The forcing work can then be compared with D-f minus P, rather than defined by that difference. In the verifier the two actual force components are evaluated separately before spatial integration.

$$
D_f=\mu LW\left[a^2\left(\frac{Hk^4}{1260}+\frac{2k^2}{105H}+\frac2{5H^3}\right)+b^2\left(\frac{Hk^4}{13860}+\frac{2k^2}{315H}+\frac2{7H^3}\right)\right].
$$

Choose positive transfer but reduce molecular viscosity. Fluctuations can then receive more power from the mean than they dissipate, while remaining steady because their own forcing extracts the excess. Negative forcing work is not negative viscosity or negative total dissipation. At exactly zero viscosity the mean forcing supplies P and the fluctuation forcing removes P. The total forcing work is zero although both exchanges can remain nonzero.

$$
P>D_f\ge0\Rightarrow F_f=D_f-P<0;\qquad\mu=0:\quad(F_m,F_f,F_{\rm total})=(P,-P,0).
$$

The table fixes the original example geometry and amplitudes while varying mu. Its columns distinguish molecular losses from exchange and forcing work. Changing b to minus b reverses P without changing either molecular loss. Setting c to zero removes the integrated exchange, although local exchange can remain. An AI model that matches only total work could still allocate mean and fluctuation powers incorrectly; test each independent integral and its zero controls. No evolved turbulent cascade or universal closure is claimed.

$$
b\mapsto-b:\ (D_m,D_f,P)\mapsto(D_m,D_f,-P),\qquad c=0:\ P=0,\ F_m=D_m,\ F_f=D_f.
$$

The table uses the declared teaching geometry above. Columns are mu in Pa s, mean loss, fluctuation loss, mean-to-fluctuation transfer and fluctuation forcing work, with all powers in watts. Its forcing column uses the derived balance for the compact example; independent actual-force quadrature supplies the separate check.

| mu | Dmean | Dfluct | transfer | Ffluct |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 0.00152320 | -0.00152320 |
| 0.00100000 | 0.01668814 | 0.00042232 | 0.00152320 | -0.00110087 |
| 0.01000000 | 0.16688140 | 0.00422323 | 0.00152320 | 0.00270004 |
| 0.10000000 | 1.66881402 | 0.04223232 | 0.00152320 | 0.04070912 |

```python
from math import pi
# H=2 m, k=1.5/m, W=1 m, rho=1 kg/m^3, Ub=0.8 m/s.
h, k, w, rho, ub, c, a, b = 2., 1.5, 1., 1., .8, .25, 1., .7
lw = 2*pi/k*w
transfer = lw*4*rho*a*b*k*ub*c/(1155*h)
for mu in (0., .001, .01, .1):
    mean_loss = mu*lw*ub*ub/h*(12+36*c*c/5)
    fluct_loss = mu*lw*(a*a*(h*k**4/1260+2*k*k/(105*h)+2/(5*h**3))
                 +b*b*(h*k**4/13860+2*k*k/(315*h)+2/(7*h**3)))
    forcing = fluct_loss-transfer
    print(" ".join(f"{v:.8f}" for v in (mu, mean_loss, fluct_loss, transfer, forcing)))
```

[INL/MOOSE Reynolds-averaging theory](https://mastodon.inl.gov/modules/navier_stokes/rans_theory.html) supplies averaging and closure background. The complete forcing-work and dissipation integrals here are derived separately for the explicitly forced streamfunction ensemble.

![Independent mean, fluctuation and total forcing powers versus viscosity](/assets/learning/continuum-reynolds-fluctuation-forcing-budget.en.svg)

### Research questions with explanations

**Can negative fluctuation forcing work violate nonnegative dissipation?** **Answer:** No. If transfer supplies more than fluctuations dissipate, their forcing removes the excess while the state stays steady. Molecular dissipation remains nonnegative.

**Why retain the vertical force when mean vertical velocity is zero?** **Answer:** Its mean work is zero, but its correlated work with vertical fluctuations need not vanish. The complete fluctuation and total budgets use both components.

**Does a correct total budget prove correct mean and fluctuation budgets?** **Answer:** No. An incorrect exchange can cancel between the two. Compare actual forces, gradient integrals and signed transfer separately.

## Six exercises with explained answers

1. **Why does changing the sign of b leave the variances unchanged?** Each diagonal entry depends on amplitude squares. R-uv contains the product a times b, so its sign reverses. Variance and cross transport encode different information.
2. **Does zero mean vertical speed imply zero horizontal momentum transport?** No. Average v is zero, but the product of the horizontal departure and v can have nonzero average. Correlated departures survive averaging.
3. **What does a nonzero stress at zero shear imply for the scalar candidate?** Its predicted covariance is zero for every finite nu-m, so no finite value matches. A division-by-zero sentinel is not a learned physical coefficient.
4. **Why can total transfer vanish while the local curve remains nonzero?** For c = 0, covariance is symmetric about the centre and mean shear is antisymmetric. Their product has equal and opposite integrals on the two halves.
5. **What changes if span width doubles?** Whole-cell powers and energies double; local velocity, covariance, stress and inferred kinematic viscosity stay unchanged. This separates geometry from local constitutive interpretation.
6. **Does matching the mean budget prove a universal closure?** No. The integrated budget permits local errors to cancel. It also concerns one declared family and forcing distribution. Test local covariance, force gradients, singular points and different physical regimes as separate requirements.

## Sources and scope

The [INL/MOOSE Reynolds-averaging theory](https://mastodon.inl.gov/modules/navier_stokes/rans_theory.html) supplies the background on averaging and viscosity-based closure. [NASA CFPOST variable definitions](https://www.grc.nasa.gov/www/winddocs/cfpost/appc.html) help compare stress conventions and distinguish variable-density definitions. The streamfunction ensemble, its forcing and the worked budgets above are our own specialization. The source discussion is background, rather than an independent certification of this implementation.
