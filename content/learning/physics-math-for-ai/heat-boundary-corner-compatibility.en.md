## A differential equation is only part of the problem

Suppose a model predicts a temperature field that satisfies the heat equation at every interior point. Has it solved the experiment you specified? Only if it also starts from the specified temperature and obeys the prescribed behavior at the ends. A differential law describes local evolution; initial and boundary data select which evolution is being requested.

This lesson deliberately constructs candidates that satisfy the interior law, then tests the remaining requirements independently. The setup is a rod of length L=1 meter with constant positive diffusivity. Temperature u is a difference from a reference, measured in kelvins. The controls specify both candidate coefficients and an independent initial profile. Making a candidate convenient does not automatically change the problem's data.

## Construct an analytic interior family

The candidate family, with x measured in meters and L=1 meter in the displayed numerical coordinates, is

$$
u(x,t)=c_0+e^{-\lambda t}\left[c_1\cos(\pi x/L)+c_2\sin(\pi x/L)\right],
\qquad \lambda=\kappa(\pi/L)^2.
$$

The constant has zero time derivative and zero second spatial derivative. Both oscillating terms acquire a factor minus (pi/L) squared after two spatial derivatives and a factor minus lambda after one time derivative. Hence u_t-kappa*u_xx is zero identically in the interior. This conclusion is an analytic identity for the family, not a residual estimated on a handful of grid points.

All three c coefficients are in kelvins. The interface uses diffusivity kappa=r/100 square meters per second. Time curves evaluate the analytic formula; they are not produced by a finite-difference time stepper. Spatial grid points determine the display resolution rather than the governing law.

## Derive the value traces at the two ends

At the endpoints, the sine disappears and the cosine is respectively one and minus one:

$$
u(0,t)=c_0+c_1e^{-\lambda t},\qquad
u(L,t)=c_0-c_1e^{-\lambda t}.
$$

Homogeneous Dirichlet data mean both endpoint temperature differences are zero for all times. Adding the two equations forces c0=0. Subtracting them forces c1=0 because the exponential is positive at every finite time. The sine coefficient c2 remains free.

Thus an interior solution with c1 nonzero can fail the endpoint data even though it decays toward zero. Waiting does not change the identity of the boundary problem. A small temperature at one late time is not an exact all-time boundary certificate.

The first guide uses c0=0,c1=1,c2=0, matching initial coefficients, and kappa=0.02. At one second the endpoint values are plus and minus exp(-0.02*pi squared) K, approximately plus and minus 0.820869 K. The interior law and initial profile match; the required Dirichlet data fail.

## Derive the derivative traces and their signs

Differentiate the spatial expression once:

$$
u_x(x,t)=\frac{\pi}{L}e^{-\lambda t}
\left[-c_1\sin(\pi x/L)+c_2\cos(\pi x/L)\right].
$$

The left and right coordinate derivatives are respectively (pi/L)*c2*exp(-lambda*t) and its negative. Homogeneous Neumann data therefore require c2=0, while c0 and c1 can remain nonzero.

The input uses the same increasing-x coordinate direction at both ends. The outward-normal derivative is minus u_x at the left and plus u_x at the right. Under a conductive law, physical heat flux also includes material factors and a minus sign relating flux to temperature gradient. A number in K/m is not already a heat rate in watts.

The third guide uses c0=1,c1=2,c2=0. Its endpoint derivatives vanish, yet the temperature profile is nonconstant and a constant offset remains. Zero derivative does not mean zero temperature. In this simple insulated model the oscillating part decays while the offset persists.

## Give the initial profile its own coefficients

The independently prescribed initial temperature is

$$
h(x)=h_0+h_1\cos(\pi x/L)+h_2\sin(\pi x/L).
$$

The candidate starts from c0+c1*cos+c2*sin. The three basis functions are linearly independent on the interval, so initial-profile equality holds exactly when all corresponding coefficients agree. Matching only a few rounded samples is not the criterion used here.

The spatial figure overlays the blue current candidate, the orange candidate initial profile and the green prescribed initial profile. The orange and green curves answer whether the candidate started correctly. Comparing blue with green at a later time answers a different question, because heat evolution generally changes the temperature after the start.

The second guide sets both coefficient triples to (0,0,2) and uses Dirichlet data. The candidate starts correctly and satisfies the boundary for every time. If only h2 is changed to one, the candidate still satisfies the boundary but no longer starts from the required initial temperature.

## Explain what corner compatibility actually requires

A spacetime corner is an endpoint at the initial time. If a requested solution and its endpoint value trace are continuous through t=0, zero Dirichlet data for positive time require the initial endpoint values to be zero as well. For this initial family that means h0=h1=0.

For the Neumann check, assume the required spatial derivative trace is continuous through t=0. Then the initial endpoint derivatives must vanish, which means h2=0. This is a statement with an explicit regularity premise. It is not a claim that every notion of a solution has the same endpoint continuity.

For example, an initial cosine profile conflicts with zero Dirichlet values at the corners if temperature is required continuous there. A weaker interpretation can allow boundary values for positive time and initial convergence in an interior or integral sense. Deciding whether a mild or weak solution exists needs a separately defined function space and solution concept. The corner flag alone does not answer that broader question.

The chosen finite family is also a restriction on candidates. Failure to find a matching candidate in it would not alone establish nonexistence among all heat fields. Here the formulas explain exactly which coefficients pass, without promoting that finite search into a general PDE theorem.

## Keep the three requirements separate

The interface reports candidate boundary satisfaction, initial-profile equality and prescribed-data corner compatibility individually. A candidate can satisfy the boundary while missing the initial profile. It can match the initial profile while violating the boundary. Compatible data do not imply that every candidate is the correct solution.

All-time boundary and initial flags are evaluated from integer coefficients. They do not use a tolerance on decayed values. Current trace numbers and curves are floating-point displays. Changing diffusivity or query time changes those numbers but does not change which coefficient subspace satisfies the same homogeneous boundary condition.

A temperature integral is also not automatically stored thermal energy. Material density, heat capacity and cross-sectional geometry are needed to turn a temperature field into the corresponding energy difference under a specified constitutive model. The figures in this lesson show temperature and gradient units explicitly.

## Relate the distinction to physics-based learning

A physics-informed model often minimizes separate interior, initial and boundary losses. This separation reflects distinct parts of the mathematical problem. A network can lower the interior loss while learning a member of the wrong boundary family; the first guide exhibits that failure without any neural-network approximation at all.

Checking only sampled boundary points creates another gap between finite evidence and an all-time claim. Analytic constraints can enforce some conditions by construction, such as multiplying a candidate by a factor that vanishes at fixed endpoints. That design still leaves the initial profile and the interior equation to be checked. The appropriate architecture depends on the requested data and solution class.

## Exercises with worked solutions

1. **Exercise:** Which coefficients satisfy zero Dirichlet data for all times? **Solution:** Adding and subtracting the endpoint equations gives c0=c1=0; c2 is unrestricted.
2. **Exercise:** Which coefficients satisfy homogeneous Neumann data? **Solution:** The endpoint gradients are opposite multiples of c2, so c2=0 is necessary and sufficient in this family.
3. **Exercise:** Does a small late-time cosine trace prove Dirichlet satisfaction? **Solution:** No. The positive exponential only shrinks the defect; it does not make a nonzero coefficient vanish.
4. **Exercise:** A candidate uses (0,0,2) and the initial data use (0,0,1). What passes? **Solution:** Both are Dirichlet-compatible, but the candidate's initial profile is wrong.
5. **Exercise:** Is the left outward derivative equal to u_x(0,t)? **Solution:** No. The left outward unit normal points toward decreasing x, giving minus u_x(0,t).
6. **Exercise:** Does a failed corner flag exclude every weak solution? **Solution:** No. It excludes the stated continuously traced class; other solution concepts require their own analysis.

## Sources and further study

The [MIT heat-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/8547cf4208529f2a7f3ce6d881f1f3c2_MIT18_152F11_lec_02.pdf) give background on initial-boundary problems and the sense of initial convergence. The trace identities and coefficient classifications here are derived directly for the displayed finite family. Continue with energy estimates, trace spaces and semigroup solutions to study wider classes of data.
