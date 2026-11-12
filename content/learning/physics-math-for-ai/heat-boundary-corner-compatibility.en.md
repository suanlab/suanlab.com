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

## Research extension: Construct a mild heat solution with incompatible corner values

### A failed continuous corner condition is not every kind of nonexistence

Normalize position by the rod length, temperature difference by a fixed amplitude, and time by L squared over kappa. On xi in (0,1), prescribe initial profile h=1 and zero Dirichlet values for every positive dimensionless time tau. A field continuous on the closed spacetime rectangle cannot satisfy both requirements at the two initial corners. We now construct a solution with a different, explicitly weaker initial interpretation: convergence to h in L2 over the open interval.

Integrating h against the sine basis gives coefficient zero for even n and 4/(n pi) for odd n. Propagating each mode yields

$$
u(\xi,\tau)={4\over\pi}\sum_{j=1}^{\infty}
{\sin((2j-1)\pi\xi)\over2j-1}
e^{-(2j-1)^2\pi^2\tau},\qquad \tau>0.
$$

For every time bounded away from zero, the Gaussian decay in mode number dominates any polynomial factors from differentiating the series. The series and its needed derivatives therefore converge uniformly there. Termwise differentiation verifies the interior heat equation, and every term has zero endpoint values. This constructs the Dirichlet heat-semigroup solution, often called a mild solution for its L2 initial datum. It is not contained in the laboratory's three-coefficient candidate family.

### Prove the initial condition in the declared norm

Sine functions of different integer frequencies are orthogonal and each has squared L2 norm one half. Parseval's identity gives the error to the constant initial field:

$$
\|u(\cdot,\tau)-1\|_2^2
={8\over\pi^2}\sum_{j=1}^{\infty}
{\left(1-e^{-(2j-1)^2\pi^2\tau}\right)^2\over(2j-1)^2}
\longrightarrow0\quad(\tau\downarrow0).
$$

Each summand tends to zero and is dominated by a summable inverse-square sequence. Dominated convergence proves the limit. Values assigned to h at just the endpoints do not affect its L2 class. However, retaining the continuous representative h=1 on the closed interval makes the corner conflict visible: for every positive tau, u(0,tau)=0 differs from h(0)=1. Uniform convergence on that closed interval is impossible. Smoothing for positive time does not restore joint continuity at the initial corners.

![A constant initial temperature profile conflicts with zero endpoint values, yet positive-time sine-series profiles satisfy those endpoints and approach the initial state in L2, with narrow boundary layers near the ends.](/assets/learning/physical-heat-mild-corner-en.svg)

### Certify a truncation before drawing it

Let u_N retain the first N odd modes. Orthogonality gives its exact squared tail as a positive series. Since all omitted frequencies are at least 2N+1 and the sum over all integers above 2N is bounded by an integral, we obtain a convenient conservative bound:

$$
\|u-u_N\|_2^2={8\over\pi^2}\sum_{j>N}
{e^{-2(2j-1)^2\pi^2\tau}\over(2j-1)^2}
\le {4\over\pi^2N}e^{-2(2N+1)^2\pi^2\tau}.
$$

Taking a square root gives two divided by pi sqrt(N), times exp(−(2N+1)² pi² tau). At tau=0.001, this upper bound is about 0.0863 for N=5, 0.00259 for N=10, and 8.9e−9 for N=20. The bound is an L2 truncation estimate, not a bound on every derivative or on floating-point summation error. The plotted curves use 80 odd modes; connecting them does not change the stated initial solution concept.

| Requirement | This constructed field | Why it matters |
|---|---|---|
| Interior heat equation for positive time | Satisfied by the differentiated series | Checks the evolution law |
| Zero endpoint values at positive time | Satisfied term by term | Checks Dirichlet traces |
| Initial convergence in L2 | Satisfied by the positive-series limit | Defines the initial condition here |
| Continuous initial corners for h=1 | Not satisfied | A stronger requested problem would be different |

### Worked research question

**Question:** Should a learner label these data simply “no solution” after a corner check fails? **Answer:** It should name the excluded class. There is no solution continuous through those corners with the stated values, yet the L2-initial heat field above exists. This also matters when evaluating neural surrogates near an initial boundary layer: a large pointwise corner discrepancy and a small integral error answer different questions. The code evaluates finite positive series and prints an initial-error enclosure whose omitted tail is bounded by 4/(pi² M). These are numerical evaluations of analytic bounds; rounded endpoints are not certified interval arithmetic. No arbitrary-initial-function solver is added to the existing browser controls.

```python
from math import pi, exp, expm1, sin, sqrt, fsum
def initial_error_enclosure(t,M=5000):
    lower=8/(pi*pi)*fsum((-expm1(-(2*j-1)**2*pi*pi*t))**2/(2*j-1)**2 for j in range(1,M+1))
    return lower,lower+4/(pi*pi*M)
previous_lower=float('inf')
for t in [.1,.01,.001,.0001]:
    lower,upper=initial_error_enclosure(t)
    assert upper<previous_lower
    previous_lower=lower
    print('time',t,'squared initial-error numerical enclosure',lower,upper)
for N in [5,10,20]:
    t=.001
    finite_tail=8/(pi*pi)*fsum(exp(-2*(2*j-1)**2*pi*pi*t)/(2*j-1)**2 for j in range(N+1,N+5001))
    bound=2/(pi*sqrt(N))*exp(-(2*N+1)**2*pi*pi*t)
    assert sqrt(finite_tail)<=bound
    print('odd modes',N,'truncation norm upper bound',bound)
def profile(x,t,N=80):
    if x in (0,1): return 0.0  # Analytically imposed endpoint values.
    return 4/pi*fsum(sin((2*j-1)*pi*x)/(2*j-1)*exp(-(2*j-1)**2*pi*pi*t) for j in range(1,N+1))
for t in [.001,.01,.1]:
    assert profile(0,t)==profile(1,t)==0
    assert abs(profile(0,t)-1)==1
    print('time',t,'midpoint',profile(.5,t),'endpoint discrepancy',1)
```

## Exercises with worked solutions

1. **Exercise:** Which coefficients satisfy zero Dirichlet data for all times? **Solution:** Adding and subtracting the endpoint equations gives c0=c1=0; c2 is unrestricted.
2. **Exercise:** Which coefficients satisfy homogeneous Neumann data? **Solution:** The endpoint gradients are opposite multiples of c2, so c2=0 is necessary and sufficient in this family.
3. **Exercise:** Does a small late-time cosine trace prove Dirichlet satisfaction? **Solution:** No. The positive exponential only shrinks the defect; it does not make a nonzero coefficient vanish.
4. **Exercise:** A candidate uses (0,0,2) and the initial data use (0,0,1). What passes? **Solution:** Both are Dirichlet-compatible, but the candidate's initial profile is wrong.
5. **Exercise:** Is the left outward derivative equal to u_x(0,t)? **Solution:** No. The left outward unit normal points toward decreasing x, giving minus u_x(0,t).
6. **Exercise:** Does a failed corner flag exclude every weak solution? **Solution:** No. It excludes the stated continuously traced class; other solution concepts require their own analysis.

## Sources and further study

The [MIT heat-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/8547cf4208529f2a7f3ce6d881f1f3c2_MIT18_152F11_lec_02.pdf) give background on initial-boundary problems and the sense of initial convergence. The trace identities and coefficient classifications here are derived directly for the displayed finite family. Continue with energy estimates, trace spaces and semigroup solutions to study wider classes of data.
