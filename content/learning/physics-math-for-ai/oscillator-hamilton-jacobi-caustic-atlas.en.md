## A smooth motion can outlive its generating-function chart

A camera can lose its view of a moving object without the object disappearing. A generating function can likewise lose the ability to describe a smooth Hamiltonian flow in its chosen independent variables. This lesson makes that distinction visible with an exactly solvable oscillator.

We compare three projections: fixing the initial position and solving an endpoint problem, fixing the initial momentum and using a mixed generating function, and evolving a whole initial momentum graph. Each projection has a different denominator. Their failures must not be reported as the same event.

The experiment uses exact rational sine and cosine values at the query. It reports undefined charts explicitly while keeping the actual phase flow and its energy available.

## Start with the exact oscillator flow

Let $m>0$ and $\omega>0$ be constant, with units kg and s⁻¹. The physical Hamiltonian is

$$
H(q,p)=\frac{p^2}{2m}+\frac{m\omega^2q^2}{2}.
$$

Write $\theta=\omega T$, $C=\cos\theta$ and $s=\sin\theta$. From initial position $q_0$ and momentum $p_0$,

$$
\begin{pmatrix}q\\p\end{pmatrix}
=\begin{pmatrix}C&s/(m\omega)\\-m\omega s&C\end{pmatrix}
\begin{pmatrix}q_0\\p_0\end{pmatrix}.
$$

The determinant is $C^2+s^2=1$. This phase map is invertible at every time. Its off-diagonal entries have reciprocal physical units; position and momentum are not interchangeable Euclidean coordinates.

Differentiating gives $\dot q=p/m$ and $\dot p=-m\omega^2q$. Direct substitution preserves the initial energy exactly. Those facts remain true at every chart failure discussed below.

## Separate a phase initial-value problem from an endpoint problem

An initial-value problem specifies both $q_0$ and $p_0$, so the displayed flow determines a unique phase state. An endpoint problem specifies $q_0$, a target final position $x$ and a duration $T$. The initial momentum may then need to be solved for.

For $s\ne0$,

$$
p_0^{\mathrm{required}}=m\omega\frac{x-Cq_0}{s},\qquad
p_f=m\omega\frac{Cx-q_0}{s}.
$$

This required momentum need not equal the experiment's selected $p_0$. The target endpoint is a separate query control. Moving it changes the requested boundary problem while the original phase trajectory stays fixed.

That separation prevents a common mistake: reading an endpoint action at an arbitrary target as the action of the already selected trajectory.

## Derive the endpoint principal action

For $s\ne0$, integrating the oscillator Lagrangian along the unique boundary solution gives

$$
A(x,q_0,T)=\frac{m\omega}{2s}
\left[C(x^2+q_0^2)-2xq_0\right].
$$

Its derivatives are

$$
A_x=p_f,\qquad A_{q_0}=-p_0^{\mathrm{required}},\qquad
A_T+H(x,A_x)=0.
$$

The minus sign at the initial endpoint comes from varying the lower boundary of the action. Treating both endpoint derivatives with the same sign would reverse the inferred initial momentum.

The mixed Hessian is $A_{xq_0}=-m\omega/s$, with units kg/s. The generating function is valid only on its declared chart, not by replacing a zero sine with a small number.

## Understand half-period focusing without a physical singularity

When $s=0$, the final position is $q=Cq_0$, independently of $p_0$. If the target is different, no initial momentum reaches it at that exact duration. If the target agrees, initial momentum is undetermined by the endpoint positions.

At a half period $C=-1$. All trajectories with one fixed initial position reach its opposite position, while their final momenta still differ. The full phase map is $-I$, not a rank-zero map. Position projection has lost information that phase space retains.

At zero duration the same algebra leaves initial momentum undetermined, but it does not describe distinct positive-duration position histories. The model marks this boundary separately. At a full period the phase map is $I$, and compatible endpoints again do not determine initial momentum.

## Use a complementary mixed complete integral

Keep final position $x$ and initial momentum $\alpha$ independent. For $C\ne0$, solve the phase flow for the initial position and integrate the final momentum with respect to $x$. This gives

$$
S_2(x,\alpha,T)=\frac{1}{C}\left[
 x\alpha-\frac{m\omega s}{2}x^2
 -\frac{s}{2m\omega}\alpha^2\right].
$$

Then

$$
S_{2,x}=\frac{\alpha-m\omega sx}{C}=p_f,\qquad
S_{2,\alpha}=\frac{x-s\alpha/(m\omega)}{C}=q_0.
$$

It also satisfies $S_{2,T}+H(x,S_{2,x})=0$. Its mixed Hessian is $1/C$. At a half period it remains regular even though the endpoint principal-action chart fails.

## Check a quarter period and switch charts

At a quarter period $C=0$ and $s=1$. The mixed chart cannot determine $q_0$ from $(x,\alpha)$: final position is already fixed to $\alpha/(m\omega)$. The endpoint chart remains regular because $s\ne0$.

Take $m=\omega=1$, $q_0=p_0=1$, with target $x=1$. The final phase state is $(1,-1)$ in its SI units, energy is $1$ J and endpoint action is $-1$ Js. The mixed generating function is undefined; the physical motion is not.

Sine and cosine cannot vanish together. At least one of these two generating charts therefore covers the exact oscillator phase map at every duration. This statement concerns this linear oscillator atlas; it is not a guarantee of two global charts for an arbitrary nonlinear system.

## Compute the physical action without dividing by a chart denominator

Along any physical oscillator trajectory,

$$
\frac d{dt}(qp)=\frac{p^2}{m}-m\omega^2q^2=2L.
$$

Therefore

$$
I(T)=\frac{q(T)p(T)-q_0p_0}{2}.
$$

This expression stays finite when either generating chart fails. On a regular endpoint chart with the selected trajectory reaching the target, $A=I$. On a regular mixed chart evaluated along the selected trajectory,

$$
S_2(q(T),p_0,T)-q_0p_0=I(T).
$$

Subtract the initial generating value. At a half period $I=0$ for every selected phase state, even when its energy is positive. Zero action is not zero energy, zero motion or proof that an arbitrary boundary problem is well posed.

## Evolve an initial action graph

Now consider many initial conditions at once:

$$
p_0=\sigma\xi,\qquad S_{\mathrm{initial}}(\xi)=\frac\sigma2\xi^2,
\qquad [\sigma]=\mathrm{kg/s}.
$$

The parameter $\xi$ is the initial position. Transporting this line by the exact phase flow gives

$$
q=a\xi,\qquad p=b\xi,\qquad
a=C+\frac{\sigma s}{m\omega},\quad
b=-m\omega s+\sigma C.
$$

The selected trajectory belongs to this family only when its initial momentum equals $\sigma q_0$. The membership flag matters: changing the graph slope should not silently replace the selected physical initial state.

The transported line remains a smooth one-dimensional phase-space object. It need not remain the graph of a single-valued momentum over position.

## Derive the initial-value action before its caustic

For $a\ne0$, the initial label reaching position $x$ is $\xi=x/a$. The momentum field and transported action are

$$
p(x,T)=\kappa(T)x,\qquad
S_g(x,T)=\frac{\kappa(T)}2x^2,\qquad \kappa=b/a.
$$

The action includes both the initial action and the physical action accumulated along its characteristic. Differentiating gives

$$
\dot a=b/m,\qquad \dot b=-m\omega^2a,\qquad
\dot\kappa=-m\omega^2-\kappa^2/m.
$$

It follows directly that $S_{g,T}+H(x,S_{g,x})=0$. This is an initial-value solution for the declared momentum graph, rather than the endpoint function with one fixed initial position.

Its Hessian can diverge while every phase point on the transported line stays finite. A failed position projection is the issue.

## Locate a graph caustic that neither endpoint chart detects

A graph caustic occurs when $a=0$. Then every initial label reaches $q=0$, but the final momenta $b\xi$ differ. Since the phase flow is invertible, $a$ and $b$ cannot both vanish.

For a concrete non-quarter-period example choose $m=\omega=1$, $C=3/5$, $s=4/5$ and $\sigma=-3/4$ kg/s. Then $a=0$ and $b=-5/4$ kg/s. Both $s$ and $C$ are nonzero, so both earlier generating charts remain regular.

Choose the family member $\xi=2$ m, $p_0=-3/2$ kg m/s. It reaches $q=0$, $p=-5/2$ kg m/s, with energy $25/8$ J and physical action $3/2$ Js. Its initial graph action is $-3/2$ Js. Their sum is zero, but different labels have different final momenta at the same position. One common action value at the focused point does not define one differentiable momentum field around it.

## Read the rational clock and the plots correctly

The controls choose a quadrant $k\in\{0,1,2,3\}$ and $u\in[0,1]$. First compute

$$
C_0=\frac{1-u^2}{1+u^2},\qquad s_0=\frac{2u}{1+u^2},
\qquad \theta=k\pi/2+2\arctan u.
$$

Rotate $(C_0,s_0)$ by the selected quadrant. Query sine, cosine, ranks, chart flags and action values are exact rational quantities. Physical time $T=\theta/\omega$ is a numerical clock label. This is a parametrization of the exact oscillator flow, not a Cayley integration step with step size $2u/\omega$.

Time plots cover one physical period. Singular action/Hessian curves are split at undefined samples and sign changes of their denominators; lines must not connect across a pole. Phase plots label position in m and momentum in kg m/s. Graph projection $a$ is dimensionless, while endpoint sensitivity $s/(m\omega)$ has units s/kg.

## Transfer the distinctions to AI and inverse problems

A learned scalar action may fit a smooth local branch while its projection fails elsewhere. Tests should ask which independent variables define the branch, what its mixed Hessian means, and whether the requested endpoint or initial graph is compatible with the data.

For a generating-function model, compare derivative momenta, Hamilton-Jacobi residuals, actual phase trajectories and independently integrated actions. Check selected-state membership before attributing a graph property to one trajectory. Keep chart failure, boundary nonuniqueness, graph caustics and phase-flow instability as different outcomes.

This example does not compute viscosity solutions, select a minimum among multivalued branches, add quantum phases, or prove that a learned action generalizes globally. Those require additional definitions and evidence. The exact oscillator provides an interpretable diagnostic setting for developing such research questions.

## Research extension: a folded phase graph and a selected value function are different objects

The native oscillator shows that a smooth phase flow can outlive a generating chart. It does not select a viscosity solution. This supplement introduces a different free-particle initial-value problem, with a nonlinear initial action, to derive what a minimum selection adds and what information it discards. The native oscillator's Hamiltonian, rational clock and chart controls remain unchanged.

### Transport a nonlinear initial momentum graph

Use position xi in meters and momentum in kg m/s. The initial action has units Js, so a has units Js/m⁴ and b has units Js/m²:

$$
H(p)=\frac{p^2}{2m},\quad S_{\rm ini}(\xi)=\frac a4\xi^4-\frac b2\xi^2,\qquad a,b,m>0.
$$

Each initial label xi carries momentum S_ini'(xi). The exact free phase flow transports the whole graph:

$$
p_0(\xi)=a\xi^3-b\xi,\quad
x(\xi,t)=\left(1-\frac{bt}{m}\right)\xi+\frac{at}{m}\xi^3,\quad
p(\xi,t)=p_0(\xi),\qquad t_c=\frac mb.
$$

At t_c the position derivative at xi=0 first vanishes. For later times the position projection folds. The phase graph itself stays smooth: its derivative is (1+t p0'/m,p0'), whose two entries cannot vanish together. The full phase map (q0,p0) to (q0+t p0/m,p0) is invertible at every time. Different initial labels sharing a final position still have different momenta.

### Define an optimization problem in addition to the flow

For t>0 define the Hopf–Lax value

$$
V(x,t)=\min_{\xi\in\mathbb R}\Phi(\xi;x,t),\qquad
\Phi=S_{\rm ini}(\xi)+\frac{m(x-\xi)^2}{2t},\quad t>0.
$$

For a free particle, the kinetic action between xi and x is m(x−xi)²/(2t). This follows from Cauchy–Schwarz, with equality for constant velocity. Adding the initial action and minimizing over initial positions defines a new value function. It is not the action of one preselected phase initial condition. Because a>0, the objective grows to positive infinity as abs(xi) grows, so a finite global minimum is attained.

Every minimizer must be a root of the stationary cubic, but a stationary root need not minimize:

$$
\Phi_\xi=a\xi^3+(m/t-b)\xi-mx/t=0,\qquad
\Phi_{\xi\xi}=3a\xi^2+m/t-b.
$$

For t<t_c the second derivative is everywhere positive and the minimizer is unique. At t=t_c and x=0, the remaining objective is a positive quartic: the minimizer is still unique despite its zero Hessian. After that time, a different situation arises at the symmetric query.

### Compare all three stationary branches

At x=0 after the critical time,

$$
x=0,\quad t>t_c:\qquad
\xi=0,\quad\xi_\pm=\pm\sqrt{\frac{b-m/t}{a}},\qquad
V(0,t)=-\frac{(b-m/t)^2}{4a}.
$$

The center is a local maximum and the two nonzero roots are equal global minima. The phrase stationary action alone would not tell us which ones define V. Negative V here reflects the negative wells of the chosen initial action; it does not mean negative kinetic energy.

For x>0, reflecting any negative xi to positive xi lowers the linear cross term −mx xi/t while leaving the even part unchanged. The positive half-line has exactly one stationary minimum that can win. The sign reverses for x<0. Thus the two winning branches exchange at x=0, producing the one-sided derivatives

$$
s(t)=\frac mt\sqrt{\frac{b-m/t}{a}}>0,\qquad
V_x(0^-,t)=s(t),\quad V_x(0^+,t)=-s(t),\quad
V_t(0,t)=-\frac{s(t)^2}{2m}.
$$

The value is continuous but has no unique spatial derivative there. Averaging the two momenta to zero would not produce the derivative of this cusp. Nor would the central stationary root become a minimizing branch because its momentum is zero.

### Check the PDE and the nonsmooth condition

On a smooth unique minimizing branch, differentiating the objective at that minimizing label cancels the derivative of the label because Phi_xi=0. Therefore

$$
\xi_*\text{ a smooth minimizing branch}:\qquad
V_x=\frac{m(x-\xi_*)}{t},\quad
V_t=-\frac{m(x-\xi_*)^2}{2t^2},\quad V_t+\frac{V_x^2}{2m}=0.
$$

At the critical flat minimum we do not invoke differentiable inversion of the stationary cubic; the localized value argument below still applies. At the cusp, a classical derivative is unavailable. A viscosity subsolution is tested by smooth functions touching the value from above; a supersolution is tested from below, with the opposite PDE inequality. For t>t_c the upper-test condition here is

$$
\varphi\text{ touches }V\text{ from above at }(0,t),\ t>t_c:\qquad
\varphi_x\in[-s,s],\quad \varphi_t=-\frac{s^2}{2m},\quad
\varphi_t+\frac{\varphi_x^2}{2m}\le0.
$$

The inequality follows since the upper-test slope lies between the two one-sided slopes. No smooth lower test can touch this downward cusp: its x slope would have to be at least the positive left slope and at most the negative right slope simultaneously. The supersolution condition there is consequently vacuous. This explains the viscosity test at the branch exchange without choosing an invented single momentum.

### Respect the initial-data assumptions of the formula

Tran's [Theorem 2.22, section 2.5.4](https://people.math.wisc.edu/~htran24/HJ%20equations-viscosity%20solutions%20and%20applications-v1.pdf) treats convex superlinear Hamiltonians with bounded uniformly continuous initial data. Our quartic initial action is not in that class globally. To apply the result locally, fix a compact interval K of query positions and choose an action level C above its maximum initial action. Set

$$
g_C(\xi)=\min(S_{\rm ini}(\xi),C),\quad C>\max_{x\in K}S_{\rm ini}(x),\qquad
S_{\rm ini}(\xi)\ge C\ \Longrightarrow\
g_C(\xi)+\frac{m(x-\xi)^2}{2t}\ge C>S_{\rm ini}(x)\ge V(x,t)\quad(x\in K).
$$

The capped function is globally bounded and Lipschitz: the original quartic is bounded below, and its derivative is bounded on the compact sublevel where it is not capped. A newly capped candidate cannot win, since the unchanged test xi=x already costs less than C. The capped and original value functions therefore coincide on K for every positive t. The bounded-data result applies there, and viscosity is a local property. This also gives the original initial value as t tends to zero locally. We do not infer uniqueness in an unspecified class of functions with arbitrary growth at infinity.

This minimum selection is appropriate to the declared convex kinetic optimization problem. It is not a claim that every mechanical stationary action is a minimum. The [earlier Morse-index example](/book/online/physics-math-for-ai/harmonic-action-stationary-saddle/) explains why that broader claim would fail.

![A smooth phase graph folds over position, a quartic endpoint objective has two equal minima, and selecting them gives a cusp in the Hopf–Lax value.](/assets/learning/analytical-hj-value-selection.en.svg)

### Calculate stationary roots and minimum values

Use a=b=m=1 in SI reference units and x=0. The critical time is 1 s. At that time the one minimizer is flat, rather than two distinct minimizers.

| t (s) | distinct stationary roots | global minimizers | V(0,t) (Js) |
|---|---|---|---|
| 0.5 | 1 | 1 | 0 |
| 1 | 1 | 1 | 0 |
| 1.5 | 3 | 2 | −0.027778 |
| 2 | 3 | 2 | −0.0625 |

The standard-library code splits the cubic into monotone intervals, searches each one, and compares objective values. It uses numerical tolerances for these specified examples; the root counts and tie at x=0 are proved analytically above. Near a general fold, numerical classification needs its own error analysis.

```python
from math import sqrt

# SI reference a=b=m=1. Split the cubic into monotone intervals.
def branches(x, t):
    if t <= 0:
        raise ValueError("Require t>0; initial data are specified separately")
    B, C = 1/t-1, -x/t
    poly = lambda y: y**3+B*y+C
    bound = 1+max(abs(B), abs(C))
    cuts = [-bound, bound]
    if B < 0:
        turn = sqrt(-B/3)
        cuts = [-bound, -turn, turn, bound]
    roots = [y for y in cuts if abs(poly(y)) < 1e-12]
    for lo, hi in zip(cuts, cuts[1:]):
        if poly(lo)*poly(hi) >= 0:
            continue
        for _ in range(80):
            mid = (lo+hi)/2
            if poly(lo)*poly(mid) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append((lo+hi)/2)
    unique = []
    for y in sorted(roots):
        if not unique or abs(y-unique[-1]) > 1e-9:
            unique.append(y)
    costs = [y**4/4-y*y/2+(x-y)**2/(2*t) for y in unique]
    best = min(costs)
    winners = [y for y,cost in zip(unique,costs) if abs(cost-best) < 1e-10]
    return unique, winners, best

for t in [.5, 1, 1.5, 2]:
    roots, winners, v = branches(0, t)
    print(t, len(roots), len(winners), round(v, 6))
assert branches(-.1, 2)[1][0] < 0 < branches(.1, 2)[1][0]
```

The independent verifier compares global minima with a separate dense objective grid, checks the bounded-data localization, differentiates V away from ties, and tests the cusp inequality with exact fractions. A finite grid check is not used as a replacement for the coercivity and branch arguments.

### Research questions with explanations

**Does a caustic mean that physical phase evolution becomes nonunique?** **Answer:** Not in this example. Each phase initial condition has a unique smooth free evolution. Nonuniqueness occurs when minimizing over initial labels or projecting the transported family onto position.

**Why not keep every stationary branch as one differentiable action?** **Answer:** The declared value function chooses the lowest objective. After the critical time the center is a maximum and the two minima tie only at the symmetric query. The lower envelope cannot retain all their momenta as one derivative.

**Does viscosity selection add physical damping to the particles?** **Answer:** No damping term is integrated here. This is a weak-solution criterion for a scalar optimization value. The full phase flow remains reversible; the selected scalar value discards branch information, so it cannot recover that entire flow by itself.

## Work through six checks with explanations

### Exercise 1

At the quarter-period example, calculate the final state and explain why the missing mixed function is harmless to the phase flow.

**Solution:** With $C=0$, $s=1$, $q_0=p_0=m=\omega=1$, the state is $q=1$, $p=-1$. The phase determinant remains one. Only the attempt to recover $q_0$ from $(x,\alpha)$ divides by zero. The endpoint action is regular and equals $-1$ Js.

### Exercise 2

At a half period with $q_0=1$ m, compare target endpoints $x=-1$ m and $x=1$ m.

**Solution:** Final position must be $-1$ m independently of initial momentum. The first target leaves momentum undetermined; the second target is unreachable at that duration. Neither result makes the phase map $-I$ singular.

### Exercise 3

Why must the initial endpoint momentum have a minus sign when extracted from the principal action?

**Solution:** Varying an action integral gives the final boundary contribution minus the initial one. Thus $\delta A=p_f\delta x-p_0\delta q_0-H_f\delta T$ for fixed initial time. Consequently $p_0=-A_{q_0}$.

### Exercise 4

For the generic graph caustic, calculate $a$ and $b$ and identify the unavailable object.

**Solution:** $a=3/5-(3/4)(4/5)=0$ and $b=-4/5-(3/4)(3/5)=-5/4$ kg/s. The smooth line becomes vertical over position. A single-valued $p(x)$ and the local graph action chart fail, while the endpoint and mixed charts remain regular.

### Exercise 5

Can changing $\sigma$ change the selected physical trajectory when $q_0,p_0$ are held fixed?

**Solution:** No. It changes the initial family and its projection, not the selected initial state. The membership flag can change. Comparing family plots without that flag can wrongly attribute a caustic to a trajectory that was never in the family.

### Exercise 6

At a full period, what are the state, physical action and endpoint-chart status?

**Solution:** The state returns to $(q_0,p_0)$, physical action is zero and energy retains its initial value. The endpoint principal-action chart is undefined because $s=0$. Compatible positions leave initial momentum undetermined, while the mixed chart is regular because $C=1$.

## Sources and the limits of this example

The generating-function framework is described in [Tong, Classical Dynamics, sections 4.4.3 and 4.7](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4) and [Sussman and Wisdom, Canonical Evolution, section 6.1](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter006.html). The oscillator charts, exact rational clock, transported initial graph and boundary classifications are derived here. Independent symbolic checks integrate the physical Lagrangian and differentiate the actions before inserting any rational-clock samples. The discussion is classical and does not import quantum propagator or phase conventions.
