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
