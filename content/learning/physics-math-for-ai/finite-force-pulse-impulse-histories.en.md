## A push has a duration and a history

Imagine pushing two identical carts on a frictionless track. In the first experiment you push steadily; in the second you start gently and increase the force. You arrange equal signed areas under the force curves. The carts finish with the same velocity, yet one has traveled farther. This is the distinction between a final change and the history that produced it.

We use a Newtonian point particle with constant positive mass m, initial position zero and known initial velocity v0. Forces are prescribed functions of time in an inertial laboratory. These are two separate experiments with identical initial data, not two forces simultaneously acting on one cart. The model contains no contact detector or collision rule.

## Specify both force profiles and the switching convention

Let duration τ be positive and amplitude F0 be signed. Experiment A uses a flat pulse; experiment B uses a linear ramp:

$$
F_A(t)=\begin{cases}F_0&0\le t<\tau,\\0&t\ge\tau,\end{cases}
\qquad
F_B(t)=\begin{cases}2F_0t/\tau&0\le t<\tau,\\0&t\ge\tau.\end{cases}
$$

The ramp reaches a left-hand limiting value of twice the flat amplitude. Its average over the pulse is still F0. We define force to be zero at the switching instant itself. Changing a finite value at a single instant would not change any integral below. Vertical force segments in the illustration mark one-sided jumps; they do not describe a physical interval during which all intermediate forces occur.

## Momentum balance integrates signed force

For constant mass, the momentum law is dp/dt=F. Integrate it from zero to the query time. Writing u=min(t,τ) gives

$$
J_A(t)=F_0u,\qquad J_B(t)=\frac{F_0u^2}{\tau},\qquad
p(t)-mv_0=J(t).
$$

Impulse has units N s, equivalent to kg m/s. It is signed in this one-dimensional problem. A negative pulse decreases signed momentum; it need not decrease speed throughout the experiment. Before the pulse finishes, the accumulated impulses generally differ. At and after the end, both equal F0τ. This is an external-impulse balance, rather than an assertion that this particle's momentum is conserved.

## Velocity follows accumulated area

Divide momentum by the same constant mass:

$$
v_A(t)=v_0+\frac{F_0u}{m},\qquad
v_B(t)=v_0+\frac{F_0u^2}{m\tau}.
$$

For positive amplitude, the flat experiment gains velocity earlier. At the pulse end the ramp catches up. Thereafter both particles move freely with their shared terminal velocity. For zero amplitude the two experiments reduce to the same uniform motion. No formula divides by amplitude, so the zero-force case remains well defined.

A finite force jump changes acceleration discontinuously. It does not create a finite jump in velocity: an interval shrinking to zero has vanishing force area when the force remains bounded. An ideal Dirac impulse is a different mathematical model and is not silently introduced here.

## Integrate again to obtain position

The time at which momentum was acquired matters for displacement. An infinitesimal force contribution at time s changes velocity for the remaining interval t−s. Therefore

$$
x(t)=v_0t+\frac1m\int_0^u(t-s)F(s)\,ds.
$$

Carrying out this integral yields the formulas used by the interactive model:

$$
x_A(t)=v_0t+\frac{F_0}{m}\left(ut-\frac{u^2}{2}\right),
$$

$$
x_B(t)=v_0t+\frac{F_0}{m\tau}\left(u^2t-\frac{2u^3}{3}\right).
$$

These expressions also cover times after the pulse. Simply substituting the query time into a during-pulse cubic would incorrectly keep accelerating the ramp particle forever. The clipped time u prevents that mistake while the unmodified t retains later free travel.

## Equal terminal impulse leaves a position gap

At t=τ the two positions differ by

$$
x_A(\tau)-x_B(\tau)=\frac{F_0\tau^2}{6m}.
$$

For positive amplitude, the earlier push puts A ahead. For negative amplitude the signed ordering reverses. After the pulse, the velocity difference is zero, so this position gap remains constant. The initial velocity cancels in the difference because both experiments started with the same value.

This gives a concrete inverse-problem warning: terminal momentum alone cannot identify the force history or final position. Even with duration known, the flat and ramp histories are distinct explanations of the same final momentum change. A peak force alone also cannot determine impulse without temporal information.

## Work uses force times actual velocity

Kinetic energy is K=mv²/2. Using dJ=F dt gives an exact expression for accumulated work in either experiment:

$$
W=\int_0^tF(s)v(s)\,ds
=\int_0^{J(t)}\left(v_0+\frac{j}{m}\right)dj
=v_0J+\frac{J^2}{2m}=K(t)-K(0).
$$

The antiderivative argument remains valid for negative impulse. Work may be negative when a force initially opposes motion. Equal terminal impulse and identical initial data imply equal terminal kinetic changes here, even though terminal positions differ. Multiplying endpoint force by total displacement fails for the time-varying ramp; after the switch endpoint force is zero despite earlier nonzero work.

## Read the three diagrams in order

The first diagram shows force against time in N. Compare signed areas, not just the heights. The second shows accumulated impulse in N s; the curves meet at the switch and remain together afterward. The third shows position in meters, where the surviving offset is visible. Each panel has its own vertical scale and unit.

Query time changes the exact table and readouts. The plotted histories retain a fixed ten-second horizon so a query change does not disguise later behavior by rescaling time. Set amplitude to zero, reverse its sign, and choose the shortest duration to inspect limiting cases. The exact rational table is the authority for current control values; sampled lines illustrate continuous histories.

## Research extension: endpoint data, invisible force histories and a minimum-effort choice

The flat and ramp pulses already show that the same final momentum can leave different positions. What if we measure both final position and final velocity? Those two observations still do not identify a whole time-dependent force. They measure two weighted integrals of it. We will characterize those integrals, construct a positive-force counterexample and explain what an extra minimum-effort assumption selects.

### Two endpoint observations are two temporal moments

Keep a one-dimensional inertial particle with constant m>0, known initial q0,v0 and a fixed horizon T>0. Allow any square-integrable prescribed force on [0,T], followed by zero force. There is no collision law. Integrating the exact Newton equation twice shows that target position X and velocity V require:

$$
J=m(V-v_0)=\int_0^T F(s)\,ds,\qquad
B=m(X-q_0-v_0T)=\int_0^T(T-s)F(s)\,ds.
$$

J has units N s and B has units N s². A late force contributes as much to momentum as an equal early impulse, but less to final position because its remaining time T−s is smaller. Knowing both moments constrains timing more strongly than impulse alone, while leaving infinitely many possible histories.

This broadens the analytical force family, not the current controls: the native widget continues to offer its flat and ramp profiles. New affine and quadratic profiles below are standalone examples with executable code.

### Select an affine force and prove why it is optimal

Suppose we additionally choose to minimize effort E(F)=integral F(s)² ds. The units are N²s; this is a mathematical control penalty. It is not mechanical work, which is integral F(s)v(s) ds in joules. An electrical-energy interpretation would need an actuator model.

The two observation kernels are 1 and T−s. Seek the minimum-effort force in their span. Substitution into the two moment constraints gives the Gram system:

$$
F_*(s)=a+b(T-s),\quad
\begin{pmatrix}T&T^2/2\\T^2/2&T^3/3\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}=\begin{pmatrix}J\\B\end{pmatrix},
\quad \det G=\frac{T^4}{12}>0.
$$

Its entries are the integrals of products of the kernels: integral 1=T, integral (T−s)=T²/2 and integral (T−s)²=T³/3. The positive determinant makes its two coefficients unique:

$$
a=\frac{4J}{T}-\frac{6B}{T^2},\qquad
b=-\frac{6J}{T^2}+\frac{12B}{T^3}.
$$

Here a has units N and b has units N/s. To prove global optimality over the full force space, let any other feasible force be F=F_*+w. Subtracting their moment equations makes w orthogonal to each kernel and therefore to F_*:

$$
\int_0^T w(s)\,ds=\int_0^T(T-s)w(s)\,ds=0,
\quad \langle F_*,w\rangle=0,\qquad
\mathcal E(F)=\mathcal E(F_*)+\int_0^T w(s)^2\,ds.
$$

The last equality comes from expanding the square and canceling twice the inner product. The remaining squared norm is nonnegative and vanishes only if w=0 almost everywhere. Thus F_* is the unique minimizer up to changes on sets of zero time measure. Values assigned at isolated switching instants do not affect the argument.

The cost simplifies to a useful nonnegative form:

$$
\mathcal E(F_*)=\frac{4J^2}{T}-\frac{12JB}{T^2}+\frac{12B^2}{T^3}
=\frac{J^2}{T}+\frac{12(B-JT/2)^2}{T^3}.
$$

The first term is the price of the required impulse. The second penalizes a position moment that differs from the constant-force value JT/2. This is a proof for unrestricted signed L2 forces. Amplitude limits, a nonnegative-force requirement or actuator dynamics can make this affine force inadmissible and require a different optimization.

### Work two target examples by hand

Set m=1 kg, T=2 s and q0=v0=0. For X=1 m,V=1 m/s, J=1 N s and B=1 N s². The coefficients are a=1/2 N,b=0: a constant 0.5 N force meets both targets. For X=1 m,V=0, J=0 and B=1, giving a=−3/2 N,b=3/2 N/s; hence F_*(s)=1.5−1.5s in the declared SI numerical units. It accelerates and then brakes, ending at rest after moving one metre.

| V (m/s) | X (m) | minimum effort (N²s) | net mechanical work (J) |
|---|---|---|---|
| 1 | 1 | 1/2 | 1/2 |
| 0 | 1 | 3/2 | 0 |

The second row makes the distinction concrete: returning to zero velocity gives zero net kinetic work while squared-force effort remains positive. Its positive and negative work contributions cancel; the actuator's actual energy consumption remains unspecified.

### Hide a force perturbation from both endpoint observations

Use the dimensionless shifted Legendre polynomial below. Direct polynomial integration proves its three identities:

$$
P_2(u)=6u^2-6u+1,\qquad
\int_0^1P_2(u)\,du=\int_0^1uP_2(u)\,du=0,
\qquad \int_0^1P_2(u)^2\,du=\frac15.
$$

A perturbation A·P2(s/T), with A in newtons, has zero impulse and zero position moment. Adding it to any feasible history preserves both endpoints. Its squared norm is A²T/5. In particular, compare these two forces with T=2 s and m=1 kg:

$$
F_A(s)=\frac12\ \mathrm N,\quad
F_B(s)=F_A(s)+A P_2(s/T),\quad A=\frac12\ \mathrm N,\quad 0\le s<T,
\qquad F_A(s)=F_B(s)=0\quad(s\ge T).
$$

The polynomial ranges from −1/2 to 1, so F_B remains between 0.25 and 1 N during the pulse. Both forces are positive; ambiguity does not depend on allowing a force reversal. Both produce q(T)=1 m and v(T)=1 m/s. Their efforts differ: 1/2 and 3/5 N²s. The extra 1/10 follows from orthogonality and A²T/5, not a numerical fit.

Integrate the perturbation to see exactly where the trajectories separate:

$$
q_B(t)-q_A(t)=\frac{A}{2m}t^2(1-t/T)^2,\qquad
v_B(t)-v_A(t)=\frac{A}{m}t(1-t/T)(1-2t/T),\quad 0\le t\le T.
$$

At t=T/2, the velocity difference is zero but the position difference is AT²/(32m)=1/16 m. After T both position and velocity differences remain zero. Even adding a midpoint velocity observation would miss this particular perturbation; a midpoint position observation separates it.

![Two positive force histories have identical final position and velocity, yet their intermediate position gap reaches 1/16 metre; switching to zero force occurs at two seconds.](/assets/learning/classical-pulse-moments.en.svg)

The upper panel uses newtons, the lower panel the signed position gap in metres. In the force panel, solid blue is the constant force and dashed orange the quadratic perturbation. Open markers at T show left limits and the filled marker shows the defined zero force at T. Vertical segments mark jumps, rather than time intervals containing intermediate forces.

### Reproduce the moments and effort exactly

This standard-library example uses rational arithmetic for polynomial integration, the Gram inverse, endpoint constraints, orthogonality and the midway discrepancy. Numerical times are in seconds. Initial state is fixed to zero in the code; the general formulas above retain it.

```python
from fractions import Fraction as F

def integral(poly, t, power=0):
    return sum(c*t**(i+power+1)/F(i+power+1)
               for i,c in enumerate(poly))

def multiply(a,b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            out[i+j] += x*y
    return out

def position(poly,t,m):
    return (t*integral(poly,t)-integral(poly,t,1))/m

# SI numerical values; m=1 kg, q0=v0=0, T=2 s.
m, T = F(1), F(2)
G = [[T,T*T/2],[T*T/2,T**3/3]]
Gi = [[4/T,-6/T**2],[-6/T**2,12/T**3]]
for i in range(2):
    for j in range(2):
        assert sum(G[i][k]*Gi[k][j] for k in range(2)) == (i==j)

for V,X in [(F(1),F(1)), (F(0),F(1))]:
    J,B = m*V,m*X
    a,b = [row[0]*J+row[1]*B for row in Gi]
    force = [a+b*T,-b]  # coefficients in increasing powers of t
    assert integral(force,T) == J
    assert position(force,T,m) == X
    effort = integral(multiply(force,force),T)
    assert effort == J*J/T+12*(B-J*T/2)**2/T**3
    print("V, X, force coefficients, effort, work:",
          V,X,force,effort,J*J/(2*m))

# P2(t/T) has zero zeroth and first time moments.
p2 = [F(1),-6/T,6/T**2]
assert integral(p2,T) == integral(p2,T,1) == 0
assert integral(multiply(p2,p2),T) == T/5
A = F(1,2)
flat = [F(1,2)]
perturbed = [flat[0]+A*p2[0],A*p2[1],A*p2[2]]
for force in [flat,perturbed]:
    assert integral(force,T)/m == 1
    assert position(force,T,m) == 1
assert integral(perturbed,T/2) == integral(flat,T/2)
gap = position(perturbed,T/2,m)-position(flat,T/2,m)
assert gap == A*T*T/(32*m) == F(1,16)
assert integral(multiply(flat,flat),T) == F(1,2)
assert integral(multiply(perturbed,perturbed),T) == F(3,5)
assert F(1,2)-A/2 == F(1,4)  # minimum force, attained at T/2
print("same endpoint; same midway velocity; midway position gap:",gap)
print("efforts in N^2 s:",F(1,2),F(3,5))
```

### Worked questions for learned control

1. **Does the unique minimum-effort answer identify the true force history?** No. The quadratic history fits the same endpoints but has larger effort. The penalty selects one explanation through an additional modeling preference; the observations alone did not supply that preference.
2. **Which extra observation distinguishes the constructed histories?** At 1 s, both velocities equal 0.5 m/s, but their positions differ by 1/16 m. Measuring position there adds information missing from endpoint position and velocity.
3. **What happens if a fixed displacement must be completed ever faster with zero terminal velocity?** With zero initial state, J=0 and B=mX, so minimum effort is 12m²X²/T³. It diverges as T approaches zero. Actual actuator constraints may make the target infeasible before that limit.

These moment, projection and polynomial calculations are direct derivations for the declared Newtonian system. They do not infer a physical force law from endpoint supervision or certify a contact model.

## Six exercises with worked solutions

1. **Exercise: end of the pulse.** Set m=1kg, F0=2N, τ=2s, v0=0. Find both states at2s. **Solution:** J=4N s and v=4m/s in both. Positions are4m and8/3m. Both kinetic energies are8J; their separation is4/3m.
2. **Exercise: halfway.** Keep the settings and query1s. **Solution:** JA=2 and JB=1N s. Velocities are2 and1m/s; positions are1 and1/3m. Equal final impulse does not imply equal partial impulse.
3. **Exercise: later free motion.** Query4s. **Solution:** both velocities remain4m/s. Adding two seconds of free travel gives xA=12m and xB=32/3m. The gap remains4/3m.
4. **Exercise: braking work.** Take m=1kg, v0=2m/s, F0=−1N, τ=1s at the end. **Solution:** J=−1N s and v=1m/s. Work is−2+1/2=−3/2J in both, matching K changing from2 to1/2J. Their signed position difference is−1/6m.
5. **Exercise: mass scaling.** Double the first exercise's mass while holding force and duration fixed. **Solution:** impulse stays4N s, velocity becomes2m/s and the gap becomes2/3m. Impulse is prescribed by force history; its velocity effect depends on mass.
6. **Exercise: switching instant.** What are both forces exactly at τ, and can their zero values explain total work? **Solution:** both are zero by convention. Both still have4N s of accumulated impulse in the first example. A single endpoint value cannot replace the full work integral.

## Connection to learning dynamical models

A learned simulator supervised only on terminal velocity can fit both histories while getting intermediate positions wrong. Adding trajectory observations constrains a different part of the model. A momentum-balance residual and a work-balance residual are useful checks, but zero residuals do not prove that the prescribed force matches an experiment. Here the force law and initial data are assumptions; validating them requires observations beyond the algebraic identities.

## Background and scope

For the standard momentum, impulse and center-of-mass framework, consult [MIT OpenCourseWare, Classical Mechanics, Week5](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-5-momentum-and-impulse/). The two pulse histories, their convolution formulas and the worked values above are derived for this lesson. Variable mass, relativistic momentum and contact-force reconstruction require additional models.
