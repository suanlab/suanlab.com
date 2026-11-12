## An impact is decided at the contact point

A rigid object can rotate while its center of mass translates. The velocity of its lowest endpoint can therefore differ from the velocity of its center. A center moving upward does not guarantee that every point is leaving the floor. To decide whether an impulse is needed, compute the normal velocity of the actual contact point first.

This lesson studies a single instantaneous impact of a uniform thin rod against a fixed horizontal floor. It connects compatible rigid velocities, linear impulse, angular impulse and energy loss. It does not integrate a later flight or infer that a rod with zero endpoint normal speed remains attached to the ground.

## Fix a geometry with only one initial contact

Place the center of mass C at(0, h), the lower endpoint O at(a, 0), and the upper endpoint at(−a, 2h). Assume a≥0, h>0 and positive constant mass m. The rod length and center-of-mass inertia are

$$
\ell=2\sqrt{a^2+h^2}, \qquad I=\frac{m(a^2+h^2)}3.
$$

Only the lower endpoint initially touches the floor; the rest of the segment lies above it. Changing a or h changes the declared rod geometry and inertia. A horizontal rectangle with two corners touching would require a different contact model.

The impact map freezes position and orientation during the velocity jump. Incoming center velocity is(Vx, Vy). Angular velocity ω is positive counterclockwise. The floor normal is n=(0, 1), and the center-to-contact offset is r=(a, −h). There is no tangential impulse. Finite forces such as gravity do not contribute a finite impulse over this ideal zero-duration event.

## Compute the compatible contact velocity

Planar rigid kinematics gives

$$
v_O=V+\omega\, \hat z\times r
=(V_x+h\omega, \;V_y+a\omega).
$$

Write the incoming normal component as c⁻=Vy⁻+aω⁻. A closing contact has c⁻<0 and receives a nonnegative normal impulse. For c⁻≥0, this map applies no impulse: positive speed is separating, while zero is nonclosing.

The zero-speed case may still require a persistent-contact force in a later dynamics calculation. An inactive impact law does not by itself decide that force. Nor does it declare a contacting body to be in static equilibrium.

## Write both impulse balances

Let jn be the scalar upward normal impulse, with units N s. The center velocity and counterclockwise angular velocity jump according to

$$
V_x^+=V_x^-, \qquad V_y^+=V_y^-+\frac{j_n}{m},
$$

$$
\omega^+=\omega^-+\frac{a j_n}{I}.
$$

The lever arm a multiplies the impulse to give angular impulse. The floor is external to the rod, so the rod's vector linear momentum generally changes. Its spin angular momentum about C changes as well. These are physical responses to an external impulse, not violations of Newton's laws.

Frictionless means that center tangential momentum is unchanged. Contact tangential speed can still change, because rotation changes. This distinction is visible in the term hω in vOx.

## Apply restitution to the contact normal speed

For a closing contact, choose a coefficient e in[0, 1] and impose

$$
c^+=-e c^-.
$$

Substituting the impulse balances gives

$$
c^+=c^-+Wj_n, \qquad
W=\frac1m+\frac{a^2}{I}>0,
$$

$$
j_n=-\frac{(1+e)c^-}{W}.
$$

W is an inverse effective normal mass, measured in1/kg. Its reciprocal is the effective mass seen at this contact. A normal impulse can generate both translation and rotation, so that effective mass need not equal the total mass.

Restitution is not a rule that sets Vy⁺=−eVy⁻. That simpler relation holds here only when the normal impulse has no rotational coupling, such as a=0. For general a, it is the endpoint normal speed that the law constrains.

## Derive the energy change instead of assuming it

Total kinetic energy includes both translation and spin:

$$
K=\frac12m(V_x^2+V_y^2)+\frac12I\omega^2.
$$

Using the impulse balances,

$$
K^+-K^-=j_nc^-+\frac12Wj_n^2
=-\frac{(1-e^2)(c^-)^2}{2W}
$$

for an active impact. Thus e=1 conserves kinetic energy while redistributing it, and e<1 loses a nonnegative amount. The inactive branch leaves all velocities and energy unchanged.

The loss is an energy budget for the declared restitution map. It does not predict temperature, deformation history or how the loss is partitioned among internal mechanisms. Those questions require a richer constitutive model.

## Find the angular momentum that stays conserved

About the fixed impact point O, the center offset is C−O=(−a, h). The total angular momentum is

$$
L_O=I\omega+(C-O)\times mV
=I\omega-a mV_y-hmV_x.
$$

The normal impulse acts through O and has zero moment about O. Substitution gives LO⁺=LO⁻ exactly. Spin momentum about C can change while total angular momentum about O is conserved. The orbital term is essential; using only Iω would test the wrong quantity.

This is an instantaneous statement at the fixed impact location. A later moving contact point or an accelerated reference would need its own derivative and transport analysis.

## Read the elastic and plastic limits carefully

An elastic contact does not require the center to bounce upward. Rotation may open the endpoint contact even while Vy⁺ remains negative. What is guaranteed is the outgoing contact normal speed and the conserved total kinetic energy.

At e=0, c⁺=0. The rod can retain tangential center motion, angular velocity and positive kinetic energy. Zero normal speed is one scalar condition, not a declaration that every velocity component vanishes. Whether subsequent forces maintain, release or change the contact is outside this instantaneous map.

A vertical rod has a=0 but still has positive inertia mh²/3. The normal impulse passes through C, so spin remains unchanged and the ordinary particle-like vertical restitution relation is recovered.

## Relate the example to a contact Jacobian

Using generalized velocity ν=(Vx, Vy, ω), define J=(0, 1, a) and mass matrix M=diag(m, m, I). Then c=Jν and

$$
M(\nu^+-\nu^-)=J^\mathsf Tj_n, \qquad
W=JM^{-1}J^\mathsf T.
$$

This exposes the structure behind effective contact mass. The example has one scalar normal contact and a positive definite mass matrix, so the normal solve is unambiguous. Adding friction, several simultaneous contacts or a compliant surface changes the closure problem. A tangential law cannot simply be appended without checking consistency and energy behavior.

## Interpret the geometry and comparisons

The rod geometry is identical before and after the ideal impact; only velocities jump. Drawn before/after configurations must therefore not imply a computed later trajectory. Velocity arrows use m/s, impulse arrows use N s, and angular quantities require their own units.

The calculation reports exact rational balances for the discrete controls. The rod length uses a numerical square root, while inertia, impulse, velocities and energy are rational. Active restitution residuals are defined only on the closing branch; an inactive law is not reported as a failed restitution equation.

## Research extension: project the velocity, then ask whether restitution is identifiable

### The mass matrix defines the relevant distance

An impact changes translation and spin together. To compare two candidate velocity changes, use the kinetic-energy metric rather than the ordinary Euclidean distance between their coordinate numbers. Keep the original frozen geometry, one frictionless floor contact, m>0, a≥0, h>0 and I=m(a²+h²)/3. Let v=(Vx,Vy,ω) be generalized velocity. Define

$$
M=\operatorname{diag}(m,m,I),\quad J=(0,1,a),\quad c=Jv,
\qquad W=JM^{-1}J^{\mathsf T}>0,\quad
n=\frac{M^{-1}J^{\mathsf T}}{W},\quad Jn=1,\quad P=\operatorname{Id}-nJ.
$$

The row J converts generalized velocity into the endpoint's normal speed. The vector n is a unit-normal-speed response, not an ordinary spatial unit vector: its components mix translation and rotation in the correct physical units. W is positive because the mass matrix is positive definite and J is nonzero.

For a closing contact c<0 and e=0, the resulting velocity Pv is the unique solution of

$$
\min_{Jz\ge0}\frac12(z-v)^{\mathsf T}M(z-v).
$$

The admissible velocities do not move into the floor. A candidate z is measured by the energy-sized cost of changing v into z. To prove the minimizer, write every feasible candidate as

$$
z=Pv+t+\alpha n,\quad Jt=0,\quad\alpha\ge0,
\qquad\frac12\|z-v\|_M^2=\frac12\|t\|_M^2+\frac{(\alpha-c)^2}{2W}
\ge\frac{c^2}{2W}\quad(c<0).
$$

Here t is a tangent generalized velocity, not the physical time. The cross term vanishes because tᵀMn=Jt/W=0. Since c<0, choosing α≥0 cannot reduce (α−c)² below c². Equality requires t=0 and α=0, giving z=Pv. If c≥0, the original v is already feasible and gives cost zero; no impulse is applied.

Thus P is the linear projection onto the boundary hyperplane on the **closing branch**. The complete plastic impact law projects onto the admissible half-space. Applying P indiscriminately to separating velocities would alter motion that needs no impact.

### Reflection and contraction explain the energy budget

The operators obey

$$
P^2=P,\quad P^{\mathsf T}M=MP,\qquad
R=2P-\operatorname{Id},\quad R^2=\operatorname{Id},\quad R^{\mathsf T}MR=M.
$$

The first two identities describe an orthogonal projection in the mass metric. R is a reflection in that metric and conserves kinetic energy. It describes the e=1 update on the closing branch. The identity R²=Id concerns the linear operator: after a real impact the contact is separating, so applying the physical impact law again gives zero impulse, rather than automatically restoring the old velocity.

Separate v into mass-orthogonal tangent and normal components. For c<0 and 0≤e≤1,

$$
v_T=Pv,\quad v_N=nJv,\quad v^+=v_T-ev_N,\qquad
K^+=K_T+e^2K_N,\quad K_N=\frac{c^2}{2W},\quad
K^--K^+=(1-e^2)\frac{c^2}{2W}.
$$

The tangent mode is unchanged, while the normal mode is reversed and scaled by e. The normal-mode kinetic energy shrinks by e². In particular, plastic contact does not remove tangent-mode energy and need not stop the object. This generalized normal mode usually includes COM motion and spin, so simply reversing COM vertical speed would be incorrect.

Try native input `[1,2,2,0,-10,0,5]`. It gives m=a=h=1, I=2/3, W=5/2, c=−1 and e=1/2. The normal impulse is 3/5 N s. Afterwards Vy=−2/5 m/s, ω=9/10 rad/s and the endpoint normal speed is +1/2 m/s. Energy falls from 1/2 to 7/20 J, a loss of 3/20 J. A descending COM can coexist with an opening contact.

### A stable state update can have an unstable parameter inverse

At fixed geometry and e, the physical impulse is

$$
j(c)=-\frac{1+e}{W}\min(c,0).
$$

It is continuous and Lipschitz in c with constant (1+e)/W. Its left derivative at zero is −(1+e)/W and its right derivative is zero. A smooth branch derivative extended across zero misses the activation change. This matters when differentiating a simulator through a contact event.

Now estimate e from measured speeds instead. Let the true incoming speed be −δ, δ>0, and the outgoing speed be eδ. Add errors d₋ and d₊ with absolute bound ζ. The ratio estimate satisfies

$$
\widehat e=\frac{e\delta+d_+}{\delta-d_-},\qquad
|\widehat e-e|\le\frac{\zeta(1+e)}{\delta-\zeta}
\quad(0\le\zeta<\delta,\;|d_-|,|d_+|\le\zeta).
$$

To obtain the bound, subtract e: the numerator becomes d₊+ed₋ and the denominator stays at least δ−ζ. A vanishingly small impulse is therefore compatible with poor restitution identification. For e=1/2 and ζ=0.001 m/s:

| Closing speed δ (m/s) | Speed error ζ (m/s) | Error bound for e | Closing sign certified? |
|---|---:|---:|---|
| 0.1 | 0.001 | 1/66 | Yes |
| 0.01 | 0.001 | 1/6 | Yes |
| 0.001 | 0.001 | Ratio bound unavailable | No |

When δ≤ζ the incoming interval can touch or cross zero. A prior 0≤e≤1 still bounds a clipped estimate, but it does not supply identifying information from the measured ratio. The failed denominator bound is not a claim that every physically constrained estimator becomes unbounded.

![Continuous impulse update and deteriorating restitution identification](/assets/learning/classical-contact-grazing.en.svg)

The upper panel uses m=a=h=1 and e=1/2. The lower panel uses ζ=0.001 m/s and only δ>ζ; both axes there are logarithmic. The vertical axes represent different quantities. Divergence of the ratio-error bound does not imply divergence of the impulse update.

Native input `[1,2,2,0,-11,10,5]` gives c⁻=−1/10, impulse 3/50, c⁺=1/20 and energy loss 3/2000. Input `[1,2,2,0,-10,10,5]` is exactly nonclosing and applies no impulse. The finer δ=0.01 and 0.001 cases belong to the standalone measurement analysis; they do not extend the native velocity grid.

### Execute exact projection and noise checks

The following standard-library Python cell verifies projection/reflection identities in 27 geometries, 324 restitution states, feasible projection competitors and the speed-error bounds using exact fractions. It uses its own mass-matrix calculations rather than importing the site evaluator.

```python
from fractions import Fraction as Q
from itertools import product


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, v):
    return [dot(row, v) for row in a]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def kinetic(mass_diag, v):
    return dot(mass_diag, [x * x for x in v]) / 2


def operators(mass, a, h, restitution):
    inertia = mass * (a * a + h * h) / 3
    mass_diag = [mass, mass, inertia]
    j = [Q(0), Q(1), a]
    w = sum(x * x / m for x, m in zip(j, mass_diag))
    normal = [x / m / w for x, m in zip(j, mass_diag)]
    identity = [[Q(int(i == k)) for k in range(3)] for i in range(3)]
    projection = [[identity[i][k] - normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    reflection = [[identity[i][k] - 2 * normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    update = [[identity[i][k] - (1 + restitution) * normal[i] * j[k]
               for k in range(3)] for i in range(3)]
    return mass_diag, j, w, normal, identity, projection, reflection, update


def restitution_error_bound(delta, noise, restitution):
    if not 0 <= noise < delta:
        raise ValueError('Incoming speed must be separated from zero.')
    return noise * (1 + restitution) / (delta - noise)


def impact_checks():
    cases = 0
    for mass, a, h in product(map(Q, [1, 2, 4]), [Q(0), Q(1, 2), Q(3, 2)],
                              [Q(1, 2), Q(1), Q(2)]):
        d, j, w, n, identity, p, r, _ = operators(mass, a, h, Q(0))
        m = [[d[i] if i == k else Q(0) for k in range(3)] for i in range(3)]
        assert mm(p, p) == p
        assert mm(transpose(p), m) == mm(m, p)
        assert mm(r, r) == identity
        assert mm(mm(transpose(r), m), r) == m
        for e in [Q(0), Q(1, 2), Q(1)]:
            *_, update = operators(mass, a, h, e)
            for velocity in [[Q(0), Q(-1), Q(0)], [Q(1), Q(1, 2), Q(-1)],
                             [Q(-1), Q(0), Q(1)], [Q(0), -a, Q(1)]]:
                c = dot(j, velocity)
                after = mv(update, velocity) if c < 0 else velocity[:]
                impulse = -(1 + e) * min(c, Q(0)) / w
                assert impulse >= 0
                assert dot(j, after) == (-e * c if c < 0 else c)
                loss = kinetic(d, velocity) - kinetic(d, after)
                assert loss == ((1 - e * e) * c * c / (2 * w) if c < 0 else 0)
                if e == 0:
                    # Feasible competitors differ by a tangent and a nonnegative
                    # normal component; compare the projection objective exactly.
                    optimum = kinetic(d, [x - y for x, y in zip(after, velocity)])
                    for tangent_x, tangent, opening in product(
                            [Q(-1), Q(0), Q(1)], [Q(-1), Q(0), Q(1)],
                            [Q(0), Q(1, 2), Q(1)]):
                        offset = [tangent_x, -a * tangent, tangent]
                        candidate = [x + y + opening * z for x, y, z in zip(after, offset, n)]
                        assert dot(j, candidate) >= 0
                        cost = kinetic(d, [x - y for x, y in zip(candidate, velocity)])
                        assert cost >= optimum
                        if tangent_x == tangent == opening == 0:
                            assert cost == optimum
                cases += 1
    e, noise = Q(1, 2), Q(1, 1000)
    for delta in [Q(1, 10), Q(1, 100)]:
        bound = restitution_error_bound(delta, noise, e)
        for incoming_error, outgoing_error in product([-noise, Q(0), noise], repeat=2):
            estimate = (e * delta + outgoing_error) / (delta - incoming_error)
            assert abs(estimate - e) <= bound
        print('Incoming speed, restitution error bound:', delta, bound)
    try:
        restitution_error_bound(noise, noise, e)
    except ValueError:
        pass
    else:
        raise AssertionError('Grazing interval needs rejection')
    print(f'27 geometries: projection/reflection identities; {cases} restitution states passed')


impact_checks()
```

### Research questions with worked answers

1. **Does e=0 solve subsequent floor contact?** No. It provides an instantaneous admissible velocity. Continued contact, gravity, changing geometry and possible further impacts require a time-evolution model, which this lesson does not infer.
2. **Why can state prediction be stable while e is poorly estimated?** The forward impulse is Lipschitz at grazing, but the inverse divides two small contact speeds. Forward continuity does not guarantee inverse conditioning.
3. **Can we apply elastic reflection twice to undo two real impacts?** The matrix squares to the identity, but the physical law switches branches. After the first update a separating contact receives no second impulse unless additional dynamics creates a new closing event.

For the general mass-matrix and contact-Jacobian framework, see [MIT Underactuated Robotics, Multi-Body Dynamics](https://underactuated.mit.edu/multibody.html). The projection proof, noise bound and single-contact assumptions above define this worked example.

## Worked examples and exercises with solutions

1. **A rising center, closing endpoint.** Set m=a=h=1, V=(0, 1/2), ω=−1 and e=1/2. **Solution:** I=2/3, c⁻=−1/2 and jn=3/10. The outgoing values are Vy=4/5, ω=−11/20 and c⁺=1/4. The endpoint impacts despite the rising center. Energy loss is3/80.

2. **Elastic redistribution.** Keep m=a=h=1, use V=(1, −1), ω=0 and e=1. **Solution:** W=5/2 and jn=4/5. After impact Vy=−1/5 and ω=6/5, so c⁺=1. The center still descends, but the endpoint opens. K is1 before and after; spin and translation redistribute energy.

3. **Plastic normal stop.** Use the second example with e=0. **Solution:** jn=2/5, Vy⁺=−3/5 and ω⁺=3/5. Endpoint normal speed is zero, but K⁺=4/5 and Vx=1 remain. The loss is1/5. No permanent-contact trajectory has been solved.

4. **Exactly nonclosing contact.** Take a=1, Vy=−1 and ω=1. **Solution:** c⁻=0, so the impact map applies jn=0 for every allowed e. The center moves downward while the endpoint has zero normal speed. Later support forces require a continuous contact calculation.

5. **Vertical limit.** Set m=4, a=0, h=3, Vy=−2, ω=2 and e=1. **Solution:** I=12, W=1/4 and jn=16. Vy⁺=2 while ω⁺=2. There is no normal-impulse spin torque because the lever arm is zero.

6. **Tangential endpoint speed.** Return to the elastic second example. **Solution:** Vx remains1 because tangential impulse is zero. Endpoint tangential speed nevertheless changes from1 to11/5 through hω. Angular momentum about the floor contact remains0 because its spin and orbital parts compensate.

## Research connection and reading

For a learned simulator, classifying impact from center velocity alone can select the wrong branch. Conserving the wrong angular momentum, applying restitution to the wrong speed, or erasing motion at e=0 produces distinct failures. Test contact kinematics, impulse balance, energy and the appropriate reference momentum separately.

The [contact and impact section of MIT's Underactuated Robotics notes](https://underactuated.mit.edu/multibody.html) provides a general mass-matrix and contact-Jacobian framework. The single-rod geometry, frictionless law, exact controls and examples above define this lesson's narrower experiment. The next lesson studies continuous wheel rolling and sliding with an explicitly declared tangential friction law.
