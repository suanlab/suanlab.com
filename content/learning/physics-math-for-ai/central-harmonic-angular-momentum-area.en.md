## A central force conserves more than a circular picture suggests

A particle can orbit on an ellipse, move along a radial line or remain stationary under the same central force law. Zero torque about the force center does not select a circle. It constrains angular momentum; initial position and velocity determine the actual trajectory.

This lesson uses the isotropic planar harmonic force F=−kq about a fixed origin, with constant positive mass m and stiffness k. We solve the physical initial-value problem and compare angular quantities about the force center and another fixed point D=d. Changing D changes the reference used for measurement, not the physical force or orbit.

## Solve both spatial components with the same frequency

Newton's law gives

$$
\ddot q=-\omega^2q,\qquad \omega^2=\frac{k}{m}.
$$

With specified initial position q0 and velocity v0,

$$
q(t)=q_0\cos(\omega t)+\frac{v_0}{\omega}\sin(\omega t),
$$

$$
v(t)=-\omega q_0\sin(\omega t)+v_0\cos(\omega t).
$$

Both components use the same frequency because the stiffness is isotropic. The position is a linear image of the unit circle in sine/cosine coordinates, so the orbit is an ellipse or a degenerate line segment or point. The period of the nonstationary solution is 2π/ω. The diagram displays one full period, while the time panels retain a ten-second horizon.

## Derive center torque and angular momentum conservation

About the force center,

$$
\tau_C=(q\times(-kq))_z=0,\qquad
L_C=(q\times mv)_z=m(q_0\times v_0)_z.
$$

The last equality follows by substituting the trigonometric solution: cross terms with identical vectors vanish, and cos²+sin²=1 combines the remaining terms. Equivalently, differentiating LC gives the vanishing center torque.

The conserved value may be positive, negative or zero. It is fixed by initial data rather than by the phrase “central force.” A nonzero value prevents the particle from passing through the center, since that would require q=0 and hence L=0 in this nonsingular model.

## Energy provides a separate invariant

The harmonic potential is V=k|q|²/2, so

$$
E=\frac12m|v|^2+\frac12k|q|^2
=\frac12m|v_0|^2+\frac12k|q_0|^2.
$$

Angular momentum and energy are different constraints. Energy is nonnegative here, while angular momentum is signed. Preserving one does not automatically verify a numerical model's other invariant or identify its phase along an orbit. The interface therefore reports separate numerical residuals for both.

Initial angular momentum and the classification tests use exact rational control values. Trigonometric positions, velocities and query cross products are floating-point evaluations, so their conservation residuals are numerical diagnostics rather than exact-rational equalities.

## Circles require additional initial conditions

For a nonzero initial radius, circular motion requires

$$
q_0\cdot v_0=0,\qquad m|v_0|^2=k|q_0|^2.
$$

The first condition makes velocity tangential. The second makes the central force match the needed centripetal acceleration. Orthogonality alone is insufficient: the wrong tangential speed gives an ellipse. The interface checks both conditions exactly, including the requirement of nonzero radius.

When q0×v0=0, the initial vectors are collinear and motion stays on a line through the center. This radial motion can have substantial speed and energy while angular momentum vanishes. If both initial vectors are zero, the trajectory is the stationary center point. The radial flag includes that degenerate case; the separate stationary flag makes the distinction explicit.

## Define swept area as a signed integral

During a short interval, the radius vector sweeps an oriented triangular area. Taking its time limit gives

$$
\frac{dA_C}{dt}=\frac12(q\times v)_z=\frac{L_C}{2m},\qquad
A_C(t)=\frac{L_C}{2m}t.
$$

This is constant areal velocity about the force center. Clockwise orientation gives negative area. A radial trajectory has zero area even while the particle moves. Multiple turns accumulate signed area rather than resetting after each completed orbit.

The plotted quantity is therefore not the nonnegative area of a filled polygon or the union of all visited sectors. Its definition retains orientation and repeated traversal. That distinction matters when comparing a long-time area integral with the ordinary area enclosed by an ellipse.

## Moving the comparison marker changes angular quantities

The point D is fixed at displacement d from the force center. With laboratory momentum retained,

$$
L_D=L_C-(d\times p)_z,\qquad
\tau_D=(q-d)\times(-kq)=k(d\times q)_z.
$$

The center torque vanishes but the comparison torque generally varies. Consequently LD and areal velocity about D generally vary too. This does not indicate a failure of the central-force law: the point about which that law has zero torque is the actual force center.

Both points are fixed in one inertial frame. This is not the moving-reference-point problem and needs no fictitious coordinate force. The trajectory and physical force readouts remain unchanged when only d is edited.

## Integrate area about the comparison point correctly

Its instantaneous areal velocity is LD/(2m). Integrating the shift term gives

$$
A_D(t)=A_C(t)-\frac12\left[d\times(q(t)-q_0)\right]_z.
$$

The position-dependent correction is essential. Substituting the constant center areal velocity for the comparison one would silently use the wrong reference. After an entire closed orbit q returns to q0, so both accumulated areas agree over that complete traversal even though their instantaneous rates differed.

The independent verification integrates the differential equations together with each area's own rate and the comparison torque. It does not obtain the comparison area by declaring it equal to the center area.

## Read the three panels

The spatial panel shows one-period motion in meters, the force center, the comparison point and radius vectors to the query state. The angular panel compares center and comparison L in kg m²/s. The area panel compares their signed time integrals in m². Spatial arrows represent radii, not a hidden conversion of force into distance.

Try the elliptic example, then the radial example, then the displaced-reference circular example. Reverse the initial transverse velocity to reverse orientation. Set both initial vectors to zero to inspect a stationary degenerate plot. Change mass or stiffness to change frequency; change only the comparison point to leave physical motion fixed.

## Research extension: reconstruct ellipse sizes from energy and center angular momentum

The original solution is a linear image of a sine/cosine circle. That observation lets us recover all semiaxis lengths from two invariants and determine exactly which information is still missing. Throughout this extension, L means angular momentum about the **force center**, with the signed orientation already defined above.

### Convert the orbit into a two-column linear map

For the same isotropic harmonic force with constant m,k>0, collect the initial vectors into C:

$$
\omega=\sqrt{k/m},\quad C=\begin{pmatrix}\vert&\vert\\
\mathbf q_0&\mathbf v_0/\omega\\\vert&\vert\end{pmatrix},\qquad
\mathbf q(t)=C\begin{pmatrix}\cos\omega t\\\sin\omega t\end{pmatrix}.
$$

Both columns have units of length. A singular value decomposition C=U Sigma V transpose first reorients the unit circle, scales its two orthogonal directions, and then reorients the result in the lab. Therefore the singular values sigma1 and sigma2 are the ellipse semiaxis lengths. Rank one produces a line segment; rank zero produces a point. This is a geometric use of SVD, not a time-integrator approximation.

The squared Frobenius norm of C is the sum of its squared singular values. Its determinant gives signed area scaling. Substituting the mechanical definitions yields:

$$
E=\frac k2(\sigma_1^2+\sigma_2^2),\qquad
L=m\omega\det C,\qquad
\sigma_1^2\sigma_2^2=\frac{L^2}{mk},\quad \sigma_1\ge\sigma_2\ge0.
$$

Here omega times mass equals sqrt(mk), so the product relation has units of length to the fourth power. E is in joules and L in kg m²/s.

### Derive a feasibility test and the circle boundary

The arithmetic–geometric mean inequality on the two semiaxis lengths gives the first identity below. Expanding the energy and using the planar dot/cross-product identity gives the second:

$$
E-\omega|L|=\frac k2(\sigma_1-\sigma_2)^2\ge0,
\qquad
E^2-\omega^2L^2
=\frac14(k\|\mathbf q_0\|^2-m\|\mathbf v_0\|^2)^2
+mk(\mathbf q_0\cdot\mathbf v_0)^2.
$$

The nonnegative polynomial expression is also a direct check from initial states. Equality E=omega|L| gives a circle when E>0. It is equivalent to perpendicular initial vectors with k|q0|²=m|v0|². The exceptional E=L=0 case is the stationary origin, not a nonzero-radius circle. Zero L with positive E instead gives radial motion and positive kinetic activity.

A learned invariant summary must satisfy E≥omega|L|. A negative energy or a negative discriminant is infeasible for this declared positive harmonic potential. Squaring the constraint alone would miss a negative-energy error.

### Invert the two invariants

The squared semiaxes have known sum 2E/k and product L²/(mk). Solving that quadratic gives:

$$
\sigma_{1,2}^2=\frac{E\pm\sqrt{E^2-\omega^2L^2}}{k},\qquad
\sigma_2^2=\frac{L^2/(mk)}{\sigma_1^2}\quad(E>0).
$$

The second formula is useful near a radial orbit: subtracting two nearly equal numbers in the minus-root formula can erase a small but nonzero minor axis. The code keeps the discriminant exact with fractions, computes the large root, then obtains the small root from their product. It rejects infeasible inputs instead of silently clamping a negative discriminant.

### Work six initial states

Use m=k=1 in the stated SI units. A has q0=(2,0),v0=(0,1); B rotates both vectors by 90 degrees to q0=(0,2),v0=(−1,0). C reverses A's velocity. D uses q0=(1,1),v0=(−1,1); E uses q0=v0=(1,0); O starts with both vectors zero.

| case | E (J) | L (kg m²/s) | squared semiaxes (m²) | motion |
|---|---|---|---|---|
| A | 5/2 | 2 | 4, 1 | ellipse |
| B | 5/2 | 2 | 4, 1 | rotated ellipse |
| C | 5/2 | −2 | 4, 1 | clockwise ellipse |
| D | 2 | 2 | 2, 2 | circle |
| E | 1 | 0 | 2, 0 | radial line |
| O | 0 | 0 | 0, 0 | stationary |

A and B have the same E and L yet different long-axis directions in the same laboratory axes. A time shift of either solution also changes its starting phase while preserving the invariants. The invariants determine axis lengths and, for nonzero L, traversal orientation; they do not specify lab orientation or the initial phase.

![Two differently oriented ellipses share energy 5/2 joules and center angular momentum 2 kg m²/s; square markers show their distinct initial positions in the same laboratory.](/assets/learning/classical-harmonic-invariant-geometry.en.svg)

Both position axes use metres with equal unit lengths. Solid blue is A and dashed orange B; arrows show increasing time. Their force center is the same origin. Each square marks t=0. These are two distinct initial conditions under one isotropic force law.

### Connect the axes to signed area

For one full nonstationary period T=2pi/omega, the center areal velocity integrates to:

$$
A_{\mathrm{signed}}(T)=\int_0^{2\pi/\omega}\frac{L}{2m}\,dt
=\frac{\pi L}{m\omega}=\pi\det C,\qquad
A_{\mathrm{ellipse}}=\pi\sigma_1\sigma_2=\pi|\det C|.
$$

The first quantity retains clockwise sign; the second is the nonnegative geometric ellipse area. Repeated traversals accumulate signed area again. For a radial segment both areas are zero even though the particle moves.

A fixed comparison origin can have a time-dependent instantaneous areal velocity while the one-period signed integral remains the same, because its displacement correction vanishes when q(T)=q0. That full-period fact does not make its instantaneous angular momentum the correct L for the axis formula.

For example, in A choose the comparison point D=(2,0), coinciding with q0. Initially L_D=0 while center L=2. Substituting that comparison value into the inversion would falsely infer a radial line. The axis reconstruction needs the force-center invariant.

### Distinguish cancellation from sensitivity near a circle

Avoiding minus-root cancellation near a radial orbit does not eliminate the inverse problem's sensitivity near a circle. Start with circle energy E_ref and keep L fixed while increasing energy by a positive delta:

$$
L=\frac{E_{\mathrm{ref}}}{\omega},\quad E=E_{\mathrm{ref}}+\delta,
\qquad
\sigma_1^2-\sigma_2^2=\frac{2}{k}\sqrt{2E_{\mathrm{ref}}\delta+\delta^2},
\quad E_{\mathrm{ref}}>0,\ \delta>0.
$$

The squared-axis split grows like sqrt(delta), not delta. With m=k=1,E_ref=2 J and delta=10^-6 J, it is about 0.0040000005 m². Thus very small invariant errors near the circle boundary need not produce proportionally small axis splits. Report uncertainty and feasibility margins before interpreting a nearly circular reconstruction.

### Reproduce with exact invariant arithmetic

The code uses rational initial data and a rational discriminant; axis-size reconstruction uses floating point. It also checks infeasible summaries, a small nonzero radial minor axis and the near-circle split. These standalone diagnostics add no new SVD control to the native lab.

```python
from fractions import Fraction as F
from math import sqrt

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def cross(a,b):
    return a[0]*b[1]-a[1]*b[0]

def axes_squared(m,k,E,L):
    if m<=0 or k<=0:
        raise ValueError("Use positive mass and stiffness")
    gap=E*E-k/m*L*L
    if E<0 or gap<0:
        raise ValueError("Infeasible oscillator invariants")
    if E==0:
        return (0.0,0.0)
    large=(float(E)+sqrt(float(gap)))/float(k)
    # Product/large avoids cancellation near a radial orbit.
    small=float(L*L/(m*k))/large
    return large,small

m=k=F(1)  # kg, N/m; omega=1 per second
cases=[("A",(2,0),(0,1)),("B",(0,2),(-1,0)),
       ("C",(2,0),(0,-1)),("D",(1,1),(-1,1)),
       ("E",(1,0),(1,0)),("O",(0,0),(0,0))]
for name,q,v in cases:
    q,v=tuple(map(F,q)),tuple(map(F,v))
    E=(k*dot(q,q)+m*dot(v,v))/2
    L=m*cross(q,v)
    gap=E*E-k/m*L*L
    assert gap==(k*dot(q,q)-m*dot(v,v))**2/4+m*k*dot(q,v)**2
    large,small=axes_squared(m,k,E,L)
    assert abs(large+small-2*float(E/k))<1e-12
    assert abs(large*small-float(L*L/(m*k)))<1e-12
    print("case, E, L, squared semiaxes:",name,E,L,(large,small))

for E,L in [(F(1),F(2)),(F(-1),F(0))]:
    try:
        axes_squared(m,k,E,L)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid invariants were accepted")

# An almost radial orbit; exact fractions retain the nonnegative gap.
q,v=(F(1),F(0)),(F(0),F(1,10**8))
E=(dot(q,q)+dot(v,v))/2;L=cross(q,v)
large,small=axes_squared(m,k,E,L)
assert small>0 and abs(small-1e-16)<1e-30
print("near-radial squared minor axis:",small)

# Fixed L=2, small energy excess above the circle E_ref=2 J.
delta=F(1,10**6)
large,small=axes_squared(m,k,F(2)+delta,F(2))
expected=2*sqrt(float(4*delta+delta*delta))
assert abs((large-small)-expected)<1e-12
print("near-circle squared-axis separation:",round(expected,10))
```

### Worked research questions

1. **Can E=1 J,L=2 kg m²/s describe m=k=1?** No. Omega=1 per second and E<omega|L|. The proposed invariants are inconsistent with this model.
2. **What does a zero center L certify?** With positive E it certifies a degenerate radial orbit, while E=0 gives the stationary point. It does not generally certify rest.
3. **What extra information reconstructs the actual initial-value orbit?** Supply lab orientation and initial position/velocity or equivalent phase data, along with the force center and coefficients. Preserving E and L alone leaves those choices open.

All geometry, feasibility and inversion formulas are derived from the declared harmonic initial-value solution. They supplement the area law without replacing trajectory, phase and force-law evaluation.

## Six exercises with worked solutions

1. **Exercise: ellipse from initial data.** Set m=k=1,q0=(2,0),v0=(0,1). **Solution:** q=(2cos t,sin t) and v=(−2sin t,cos t). The orbit is elliptical, LC=2, energy=5/2J and areal velocity=1m²/s.
2. **Exercise: area at a query time.** For exercise1, what is the area at1s? **Solution:** AC=1m². No angle inversion or polygon approximation is needed. A full period gives area2πm², matching the ellipse with semiaxes2 and1.
3. **Exercise: radial motion.** Set q0=(1,0),v0=0,m=k=1. **Solution:** q=(cos t,0),v=(−sin t,0). Angular momentum and area vanish, but the particle moves and energy remains1/2J.
4. **Exercise: the circle condition.** Take m=1,k=4,q0=(1,0). What tangential velocity makes a circle? **Solution:** v0=(0,2) or(0,−2), since m|v0|²=k|q0|². Speed1 gives an ellipse despite being perpendicular to q0.
5. **Exercise: displaced reference.** Set m=k=1,q0=(1,0),v0=(0,1),D=(1,0). **Solution:** LC=1,LD=1−cos t and τD=sin t. Areas are AC=t/2 and AD=(t−sin t)/2. They agree after2πs but generally differ before then.
6. **Exercise: clockwise repeated turns.** Reverse v0 in exercise5 and set D=0. At10s? **Solution:** LC=−1 and A=−5m². Its magnitude exceeds the disk areaπ because the integral counts repeated signed sweeping, rather than only the union of visited locations.

## A central-force statement is not the entire Kepler model

Zero torque and equal areas follow from the force direction. They do not by themselves imply an inverse-square law, a focus-centered gravitational ellipse or Kepler's period–size relation. Our harmonic ellipse is centered on the force origin and has a frequency independent of amplitude. More general central forces can produce different orbit shapes and precession.

For learned simulators, evaluate trajectory phase and force law alongside invariant residuals and reference-point consistency. Matching angular momentum alone is insufficient to identify the dynamics.

## Background and scope

For central-force angular conservation, see [MIT Physics8.01, Central Force Motion](https://web.mit.edu/8.01t/www/materials/Presentations/Presentations_f10/Presentation_W14D1.pdf). The harmonic trajectory and signed-area examples above follow from the stated model. Singular forces, moving reference points and rigid-body spin are separate topics.
