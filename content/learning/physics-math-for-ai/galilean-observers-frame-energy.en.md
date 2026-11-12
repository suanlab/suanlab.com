## One event, two coordinate descriptions

Imagine watching a cart from the laboratory while a camera moves alongside it at constant speed. The laboratory can assign a nonzero velocity while the camera assigns zero. Neither reading by itself describes a different cart. A meaningful comparison must specify the observer, transform the initial data and reconstruct the same physical event.

This lesson uses Newtonian mechanics, one spatial dimension, a constant positive mass and a prescribed uniform force. The second observer moves with a constant velocity relative to the first, so both are inertial. Relativistic motion, accelerated observers, rotating frames and arbitrary electromagnetic force transformations require additional rules and are outside this model.

We will distinguish numerical agreement of coordinates, covariance of an equation, and conservation of a quantity. These are separate questions. The equation can retain its form in both observers while momentum changes under an external force and the kinetic energies assigned by the observers differ.

## Specify the coordinate convention

Let the moving observer's origin have laboratory position b+Ut at laboratory time t. The clocks share the same Newtonian time coordinate. If x is a particle's laboratory coordinate, the other observer assigns

$$
x'=x-b-Ut,\qquad t'=t.
$$

Here b is an initial displacement in meters and U is a constant velocity in meters per second. A prime labels another coordinate description; it is not a time derivative. The inverse relation is x=x'+b+Ut.

Different texts can reverse which observer is primed or which relative velocity is called U. A sign discrepancy is resolved by stating the origin's motion and checking a simple event: the moving observer's own origin must have x'=0. Substituting x=b+Ut into the displayed rule gives exactly that result.

## Differentiate before comparing force laws

Because b and U are constant,

$$
v'=v-U,\qquad a'=a.
$$

The observer displacement and velocity affect positions and velocities but cancel from acceleration. With unchanged mass and the same prescribed Newtonian uniform force, both descriptions satisfy m a=F. This is a statement about the specified model. It is not an assertion that every imaginable force formula is unchanged by blindly retaining all numerical parameters.

If the second origin instead has a time-dependent acceleration, differentiating its position twice introduces an additional term. The simple inertial-frame rule would no longer apply. Deriving a transformation from the coordinate definition is safer than memorizing an equation outside its assumptions.

## Transform the initial conditions too

At t=0, let the laboratory initial position and velocity be x0 and v0. The corresponding initial data for the moving observer are

$$
x'_0=x_0-b,\qquad v'_0=v_0-U.
$$

Under a constant prescribed force, the laboratory solution is

$$
x(t)=x_0+v_0t+\frac{F}{2m}t^2,
\qquad v(t)=v_0+\frac{F}{m}t.
$$

Solving the same equation with the transformed initial data gives

$$
x'(t)=x'_0+v'_0t+\frac{F}{2m}t^2.
$$

Substitution reproduces x-b-Ut. The equation and its initial data therefore describe the same motion in two observers. Keeping x0 and v0 unchanged in the second coordinate system generally starts a different physical experiment.

## Compare reconstructions rather than raw coordinates

A useful plot can place x(t) beside x'(t), showing different coordinate values. A second comparison reconstructs x'(t)+b+Ut and overlays it with x(t). These reconstructed curves agree for every time in the declared model, not merely at one sampled point.

Consider a particle with m=2 kg, x0=0 m, v0=1 m/s and F=0. Let b=0 and U=1 m/s. At t=2 s the laboratory gives x=2 m and v=1 m/s, while the moving observer gives x'=0 and v'=0. Reconstructing gives x'+Ut=2 m and v'+U=1 m/s.

The particle did not stop because a camera changed its coordinates. Its relative velocity to that camera is zero. A model evaluation should specify which comparison it is making before labeling coordinate differences as prediction errors.

## Momentum and kinetic energy depend on the observer

For one particle, P=m v and K=m v squared/2. Applying the velocity rule gives

$$
P'=P-mU,
\qquad
K'=K-UP+\frac12mU^2.
$$

The kinetic energies need not be equal. In the previous example K=1 J and K'=0 J. This numerical difference is a change of description; it is not energy removed by a physical brake or a projection algorithm.

The cross term uses the particle's momentum at the event being compared. It cannot generally be replaced by an arbitrary initial momentum when an external force changes v. The mass, observer velocity, time and state must refer to the same declared setup.

Conservation is a different statement: it compares a quantity at different times within a specified observer and system. With F=0 each observer's momentum and kinetic energy are constant, even though their constants can differ. With F nonzero, neither covariance of the acceleration law nor the existence of two descriptions makes the single particle's momentum conserved.

## Work is observer-dependent in the matching way

For a constant uniform force, work along the laboratory displacement is W=F[x(t)-x0]. The moving observer uses W'=F[x'(t)-x0']. Since the displacement transforms by subtracting Ut,

$$
W'=W-FUt.
$$

Each observer still satisfies its own kinetic work identity: the change in that observer's kinetic energy equals that observer's work. Different numerical work values do not contradict this identity.

For example, choose m=2 kg, x0=1 m, v0=-1 m/s and F=2 N. After t=2 s the laboratory position returns to1 m and velocity is1 m/s. Its initial and final kinetic energies are both1 J and its net work is0 J. Let U=1 m/s and b=2 m. The transformed initial position is-1 m and final position-3 m; the initial velocity is-2 m/s and final velocity0. The other observer's kinetic energy changes from4 J to0 J and its work is-4 J. Both identities hold.

## A material medium is also part of the model

As a separate force-law example, suppose a medium has a constant laboratory velocity w and a drag law Fdrag=-c(v-w). The same medium has velocity w'=w-U. Transforming both velocities leaves their difference unchanged: v'-w'=v-w.

If a program transforms the particle velocity but leaves the medium's numerical velocity fixed, it changes the modeled relative motion. The resulting force differs by cU. A preferred material rest state does not prevent consistent observer changes; it means the medium's state must be transformed as well.

This constitutive example is separate from the uniform-force trajectory above. We have not integrated a drag trajectory while pretending that its force stays constant. A complete simulation would need the drag equation and its corresponding solution or numerical evolution.

## Recognize accidental agreement

At t=0 with b=0, the two position coordinates agree regardless of U. Their velocities can still differ, and later coordinates generally diverge. A single position match therefore does not certify that initial data or an entire coordinate transformation were handled correctly.

Similarly, kinetic energies can accidentally match when the two velocities have equal magnitudes. That coincidence does not establish that kinetic energy is invariant under every observer change. Evaluate the transformation identity over its stated domain rather than interpreting one equality as a general symmetry theorem.

For the supported rational input grid, coordinate reconstruction, momentum transformation, kinetic transformation and both work identities can be checked exactly with fractions. Plotted samples and decimal readouts illustrate those identities; they do not provide the proof by themselves.

## Implications for learned dynamics

A learned trajectory predictor may be tested under observer changes by transforming both its inputs and its outputs. Raw coordinate discrepancies are expected when the target is a different observer's position. Reconstructing the same physical event is the relevant comparison.

The model's context also matters. A medium velocity, an imposed field or an observer-dependent sensor input cannot be silently omitted from the transformation. An equivariance test should report the transformation convention and the quantities it expects to agree. Kinetic energy in a chosen frame can be a valid prediction target without being an observer-independent scalar.

## Research extension: Separate center motion from intrinsic translational kinetic energy

### Generalize the observer comparison to several particles

The native example has one particle, whose kinetic energy changes under a Galilean boost. For N particles with positive fixed masses, define total mass M, momentum P, center velocity V, and total translational kinetic energy K. Velocities can be vectors in the same inertial frame. Under a constant observer boost U, all velocities change by minus U, so V changes by the same amount.

Expand each velocity as V plus its deviation. The cross term vanishes because the mass-weighted deviations sum to zero:

$$
M=\sum_i m_i,\quad P=\sum_i m_iv_i,\quad V=P/M,\qquad
K={\|P\|^2\over2M}+K_{\rm rel},\qquad
K_{\rm rel}=\tfrac12\sum_i m_i\|v_i-V\|^2.
$$

The relative velocities v_i−V are unchanged by the boost, hence K_rel is invariant while K generally is not. K_rel is the kinetic energy of relative translational motion in this particle model. It is not automatically the full thermodynamic internal energy: unresolved rotation, deformation, potentials and thermal states would need their own variables.

### Obtain an independent pairwise expression

Expanding squared pair differences and collecting each mass-weighted velocity term gives another useful identity:

$$
K_{\rm rel}=K-{\|P\|^2\over2M}
={1\over2M}\sum_{i<j}m_im_j\|v_i-v_j\|^2.
$$

This expression makes boost invariance visible without computing a center velocity first. For one particle the pair sum is empty and K_rel=0, even when its laboratory kinetic energy is large. For two particles it becomes one half the reduced mass times relative speed squared, exactly the term controlling the restitution example.

### Work a boost without calling the energy change dissipation

Use masses (2,1,3) kilograms and velocities (1,−2,4) meters per second. M=6, P=12, V=2, K=27 joules and K_rel=15 joules. A boost U=3 gives velocities (−2,−5,1), P'=−6, V'=−1 and K'=18 joules. The nine-joule difference in total kinetic energy is a frame change; K_rel stays 15. It is not evidence of a physical energy-loss event.

| Quantity | Effect of a common constant boost | Interpretation |
|---|---|---|
| Total momentum | Subtract M U | Frame-dependent coordinate quantity |
| Total translational kinetic energy | K−U dot P+M times norm(U)²/2 | Frame-dependent |
| Relative velocities and K_rel | Unchanged | Boost-invariant information |

The decomposition is instantaneous kinematics. A dynamical covariance claim still needs the force model and any medium velocity to transform as in the original lesson. Accelerated observers are not included in the inertial Galilean statement. Under an event with conserved total momentum, the difference in kinetic energy between before and after is boost-invariant because the term involving the momentum change vanishes; with external impulse that simplification need not hold.

**Worked question:** Can a model use K_rel alone to reconstruct laboratory trajectories? **Answer:** No. It discards center motion and compresses many different relative configurations to one scalar. An invariant feature is useful only when its lost information is irrelevant to the prediction task or provided separately. The code checks the center and pairwise formulas and the exact momentum/energy transformation under five boosts. It is a multi-particle standalone extension, not an extra hidden state in the single-particle widget.

```python
from fractions import Fraction as F
from itertools import combinations
m=list(map(F,[2,1,3]));v=list(map(F,[1,-2,4]));M=sum(m)
def quantities(w):
    P=sum(a*b for a,b in zip(m,w));V=P/M
    K=sum(a*b*b for a,b in zip(m,w))/2
    relative=sum(a*(b-V)**2 for a,b in zip(m,w))/2
    pair=sum(m[i]*m[j]*(w[i]-w[j])**2 for i,j in combinations(range(3),2))/(2*M)
    assert relative==pair==K-P*P/(2*M)
    return P,K,relative
P,K,relative=quantities(v)
assert (P,K,relative)==(12,27,15)
for U in map(F,[-7,-2,0,3,10]):
    pp,kp,rp=quantities([w-U for w in v])
    assert pp==P-M*U and kp==K-U*P+M*U*U/2 and rp==relative
    print('boost',U,'momentum',pp,'kinetic',kp,'relative kinetic',rp)
```

## Exercises with worked solutions

1. **Exercise:** A particle has laboratory x=5 m at t=2 s. The other observer has b=1 m and U=1 m/s. Find x'. **Solution:** x'=5-1-2=2 m. Reconstructing gives2+1+2=5 m.
2. **Exercise:** A2 kg particle moves at1 m/s while U=1 m/s. Compare kinetic energies. **Solution:** K=1 J; v'=0 and K'=0 J. This is not physical energy loss caused by changing observers.
3. **Exercise:** Which initial velocity should the moving observer use? **Solution:** v0'=v0-U. Keeping v0 numerically unchanged generally defines another experiment.
4. **Exercise:** Does unchanged acceleration imply conserved momentum under a nonzero uniform force? **Solution:** No. In either inertial observer, dP/dt=F. Covariance and conservation are different statements.
5. **Exercise:** Why must a drag model also transform w? **Solution:** The force depends on particle-medium relative velocity. Both velocities shift by-U, preserving their difference.
6. **Exercise:** If b=0 and coordinates agree at t=0, has global observer covariance been verified? **Solution:** No. Initial velocities and future reconstructions still need checking; nonzero U produces an immediate velocity difference.

## Sources and scope

[MIT8.20's Galilean transformation lecture](https://ocw.mit.edu/courses/8-20-introduction-to-special-relativity-january-iap-2021/resources/lecture-2-1/) gives background on Newtonian inertial observers. The uniform-force, kinetic-energy, work and medium examples above are direct derivations under their declared assumptions.
