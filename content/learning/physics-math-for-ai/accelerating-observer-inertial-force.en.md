## Acceleration can depend on the observer

An unforced particle is stationary in an inertial laboratory. A camera accelerating away from it records a moving particle with a nonzero coordinate acceleration. The camera did not create a new physical interaction with the particle. Its coordinate description requires an additional inertial term.

This lesson extends the earlier Galilean observer change to a nonrotating observer with constant translational acceleration. We derive the coordinate law, reconstruct the same physical event and examine what goes wrong when the inertial term is omitted. Rotating frames, contact forces and apparent-weight measurements require further models and are not silently inferred here.

## Specify the moving origin before changing the law

Let the laboratory be inertial. The other observer's origin has laboratory position

$$
R(t)=b+Ut+\frac12At^2,
\qquad x'=x-R(t).
$$

The parameters b,U,A have units meters,m/s and m/s squared. The clocks share Newtonian time. A prime labels coordinates assigned by the other observer, not a time derivative.

The displacement convention determines all signs. The moving origin itself has x'=0, and a laboratory event is reconstructed as x=x'+R(t). Stating that convention is more reliable than memorizing a force sign without specifying which frame moves.

## Differentiate the coordinate definition

Differentiating once and twice gives

$$
v'=v-U-At,\qquad a'=a-A.
$$

When A=0 this reduces to the Galilean case: velocities differ by a constant while accelerations agree. A nonzero A changes acceleration components as well.

The observer is assumed not to rotate. Rotation produces additional terms involving angular velocity and relative motion. Applying only minus A to a rotating coordinate system would omit part of its kinematics.

## Derive the inertial coordinate term

For a constant positive particle mass and prescribed constant physical force, the laboratory law is m*a=F. In the accelerating coordinates,

$$
m a'=F-mA.
$$

The term Finertial=-mA is introduced so that the coordinate acceleration satisfies the modified equation. Its effective force is Feffective=F+Finertial.

The inertial term is not another physical interaction with a new partner body. One should not invent a separate action-reaction partner for it inside this one-particle model. Physical force and inertial coordinate term are reported separately even though they appear together in the primed equation.

## Transform initial data and integrate consistently

At t=0, the appropriate initial data are

$$
x'_0=x_0-b,\qquad v'_0=v_0-U.
$$

The acceleration A of the origin does not add another initial-velocity shift at t=0. With constant F,m,A, the primed solution is

$$
x'(t)=x'_0+v'_0t+\frac12\left(\frac{F}{m}-A\right)t^2.
$$

Adding R(t) reproduces the laboratory trajectory x0+v0*t+F*t squared/(2m). Velocity reconstruction adds U+At. The consistency is exact for the declared model, including transformed initial conditions.

Keeping the physical law but forgetting the initial-data transformation would define another error. The interface separates a consistent solution from a counterfactual that uses the correct primed initial data but deliberately omits the inertial acceleration term.

## Work an unforced-particle example

Take m=1 kg,F=0,x0=0,v0=0,b=0,U=0 and A=1 m/s squared. The particle remains at laboratory position zero with laboratory acceleration zero.

The accelerating observer assigns x'=-t squared/2, v'=-t and a'=-1 m/s squared. At2 s, its position and velocity are-2 m and-2 m/s. The inertial coordinate term is-1 N and the modified equation gives m*a'=-1 N.

Reconstruction adds the origin's position2 m, returning the same laboratory position zero. The nonzero coordinate acceleration does not require an additional physical force on the particle.

## Zero primed acceleration does not eliminate physical force

Now use m=2 kg,F=2 N and A=1 m/s squared, with initial data and origin offsets zero. Laboratory acceleration is1 m/s squared. The observer accelerates at the same rate, giving a'=0.

At2 s the laboratory position is2 m, while the primed position remains zero. Physical force2 N and inertial term-2 N cancel in the primed equation. The physical force itself has not vanished.

A statement about apparent weight would additionally require a contact or measuring-device model. This free-particle coordinate calculation does not determine a support force, scale reading or elevator-wall interaction merely because an effective term is zero.

## Quantify the error from omitting the inertial term

Suppose a program uses the correct primed initial data but integrates xwrong'=x0'+v0'*t+F*t squared/(2m), as if the observer were inertial. Subtracting the correct primed trajectory gives

$$
x'_{\mathrm{wrong}}-x'=\frac12At^2.
$$

The error grows quadratically with time for fixed A. At t=0 the positions agree, so an initial-position check alone will miss the mistake. If A=0 the error vanishes for all time, recovering the inertial limit.

A plot can show the laboratory path, the correct physical reconstruction and the incorrect reconstruction. Their difference is a modeling error in coordinate dynamics, not an observed change of the physical laboratory experiment.

## Separate coordinate work from extra physical energy

For the constant effective acceleration, the primed kinetic-work identity is

$$
K'(t)-K'_0=(F-mA)\,[x'(t)-x'_0].
$$

Its right-hand side can be split into the physical-force term F*DeltaXprime and the inertial coordinate term(-mA)*DeltaXprime. Both are needed to match the primed kinetic-energy change in this representation.

In the unforced example at2 s, primed kinetic energy increases from0 to2 J. Physical force work in the primed coordinates is0 and inertial coordinate work is2 J. This does not insert a physical2 J source into the stationary laboratory particle. Kinetic energy is observer-dependent, and the inertial term belongs to the accelerating description.

Under the stated uniform terms this algebraic identity is exact. It is not a universal assertion that every non-inertial force model has a simple time-independent potential or that every energy interpretation transfers unchanged to rotating frames.

## Keep parameter changes and observer changes distinct

Changing A changes the observer's description while the laboratory force, mass and initial conditions can remain fixed. Changing F changes the physical laboratory model. Changing m at fixed F changes laboratory acceleration as well as the magnitude of the inertial term.

These controls should not be treated as interchangeable ways to make a plotted curve move. Inspecting one change at a time reveals which quantities are physical laboratory inputs and which describe the observer.

A negative A simply means acceleration opposite the selected positive axis. The inertial term reverses accordingly. A signed component is not inherently an invalid physical input.

## Interpret exact checks and sampled curves

Supported inputs are rational: mass uses kilograms directly, while other codes are divided by10 in their stated units. Reconstruction, modified-force balance, omission error and primed kinetic-work residuals are checked with exact reduced fractions.

Rendered curves use sampled decimal coordinates. They illustrate the analytic trajectory but are not the proof that every time satisfies the equation. At the selected time, exact labels keep the declared mathematical identities separate from plotting precision.

The figures compare actual time-dependent trajectories in this uniform-acceleration model. A force value or an inertial term should not be plotted as a spatial displacement without specifying an additional conversion rule. Position graphs use meters and time graphs use seconds.

## Implications for physical models learned by AI

A trajectory predictor trained on accelerating cameras can confuse coordinate acceleration with a new physical force if observer motion is absent from its inputs. A model may fit the observations while assigning an incorrect physical interpretation to its coefficients.

A useful evaluation transforms initial conditions and the observer trajectory consistently, reconstructs the same laboratory event and checks the modified equation. It also reports the scope: nonrotating constant translational acceleration is a tractable example, not a certificate for arbitrary camera motion or all non-inertial mechanics.

## Exercises with worked solutions

1. **Exercise:** An unforced1 kg particle is at rest in the lab while A=1 m/s squared. What is a'? **Solution:** a'=0-1=-1 m/s squared and the inertial term is-1 N.
2. **Exercise:** At2 s in that example, reconstruct the laboratory position. **Solution:** x'=-2 m and R=2 m, so x=x'+R=0.
3. **Exercise:** If F=2 N,m=2 kg,A=1 m/s squared, has physical force vanished because a'=0? **Solution:** No. The physical2 N force is balanced by a-2 N inertial term in the coordinate equation.
4. **Exercise:** What position error follows from omitting the inertial term with otherwise correct primed initial data? **Solution:** A*t squared/2.
5. **Exercise:** Does nonzero inertial coordinate work imply an extra physical laboratory energy source? **Solution:** No. It contributes to the accelerating-coordinate kinetic-work identity; its interpretation differs from a new laboratory interaction.
6. **Exercise:** What changes when A=0? **Solution:** The inertial term and omission error vanish, and the Galilean inertial-observer transformation is recovered.

## Sources and further study

[MIT8.01, Chapter31, section31.2](https://live.ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/mit8_01scs22_chapter31.pdf) gives background on translating non-inertial frames and the inertial term. The constant-acceleration origin, exact initial-value comparisons, omission error and coordinate-work examples here are direct derivations under the stated assumptions.
