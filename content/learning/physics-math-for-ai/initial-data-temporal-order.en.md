## Begin with a question about a complete state

A photograph of a stretched string shows its shape, but does not reveal whether each point is moving upward or downward. Two strings can have the same shape and different futures. A temperature profile under a specified heat law behaves differently: the initial temperature is the evolving state, and its initial rate of change is constrained by that law. Counting initial data therefore requires knowing which equation and which state variables are being modeled.

This lesson compares two explicit single-mode families on an interval of length L=1 meter. Both have zero values at the endpoints for every time. This is a deliberate finite-dimensional example, not a numerical solver for arbitrary initial functions. It makes the distinction visible without hiding it inside a large simulation. We compare independently prescribed data A,V with candidate coefficients a,b; changing a candidate does not silently change the prescribed problem.

## Keep length, time and amplitude units explicit

Write the spatial factor as sin(pi*x/L). Its argument is dimensionless. In the heat model u is a temperature difference in kelvins and diffusivity kappa has units square meters per second. In the wave model u is transverse displacement in meters and speed c has units meters per second. Numerically setting L=1 does not erase these dimensional distinctions.

The rate-code control r means kappa=r/100 in the heat case and c=r/2 in the wave case. It is not one physical coefficient shared by both models. Time is in seconds; the integer time control is divided by ten. Negative heat amplitudes are differences below a reference temperature, not negative absolute temperatures.

## Reduce the heat equation to one first-order equation

Insert a separated candidate into the heat law:

$$
u(x,t)=q(t)\sin(\pi x/L),\qquad u_t=\kappa u_{xx}.
$$

Differentiating the spatial factor twice multiplies it by minus (pi/L) squared. For this mode, the amplitude therefore satisfies

$$
q'(t)=-\lambda q(t),\qquad \lambda=\kappa(\pi/L)^2,
\qquad q(t)=a e^{-\lambda t}.
$$

Setting q(0)=A fixes a=A. The candidate's initial time derivative is then minus lambda times a; the prescribed-data solution has initial derivative minus lambda times A. This derivative is not an independently adjustable extra condition in this model. The two velocity controls remain visible so the contrast can be explored, but their labels and readouts identify them as inactive for heat.

For example, A=2 K and kappa=0.04 square meters per second give q'(0)=-0.08*pi squared K/s, approximately -0.789568 K/s. Specifying an additional initial rate of zero would be incompatible with this nonzero single mode and this heat law.

## Reduce the wave equation to a second-order equation

Use the same spatial factor for a different physical field:

$$
u_{tt}=c^2u_{xx},\qquad
q''(t)+\omega^2q(t)=0,\qquad \omega=c\pi/L.
$$

The complete amplitude family for this ordinary differential equation is

$$
q(t)=a\cos(\omega t)+\frac{b}{\omega}\sin(\omega t).
$$

Direct evaluation and differentiation give q(0)=a and q'(0)=b. Here a has units meters while b has units meters per second. Dividing b by omega gives meters, so the two terms can be added. Prescribing only A fixes a but leaves every real value of b possible. Prescribing both A and V fixes both coefficients within this mode.

A familiar analogy is a pendulum released from the same angle with different initial pushes. The shape alone does not encode the push. For a string, the velocity is a function across space; the single-mode restriction here makes that function b*sin(pi*x/L).

## Work through the first guide before moving the controls

Choose the wave law, speed c=1 m/s, prescribed data A=1 m and V=0, and candidate a=1 m and b=2 m/s. The two initial shapes agree exactly. At t=0.5 seconds, omega*t=pi/2, so the reference amplitude is zero while the candidate amplitude is 2/pi meters, approximately 0.63662 meters.

The difference is not a numerical error or an unstable integrator. Both curves are evaluations of analytic solutions of the same interior law and endpoint conditions. They solve different initial-velocity problems. In the second guide, changing the prescribed V to two makes both coefficients agree and the entire displayed amplitude curves coincide.

The third guide switches to heat. Changing either velocity input no longer changes the curves. This is a declared model dependence, not an interface failure. The initial derivative readout still changes when the initial amplitude or diffusivity changes.

## Distinguish a snapshot from the complete initial conditions

At time zero, the mismatched wave-velocity example gives identical displacement values. The equality of this one snapshot does not establish equality of trajectories. At later times, sine factors can also vanish and create another snapshot coincidence.

The interface therefore checks required initial data through the integer coefficients, separately from floating-point curve coordinates. It does not certify a solution by testing whether two rounded values happen to match. Trigonometric and exponential evaluations, graph positions and decimal readouts remain numerical approximations. Endpoint zeros in the spatial drawing are the explicitly imposed boundary values.

The second figure shows the current spatial profile. It answers what the chosen single mode looks like across the string or rod at one time; the first figure shows how its amplitude changes with time. These axes describe different questions and should not be interchanged.

## State the scope of uniqueness and compatibility

Within the displayed heat family, one initial amplitude selects one curve. Within the displayed wave family, an initial amplitude and velocity select one curve. These elementary coefficient arguments are complete for the stated families. They do not establish existence or uniqueness for every PDE, every initial function, every boundary condition or every regularity class.

A more general wave field requires two initial functions, not merely two scalar numbers. Initial functions must also be compatible with any endpoint conditions in the solution class being requested. This lesson's sine factor satisfies zero endpoint values automatically; the next lesson removes that convenience and checks traces explicitly.

For sufficiently regular solutions with zero endpoint values, uniqueness can also be studied using energy estimates. In the wave case, the difference of two solutions with identical initial displacement and velocity has zero initial energy; a suitable energy identity keeps it zero. Applying that argument requires stating the domain, boundary conditions and regularity assumptions. A few coincident plotted samples cannot replace it.

## Connect the missing data to AI models

A learning system asked to predict wave futures from displacement-only snapshots faces an incomplete state representation. Distinct velocities can produce the same input image and different targets. More training examples do not make that mapping deterministic without additional information or a probabilistic formulation. Supplying velocity, a sequence of observations, or an explicitly inferred hidden state changes the prediction problem.

For a heat model, independently predicting an arbitrary time derivative in addition to the temperature can violate the governing relation. Physics-based learning should distinguish independent state data from quantities implied by the law. It should also check the intended initial and boundary conditions, rather than relying only on an interior residual loss.

## Exercises with worked solutions

1. **Exercise:** For the wave mode, find q(0) and q'(0). **Solution:** Substitute t=0 to obtain a; differentiating first and then substituting gives b. These are separate coefficients.
2. **Exercise:** With c=1, a=1 and b=2, compute q(0.5). **Solution:** The cosine vanishes and the sine equals one, giving 2/pi meters.
3. **Exercise:** What is the heat initial derivative for A=2 and kappa=0.04 on L=1? **Solution:** It is -kappa*pi squared*A=-0.08*pi squared K/s.
4. **Exercise:** Do equal wave displacements at t=0 prove matching initial conditions? **Solution:** No. Initial velocity can differ, as the first guide demonstrates.
5. **Exercise:** Why is b divided by omega in the wave solution? **Solution:** Differentiation then gives q'(0)=b, and b/omega has displacement units.
6. **Exercise:** Does matching both coefficients prove uniqueness for arbitrary initial profiles? **Solution:** No. It proves equality within this single-mode family. General PDE uniqueness requires a stated function class and argument.

## Sources and further study

The [MIT heat-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/8547cf4208529f2a7f3ce6d881f1f3c2_MIT18_152F11_lec_02.pdf) introduce separated heat modes and distinguish boundary conditions from initial data. The [MIT wave-equation notes](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/ccd4ae63858e855c18c96ce96a797b9b_MIT18_152F11_lec_10.pdf) discuss the two wave initial functions. The finite families, control choices and worked numerical comparisons above are direct derivations for this lesson.
