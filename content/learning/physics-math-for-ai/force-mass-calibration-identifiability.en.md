## Predicting motion is not the same as identifying its causes

Suppose a camera records a cart accelerating at2 meters per second squared. With a known initial position and velocity, we can predict its entire ideal constant-acceleration trajectory. Can we also conclude that the cart has mass1 kilogram and experiences force2 newtons? A2 kilogram cart driven by4 newtons produces the same acceleration. More accurate observation of that same motion does not automatically separate the two quantities.

This lesson distinguishes a state or trajectory from physical coefficients in a model. We use one spatial dimension, an unknown constant positive mass m and an unknown constant background force F. The equations are Newtonian and noiseless for the exact identification statements. We will explicitly introduce a second controlled experiment to remove the ambiguity, then examine the assumptions that make that intervention informative.

## State the model and what is observed

The model is m times acceleration equals F. With known initial data x0 and v0, its trajectory is

$$
a_0=\frac{F}{m},\qquad
x(t)=x_0+v_0t+\frac12a_0t^2.
$$

Acceleration has units m/s squared, force has units newtons and mass has units kilograms. Even if the entire ideal trajectory determines a0 exactly, it supplies the ratio F/m rather than the two coefficients separately.

If initial data are also unknown, an observation design must first justify recovering those quantities. We are temporarily assuming them known, or that the observations are sufficient to determine the trajectory coefficients. This stronger observation assumption still does not remove the force-mass ambiguity.

## Derive the complete continuous equivalence family

For any positive dimensionless scale lambda, replace the parameters by

$$
m_\lambda=\lambda m,\qquad F_\lambda=\lambda F.
$$

Their ratio remains F/m. With the same initial data, the whole ideal trajectory remains unchanged. These physically different parameter pairs are observationally equivalent under this experiment.

For a known acceleration a0, the full compatible parameter set is

$$
\{(m,F):m>0,\ F=a_0m\}.
$$

It is a ray in the positive-mass portion of the mass-force plane. It is not merely a list of the integer candidates currently exposed by a user interface. A finite control grid can illustrate several points on the ray, but an exact continuous derivation is needed to describe the whole family.

For a0=2, the pairs(1 kg,2 N),(2 kg,4 N) and(3 kg,6 N) all fit. Values between those examples fit too. Counting a few supported candidates must not be presented as a proof that only those masses are possible.

## Explain why repeated observation need not help

Taking another position sample can improve estimation of the trajectory or detect a failure of the constant-acceleration assumption. But after the trajectory is known exactly, more observations from the same unchanged experiment still depend on m and F only through their ratio.

An optimization routine may return one parameter pair because it has an initialization, a regularizer or a numerical preference. That output does not make the underlying pair uniquely identified by the data. A prior can select a representative of the family, but the distinction between information supplied by the experiment and information supplied by the prior should remain visible.

The same issue arises in learned dynamics: excellent predictions along the observed trajectory do not by themselves establish unique recovery of interpretable physical coefficients.

## Add a calibrated intervention

Run a second experiment with the same mass and background force, but add a known constant force deltaF. The second acceleration is

$$
a_1=\frac{F+\Delta F}{m}.
$$

Subtract the first relation:

$$
a_1-a_0=\frac{\Delta F}{m}.
$$

If deltaF is nonzero and the observations follow this model, the acceleration difference is nonzero. The unique recovery formulas are

$$
m=\frac{\Delta F}{a_1-a_0},\qquad F=m a_0.
$$

The intervention contributes an independent physical scale. The initial conditions of the two runs can differ if their accelerations are correctly determined, but the mass and background-force assumptions must remain the same. A changed background force cannot be silently attributed entirely to the calibrated added input.

## Work a complete recovery example

Let the unknown system actually have mass2 kg and background force4 N. The first experiment gives a0=2 m/s squared. By itself, it leaves the entire ray F=2m.

Now add deltaF=2 N. The second acceleration is3 m/s squared. The difference is1 m/s squared, so the recovery formula gives mass2 kg and then background force4 N.

In a mass-force diagram, the first experiment supplies F=a0*m. The second supplies F=a1*m-deltaF. Their intersection is the recovered pair. Each line is a parameter constraint, not a physical path traveled by the cart. The horizontal coordinate is mass and the vertical coordinate is force, with their units labeled separately.

## Keep the zero-input case and inconsistent cases separate

If deltaF=0 and the accelerations agree, the second run repeats the same parameter constraint. The continuous family remains. If deltaF=0 but the exact reported accelerations differ, the two observations are inconsistent with the declared fixed mass and background force.

If deltaF is nonzero but the accelerations are equal, no finite mass can satisfy the model. The division formula is unavailable. It would be misleading to display an infinite number as a successfully identified physical mass.

For a positive mass, the acceleration difference and deltaF must have the same sign. Opposite signs produce a negative algebraic mass, which lies outside the declared physical domain. Separate an algebraic calculation, compatibility with the measurement equations and admissibility under physical assumptions.

Multiplying the acceleration equations by mass can also produce a formal zero-mass solution in a linear system. That point is outside m>0 and makes the original ratio F/m undefined. A zero linear residual must not be promoted into a physically admissible mass.

An inconsistent case is useful information. It can indicate measurement error, an incorrect input calibration or failure of the fixed-environment model. It does not justify silently changing assumptions until a positive answer appears.

## Zero background acceleration can still be informative after intervention

If the background force is zero, a0=0 for every positive mass. The original experiment cannot identify mass at all, even if a stationary or uniformly moving trajectory is measured perfectly.

A nonzero calibrated added force changes this: a1=deltaF/m, so a nonzero measured response identifies mass under the model. The zero background signal and the informative intervention answer different questions.

This illustrates why experimental design matters. Repeating an uninformative zero-excitation experiment with greater precision is not always equivalent to applying an input that separates candidate models.

## Calibration is an assumption about physical force

The formula assumes deltaF is known in newtons. Knowing an actuator command does not automatically make its applied physical force known. If an unknown actuator gain scales the command into force, that gain may become another unidentified coefficient.

If m, F and deltaF are all multiplied by the same positive scale, both accelerations remain unchanged. The two experiments therefore do not create an absolute force or mass reference when the intervention scale is itself unknown.

A meaningful report should state what was calibrated, how it relates to the model input and which parameters were assumed fixed across experiments. These requirements are part of interpreting the inverse problem, rather than extra decoration around a successful fit.

## Exact identifiability differs from sensitivity to noise

The nonzero difference condition gives exact uniqueness in the noiseless model. It does not imply that the recovered mass is insensitive to small acceleration errors.

Writing d=a1-a0 gives m=deltaF/d. A small perturbation in d changes the mass with derivative minus deltaF divided by d squared. When the acceleration difference is small compared with its measurement uncertainty, the estimate can be unstable even though d is numerically nonzero.

The derivative carries units and depends on the chosen uncertainty model. A plotted point estimate should not replace uncertainty analysis. In particular, an uncertainty interval for d that crosses zero needs careful treatment; one should not present an arbitrarily large finite mass as a well-determined result merely because a floating-point denominator did not round to zero.

## Implications for physical models learned by AI

A model can learn the trajectory ratio accurately while its separate force and mass estimates depend strongly on training conventions. Such estimates may look interpretable without being identified by the available observations.

An informative evaluation asks which candidate parameter transformations leave every observed experiment unchanged. It then asks which additional calibrated inputs or measurements remove those equivalences. Structural assumptions, input calibration, exact uniqueness, physical admissibility and statistical uncertainty should be reported separately.

The simple cart is a manageable example of a general habit: first characterize the complete observation-equivalence family, then design a new experiment that changes it. A smaller residual from the same uninformative experiment is not automatically new identifying information.

## Exercises with worked solutions

1. **Exercise:** An exact trajectory has a0=2 m/s squared. Are m=1 kg,F=2 N uniquely identified? **Solution:** No. Every positive mass with F=2m fits the same ratio and trajectory under the stated initial data.
2. **Exercise:** Let a0=2,a1=3 m/s squared and deltaF=2 N. Recover the parameters. **Solution:** m=2/(3-2)=2 kg and F=m*a0=4 N.
3. **Exercise:** What if deltaF=0 and a1=a0? **Solution:** The second equation repeats the first; the continuous parameter family remains.
4. **Exercise:** What if deltaF=2 N but a1=a0 exactly? **Solution:** No finite mass fits this declared model. The inverse division is unavailable, not an identified infinite mass.
5. **Exercise:** Does a0=0 imply that no experiment can identify mass? **Solution:** No. A calibrated nonzero added force and its acceleration response can identify mass under the fixed-background assumptions.
6. **Exercise:** If only an actuator command is known and its force gain is unknown, does the intervention necessarily identify absolute mass? **Solution:** No. An unknown common force scale can preserve a scaling ambiguity.

## Sources and further study

[MIT Underactuated Robotics on system identification](https://underactuated.mit.edu/sysid.html) provides background on distinguishing state estimation from structured parameter identification. The continuous force-mass family, intervention inverse, compatibility cases and sensitivity calculation here are direct derivations for the declared cart model.
