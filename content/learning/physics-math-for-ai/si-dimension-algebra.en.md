## Treat dimensions as structured information

A scientific measurement is more than a bare floating-point number. A value needs a quantity definition, a unit, and a numerical coordinate in that unit. This lesson focuses on the dimension information that survives many changes of units. It extends the earlier conversion lesson into an algebra that can inspect a physical equation and reveal some mistakes before any model is trained or any numerical solver is run.

The laboratory represents a candidate monomial with seven integer exponents and compares it with one selected reference quantity. The supported range is deliberately finite so every displayed example can be understood directly. A passing comparison means that the exponents agree. It does not prove that the candidate is the intended quantity, that addition is meaningful, or that a proposed physical law is correct.

## Fix a seven-coordinate convention

We use the coordinate order length, mass, time, electric current, thermodynamic temperature, amount of substance, and luminous intensity. Their dimension symbols in this lesson are L, M, T, I, Theta, N, and J. The corresponding base-unit symbols are m, kg, s, A, K, mol, and cd. Notice that dimension symbol J in this list denotes luminous intensity; the unit symbol J used for a joule belongs to a different notation layer.

Write a dimension as an exponent vector:

$$
\dim Q=L^{d_1}M^{d_2}T^{d_3}I^{d_4}\Theta^{d_5}N^{d_6}J^{d_7},
\qquad d(Q)=(d_1,\ldots,d_7).
$$

The coordinate order is a convention that must remain fixed in both data and code. Swapping mass and time entries without updating the rest of a pipeline is an indexing error, not a new physical result. The laboratory shows the units next to the coordinate names to make that convention inspectable.

## Derive multiplication from exponent rules

Consider two quantities whose dimensions are monomials in the base dimensions. Multiplying the quantities multiplies those monomials. Powers of the same base combine by adding their exponents, giving

$$
d(AB)=d(A)+d(B),\qquad
d(A/B)=d(A)-d(B),\qquad
d(A^k)=k\,d(A).
$$

For example, velocity has length divided by time dimensions, and multiplying it by time cancels the time exponent. Squaring velocity doubles both its length exponent and its negative time exponent. Negative powers are not exceptional: inverse time has a time exponent of minus one. The operations use exact integer coordinates here; no floating-point tolerance is needed to decide equality of these supported exponent vectors.

This laboratory does not implement arbitrary symbolic fractional powers or a complete symbolic expression parser. To analyze a more elaborate expression, first state the algebraic operations and any domain restrictions of the actual quantities. A dimension calculation alone cannot determine whether a real square root exists or whether a divisor has numerical value zero.

## Work the force comparison

Load the force example. The candidate vector is (1, 1, minus 2, 0, 0, 0, 0). It represents the dimension of a mass multiplied by a length and divided by time squared. The force reference has the same vector. Subtracting the reference gives a zero vector:

$$
d(\text{candidate}/\text{force})=(0,0,0,0,0,0,0).
$$

Blue marks show candidate exponents and orange marks show reference exponents. Matching values appear next to each other; zero exponents are marked on the baseline rather than omitted. The comparison table retains each coordinate and its defect, allowing you to locate a disagreement rather than relying on a single pass indicator.

Now change only the time exponent from minus two to minus one. The time defect becomes plus one, and the comparison fails. The resulting dimension is that of momentum. A bare numerical coincidence between a momentum value and a force value would not repair the missing inverse-time factor.

## Understand what addition requires

If two terms are to be added as values of the same kind of quantity, their dimensions must agree. In a position equation with constant velocity, both the initial position and velocity-times-time terms have length dimensions. Replacing the latter term by velocity alone creates a mismatch.

$$
x(t)=x_0+vt,\qquad d(x_0)=d(vt)=d(x).
$$

Equal dimensions are only a necessary condition. A proposed position equation containing twice the correct coefficient can pass a dimension check and still make incorrect predictions. Two expressions may also share dimensions while representing different quantity kinds, different coordinate components, or different measurement contexts. Keep the mathematical equality, the quantity definition, and the dimensional compatibility as separate statements.

## Compare energy with torque without identifying them

The energy and torque reference entries deliberately share the vector (2, 1, minus 2, 0, 0, 0, 0). Selecting the torque reference while entering those exponents therefore passes the dimension comparison. The result illustrates information that this algebra discards.

For a constant force, work involves a scalar product with displacement. Torque involves a cross product with a position vector and describes rotational action. Their shared dimensional exponents do not turn these definitions into the same operation. In scientific software, an exponent-vector check can be one layer of a richer type system, alongside quantity-kind tags and vector or tensor structure. The laboratory provides the exponent layer and explicitly avoids claiming the richer semantic system.

## Keep dimension-one quantities meaningful

A vector of seven zeros represents dimension one. A ratio of two compatible lengths has this vector, as does a ratio of two times. Their numerical values may answer very different questions. Probability, a normalized error measure, and an angle also require definitions even when a dimension calculation produces no nonzero exponents.

When applying a logarithm to a measured length, specify a reference length and use a ratio, for example log of length divided by reference length. This removes unit dependence from the argument but does not choose the scientifically appropriate reference for you. Likewise, numerical feature standardization in an AI model does not automatically remove all unit choices from its penalties or model coefficients.

## Read the reference selector and its limits

Codes zero through thirteen select dimension one, length, time, mass, force, energy, torque, electric charge, temperature, amount, luminous intensity, pressure, entropy, and frequency. The full mapping appears beside the experiment. Changing a reference can change the expected vector; energy and torque intentionally retain the same vector while changing the quantity label.

Try an entropy reference. Its temperature exponent is minus one, making the otherwise similar energy vector fail. Try electric charge, which combines current and time rather than any length or mass powers. These examples make the additional four base coordinates operational rather than merely a decorative list. Supported integer ranges are a teaching choice, not a claim that all physical dimensions fit between minus three and three.

## Exercises with worked solutions

1. **Exercise:** Derive pressure dimensions from force divided by area. **Solution:** Subtract two length exponents from force, giving length exponent minus one, mass exponent one, and time exponent minus two. The pressure reference verifies those coordinates.
2. **Exercise:** Derive the dimensions of kinetic energy from mass times velocity squared. **Solution:** Squaring velocity gives length squared and inverse time squared; multiplying mass adds its exponent one. The numerical coefficient one half has dimension one and cannot be checked from exponents.
3. **Exercise:** Can the checker distinguish kinetic energy from twice kinetic energy? **Solution:** No. The factor two does not change any exponent. A physical derivation or experiment must establish the coefficient.
4. **Exercise:** Why does selecting torque not reject the energy exponent vector? **Solution:** Both definitions map to the same dimension vector. The mapping forgets quantity-kind and scalar-versus-rotational information.
5. **Exercise:** Enter the charge reference. Which exponents are nonzero? **Solution:** Time and current each have exponent one. All other entries are zero.
6. **Exercise:** Is a logarithm of a dimensional measurement fixed by changing meters to centimeters? **Solution:** No. Use a stated dimensionless ratio; otherwise its numerical value depends on the chosen unit.

## Source and next research step

The notation and quantity distinctions follow the [BIPM SI Brochure, ninth edition, updated in 2026](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf), especially its discussion of dimensions and derived quantities. The seven-vector calculations and laboratory examples are direct derivations. A useful research extension is a scientific-data schema combining dimensions, quantity kinds, coordinate frames, uncertainty, and provenance. Each layer catches different mistakes; a dimension-only pass should remain labeled as such.
