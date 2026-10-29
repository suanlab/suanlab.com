## One quantity can have many representations

The fraction 1/3, the fraction 2/6 and a repeating decimal describe the same rational number. A displayed decimal such as 0.333 is only an approximation. By contrast, 150 centimeters and 1.5 meters describe the same physical length with different numerical values and units. These are two distinct kinds of representation change.

This distinction matters in AI preprocessing and scientific data. Rescaling coordinates need not change a physical system, but it can change a numerical loss, penalty or optimizer unless the rest of the model is transformed consistently.

## Integers, fractions and equivalence

Integers include negative values, zero and positive values. A rational number is represented by a/b with integer a and nonzero integer b. Two such fractions represent the same number when ad=bc. Multiplying numerator and denominator by the same nonzero integer preserves the ratio.

$$
\frac13=\frac26,\qquad \frac16+\frac13=\frac16+\frac26=\frac12.
$$

The common denominator expresses comparable pieces before adding. Adding numerators and denominators separately would produce 2/9 instead of 1/2. Multiplication is different: multiply the numerators and denominators, then reduce common factors.

The lab reduces fractions using the greatest common divisor. For 0/3 the reduced form is 0/1. For 1/0 there is no rational value. The expression 0/0 is not rescued by numerator and denominator being equal; cancellation requires a nonzero divisor.

## Decimal displays and finite precision

A rational has a terminating decimal exactly when its reduced denominator has no prime factors other than 2 and 5. Thus 1/8=0.125 terminates, while 1/3 repeats. Neither form changes the underlying rational value.

Computers often store approximations in binary rather than exact rational arithmetic. Even 0.1 need not have a finite binary representation. The lab shows a rounded quotient while retaining the exact small-integer reduced fraction separately. A rounded equality is not a theorem about the original numbers.

## Ratios and percentages

A dimensionless ratio compares compatible quantities, such as two lengths expressed consistently. A percentage is one hundred times such a ratio. Moving from 20% to 30% is an increase of ten percentage points and a relative increase of 50%; the two comparisons answer different questions.

A rate can carry units. Meters per second is not a pure percentage, and a numerical value of 75 cm/s equals 0.75 m/s. Record both the reference quantity and units before interpreting a ratio in an evaluation report.

## Convert by multiplying by one

Use an equality of units to form a conversion factor:

$$
150\,\mathrm{cm}\times\frac{1\,\mathrm m}{100\,\mathrm{cm}}=1.5\,\mathrm m.
$$

For elapsed time 2 seconds, average speed over this path length is 0.75 m/s. Zero elapsed time does not supply a finite average speed by division. A physical limiting velocity requires a different limiting definition, not assigning a value to 1.5/0.

Powers of units must also transform. One square meter is 10,000 square centimeters, not 100. One cubic meter is one million cubic centimeters. Converting only the written prefix and forgetting the power changes the quantity.

## Dimensions are an algebraic consistency check

Denote length, time and mass dimensions by L,T,M. Speed has dimension LT⁻¹, acceleration LT⁻², force MLT⁻² and mechanical energy ML²T⁻². Addition requires compatible dimensions; multiplication combines dimension powers.

In x(t)=x₀+vt, both terms on the right have length dimension. In exp(−kt), the exponent must be dimensionless, requiring k to have inverse-time units. A logarithm of a dimensional measurement should instead be written using a dimensionless ratio, such as log(L/L₀).

Dimensional consistency is necessary for a physical equation but does not establish its correctness. Many wrong laws have the right units. Nondimensionalizing with reference length L₀ and time T₀ gives x*=x/L₀ and τ=t/T₀; it clarifies scale while preserving the need to transform the equation.

## Read the experiment and connect to AI

The number-line plot shows only the dimensionless fraction. Its vertical marker has no physical magnitude. Length conversion and speed are separate readouts, so incompatible quantities are not placed on a shared numeric axis.

Load the equivalent fraction and observe the unchanged location. Then load zero denominator and zero time: undefined values are shown as —, and no fraction point is fabricated. In real datasets, store units and conversion history explicitly. Normalizing inputs may improve numerical behavior while changing the meaning of an unadjusted regularizer.

## Practice and sources

1. Convert 36 km/h to m/s. **Answer:** 10 m/s.
2. Reduce −4/6. **Answer:** −2/3.
3. Convert 2 m² to cm². **Answer:** 20,000 cm².
4. Check the dimensions of E=mv²/2 and explain why this check does not prove the law.

For algebra background see [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e). For unit conventions consult the [BIPM SI Brochure](https://www.bipm.org/en/publications/si-brochure). The conversion and dimension examples above are direct worked calculations.
