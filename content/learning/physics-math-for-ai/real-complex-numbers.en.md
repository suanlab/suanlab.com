## Why extend a number system?

Whole-number counting does not solve every subtraction; integers do. Integers do not solve every nonzero division; rationals do. Rational numbers still leave gaps such as a positive length whose square is two. Real numbers complete the ordered number line, while complex numbers allow solutions such as a square root of −1.

These extensions preserve useful arithmetic while adding new structure. Complex numbers do not have an order compatible with all the ordered-field rules of the reals: every square would have to be nonnegative, conflicting with i²=−1.

## Rational and irrational real numbers

To prove √2 is irrational, suppose √2=a/b in lowest terms with integers a,b and b≠0. Then a²=2b², so a is even by the parity theorem from the proof unit. Write a=2k; then b²=2k² makes b even too. Both share factor two, contradicting lowest terms.

This is a general contradiction proof, not evidence from a long decimal expansion. Completeness of the real numbers is a stronger structural property than displaying many digits. Later analysis will formulate least upper bounds and convergence precisely; here it explains why ℚ and ℝ must be distinguished.

## Represent complex arithmetic by pairs

Write z=a+bi with real a,b and i²=−1. Addition is componentwise. Multiplication follows distributivity:

$$
(a+bi)(c+di)=(ac-bd)+(ad+bc)i.
$$

The conjugate is a−bi, and z times its conjugate equals a²+b². Therefore |z|=√(a²+b²), and for nonzero z its inverse is conjugate(z)/|z|². Zero has no multiplicative inverse.

## Rotation and magnitude

A unit complex multiplier cosθ+i sinθ rotates the coordinate pair. With θ expressed in radians, it is also written exp(iθ); the trigonometry unit develops the angle conventions in more detail. Multiplication gives

$$
(a,b)\mapsto(a\cos\theta-b\sin\theta,\ a\sin\theta+b\cos\theta).
$$

Its squared length expands to a²+b² because cos²θ+sin²θ=1. Thus the rotation preserves magnitude. At z=1+i and θ=90°, the result is −1+i. Both magnitudes are √2.

At z=i and θ=180°, the result is −i. Tiny nonzero real coordinates in a floating-point display can come from approximating trigonometric values, not a failure of the exact identity.

## Argument is not a globally continuous coordinate

For z≠0, an angle describes its direction only up to full turns. A principal argument chooses one branch; atan2 returns values between −180° and 180° in the display convention. Crossing the negative real axis can make the displayed angle jump even though the point moves continuously.

At z=0, every rotation still gives zero and no direction is defined. The lab reports no argument rather than inventing zero degrees. This matters for phases in signals and for avoiding artificial discontinuities in learning targets.

The ordinary real logarithm does not apply unchanged to complex values. Complex logarithms require choices of argument and branch, and are outside this introductory model. A nonzero magnitude alone does not resolve those choices.

## Read the complex-plane experiment

The horizontal coordinate is real part and vertical coordinate is imaginary part. Both axes use the same unit length so angles and lengths are not visually distorted. Blue is the original vector from zero and orange is the rotated vector. The figure is a representation of a complex number, not a third physical direction or an extra observable probability.

Change the real and imaginary parts before rotating. Then set both to zero and compare the valid magnitude zero with the undefined argument. The numeric tuple and its plotted endpoint are computed from the same values.

## AI connections and exercises

Fourier representations use complex amplitude and phase. Complex arithmetic can encode two coupled real coordinates, but a complex-valued model must specify its loss and differentiation convention. It is not automatically more expressive or physically meaningful merely because i appears in an equation.

1. Compute (1+i)². **Answer:** 2i.
2. Find 1/(1+i). **Answer:** (1−i)/2.
3. Explain why a principal phase jump need not mean the underlying signal jumped.
4. Prove |zw|=|z||w| using conjugates or pair multiplication, and state why z=0 still has a well-defined magnitude but no angle.

For arithmetic and complex numbers see [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e). The pair rotations and boundary checks here are independent worked examples.
