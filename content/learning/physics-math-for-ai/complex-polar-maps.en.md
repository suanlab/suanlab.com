## A pair of reals with a special product

Write z=x+iy, where i²=−1. Multiplication gives (a+ib)(c+id)=(ac−bd)+i(ad+bc), rather than coordinatewise multiplication.

$$
z=r(\cos\theta+i\sin\theta)=re^{i\theta},\qquad r=\sqrt{x^2+y^2}.
$$

For r>0 the angle is defined modulo 2π. Zero has magnitude zero but no unique argument; the lab displays — rather than assigning a direction.

## Multiplication rotates and scales

Angle-addition formulas yield

$$
(r_1e^{i\theta_1})(r_2e^{i\theta_2})=r_1r_2e^{i(\theta_1+\theta_2)}.
$$

Multiplication by nonzero α is an invertible real-plane map combining rotation and uniform scaling. Addition instead adds Cartesian coordinates. Do not transfer the polar multiplication rule to sums.

## De Moivre and power maps

Repeated multiplication proves, for positive integer k,

$$
z^k=r^ke^{ik\theta}.
$$

For r=1,θ=45 degrees,k=2, the output is i. At r=2 its magnitude becomes four with the same angle. Doubling radius does not merely double a squared output.

A power maps circles to circles while winding k times. Nonzero outputs have k roots separated by angles 2π/k. The map is not globally one-to-one for k>1.

## Principal versus accumulated angle

The readout separates accumulated kθ from the atan2 principal argument. At θ=135 degrees,k=2, these are 270 and −90 degrees, representing the same number. Conventions at the negative-real-axis boundary also matter.

Convert displayed degrees to radians before trigonometric evaluation. Do not insert a degree number directly into exp(iθ). Floating-point trigonometry can leave tiny residuals in theoretically zero components.

## Read the geometry and connect to AI

Blue is z and orange zᵏ with equal axis units. This compares two complex-plane values, not an embedding of a whole high-dimensional complex vector space.

Fourier phase, rotational representations and wave amplitudes use this structure. A rotation-aware representation alone does not make a whole model equivariant; nonlinearities, normalization and objectives must respect the intended symmetry too.

## Exercises and answers

1. For r=2,θ=30 degrees,k=3: **magnitude eight, accumulated angle 90 degrees**.
2. z and −z have **the same square**, showing noninjectivity.
3. Zero has no argument because **all angles represent the same zero value**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
