## Kernels connect feature inner products and function spaces

A PSD kernel requires a nonnegative quadratic form for every finite input set and coefficient vector. Positive entries alone do not suffice. An RKHS is a Hilbert function space with continuous point evaluation, represented by kernel section kₓ.

$$
f(x)=\langle f,k_x\rangle_H,\qquad \|k_x\|_H^2=k(x,x).
$$

Consequently |f(x)|≤‖f‖H√k(x,x). Unlike general L2, stable evaluation is built into this space.

## Calculate everything with two features

For kρ(x,z)=1+ρxz on [−1,1], positive rho gives features (1,√ρx). Writing f=a+bx uses coefficients (a,b/√ρ):

$$
\|a+bx\|_H^2=a^2+b^2/\rho.
$$

At rho=1, f=1+x and query .5, squared norm is two, evaluation norm √1.25 and reproduction gives 1.5. Changing the kernel changes the norm of the same function. At rho=0 only constants remain. A sloped target lies outside the space: displayed ∞ is extended notation, not a legitimate finite RKHS norm; reproduction is undefined.

## Mercer theory depends on a measure

Continuous symmetric PSD kernels on compact domains, with suitable finite Borel measures, admit integral-operator spectral descriptions. Fix uniform probability measure dx/2. Orthonormal functions e₀=1 and e₁=√3x have eigenvalues one and rho/3:

$$
k(x,z)=1\,e_0(x)e_0(z)+\frac\rho3e_1(x)e_1(z).
$$

The three-point Gram matrix at −1,0,1 instead has eigenvalues 3,2ρ,0. Finite Gram and integral spectra differ because sample count and weights differ.

## Read the curves correctly

Blue is a kernel section and orange the target. Reproduction concerns their **RKHS inner product**, not equality of their heights. Kernel scaling encodes which functions have small norm. This two-feature space is finite dimensional; an RBF RKHS need not be.

## Exercises with explanations

1. Squared norm of x at rho=.5? **Two**, since slope cost is 1/rho.
2. Does a zero Gram eigenvalue invalidate the kernel? **No.** PSD allows singular matrices.
3. Replace dx/2 by dx: what happens to integral eigenvalues? **They double**, while the kernel formula remains unchanged.

Further reading: [MIT 9.520, RKHS and Mercer](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/resources/class14/).
