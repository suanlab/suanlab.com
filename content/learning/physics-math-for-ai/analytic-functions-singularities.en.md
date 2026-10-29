## Analyticity is local

A holomorphic function has a convergent power series in a sufficiently small disk around each point:

$$
f(z)=\sum_{n=0}^{\infty}a_n(z-a)^n.
$$

Nearby complex singularities can limit the radius. Smooth-looking behavior along the real axis does not guarantee global complex-series convergence.

## A removable point

Divide the sine series by z:

$$
\frac{\sin z}{z}=1-\frac{z^2}{3!}+\frac{z^4}{5!}-\cdots.
$$

Every complex approach to zero gives limit one. Defining the missing value as one gives a holomorphic extension. The lab uses that extension rather than performing zero division, with a sufficiently long series on its small |z|≤1 domain.

## Poles and essential singularities

A finite negative Laurent principal part with lowest power −m gives a pole of order m. The example 1/z has a simple pole. Infinitely many negative powers instead give an essential singularity:

$$
e^{1/z}=1+\frac1z+\frac1{2!z^2}+\frac1{3!z^3}+\cdots.
$$

Along the positive real axis it grows rapidly near zero; along the negative real axis it tends to zero; on the imaginary axis its magnitude is one while it oscillates. One path cannot determine a complex limit. Its residue is still one, separately from its essential classification.

## What the classification means

Removable, pole and essential are Laurent-structure properties in a punctured neighborhood. The lab classifies explicit known examples; it does not infer every singularity type from finite samples.

Boundedness along one ray differs from boundedness throughout a punctured neighborhood. The latter supports the removable-singularity theorem. Small values on the negative real axis do not make exp(1/z) removable.

## Plot domain and numerical range

The graph samples radii 0.1 through one at the selected angle. The separate probe can include zero. At zero, 1/z and exp(1/z) are undefined. At positive radius, a value can instead exceed machine range; that representation limit is reported separately.

Do not replace a large but finite mathematical value by a claim of infinity. The reciprocal calculation scales components to reduce avoidable underflow in x²+y².

## Exercises and answers

1. The sine quotient’s extended value and residue are **one and zero**.
2. The residue of exp(1/z) is **one**.
3. Tending to zero along one path **does not establish a complex limit**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
