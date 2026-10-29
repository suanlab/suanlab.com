## Parametrize the contour

For a piecewise smooth path γ,

$$
\int_\gamma f(z)\,dz=\int_a^b f(\gamma(t))\gamma'(t)\,dt.
$$

Do not sum only function values: the complex tangent factor matters. Orientation-preserving regular reparametrization preserves the integral; reversal changes its sign. This differs from an arc-length integral of magnitudes.

## Primitives give path independence

If F′=f throughout the domain, the chain rule gives the endpoint difference F(γ(b))−F(γ(a)). Thus a closed contour has zero integral. For f=z, the primitive z²/2 exists on the entire plane:

$$
\oint z\,dz=0.
$$

A tiny numerical residual is distinct from that exact mathematical zero. The lab calculates midpoint quadrature separately from the analytic reference.

## Cauchy’s theorem needs a domain

A holomorphic function on a suitable simply connected open domain has zero integral around closed contours within it. More general forms use homotopy or homology conditions; the introductory statement fixes the simpler domain assumption.

Smooth boundary values alone are insufficient. Interior singularities or holes can prevent a global primitive. There is no general rule that every closed complex integral vanishes.

## Directly integrate 1/z

For R>0 and a counterclockwise circle, γ=R exp(it) and γ′=iR exp(it):

$$
\oint\frac{dz}{z}=\int_0^{2\pi}\frac{iRe^{it}}{Re^{it}}\,dt=2\pi i.
$$

The result is independent of radius but reverses under clockwise traversal. The origin violates interior holomorphicity, so the zero-integral theorem does not apply. This also explains the obstruction to a global logarithm primitive.

## Read direction and degenerate cases

Blue is the input contour; orange arrows indicate direction, not integral magnitude. At radius zero, the constant path for f=z integrates to zero, while 1/z is undefined along the whole path. Distinguish direct definitions on degenerate paths from regular-contour theorem assumptions.

Finite midpoint quadrature illustrates these examples without proving the theorem for every holomorphic function.

## Applications and exercises

Fourier and Laplace calculations, wave responses and contour-based algorithms depend on singularity location. Changing a numerical contour requires checking poles crossed and the valid domain.

1. Clockwise ∮dz/z is **−2πi**.
2. Doubling radius keeps the CCW result **2πi**.
3. The closed integral of z vanishes because **z²/2 is a global primitive**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
