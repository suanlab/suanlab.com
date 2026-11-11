## Ask three different discrete questions

Solve -u″=f on [0,1]. Choose the exact function u=L(1-x)+Rx+x(1-x)+A sin(πx), then manufacture f=2+Aπ²sin(πx). Knowing the solution independently lets us inspect boundary treatment, assembly, linear-solve error and discretization error separately.

Finite differences begin with a derivative approximation at nodes. Finite volumes begin with flux balance over cells. Finite elements begin with a weak equation against test functions. Rather than attaching three names to one stencil, this experiment actually changes unknown locations and load assembly. Compare the upper location diagram with the numeric table: FV centers differ from FD/FEM interior nodes.

## Difference and volume equations

For uniform spacing h, nodal central finite differences solve

$$
\frac{2u_i-u_{i-1}-u_{i+1}}{h^2}=f(x_i).
$$

Finite volume uses J=-u′ and J_right-J_left≈h f_center. Each interior face flux enters adjacent cells with opposite signs. The first cell center is only h/2 from the boundary, so its boundary derivative must use a half-cell distance. The resulting endpoint diagonal is 3/h², compared with 2/h² internally, and the boundary-value load coefficient is 2/h². Here FV unknowns are interpreted as cell-center point approximations, not exact cell averages.

## The finite-element weak equation

Multiply by a test function vanishing at Dirichlet boundaries and integrate by parts:

$$
\int_0^1u'v'\,dx=\int_0^1fv\,dx,\qquad
K_{ij}=\int\phi_i'\phi_j'\,dx,\quad b_i=\int f\phi_i\,dx.
$$

Piecewise-linear hat functions produce diagonal 2/h and neighboring entries -1/h. The load is integrated over elements using two-point Gauss quadrature, rather than inserting f_i directly. This quadrature is exact for appropriate polynomial integrands but remains approximate for the sinusoidal load. Rescaling K to resemble an FD matrix does not erase the difference between their right-hand sides.

## Exact nodal values can coexist with nonzero function error

With exact weak loads, the constant-coefficient 1D linear FEM solution is the nodal interpolant Iₕu. To prove it, note that each test derivative vₕ′ is constant on an element. Therefore ∫(u−Iₕu)′vₕ′ is that constant times the difference of the endpoint interpolation errors, both zero. Galerkin orthogonality and uniqueness identify uₕ=Iₕu. More generally, orthogonality gives the energy best-approximation property for this symmetric coercive bilinear form.

This does not make the interpolant equal to u between nodes. For u=x(1−x), its interpolation error on an element starting at a is ξ(h−ξ), ξ=x−a. Integrating its square and its derivative square over N uniform elements gives

$$
\|u-I_hu\|_{L^2}^2=\frac{h^4}{30},\qquad
\|(u-I_hu)'\|_{L^2}^2=\frac{h^2}{3}.
$$

| N | Squared L2 error | Squared energy error |
|---|---|---|
| 4 | 1/(30·4⁴) | 1/(3·4²) |
| 8 | 1/(30·8⁴) | 1/(3·8²) |
| 16 | 1/(30·16⁴) | 1/(3·16²) |

```python
from fractions import Fraction as F
def square(c):
    out=[F(0)]*(2*len(c)-1)
    for i,a in enumerate(c):
        for j,b in enumerate(c):out[i+j]+=a*b
    return out
def integral(c,h):return sum(a*h**(i+1)/F(i+1) for i,a in enumerate(c))
for N in (4,8,16):
    h=F(1,N)
    l2=N*integral(square([F(0),h,F(-1)]),h)
    energy=N*integral(square([h,F(-2)]),h)
    assert l2==h**4/30 and energy==h*h/3
    print(N,l2,energy)
```

The exact polynomial integration checks errors across entire elements, not just at nodes. The proof assumes exact loads and a constant diffusion coefficient. The live two-point quadrature is exact for the quadratic example's constant source, but approximate for its sine source. Quadrature error then adds a separate consistency term; this nodal argument must not be transferred unchanged to that approximate load or to variable coefficients.

## Worked example and exercise

Set A=L=R=0. The exact solution x(1-x) has constant source two. Central second differences reproduce its second derivative exactly, so FD nodal errors are near roundoff. Linear FEM with exact loads also has exact nodal values for this one-dimensional constant-coefficient problem, although its linear interpolant is not the exact quadratic inside an element.

Exercise: insert the exact quadratic into the first FV center equation at x=h/2. Solution: (3u(h/2)-u(3h/2))/h²=3/2, different from source two. In this special model the computed FV center values exceed exact point values by h²/4. Assuming all three methods always give identical point values would conceal this boundary effect.

## Learning a simulator

When evaluating surrogates or physics-informed models, distinguish point samples from cell averages and weak residuals from pointwise residuals. A small matrix residual means that the assembled discrete problem was solved accurately; it does not by itself certify the continuous PDE approximation. Higher-dimensional FEM, unstructured meshes and nonlinear fluxes require additional machinery. This finite experiment supplies a transparent comparison rather than a universal implementation of those methods.

References: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/) and [the Clawpack authors on finite-volume methods](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html).
