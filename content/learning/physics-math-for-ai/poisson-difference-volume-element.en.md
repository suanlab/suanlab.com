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

## Worked example and exercise

Set A=L=R=0. The exact solution x(1-x) has constant source two. Central second differences reproduce its second derivative exactly, so FD nodal errors are near roundoff. Linear FEM with exact loads also has exact nodal values for this one-dimensional constant-coefficient problem, although its linear interpolant is not the exact quadratic inside an element.

Exercise: insert the exact quadratic into the first FV center equation at x=h/2. Solution: (3u(h/2)-u(3h/2))/h²=3/2, different from source two. In this special model the computed FV center values exceed exact point values by h²/4. Assuming all three methods always give identical point values would conceal this boundary effect.

## Learning a simulator

When evaluating surrogates or physics-informed models, distinguish point samples from cell averages and weak residuals from pointwise residuals. A small matrix residual means that the assembled discrete problem was solved accurately; it does not by itself certify the continuous PDE approximation. Higher-dimensional FEM, unstructured meshes and nonlinear fluxes require additional machinery. This finite experiment supplies a transparent comparison rather than a universal implementation of those methods.

References: [MIT numerical PDE course](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/) and [the Clawpack authors on finite-volume methods](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html).
