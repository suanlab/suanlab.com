## Solve a relation locally

The circle x²+y²=1 is not globally one function y=f(x), since many x values have two corresponding y values. Near (0.6,0.8), however, its upper branch is differentiable:

$$
F(x,y)=x^2+y^2-1,\qquad F_x=2x,\quad F_y=2y.
$$

For C¹ F with F(x₀,y₀)=0 and Fᵧ≠0, the implicit function theorem supplies a locally unique differentiable y(x). Locality matters; it does not select a globally unique branch around the entire circle.

## Derive implicit differentiation

Differentiate F(x,y(x))=0 using the chain rule:

$$
y'(x)=-\frac{F_x}{F_y}=-\frac xy.
$$

At the reference this is −0.75. At (1,0), the denominator vanishes; the upper branch has no finite derivative. The lab does not replace a vertical tangent with zero slope.

For vector-valued y, Fᵧ is a square Jacobian and must be invertible. The derivative becomes −(Fᵧ)⁻¹Fₓ. Solving a linear system can be preferable to explicitly materializing an inverse.

## Inverse function theorem: a sufficient condition

A C¹ map f:Rⁿ→Rⁿ with invertible Jacobian at a point has a local C¹ inverse, with

$$
D(f^{-1})(f(x_0))=[Df(x_0)]^{-1}.
$$

Invertibility of the derivative is sufficient for this differentiable local inverse, not necessary for mere existence of an inverse. The cube x³ is globally one-to-one despite f′(0)=0. Its cube-root inverse exists but has no finite derivative at zero.

## Interpret singular cases carefully

At the circle endpoint, expressing y as a differentiable function of x fails, but expressing x in terms of y can succeed. Changing the dependent variable changes the relevant Jacobian block. The curve itself has not disappeared.

Failure of a theorem’s assumption is not automatically the negation of its conclusion; it removes the theorem’s guarantee and calls for separate analysis.

## AI connections and exercises

Implicit differentiation of stationary conditions G(θ,z)=0 can give solution sensitivity to data z. Check existence, solution selection, Jacobian invertibility and stability. Unrolling an iterative solver and implicitly differentiating its limiting condition have different computational and mathematical requirements.

Mode zero uses the upper circle branch; mode one compares a cube with its inverse. Axes have equal unit length. Undefined derivatives are distinct from missing function values.

1. At (0,1), the circle slope is **zero**.
2. At x=1 for the cube, the inverse derivative is **one third**.
3. An invertible Jacobian guarantees **a local inverse, not global invertibility**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
