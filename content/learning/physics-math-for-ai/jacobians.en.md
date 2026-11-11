## From one sensitivity to a map of sensitivities

A scalar derivative answers one input-output question. An AI model may instead turn thousands of inputs into hundreds of features. We need a systematic way to describe how each output responds to each input. The **Jacobian** collects these partial derivatives into a matrix.

Its purpose is local prediction. Near a point, a nonlinear map behaves approximately like an affine map: its current output plus a matrix applied to the input displacement. This connects the linear-transformation unit to calculus without claiming that the original map is globally linear.

We will compute one Jacobian by hand, propagate an input direction forward, propagate output importance backward and explain why shared computation paths require addition. All examples use real column vectors and the Euclidean inner product.

## The example and its shapes

Use the two-input, two-output function

$$
F(x,y)=\begin{pmatrix}xy\\x^2+y\end{pmatrix}.
$$

| Symbol | Shape and role |
| --- | --- |
| $p=(x,y)^T$ | Current input, $2\times1$ |
| $F(p)$ | Current output, $2\times1$ |
| $J(p)$ | Output-by-input derivative matrix, $2\times2$ |
| $d$ | Input direction or displacement, $2\times1$ |
| $q$ | Fixed output weights, $2\times1$ |
| $Jd$ | Predicted output change, $2\times1$ |
| $J^Tq$ | Input gradient of $q^TF$, $2\times1$ |

A square example hides dimension mistakes, so keep the general case in mind: for $F:\mathbb R^n\to\mathbb R^m$, $J$ has $m$ rows and $n$ columns. Rows index outputs; columns index inputs. This convention will be used throughout the book.

## Build the Jacobian one row at a time

For the first output, $F_1=xy$, differentiate with respect to $x$ while holding $y$ fixed, then with respect to $y$ while holding $x$ fixed. The row is $(y,x)$. For the second output, $F_2=x^2+y$, the row is $(2x,1)$.

$$
J(x,y)=\begin{pmatrix}y&x\\2x&1\end{pmatrix}.
$$

At $p=(2,3)^T$, the forward output is $(6,7)^T$ and

$$
J(p)=\begin{pmatrix}3&2\\4&1\end{pmatrix}.
$$

For example, the entry 4 means that a small increase in $x$, with $y$ fixed, changes the second output by approximately four times that increase. It is a sensitivity, not the second output value.

## Push an input direction forward: JVP

Let $d=(0.1,-0.2)^T$. Multiplication gives the Jacobian-vector product, often abbreviated JVP:

$$
Jd=\begin{pmatrix}3(0.1)+2(-0.2)\\4(0.1)+1(-0.2)\end{pmatrix}
=\begin{pmatrix}-0.1\\0.2\end{pmatrix}.
$$

Each output receives contributions from both inputs. A JVP answers: “If I perturb the inputs in this direction, what is the first-order output response?” The direction need not have unit length. If you want change per unit distance, normalize the direction and explicitly state that choice.

For differentiable maps, the underlying statement is

$$
F(p+d)=F(p)+J(p)d+o(\lVert d\rVert).
$$

All matrices and local derivatives are evaluated at the original point $p$.

## Compare with the actual finite displacement

The new point is $(2.1,2.8)^T$ and its output is $(5.88,7.21)^T$. The actual change is therefore $(-0.12,0.21)^T$, slightly different from $Jd$.

For this polynomial example, expansion yields the exact remainder:

$$
F(p+d)-F(p)-Jd
=\begin{pmatrix}d_xd_y\\d_x^2\end{pmatrix}.
$$

At the reference settings this is $(-0.02,0.01)^T$. The derivative is correct; the difference comes from using a local linear prediction for a finite move.

Reduce $d$ tenfold to $(0.01,-0.02)^T$. The JVP becomes $(-0.01,0.02)^T$ and the remainder becomes $(-0.0002,0.0001)^T$, one hundredth of its previous size. This is a useful experiment for separating a derivative implementation bug from ordinary approximation error.

## Total differentiability controls every small displacement

For an open-domain map F:Rⁿ→Rᵐ, total (Fréchet) differentiability at p means there is one linear map A such that

$$
\lim_{d\to0,\,d\ne0}\frac{\|F(p+d)-F(p)-Ad\|}{\|d\|}=0.
$$

The approach may bend or change direction as it shrinks. This is stronger than checking each fixed straight direction separately. If the limit holds, coordinate directions identify A’s columns as partial derivatives, so A is the Jacobian. Continuous partial derivatives on a neighborhood are a sufficient condition, although not a necessary one.

For the lab’s polynomial, the exact remainder already computed is (dₓdᵧ,dₓ²). Each entry has absolute value at most ‖d‖₂², hence its vector norm is at most √2‖d‖₂². Dividing by ‖d‖₂ bounds the quotient by √2‖d‖₂→0. This establishes the displayed Jacobian as the total derivative, beyond testing finitely many directions.

## A continuous function with every directional derivative but no total derivative

Define a separate scalar example, with value zero at the origin:

$$
f(x,y)=\begin{cases}\dfrac{x^3y}{x^4+y^2}&(x,y)\ne(0,0),\\0&(x,y)=(0,0).\end{cases}
$$

Because 2|x²y|≤x⁴+y², we have |f(x,y)|≤|x|/2. Thus f is continuous at the origin. For any fixed v=(a,b) with b≠0,

$$
\frac{f(ta,tb)}t=\frac{t a^3b}{t^2a^4+b^2}\longrightarrow0.
$$

For b=0 the quotient is exactly zero. Every directional derivative therefore exists and equals zero; the proposed directional map is even linear. Both coordinate partial derivatives are zero, so a total derivative, if it existed, would have to be A=0.

Now take the curved approach (x,y)=(t,t²), t≠0. Direct substitution gives f(t,t²)=t/2 and

$$
\frac{|f(t,t^2)|}{\|(t,t^2)\|_2}=\frac1{2\sqrt{1+t^2}}\longrightarrow\frac12.
$$

The required remainder limit fails. The paths become increasingly tangent to the x-axis, while each directional derivative holds its ray fixed. This difference in quantifiers explains how all the straight-line checks can miss the failure. Also ∂f/∂y(x,0)=1/x for x≠0, so the partial derivative is not continuous at the origin.

| Positive t | f(t,t)/t on a fixed ray | f(t,t²)/t on the curved path |
|---|---|---|
| 1/2 | 2/5 | 1/2 |
| 1/8 | 8/65 | 1/2 |
| 1/32 | 32/1025 | 1/2 |

## Reproduce the path comparison in Python

The code uses exact fractions and checks the table and fixed-ray formulas, including negative t and coordinate directions. Its finite cases illustrate the formulas; the limits and continuity arguments above establish the claims for all directions. The interactive lab continues to display F=(xy,x²+y), not this separate counterexample.

```python
from fractions import Fraction as Q

def f(x, y):
    if x == 0 and y == 0:
        return Q(0)
    return x**3 * y / (x**4 + y**2)

for t in [Q(1, 2), Q(1, 8), Q(1, 32)]:
    ray = f(t, t) / t
    curved = f(t, t * t) / t
    assert ray == t / (1 + t * t)
    assert curved == Q(1, 2)
    print(t, ray, curved)

for a, b in [(Q(1), Q(0)), (Q(0), Q(1)),
             (Q(2), Q(-3)), (Q(-1), Q(2))]:
    for t in [Q(1, 4), Q(-1, 8), Q(1, 32)]:
        quotient = f(t * a, t * b) / t
        expected = Q(0) if b == 0 else t * a**3 * b / (t*t*a**4 + b*b)
        assert quotient == expected
        assert abs(f(t*a, t*b)) <= abs(t*a) / 2
```

Research lesson: a successful finite set of directional gradient checks does not prove total differentiability. A check should record the model’s differentiability assumptions, the paths sampled and the scale of the remainder. For this example, explain why a Jacobian assembled from coordinate partials cannot justify the usual linear approximation at the origin.

## Pull output importance backward: VJP

Suppose we care about the scalar score $L=F_1-F_2$. Define fixed output weights $q=(1,-1)^T$, so $L=q^TF$. The input gradient is

$$
\nabla_p L=J^Tq
=\begin{pmatrix}3&4\\2&1\end{pmatrix}
\begin{pmatrix}1\\-1\end{pmatrix}
=\begin{pmatrix}-1\\1\end{pmatrix}.
$$

The usual row-vector notation is $q^TJ$, called a vector-Jacobian product or VJP. Our display uses its transpose, $J^Tq$, to keep gradients as columns. Both represent the same sensitivities with a stated orientation.

Here $q$ is **not** an input displacement. It assigns importance to the outputs. The backward operation answers: “Which inputs matter to this weighted output score?” For non-Euclidean inner products, the adjoint needs the corresponding metric; the ordinary transpose here relies on Euclidean coordinates.

## The same scalar in two directions

Forward and reverse products agree when paired:

$$
q^T(Jd)=(J^Tq)^Td=-0.3.
$$

You can first push the input direction into output space and score it there, or first pull the score sensitivity into input space and pair it with the direction. This identity is a practical check of orientation and implementation. It holds for any compatible matrix and vectors; differentiability is needed to interpret the matrix as a derivative.

For the actual finite displacement, $q^T\Delta F=-0.33$. That differs from $-0.3$ because of the same nonlinear remainder. The duality identity is exact for the local linear map, not for arbitrary finite nonlinear changes.

## Branches require accumulation

The computation graph contains a branch from $x$ to both $xy$ and $x^2+y$. For $L=F_1-F_2$, the contribution to the $x$ gradient from the first output is 3. The second output contributes $-4$. Their sum is $-1$.

For $y$, the contributions are 2 and $-1$, summing to 1. A reverse implementation must **add** contributions arriving through different paths. Overwriting an earlier contribution with a later one silently computes the wrong gradient. This principle applies to shared parameters, residual connections and reused tensors in neural networks.

For a composition $H=G\circ F$, the matrix chain rule is $J_H(p)=J_G(F(p))J_F(p)$. The product order follows the forward flow of column directions. Reverse propagation applies the transposes in the opposite order; it need not construct the full product matrix.

## A singular Jacobian is a local statement

The determinant is $y-2x^2$. At $p=(0.5,0.5)^T$, it is zero and

$$
J=\begin{pmatrix}0.5&0.5\\1&1\end{pmatrix}.
$$

For $d=(0.1,-0.1)^T$, $Jd=0$. Yet the exact output change is $(-0.01,0.01)^T$. A direction is invisible to first order, but higher-order effects survive.

Do not infer global constancy or a global inverse from a Jacobian at one point. The inverse function theorem provides a sufficient local invertibility condition when the Jacobian is nonsingular and the map is continuously differentiable nearby. Failure of that sufficient condition does not, by itself, prove every possible global claim.

## Read the lab and check the derivative

The graph bundles the two inputs for readability; the matrix and readout provide the individual sensitivities. Blue and orange arrows distinguish forward values from reverse sensitivities, not physical motion. The output-displacement chart uses the same axes for the actual change and $Jd$; its positive display limits are labeled and automatically rescale.

The central-difference readout estimates a directional derivative using

$$
\frac{F(p+\varepsilon d)-F(p-\varepsilon d)}{2\varepsilon}.
$$

It estimates $Jd$, not the finite displacement $F(p+d)-F(p)$. For these quadratic outputs it agrees in exact arithmetic; tiny floating-point differences can remain. General nonlinear functions also have truncation error. Change $d$ to test finite-move accuracy and change $\varepsilon$ to test numerical differentiation.

## Exercises and further study

1. At $(1,2)^T$, calculate $J$, then $J(1,0)^T$. **Check:** $J=[[2,1],[2,1]]$, and the product is $(2,2)^T$.
2. At the reference point, choose $q=(1,1)^T$. **Check:** $J^Tq=(7,3)^T$.
3. Derive the exact remainder for the example without looking at the formula. Identify which operations produce the two quadratic terms.
4. For an $m$-output, $n$-input model, write the shapes of $J,d,q,Jd,J^Tq$ before writing any code.

For the calculus background, see [MIT OpenCourseWare: Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). For a concrete implementation perspective on reverse-mode automatic differentiation, consult [PyTorch’s Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). The next unit uses these local products to derive every gradient of a small neural network by hand.
