## Differentiate the gradient

For a smooth scalar f:Rⁿ→R, the gradient is a vector and its derivative is an n×n Hessian. Its directional action is

$$
H_{ij}=\partial_i\partial_j f,\qquad H v=D(\nabla f)[v].
$$

Continuous mixed-partial assumptions give symmetry, not positive definiteness. Computing derivatives does not establish convexity.

## Work a nonlinear example

For f(x,y)=½x²y+sin y,

$$
\nabla f=(xy,\tfrac12x^2+\cos y),\qquad
H=\begin{pmatrix}y&x\\x&-\sin y\end{pmatrix}.
$$

At (1,0), gradient=(0,1.5) and H=[[0,1],[1,0]], with eigenvalues ±1. For v=(1,1), Hv=(1,1) and vᵀHv=2. Since v is not unit length, two is not a claim about the maximum eigenvalue.

## Directional curvature and a quadratic model

Along g(t)=f(z+tv),

$$
g'(0)=\nabla f(z)^Tv,\qquad g''(0)=v^TH(z)v.
$$

The quadratic approximation is f(z)+t∇f·v+½t²vᵀHv. For the reference this is 1.5t+t². The actual path is ½(1+t)²t+sin t and retains higher-order terms. The plot compares them.

Along v=(1,0), the reference path is constant zero. A zero direction makes any path constant; it does not remove the input space.

## Products without materializing a Hessian

A full Hessian has n² entries. Differentiating the gradient in direction v can compute Hv without storing all entries. Differentiating the scalar ∇f·v while treating v as constant gives Hᵀv, equal to Hv under symmetry.

The interactive engine evaluates analytic expressions; it does not claim to run an autodiff library. An implementation with JVPs, VJPs or nested differentiation must state shapes, graph retention and whether v is treated as constant.

## Independent checks and AI use

$$
Hv\approx\frac{\nabla f(z+hv)-\nabla f(z-hv)}{2h}.
$$

The regression checks use central differences of an independently evaluated gradient. Zero steps are invalid; very small steps can suffer cancellation.

Newton-CG, influence calculations and curvature diagnostics use HVPs. Nonsmooth operations, stochastic samples or changing batches can invalidate a naive derivative comparison; hold the relevant computation fixed.

## Exercises and answers

1. At the reference with v=(1,−1): **Hv=(−1,1), vᵀHv=−2**.
2. Symmetry **does not imply PSD**; the reference is indefinite.
3. Hv is **a vector**; vᵀHv is scalar.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
