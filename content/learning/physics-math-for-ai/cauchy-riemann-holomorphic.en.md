## A stronger directional requirement

A complex derivative requires the same quotient limit as complex h approaches zero from every direction. Write f=u+iv:

$$
f'(z)=\lim_{h\to0}\frac{f(z+h)-f(z)}h,\qquad
u_x=v_y,\quad u_y=-v_x.
$$

Read u as the real component. Partial derivatives alone are not sufficient: use real differentiability, or suitable continuity of partials nearby, together with CR. All three models here are real C¹.

## Compare two quotient directions

$$
\lim_{t\to0}\frac{f(z+t)-f(z)}t=u_x+iv_x,\qquad
\lim_{t\to0}\frac{f(z+it)-f(z)}{it}=v_y-iu_y.
$$

For z², u=x²−y² and v=2xy give the same value 2z. For conjugation, u=x,v=−y give one and minus one, ruling out a complex derivative everywhere. Conjugation is nevertheless a differentiable real-linear map.

## One differentiable point is not a neighborhood

For |z|², CR holds only at zero. Its quotient there is |h|²/h=conj(h)→0, so the complex derivative really exists and equals zero. Every open neighborhood also contains nonzero points where differentiability fails. Hence the function is not holomorphic near zero.

Holomorphicity means complex differentiability throughout an open set. The strong conclusions of complex analysis require that neighborhood condition, not merely one successful derivative.

## Restricted real Jacobian

At a CR point, the real Jacobian has the multiplication form

$$
J_f=\begin{pmatrix}a&-b\\b&a\end{pmatrix},\qquad f'=a+ib.
$$

When f′≠0 this gives local rotation and uniform scaling to first order. The same angle-preservation explanation cannot be assumed at a zero derivative. A general real two-by-two layer need not have this structure.

## Plot and AI connection

The plot shows images of horizontal and vertical input lines through the probe. Its axes are output real and imaginary parts, not the original input grid. Exact partials and CR predicates, rather than finite drawn curves, support the readout.

Complex networks need not use only holomorphic operations. Nonconstant real-valued objectives cannot be holomorphic on an open domain; optimize them with real or Wirtinger derivatives instead.

## Exercises and answers

1. The derivative of z² at i is **2i**.
2. Conjugation is **real differentiable but not complex differentiable**.
3. A zero derivative for |z|² at zero **does not provide a holomorphic neighborhood**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
