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

## Prove when CR is sufficient

Assume f is differentiable as a real map at p. Its real differential can be written uniquely as the sum of a complex-linear and a conjugate-linear part:

$$
f(p+h)-f(p)=a h+b\bar h+o(|h|),\qquad
a=\tfrac12(f_x-if_y),\quad b=\tfrac12(f_x+if_y).
$$

This follows by substituting dx=(h+conj h)/2 and dy=(h−conj h)/(2i) into fₓdx+fᵧdy. Dividing by h gives a+b·conj(h)/h+o(1). The real and imaginary approaches yield a+b and a−b. Thus b=0 is necessary; if b=0 the remainder divided by h tends to zero for every approach, so it is sufficient too.

Writing f=u+iv, the condition b=0 is exactly uₓ=vᵧ and uᵧ=−vₓ. The coefficient a is the complex derivative. This proof uses real differentiability, not only existence of coordinate partials. It also identifies a and b as the two Wirtinger derivatives introduced later.

## A continuous counterexample with CR at one point

Use a real-valued complex function, with f(0)=0:

$$
f(x+iy)=\begin{cases}\dfrac{x^3y}{x^4+y^2}&(x,y)\ne(0,0),\\0&(x,y)=(0,0).\end{cases}
$$

The estimate |f(x+iy)|≤|x|/2 proves continuity at zero. Both coordinate partials of its real part are zero there, and the imaginary part is identically zero. CR therefore holds at zero. Nevertheless:

| Approach h | Quotient f(h)/h | Limit |
|---|---|---|
| t, real t→0 | 0 | 0 |
| it, real t→0 | 0 | 0 |
| t+it², real t→0 | 1/[2(1+it)] | 1/2 |

Two coordinate tests agree but the complex limit fails along a curved approach. There is no contradiction with the sufficient condition: this function is not real differentiable at zero, as the [total-differentiability example](/book/online/physics-math-for-ai/jacobians/) establishes. Its y partial at (x,0), x≠0, is 1/x, so nearby continuity of partials also fails.

## Check the counterexample in Python

The following standard-library code evaluates exact rational real and imaginary parts of the quotient, including negative t. The finite checks illustrate the formulas; the distinct limits above prove failure of complex differentiability. The interactive lab retains its original three C¹ models rather than displaying this additional function.

```python
from fractions import Fraction as Q

def f(x, y):
    return Q(0) if x == 0 and y == 0 else x**3 * y / (x**4 + y**2)

def quotient(x, y):
    value = f(x, y)
    denominator = x*x + y*y
    assert denominator > 0
    return (value*x/denominator, -value*y/denominator)

for t in [Q(1, 2), Q(1, 8), Q(1, 32), Q(-1, 16)]:
    assert quotient(t, Q(0)) == (0, 0)
    assert quotient(Q(0), t) == (0, 0)
    curved = quotient(t, t*t)
    assert curved == (1/(2*(1+t*t)), -t/(2*(1+t*t)))
    assert abs(f(t, t*t)) <= abs(t)/2
    print(t, curved)
```

Research exercise: before applying a holomorphic-layer argument, identify which regularity hypothesis justifies the first-order real expansion. Testing only real and imaginary perturbations can reject many wrong derivatives but cannot prove complex differentiability for every function.

## Exercises and answers

1. The derivative of z² at i is **2i**.
2. Conjugation is **real differentiable but not complex differentiable**.
3. A zero derivative for |z|² at zero **does not provide a holomorphic neighborhood**.

Further reading: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). The official course scope is linked as supplementary reading; the small calculations and examples here are independently constructed.
