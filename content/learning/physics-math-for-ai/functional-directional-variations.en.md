## A functional accepts a whole function

A usual objective takes a vector and returns a number. A functional can instead accept a curve, field or probability density. To ask how it changes, perturb the entire input function in a specified direction: yε=y+εh. The direction h is itself a function, while ε is one scalar controlling its amplitude.

The lab uses dimensionless t∈[0,1], fixed zero endpoints and the quadratic energy

$$
J[y]=\frac12\int_0^1\bigl((y')^2+\alpha y^2\bigr)\,dt,
\qquad y_\epsilon=y+\epsilon h,\quad\alpha\ge0.
$$

Sine basis functions make the endpoint conditions exact. Blue is the original curve and orange is the perturbed curve, rather than an optimizer trajectory.

## First and second variations

Expand the integrand before integrating. The first variation is linear in h:

$$
\delta J[y;h]=\left.\frac{d}{d\epsilon}J[y+\epsilon h]\right|_{\epsilon=0}
=\int_0^1(y'h'+\alpha yh)\,dt,
\qquad
\delta^2J[y;h,h]=\int_0^1((h')^2+\alpha h^2)\,dt.
$$

A directional, or Gâteaux, derivative considers one direction at a time. A Fréchet derivative additionally controls the approximation uniformly over small perturbations in a specified norm. For this quadratic on the zero-trace Sobolev space H₀¹, the displayed bounded bilinear forms provide that stronger differentiability. One must specify a function space and norm before making such a claim for a general functional.

## Exact two-mode calculation

Let y=a sin πt+b sin 2πt and h=c sin πt+d sin 2πt. Orthogonality of both sine and derivative cosine modes gives A=π²+α and B=4π²+α:

$$
J[y]=(Aa^2+Bb^2)/4,\quad
\delta J=(Aac+Bbd)/2,\quad
\delta^2J=(Ac^2+Bd^2)/2.
$$

The factor one-half in the sine mass integral matters. Treating the two coefficients as an orthonormal basis without rescaling would give a different coefficient differential.

## Finite changes are not derivatives

Here J[y+εh]−J[y]=εδJ+ε²δ²J/2 exactly. For nonzero ε, the finite quotient is δJ+εδ²J/2. The code evaluates this expansion directly to avoid subtracting nearly equal energies. The varied energy is evaluated directly from perturbed coefficients, preserving a zero-energy curve when the perturbation cancels it. At ε=0 the quotient is undefined, while its limit, the first variation, remains defined.

A zero first variation in one direction does not mean a constant functional. It can indicate an orthogonal direction with a nonzero quadratic energy cost. Stationarity requires vanishing first variation for every admissible direction.

## Why the function-space norm changes the theorem

The assertion of Fréchet differentiability can be proved directly. Equip H₀¹(0,1) with the derivative norm ‖v‖V=‖v′‖₂. Since v(t)=∫₀ᵗv′(s)ds, Cauchy–Schwarz gives ‖v‖₂≤‖v′‖₂. Write B(y,h)=∫(y′h′+αyh). Then |B(y,h)|≤(1+α)‖y‖V‖h‖V. Expansion yields the uniform remainder estimate

$$
J[y+h]-J[y]-B(y,h)=\tfrac12 B(h,h),\qquad
\frac{|J[y+h]-J[y]-B(y,h)|}{\|h\|_V}
\le\tfrac12(1+\alpha)\|h\|_V\longrightarrow0.
$$

The bound holds for every small admissible h, including directions that change as h shrinks. This uniformity is what a list of directional derivative checks alone does not establish.

Now keep the same set of functions but measure perturbations only in L2. For hₙ(t)=sin(nπt)/n, the amplitude and L2 norm vanish, yet slope energy does not:

$$
\|h_n\|_2^2=\frac{1}{2n^2},\qquad
\|h_n'\|_2^2=\frac{\pi^2}{2},\qquad
J[h_n]=\frac{\pi^2}{4}+\frac{\alpha}{4n^2}.
$$

Thus J is not even continuous at zero on H₀¹ equipped only with the L2 norm. There is no Fréchet derivative in that topology. A visually small, rapidly oscillating curve can carry large derivative energy. This is also why a low-amplitude residual plot is not an energy-norm certificate for a learned field.

With α=0, the comparison is:

| n | $\|h_n\|_2^2$ | $\|h_n'\|_2^2/\pi^2$ | $J[h_n]/\pi^2$ |
|---|---:|---:|---:|
| 1 | 1/2 | 1/2 | 1/4 |
| 10 | 1/200 | 1/2 | 1/4 |
| 100 | 1/20000 | 1/2 | 1/4 |

## Execute an exact remainder and a high-frequency counterexample

The following standard-library program represents energies as rational multiples of π², with α=0. It verifies the remainder at ε=10⁻²⁰ without subtracting rounded binary64 energies. The final three rows describe a separate high-frequency family; the interactive lab above still uses only its original two modes.

```python
from fractions import Fraction as F

# alpha=0; energies and H1 seminorms are divided by pi**2.
a, h, eps = (F(1), F(1, 3)), (F(1, 2), F(-1, 4)), F(1, 10**20)
energy = lambda v: sum(F(k*k, 4)*x*x for k, x in enumerate(v, 1))
first = sum(F(k*k, 2)*x*z for k, (x, z) in enumerate(zip(a, h), 1))
second = sum(F(k*k, 2)*z*z for k, z in enumerate(h, 1))
changed = tuple(x+eps*z for x, z in zip(a, h))
remainder = energy(changed)-energy(a)-eps*first
assert remainder == eps*eps*second/2
assert remainder > 0
print("exact quadratic remainder / pi^2:", remainder)
for n in (1, 10, 100):
    l2_squared, h1_squared_over_pi2 = F(1, 2*n*n), F(1, 2)
    energy_over_pi2 = F(1, 4)
    print(n, l2_squared, h1_squared_over_pi2, energy_over_pi2)
```

For further study, distinguish an L2 gradient at a sufficiently regular individual y from differentiability of the entire energy under the L2 topology. Integration by parts may produce −y″+αy when it belongs to L2 and boundary terms vanish. That representation does not make the high-frequency counterexample disappear.

## Exercise and explained answer

Choose a=c=1,b=d=α=0,ε=0.1. The first and second variations are both π²/2. Finite energy change is 0.0525π²≈0.518154. Now choose c=0,d=1: first variation is zero, but the finite change is 0.01π²≈0.098696. Explain this using orthogonality, then set ε=0 and distinguish no perturbation from the derivative's value.

Further reading: [MIT, Matrix Calculus for Machine Learning: Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
