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

## Exercise and explained answer

Choose a=c=1,b=d=α=0,ε=0.1. The first and second variations are both π²/2. Finite energy change is 0.0525π²≈0.518154. Now choose c=0,d=1: first variation is zero, but the finite change is 0.01π²≈0.098696. Explain this using orthogonality, then set ε=0 and distinguish no perturbation from the derivative's value.

Further reading: [MIT, Matrix Calculus for Machine Learning: Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
