## Ask how delta acts, not how tall it is

A distribution is a continuous linear functional on C_c^∞. Test-function convergence is stronger than pointwise convergence: supports stay in a common compact set and derivatives of every order converge uniformly. Dirac delta returns one test value.

$$
\delta_a[\varphi]=\varphi(a),\qquad T_f[\varphi]=\int f\varphi.
$$

Locally integrable functions define distributions through the second formula, but not every distribution is an ordinary function. Defining delta merely as an infinite point height obscures its calculus.

## Approximate with unit-area triangles

Use ηε(x)=ε⁻¹ max(1−|x−a|/ε,0). Support length is 2ε, height 1/ε and area one. For continuous tests,

$$
\int\eta_\varepsilon(x)\varphi(x)\,dx\longrightarrow\delta_a[\varphi].
$$

The test equals xᵈ on [−2,2] and has a smooth compactly supported extension outside. All triangle supports lie inside this interval, so the undrawn extension does not affect pairings. A polynomial alone is not compactly supported.

## Compute actions exactly

The centered triangular variance is ε²/6. For a quadratic test, the approximate action is a²+ε²/6. At a=.5, ε=.5, exact delta action is .25, triangle action 7/24 and error 1/24. Width .1 reduces error to 1/600.

Blue is a density and orange a local test function; they are not two approximations of the same quantity.

## Differentiate through the test

Transfer differentiation with a minus sign:

$$
T'[\varphi]=-T[\varphi'],\qquad \delta'_a[\varphi]=-\varphi'(a).
$$

For φ=x, the answer is −1. For x² it is −2a, also obtained by the triangle derivative. For x³, delta gives −3a² while the approximation gives −3a²−ε²/2. These signs matter in weak PDEs, point loads and Green functions.

## Exercises with explanations

1. Halve the width: what happens to quadratic action error? **It becomes one quarter**, because variance scales as ε².
2. Does growing height imply growing mass? **No.** Width shrinks to preserve area one.
3. Do equal actions on one test prove equal distributions? **No.** Equality requires every test; finite experiments do not prove general distributional convergence.

Further reading: [MIT 18.155, Test Functions and Distributions](https://ocw.mit.edu/courses/18-155-differential-analysis-fall-2004/pages/lecture-notes/).
