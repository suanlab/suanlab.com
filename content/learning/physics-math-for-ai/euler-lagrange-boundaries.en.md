## Interior equations come with boundary assumptions

Consider minimizing J[y]=∫₀¹[(y′)²/2−qy]dt for a constant load q≥0. The derivative term penalizes large slopes, while the load rewards larger y. The admissible endpoint conditions determine which curves and which variations are allowed.

The lab compares zero values at both endpoints with a fixed left endpoint and a free right endpoint. These are different optimization problems, even though their interior integrand is identical.

## Derive the Euler equation

For a smooth candidate and direction h, integrate by parts:

$$
\delta J[y;h]=\int_0^1(y'h'-qh)\,dt
=[y'h]_0^1+\int_0^1(-y''-q)h\,dt.
$$

Directions supported strictly inside the interval force −y″−q=0 at a stationary curve. For weak Sobolev solutions the same statement begins with an integral identity; pointwise differentiation requires sufficient regularity. This polynomial example is smooth, so both descriptions agree.

If both endpoint values are fixed, h(0)=h(1)=0 and the boundary term vanishes automatically. If only the left endpoint is fixed, h(1) is arbitrary. Stationarity then also requires the natural condition y′(1)=0. A free endpoint does not mean that its variation vanishes.

## Solve the two boundary-value problems

With fixed endpoints the unique minimizing curve is y*=(q/2)t(1−t), with J*=−q²/24. With a free right endpoint it is y*=q(t−t²/2), with J*=−q²/6. The latter has nonzero endpoint value but zero endpoint slope.

For q=2 these become t(1−t) and 2t−t². Their minima are −1/6 and −2/3. Removing a constraint enlarges the feasible set, so the lower free-end minimum is consistent with optimization geometry.

## Stationarity versus minimality

The Euler equation is generally a stationarity condition, not a universal global-minimum theorem. Here convexity supplies the stronger conclusion. For an admissible h=y−y*, stationarity removes the linear term and

$$
J[y]-J[y^*]=\frac12\int_0^1(h')^2\,dt\ge0.
$$

Because the left endpoint is fixed, a nonzero constant difference is not admissible; equality therefore identifies the unique minimizer. Without appropriate boundary or function-space assumptions, uniqueness and even existence require further work.

## Exercise and explained answer

The candidate is a polynomial basis curve plus b sin πt. In free-end mode use q=a=2 and b=0.2. Its right-end slope is −0.2π≈−0.628319. Its interior residual is π²·0.2 sin πt, so the candidate violates both interior stationarity and the natural condition. Its objective excess is 0.01π²≈0.098696.

Set b=0 and verify both residuals vanish. Then switch boundary mode and update a to q/2. Explain why reusing the same interior differential equation without changing the boundary conditions would solve the wrong problem. This distinction carries directly into variational PDE models and physics-informed learning.

Further reading: [MIT, Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
