## Why two gradients can chase each other

In ordinary minimization, moving against a gradient aims to reduce one objective. In a game, one participant's move changes the gradient faced by another. Even an extremely simple objective can produce rotation rather than a direct approach to equilibrium.

Use the unconstrained two-variable game

$$
\min_{x\in\mathbb R}\max_{y\in\mathbb R}
 f(x,y)=\frac{\mu}{2}(x^2-y^2)+a xy,\qquad \mu\ge0.
$$

X minimizes and y maximizes. The coupling a controls the direction and strength of interaction; mu supplies convexity in x and concavity in y. When mu>0 or a is nonzero, the unique global saddle is (0,0). If both are0, the objective is constant and EVERY point is a saddle. Measuring error only from a chosen origin would be wrong in that flat case.

The lab uses exact deterministic derivatives and floating trajectories. It has no constraints, projection, clipping, stochastic gradients or momentum. A statement about these updates is not a general guarantee for neural-network training.

## Turn the two objectives into one field

For a descent/ascent update, use the signed field

$$
F(z)=\begin{pmatrix}f_x\\-f_y\end{pmatrix}
=\begin{pmatrix}\mu x+ay\\\mu y-ax\end{pmatrix}
=Az,\qquad A=\begin{pmatrix}\mu&a\\-a&\mu\end{pmatrix},
\quad z=\begin{pmatrix}x\\y\end{pmatrix}.
$$

Both coordinates then follow z'=−F(z) in the continuous comparison. The symmetric part of A is mu I; its skew part rotates. Taking an inner product gives

$$
\frac{d}{ds}\|z(s)\|^2=-2\mu\|z(s)\|^2,
\qquad z(s)=e^{-\mu s}
\begin{pmatrix}\cos(as)&-\sin(as)\\\sin(as)&\cos(as)\end{pmatrix}z_0.
$$

Thus pure coupling, mu0 and a nonzero, produces circles of constant radius. Positive mu gives an inward spiral or straight contraction when a0. This continuous ODE is a mathematical reference, not a physical law for optimization time. The orange phase path samples it at s=t h; the straight segments between samples do not assert that the ODE moves along those chords.

## Four ways to discretize the same game

Write h for the step size, d=1−h mu and q=h a. Each method produces z_{t+1}=Mz_t, but with a different M.

Simultaneous descent/ascent uses both old coordinates:

$$
x^+=x-h(\mu x+ay),\qquad y^+=y+h(ax-\mu y),
\qquad M_{\rm sim}=\begin{pmatrix}d&-q\\q&d\end{pmatrix}.
$$

X-first alternating updates x first, then uses the NEW x in y's derivative while retaining the old y in its damping term:

$$
x^+=dx-qy,\qquad y^+=dy+q x^+,
\qquad M_{\rm alt}=\begin{pmatrix}d&-q\\qd&d-q^2\end{pmatrix}.
$$

Changing the update order changes the algorithm. It is not merely a different drawing of simultaneous updates.

Extragradient first predicts z_{half}=z−hF(z), evaluates the field there, then corrects from the ORIGINAL z:

$$
z^+=z-hF(z_{half}),\qquad M_{\rm extra}=I-hA+h^2A^2.
$$

Two ordinary gradient steps, the second starting from the predictor, would instead give (I−hA)^2. The green square in the phase diagram is the predictor; the dark update line starts at the original state, not at that square.

Implicit updates solve z^+=z−hF(z^+) together. In this linear example the solution is explicit:

$$
M_{\rm impl}=(I+hA)^{-1}
=\frac{1}{(1+h\mu)^2+q^2}
\begin{pmatrix}1+h\mu&-q\\q&1+h\mu\end{pmatrix}.
$$

This is an exact linear solve, not one alternating step pretending to solve a coupled system. A nonlinear implicit method would require its own solver and accuracy analysis.

## Hand calculation: a half step changes everything

Take a1, mu0, h1/2 and z0=(1,0). Simultaneous and x-first alternating both first reach (1,1/2), although later trajectories differ. Its squared norm is5/4, greater than the initial1. The continuous reference still has norm1.

Extragradient has the same predictor (1,1/2), but F at the predictor is (1/2,−1). Correcting the original state gives (3/4,1/2), with squared norm13/16. Starting a second step at the predictor would give a different point and would miss this calculation.

Implicit solves x^+=1−y^+/2 and y^+=x^+/2. Therefore x^+=4/5, y^+=2/5, squared norm4/5. These four rational results make a useful implementation check independent of a plot.

For the pure game, simultaneous, extragradient and implicit have constant squared-norm ratios:

$$
R_{\rm sim}=1+q^2,\qquad
R_{\rm extra}=1-q^2+q^4,\qquad
R_{\rm impl}=\frac{1}{1+q^2}.
$$

Extragradient contracts only for 0<|q|<1. At |q|1 it rotates without contraction, and beyond1 it expands. Implicit contracts for any nonzero q in this toy. These are different stability regions, not a ranking that holds for every optimization problem.

## Alternating motion preserves a tilted metric

For mu0, the alternating matrix has determinant1. When 0<|q|<2, its eigenvalues have modulus1 and its powers are bounded, but nonzero iterates do not approach the origin. A more informative quantity than the Euclidean norm alone is

$$
H(x,y)=x^2+y^2-qxy,\qquad
M_{\rm alt}^{\mathsf T}
\begin{pmatrix}1&-q/2\\-q/2&1\end{pmatrix}
M_{\rm alt}=\begin{pmatrix}1&-q/2\\-q/2&1\end{pmatrix}.
$$

The metric is positive definite for |q|<2 and bounds the squared Euclidean norm between factors1−|q|/2 and1+|q|/2. It explains a bounded ellipse even when an individual step increases the ordinary norm. The lab exposes this positive-definite invariant only in its valid pure-game range. An algebraically preserved indefinite quadratic outside that range would not prove stability and can suffer severe floating cancellation on expanding paths.

At mu>0, simultaneous contraction requires

$$
(1-h\mu)^2+(ha)^2<1
\quad\Longleftrightarrow\quad
0<h<\frac{2\mu}{\mu^2+a^2}.
$$

For alternating updates in this model, strict stability requires mu>0, h>0 and h(mu+|a|)<2. Damping helps, but it does not make every large discrete step safe.

## Last iterate and average answer different questions

The second preset uses alternating updates with a1, mu0, h1. Starting at (1,0), the states are

$$
(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1),(1,0),\ldots
$$

There is a six-cycle. After T6 updates the terminal state is again (1,0), with residual1. Yet the mean of states0 through5 is exactly (0,0). Averages can cancel the rotation without the last iterate converging.

The convention is explicit:

$$
\bar z_T=\frac1T\sum_{t=0}^{T-1}z_t\quad(T>0).
$$

The terminal state z_T is excluded. At T0 the average and update query are unavailable; the lab does not invent a zero mean. For bounded alternating powers with nonzero q, a telescoping sum gives an O(1/T) bound for the mean norm. An unconditional positive lower bound of that order would be false: complete periods in this example average to exactly0.

The mean is a summary of iterates, not the same policy evaluation or rerun comparator used in the repeated-game unit. In this linear model F(mean) equals the mean field, but that relation does not extend automatically to nonlinear fields.

## Radius one does not settle boundedness

For any finite matrix, spectral radius below1 implies powers tend to0. At radius1 one must inspect the eigenstructure. With alternating a1, mu0, h2,

$$
M=\begin{pmatrix}1&-2\\2&-3\end{pmatrix}
$$

has a repeated eigenvalue−1 and a nontrivial Jordan block. From (1,0), the first states are (1,2),(−3,−4),(5,6),(−7,−8), which grow. From the exceptional initial state (1,1), however, the orbit alternates between (1,1) and (−1,−1). The all-initial bounded-powers property is false even though this particular orbit is bounded.

The stability panel therefore separates strict stability, bounded powers, a unit-modulus Jordan block and all-initial convergence to the saddle set. The displayed spectral radius is approximate. The classifications use exact integer/rational coefficients, including the real two-by-two Schur inequalities for trace tau and determinant delta:

$$
1-\tau+\delta>0,\qquad1+\tau+\delta>0,\qquad1-\delta>0.
$$

A finite numerical trace alone is not a proof of an asymptotic property.

## A zero step is not an equilibrium certificate

With h0 every update stays at its initial state even when F is nonzero. A subtler example uses a0, mu1, h1 and extragradient. The predictor from (1,0) is0, so its field is0. But the corrector starts from the ORIGINAL (1,0), leaving it unchanged. The extragradient matrix is the identity, update size0, residual1.

Use the actual stationary residual

$$
\|F(z)\|=\sqrt{\mu^2+a^2}\,\|z\|
$$

and distance to the saddle SET. In the flat a=mu0 game that set is all of R² and set distance is0, regardless of origin distance. Global matrix instability also does not prevent the origin itself from staying at its saddle.

The objective value is not a substitute: in the pure game f(1,0)=0 equals the saddle value, but the point is not a saddle and has nonzero field. The unrestricted global saddle gap is infinite at nonzero pure-game points, so the lab does not relabel a finite objective difference as that gap.

## Read the plots without changing the problem

The phase diagram includes all displayed states, the mean and the extragradient predictor on one equal-coordinate scale. It never clips or renormalizes an expanding trajectory. Consequently earlier points can appear compressed near the origin when later states are huge. Change the horizon to inspect the early geometry.

The lower curves show log10(1+stationary residual): blue for the selected state sequence, orange for the continuous reference at t h, green for the prefix mean. This transformation makes growth readable without changing the iterates. Raw residuals remain in the readout/table. Very small transformed values use log1p internally; raw residuals are not reconstructed by subtracting1 from a rounded logarithm.

The queried transition clamps to T−1 and changes only inspection. At T0 no transition exists. Small numbers and large phase-axis labels use scientific notation where needed, while the stored dynamics remain unscaled.

## Exercises and worked solutions

1. For simultaneous a1, mu0, h1/2, what is squared norm after eight updates from (1,0)? It is (5/4)^8=390625/65536, approximately5.96046. The norm is its square root, not that ratio itself. For extragradient it is (13/16)^8 instead.
2. With a1, mu0, h1, does extragradient converge? Its matrix is a quarter-turn, so (1,0) visits (0,1),(−1,0),(0,−1), then returns. Its squared-norm factor is1. A stable-looking finite plot or zero average does not prove last-iterate convergence.
3. For a0, mu1, h1, compare simultaneous and extragradient from (1,0). Simultaneous reaches0 in one update. Extragradient predicts0 but corrects from the old point, staying at(1,0). Checking the residual distinguishes the genuine solution from stagnation.
4. Can one bounded trajectory refute a unit Jordan block's unbounded powers? No. The special eigenline at alternating h2 gives a two-cycle, but another initial condition grows. A matrix property quantifies over all initial states.

## Connection to research

Coupled generator/discriminator or actor/critic parameters motivate examining the game field, its symmetric/skew components and the discretization, rather than monitoring one scalar loss alone. A practical study should specify update order, evaluation points, gradient noise, constraints, averaging convention, computational cost and the exact convergence quantity measured.

The standard toy-method context is [Gidel et al., A Variational Inequality Perspective on GANs, §3 and Appendix B](https://arxiv.org/pdf/1802.10551). This unit's damping extension, exact stability classification, boundary fixtures and numerical safeguards are derived and checked for the explicit quadratic model above. They do not establish a theorem for nonlinear GANs or a universal benefit of an optimizer.
