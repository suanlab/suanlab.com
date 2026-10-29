## Does knowing the slope take you straight to the answer?

Learning in AI is often expressed as changing parameters to reduce an objective. Which direction should we take, and how far? This experiment separates **gradient direction from learning-rate magnitude**.

$$
L(x,y)=\tfrac12(ax^2+by^2),\qquad a>0,\ b>0.
$$

The parameters x and y and the objective L are dimensionless. This positive quadratic has its minimum at the origin. It is not a reproduction of an entire neural network's complex loss landscape.

## What you will gain

Begin with small finite changes, even if derivative notation is unfamiliar. Define loss first, then move from changing one variable to partial derivatives and the gradient.

- Distinguish current parameters, loss and gradient.
- Calculate one update by hand and check whether the actual loss decreases.
- Explain how learning rate and curvature jointly affect convergence.

| Symbol | Meaning | Reference example |
| --- | --- | --- |
| $\mathbf p=(x,y)$ | Two adjustable parameters | Start at `(2,2)` |
| $L(\mathbf p)$ | Loss at the current parameters | Initial loss 6 |
| $a,b$ | Positive constants controlling curvature along the axes | 2 and 1 |
| $\nabla L$ | Vector of partial derivatives | Initially `(4,2)` |
| $\eta$ | Learning rate controlling update size | Default 0.25 |
| $k$ | Update count | $k=0$ is the start |

Here x and y are **parameters the algorithm changes**, rather than two features in the training data. The constants a and b define the experimental loss surface. Changing them compares different optimization problems.

## What does a loss measure?

Learning can express prediction error, or disagreement with a chosen objective, as one number. Parameters are updated to reduce that number. Our quadratic isolates this principle in a small example.

Consider a separate one-dimensional example. For an input of 1, a model predicts w and the target is 3:

$$
L(w)=\tfrac12(w-3)^2.
$$

At w=1, loss is 2 and the derivative is $w-3=-2$. With learning rate 0.25, $w_{\rm next}=1-0.25(-2)=1.5$ and loss becomes 1.125. **With a negative slope, increasing the parameter can reduce loss.** “Descent” does not mean decreasing every coordinate.

This example minimizes at w=3, while the 3D experiment minimizes at the origin. Zero is not intrinsically special: the objectives are defined differently.

## A partial derivative asks about one direction

At `(2,2)`, hold y at 2 and increase x by 0.1. Loss rises from 6 to 6.41. The estimated increase per unit change in x is

$$
\frac{L(2.1,2)-L(2,2)}{0.1}=4.1.
$$

With a change of 0.01, the estimate is 4.01. The limit as the change approaches zero is the partial derivative with respect to x: here, 4. Asking the same question for y gives 2. Collecting these answers in order gives gradient `(4,2)`.

A partial derivative holds the other variables fixed. Changing x and y together, then dividing by only the change in x, answers a different question. Record which variable changes when checking derivatives with finite differences.

## From derivative to update

The derivative of x² with respect to x is 2x. For the partial derivative with respect to x, hold y fixed: the term containing only y contributes zero. The symbol ∂ denotes a partial derivative.

$$
\frac{\partial L}{\partial x}=\tfrac12 a(2x)+0=ax,\qquad
\frac{\partial L}{\partial y}=0+\tfrac12 b(2y)=by.
$$

The factor 1/2 cancels the 2 from differentiating the square. Multiplying an objective by a positive constant preserves the minimum's location, but can change gradient magnitude and update size at the same learning rate.

Differentiate with respect to each parameter:

$$
\nabla L(x,y)=(ax,by).
$$

Under the Euclidean metric, the gradient points in the direction of locally steepest increase. Gradient descent updates in the opposite direction:

$$
\mathbf p_{k+1}=\mathbf p_k-\eta\nabla L(\mathbf p_k),\qquad \eta>0.
$$

Here, “steepest” compares very small movements of equal length. In this unit, length is the square root of the sum of squared coordinate changes: the Euclidean distance.

Here η is the learning rate. The gradient is not a velocity or a force pulling a physical object to the origin. It is the derivative evaluated at the current parameters.

## Calculate the first step

Set `a=2`, `b=1`, start at `(2,2)`, and use `η=0.25`.

$$
\nabla L(2,2)=(4,2),\qquad \mathbf p_1=(2,2)-0.25(4,2)=(1,1.5).
$$

Check the objective as well:

$$
L(2,2)=6,\qquad L(1,1.5)=2.125.
$$

Take one step in the experiment and compare the coordinates, gradient, and objective with these numbers. Surface height represents the objective; changing the visual height scale does not change the calculation.

## The second step and the loss history

Separate the calculation to make implementation errors easier to locate:

1. Evaluate gradient `(4,2)` at the current point.
2. Multiply by the rate to obtain update vector `(1,0.5)`.
3. Subtract it from the current point to reach `(1,1.5)`.
4. **Evaluate the objective at the new point** to obtain loss 2.125.

Do not reuse the initial gradient at the next step. The new gradient is `(2,1.5)`, giving next coordinates `(0.5,1.125)`.

| Step k | x | y | Loss L |
| --- | --- | --- | --- |
| 0 | 2 | 2 | 6 |
| 1 | 1 | 1.5 | 2.125 |
| 2 | 0.5 | 1.125 | 0.8828125 |
| 3 | 0.25 | 0.84375 | 0.41845703125 |

The 3D point represents one row of this table; the connected path records the update history. The horizontal axis of the loss chart counts iterations k, not seconds. Changing the rate changes the point reached at the same k.

### Does the negative gradient point straight at the origin?

At the start, the direction to the origin is `(-2,-2)`, while the negative gradient is `(-4,-2)`. These are different directions. Loss has greater curvature along x, so changing more along that direction produces the locally steepest decrease. The gradient does not know the shortest route to the minimum in advance.

## Why is there a learning-rate boundary?

Each component updates independently:

$$
x_{k+1}=(1-\eta a)x_k,\qquad y_{k+1}=(1-\eta b)y_k.
$$

For both components to shrink to zero from every starting point, both multipliers must have magnitude less than one. Therefore,

$$
0<\eta<\frac{2}{\max(a,b)}.
$$

The reference range is `0<η<1`. **This condition applies to this positive quadratic.** It cannot be used unchanged for arbitrary nonlinear functions or real neural-network training.

## Understand the rate through multipliers

In the reference example, x is multiplied by $1-2\eta$ and y by $1-\eta$. Ask what repeated multiplication does to a sequence:

| Multiplier q | Repeated behavior | Example |
| --- | --- | --- |
| $0<q<1$ | Same sign, approaching zero | `2, 1, 0.5, …` |
| $-1<q<0$ | Alternating sign, approaching zero | `2, -1, 0.5, …` |
| $q=-1$ | Alternation without shrinkage | `2, -2, 2, …` |
| $|q|>1$ | Growing magnitude from a nonzero start | `2, -3, 4.5, …` |

For q=0, that component reaches zero in one step. For q=1, it stays unchanged. Both multipliers must have magnitude below one for both components to approach zero from every starting point. Greater curvature a or b reduces the upper bound on the rate.

This is an exact result for a particular function. Curvature or loss reduction observed at one point of a real neural network does not establish convergence for its entire training process.

## Compare convergence, oscillation, and growth

- **η=0.25:** Both components shrink toward the origin.
- **η=0.75:** The sign of x alternates while its magnitude shrinks. Zigzagging need not mean divergence.
- **η=1:** x alternates `2,-2,2,…` and y becomes zero. After the first step, the objective stays at 4.
- **η=1.25:** The first update is `(-3,-0.5)` and the objective increases to 9.125.

Compare presets from the same starting point. Starting at the origin keeps the point there for any positive learning rate. Success from one initial point does not establish convergence from every point.

## Read the scene

The two floor axes are parameters x and y; height is L. The connected dots show the update history. Compare the surface's slope with a top-down view of the path.

The scene scale may adapt to a larger displayed range, but parameter and objective values in the table stay unchanged. If a display limit is exceeded, automatic updates stop rather than clipping the mathematical result to suggest convergence.

## One step deeper: why can a downhill direction increase loss?

For a small change, the objective change is approximated by

$$
\Delta L\approx\nabla L^T\Delta\mathbf p.
$$

Substituting $\Delta\mathbf p=-\eta\nabla L$ gives $-\eta\|\nabla L\|^2$. At a point with nonzero gradient, this predicts decrease for a sufficiently small step. It is not an exact prediction for arbitrarily large steps.

For this quadratic, the curvature contribution can be written exactly. Define $H=\operatorname{diag}(a,b)$ as the matrix of second partial derivatives, called the Hessian:

$$
\Delta L=-\eta\|\mathbf g\|^2+\tfrac12\eta^2\mathbf g^TH\mathbf g,\qquad \mathbf g=\nabla L.
$$

At the reference start, $\|\mathbf g\|^2=20$ and $\mathbf g^TH\mathbf g=36$, so $\Delta L=-20\eta+18\eta^2$. For rate 0.25, this is −3.875; for 1.25, it is +3.125. The decrease term scales with η, while the curvature penalty scales with η². Their competition explains why an excessive rate can increase loss.

If Hessians are new to you, first understand the roles of the two terms. Multivariable calculus and optimization will later connect them to quadratic approximations and eigenvalues.

## Connecting AI and physics carefully

In a neural network, backpropagation computes gradients for many parameters, and an optimizer uses them to update the model. Sampling noise, regularization, constraints, and nonconvexity add further conditions.

A physical ball rolling in a bowl needs mass, velocity, inertia, force, and a law of motion. This point is **the algorithm's current parameters**, not such a particle. Distinguishing similar pictures from identical laws also matters in research.

## Common misconceptions and research checks

| Thought | Counterexample or check |
| --- | --- |
| Descent decreases every coordinate | A negative gradient component can make a coordinate increase |
| The gradient points straight to the minimum | Negative gradient `(-4,-2)` is not parallel to displacement `(-2,-2)` toward the origin |
| Alternation means failure | Rate 0.75 alternates signs and still converges |
| One decrease proves convergence | Rate 1 reduces loss from 6 to 4, then keeps oscillating |
| Small loss means the real task is solved | Check the objective definition and evaluation data separately |

When reading research code, record initial values, learning rate, sum-versus-mean loss conventions, and where the gradient is recomputed. Multiplying loss by a constant c multiplies its gradient by c, changing update size at the same rate. A rate copied as a single number can therefore behave differently in another implementation.

## Challenge and check

Bring the reference objective below 0.05. Record the learning rate and step count, then explain a case that converges while changing sign.

**Explanation:** `η=0.25` reaches the target after several steps. `η=0.75` also converges for this function. Check the component multipliers and actual objective, rather than assuming that a visually small step must work.

Learning reference: [Boyd and Vandenberghe — Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).
