## Universal approximation is an existence statement

A typical result concerns a continuous target on a compact domain: with appropriate activation, biases and sufficient width, a finite network can achieve any prescribed uniform accuracy. Continuous nonpolynomial activations such as ReLU support standard results when their other hypotheses hold.

$$
\forall f\in C(K),\ \forall\varepsilon>0,\ \exists g_\theta:\|f-g_\theta\|_\infty<\varepsilon.
$$

One fixed width is not promised to work for every target and tolerance. Neither practical size nor a successful training algorithm follows merely from existence.

## Assemble a ReLU approximation

Split [−1,1] into n intervals and interpolate x². Initial value is one, initial slope s₀=−2+2/n, and each internal knot tⱼ=−1+2j/n increases slope by 4/n.

$$
g_n(x)=1+s_0(x+1)+\sum_{j=1}^{n-1}\frac4n\max(0,x-t_j).
$$

This uses an affine skip plus n−1 ReLU hinges. Four intervals require three hinges and give error 1/16; eight give 1/64. The interval length is 2/n and quadratic interpolation error is h²/4. This construction illustrates one target, not the entire general theorem.

## Why linear activations fail

Sums and compositions of affine maps remain affine. The best uniform affine approximation of x² on [−1,1] is constant 1/2:

$$
\inf_{a,b}\sup_{|x|\le1}|x^2-(a+bx)|=1/2.
$$

Symmetry of the endpoints and the center establishes the lower bound. The lab's linear mode shows that best baseline. The n=1 ReLU interpolation construction instead gives constant one and error one. There is no contradiction: a specified construction need not be optimal in its family.

## Separate research questions

Representability, compact representation, finite-sample generalization and efficient optimization are distinct. Depth-efficiency claims also need a target class and cost model. A small successful width experiment cannot settle all of them.

## Exercises with explanations

1. Does this finite ReLU spline equal x² everywhere? **No.** It is piecewise linear and approaches it as width grows.
2. Can biases be removed without rechecking assumptions? **No.** Even control of knot locations changes.
3. How large n ensures error below ε here? **Choose n>1/√ε**, specifically for this construction and target.

Further reading: [Leshno et al., Nonpolynomial Activation and Approximation](https://doi.org/10.1016/S0893-6080(05)80131-5).
