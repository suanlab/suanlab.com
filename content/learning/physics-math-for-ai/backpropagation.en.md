## What backpropagation actually computes

A neural network can contain millions of parameters but only one scalar training loss for a batch. Computing a separate finite difference for every parameter would repeatedly evaluate almost the same network. **Backpropagation** reuses the computation graph: evaluate values forward, then propagate loss sensitivities backward through local derivatives.

It is an application of the chain rule, organized to avoid repeated work. It computes gradients; an optimizer subsequently decides how to change parameters. Keeping these roles separate helps diagnose whether a failed experiment comes from an incorrect derivative, a nonsmooth point or an unsuitable update size.

We will use one sample, one hidden neuron and five scalar parameters. The small size makes every intermediate value and every gradient inspectable. It is an educational regression model, not a claim about the capacity of large networks.

## Define the complete model

For fixed input $x=(x_1,x_2)^T$ and target $t$, define

$$
z=w_1x_1+w_2x_2+b,\quad h=\phi(z),\quad
\hat y=vh+c,\quad L=\frac12(\hat y-t)^2.
$$

| Symbol | Role |
| --- | --- |
| $w_1,w_2,b$ | Hidden affine parameters |
| $v,c$ | Output weight and output bias |
| $\phi$ | ReLU by default; tanh is also available |
| $z,h,\hat y$ | Pre-activation, hidden activation and prediction |
| $e=\hat y-t$ | Signed prediction error |
| $\eta>0$ | Learning rate for a later update |

The factor $1/2$ simplifies the loss derivative. The target is fixed, not a trainable parameter. We also calculate input sensitivity, but the update button changes only $\theta=(w_1,w_2,b,v,c)$.

## Forward pass: record values before differentiating

Use $x=(1,2)^T$, $t=0$ and

$$
(w_1,w_2,b,v,c)=(0.5,-0.25,0.5,2,0).
$$

First, $z=0.5(1)-0.25(2)+0.5=0.5$. ReLU gives $h=\max(0,z)=0.5$. The prediction is $\hat y=2(0.5)+0=1$, error is $e=1$, and loss is $L=0.5$.

| Quantity | Reference value |
| --- | --- |
| $z$ | $0.5$ |
| $h$ | $0.5$ |
| $\hat y$ | $1$ |
| $e$ | $1$ |
| $L$ | $0.5$ |

These saved values are the operating points for the local derivatives. Evaluating a derivative at a later updated parameter state would change the calculation.

## Start from a scalar loss

Let $\bar a=\partial L/\partial a$ denote the sensitivity of the final loss to an intermediate variable. Begin with $\bar L=1$. From the squared error,

$$
\bar{\hat y}=\hat y-t=e.
$$

For $\hat y=vh+c$, reverse propagation gives

$$
\bar h=ev,\qquad
\frac{\partial L}{\partial v}=eh,\qquad
\frac{\partial L}{\partial c}=e.
$$

At the reference point these are $\bar h=2$, $\partial L/\partial v=0.5$ and $\partial L/\partial c=1$. The same upstream error is reused with the appropriate local derivative for each incoming argument.

## Pass the activation gate

For ReLU away from zero,

$$
\phi'(z)=\begin{cases}1&z>0\\0&z<0.\end{cases}
$$

Since the reference $z=0.5$ is positive, the gate is open and $\bar z=\bar h\phi'(z)=2$. A negative pre-activation closes the gate and makes this path’s first-order sensitivity zero.

At $z=0$, ReLU has no classical derivative. **This lab explicitly chooses the computational convention $\phi'(0)=0$.** That choice must not be presented as a uniquely defined classical derivative or as a result that every numerical difference must reproduce.

Switching to $\tanh$ uses $\phi'(z)=1-h^2$. This activation is smooth, making it useful for a contrasting gradient check. Smoothness alone does not prevent small gradients: tanh saturates for large absolute pre-activations.

## Finish the parameter and input gradients

The affine hidden stage yields

$$
\frac{\partial L}{\partial w_1}=\bar z x_1,\quad
\frac{\partial L}{\partial w_2}=\bar z x_2,\quad
\frac{\partial L}{\partial b}=\bar z.
$$

Thus the full parameter gradient in the declared order is

$$
\nabla_\theta L=(2,4,2,0.5,1)^T.
$$

The input gradient is $(\bar z w_1,\bar z w_2)^T=(1,-0.5)^T$. It answers a different question from the parameter gradient: how the loss responds to changing the input while holding parameters fixed.

For graphs with shared variables, contributions from every downstream path must be added. This tiny chain has no complicated shared parameter, but the accumulation principle from the Jacobian unit remains essential in larger architectures.

## One simultaneous learning step

Gradient descent updates all five parameters from the **same old state**:

$$
\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta L.
$$

At $\eta=0.01$, the new parameters are $(0.48,-0.29,0.48,1.995,-0.01)$. Re-evaluating the complete forward model gives

$$
z_{\mathrm{new}}=0.38,\quad \hat y_{\mathrm{new}}=0.7481,
\quad L_{\mathrm{new}}=0.279826805.
$$

This is a new forward pass, not an estimate obtained by subtracting the loss gradient from the old loss. Do not update one weight and then use that changed weight while computing the remaining gradients: that is a different algorithm.

The lab shows the previous and new loss after a manual update. It permits at most 50 updates and keeps persisted parameters within $[-100,100]$. A candidate outside that range is displayed but not applied; values are not silently clipped. These limits keep an educational control usable and are not assumptions of the mathematical algorithm.

## A correct gradient can still increase loss

Load the large-rate example, which starts from the same reference parameters but uses $\eta=2$. The simultaneous update gives $(-3.5,-8.25,-3.5,1,-2)$. Now $z=-23.5$, ReLU outputs zero, the prediction is $-2$ and the loss is 2, larger than the original 0.5.

A negative gradient is a local descent direction at a differentiable nonstationary point. That statement does not guarantee improvement for an arbitrarily large finite step. Crossing activation boundaries and changing several interacting parameters can invalidate the local approximation. Inspect update size and the new forward loss before declaring the derivative wrong.

## A closed gate does not stop every parameter

Load the inactive-ReLU example: set hidden bias $b=-1$ and target $t=1$, retaining the other reference parameters. Then $z=-1$, $h=0$, prediction is 0 and loss is 0.5.

The gradients of $w_1,w_2,b$ vanish because the activation gate is closed. The gradient of $v$ is also zero because $h=0$. But the output-bias gradient is $\partial L/\partial c=-1$. A parameter after the closed gate can still learn. Avoid the blanket claim that an inactive hidden unit makes every network gradient zero.

## Compare against finite differences carefully

For each parameter $\theta_i$, the lab recomputes two losses while holding every other parameter fixed:

$$
g_i^{\mathrm{FD}}=
\frac{L(\theta+\varepsilon e_i)-L(\theta-\varepsilon e_i)}{2\varepsilon}.
$$

The table compares this estimate with backpropagation and reports absolute difference. On sufficiently smooth intervals, central differences provide an independent check, subject to truncation and floating-point error. The smallest possible epsilon is not necessarily the best choice.

The table also marks whether the sampled ReLU interval stays within one branch. At the kink example, $b=0,t=1$ gives $z=0$. The chosen backward convention returns zero for the hidden weight gradients, but the central difference for $w_1$ is approximately $-1$, not zero. The two sides cross the kink; there is no classical derivative for them to agree on. The warning identifies that situation rather than labeling a large numerical difference as a universal backpropagation failure.

## Automatic differentiation, numerical differentiation and memory

Backpropagation composes derivatives of elementary operations evaluated at recorded values. It is not finite differencing the full network, and it is not expanding one enormous symbolic expression. Reverse-mode automatic differentiation implements this pattern for a supported computation graph.

A reverse pass needs appropriate forward information. Implementations may save intermediate values or recompute some of them, trading memory for additional computation. For a scalar objective with many parameters, reverse mode is often more efficient than separately probing every parameter. That does not mean the full Jacobian of an arbitrary many-output map is obtained for free.

The lab uses explicit JavaScript formulas so every number is auditable. It does not load PyTorch in the browser or simulate a general automatic-differentiation engine. For framework behavior, consult the [official PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), including its discussion of saved tensors and nondifferentiable functions.

## Practice before moving on

1. Recompute all five reference gradients without looking at the readout. Explain which saved forward value each derivative needs.
2. Apply a reference update at rate 0.01. Verify $z=0.38$ and loss $0.279826805$ with a fresh forward pass.
3. Explain why only the output bias has a nonzero parameter gradient in the inactive-ReLU example.
4. Switch to tanh and compare several epsilons. Explain why a numerical mismatch alone does not identify its cause.
5. Imagine that one hidden value feeds two output terms. Draw both reverse paths and state where their contributions must be added.

You now have the computational foundation for studying optimization, shared computation and differentiable scientific models. The next linear-algebra topic is singular value decomposition: it generalizes local stretching to rectangular maps and prepares us for low-rank models and conditioning.
