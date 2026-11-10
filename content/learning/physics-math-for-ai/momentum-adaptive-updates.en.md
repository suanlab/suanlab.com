## Fix the algorithm convention before comparing

Momentum and adaptive methods have several variants whose parameter names look similar. This lab explicitly compares heavy ball and the original bias-corrected Adam update on f(x)=(x₁²+κx₂²)/2, starting from (1,1). Blue is ordinary GD with the same learning rate and iteration count. Orange is the selected method, so every comparison uses the same objective and initial vector.

The example is a finite deterministic quadratic. It does not test generalization, noisy training, decoupled weight decay, asynchronous updates or a universal convergence theorem.

## Heavy ball stores velocity

With zero initial velocity the implemented convention is

$$
v_k=\beta v_{k-1}+g_k,\qquad x_{k+1}=x_k-\eta v_k.
$$

There is no factor 1−β multiplying the gradient. Other conventions can absorb that factor into the learning rate, changing the meaning of the same numerical η. Setting β=0 exactly recovers GD.

In a scalar curvature direction λ, eliminating velocity gives characteristic polynomial r²−(1+β−ηλ)r+β. For 0≤β<1 its roots are inside the unit circle when 0<ηλ<2(1+β). The lab checks this condition for both quadratic modes. It is a stability condition for this linear recurrence, not a statement that every iteration reduces the objective. Complex roots and overshoot can create oscillation.

## Adam stores two exponential moments

Adam instead uses

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
\qquad
x_{k+1}=x_k-\eta\frac{m_k/(1-\beta_1^k)}{\sqrt{v_k/(1-\beta_2^k)}+\epsilon}.
$$

Squares, square roots and division are coordinatewise. The experiment fixes ε=0.001 outside the square root. Moving it inside the square root changes the algorithm. Beta2 is unused in heavy-ball mode.

At the first step g=(1,κ). Bias correction makes the first moment equal g and the second equal g², regardless of the allowed beta values. For κ=8 and η=0.1, the new vector is approximately (0.900100,0.900012). Ordinary GD instead gives (0.9,0.2). Similar initial Adam displacements arise from normalization; they are not evidence that the curvatures are equal.

## What adaptive scaling does not establish

The squared-gradient memory is not the Hessian and contains no off-diagonal curvature terms. Coordinate normalization is not invariant under an arbitrary rotation of parameters. A favorable finite curve does not establish convergence on all convex or nonconvex problems. Convergence arguments depend on the precise variant, step schedules, stochastic assumptions and regularity conditions.

## Exercise and explained answer

Select heavy ball with κ=8,η=0.1,β=0.8 and ten steps. Compare its objective with GD: stability does not make it better at every finite iteration. Then set β=0 and verify the two curves coincide. Switch to one Adam step and independently calculate corrected moments (1,8) and (1,64). Explain why omitting correction would change the first displacement.

Further reading: [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980) and [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
