## 1. Ask how a final measurement depends on earlier controls

A discrete model evolves by $x_{n+1}=a x_n+u_n$ from a known initial state. The parameter $a$ models persistence, and $u_n$ is the input at step $n$. Suppose the task is to match a terminal observation $y$ while controlling input energy. Define

$$
J=\tfrac12(x_N-y)^2+\tfrac{\gamma}{2}\sum_{n=0}^{N-1}u_n^2.
$$

The forward pass stores the states and evaluates the loss. Sensitivity asks how the same loss changes when a parameter or input changes. [Giles and Pierce, An Introduction to the Adjoint Approach to Design](https://doi.org/10.1023/A:1011430410075)

## 2. Propagate the terminal sensitivity backward

Write $p_n$ for the derivative of the terminal loss with respect to $x_n$, holding later controls fixed. The terminal condition and reverse recurrence follow from the chain rule:

$$
p_N=x_N-y,\qquad p_n=a p_{n+1}.
$$

Each reverse step multiplies by the derivative of one forward step with respect to its input state. In a vector model this factor is the transpose of the state Jacobian.

## 3. Read off all control derivatives

A change in $u_n$ shifts $x_{n+1}$ directly, and the control penalty contributes $\gamma u_n$. Therefore

$$
\frac{\partial J}{\partial u_n}=p_{n+1}+\gamma u_n,
\qquad
\frac{dJ}{da}=\sum_{n=0}^{N-1}p_{n+1}x_n.
$$

For $x_0=1$, $a=0.5$, $u=(0,0)$, $y=0$, and $\gamma=0.1$, the state sequence is $(1,0.5,0.25)$. The two control derivatives are $(0.125,0.25)$ and the derivative with respect to $a$ is $0.25$.

## 4. Decide which sensitivity calculation fits the task

Forward sensitivities propagate one perturbation at a time and can return its effect on many outputs. A reverse pass propagates the derivative of one scalar loss and collects the effects of many controls. The reverse computation needs forward states or a way to reconstruct them. For a long simulation, checkpointing trades storage for recomputation. The discrete adjoint differentiates the actual update rule used in code.

## 5. Compare with a directional finite difference

```python
import numpy as np

u = np.array([.15, -.1, .2])
a, target, gamma = .7, .3, .1

def objective(a, u):
    states = [1.]
    for value in u:
        states.append(a * states[-1] + value)
    loss = .5 * (states[-1] - target)**2 + .5 * gamma * (u @ u)
    return loss, np.array(states)

loss, states = objective(a, u)
adjoint = np.empty(len(u) + 1)
adjoint[-1] = states[-1] - target
for n in range(len(u)-1, -1, -1):
    adjoint[n] = a * adjoint[n+1]
du = adjoint[1:] + gamma * u
da = adjoint[1:] @ states[:-1]

eps = 1e-6
direction, direction_a = np.array([.2, .4, -.3]), .1
numeric = (objective(a+eps*direction_a, u+eps*direction)[0]
           - objective(a-eps*direction_a, u-eps*direction)[0]) / (2*eps)
assert np.isclose(numeric, du @ direction + da * direction_a, atol=1e-9)
print(du, da)
```

## 6. Check derivatives before trusting optimization

Compare the adjoint with centered differences over several perturbation sizes. Large steps expose nonlinear truncation effects, and very small steps can expose roundoff. Include representative boundary and control conditions. A derivative check verifies the implementation of the chosen forward model. Check model predictions against independent observations as well.
