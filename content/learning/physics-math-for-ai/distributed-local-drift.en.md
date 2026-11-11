## Communication changes where gradients are evaluated

Two workers have objectives F₁(w)=(w+1)²/2 and F₂(w)=κ(w−1)²/2. The global objective is their equal-weight average. Its optimizer is w*=(κ−1)/(κ+1), not the unweighted average of the two local minimizers when curvatures differ.

Each round starts both workers at the same broadcast point. Each performs E exact local gradient steps, then their endpoints are averaged. The centralized comparator performs E full-gradient steps per round. Both use the same number of component-gradient evaluations, but their intermediate evaluation points differ. The experiment is synchronous and deterministic; it does not model delays, compression, stochastic batches or uneven worker weights.

## Why one local step agrees with centralized GD

For E=1, averaging w−η∇Fᵢ(w) gives w−η∇F(w). Thus the two trajectories agree exactly from a common start. This identity depends on gradients being evaluated at the same point and on matching objective weights.

For E>1 each worker follows its own objective between communications. Averaging endpoints is generally different from repeatedly averaging gradients at synchronized intermediate points. This is local drift, rather than a numerical failure of the averaging operation.

## Derive the round map and its fixed point

For scalar quadratic curvature hᵢ and minimizer aᵢ,

$$
w_i^{(E)}=a_i+(1-\eta h_i)^E(w-a_i).
$$

Let Aᵢ=(1−ηhᵢ)ᴱ and mᵢ=1−Aᵢ. With a₁=−1,a₂=1, averaging gives

$$
w^+=Aw+B,\qquad A=1-(m_1+m_2)/2,
\quad B=(m_2-m_1)/2,
\quad w_{fixed}=\frac{m_2-m_1}{m_1+m_2}.
$$

The last formula requires a nonzero denominator. At η=0 the round map is identity, so there is no unique fixed point. Even when a unique algebraic fixed point exists, convergence to it requires |A|<1. An unstable fixed point is not a trajectory prediction.

## Bias can persist after apparent convergence

Choose κ=4,η=0.1. At E=1 the fixed point is the global optimizer 0.6. At E=4, m₁=1−0.9⁴=0.3439 and m₂=1−0.6⁴=0.8704, giving fixed point approximately 0.433583. The round map contracts with A≈0.39285, but its limit has a nonzero global gradient.

This is a concrete example where a converged communication-round sequence does not solve the intended global objective. Equal curvatures remove this particular fixed-point bias; more local computation is therefore not automatically harmful or automatically beneficial.

## A broadcast-gradient correction can remove this fixed-point bias

At a broadcast point w̄, replace each local gradient by gᵢ(w)−gᵢ(w̄)+G(w̄). Every worker initially sees the same full gradient. For quadratic curvature hᵢ, solving the local recurrence gives

$$
w_i^{(E)}=\bar w-\frac{1-(1-\eta h_i)^E}{h_i}G(\bar w),\qquad
w^+=\bar w-sG(\bar w),\quad
s=\frac12\sum_{i=1}^2\frac{1-(1-\eta h_i)^E}{h_i}.
$$

Thus, when this corrected round map contracts, its fixed point solves the global gradient equation. With κ=4,η=0.1,E=4, s=0.28075 and its error factor is 0.298125. The uncorrected map instead converges to approximately 0.433583. The correction uses fresh anchor gradients and has an additional work/information cost.

| Method in this example | Contracting fixed point |
|---|---:|
| Uncorrected local averaging | Approximately 0.433583 |
| Exact broadcast-gradient correction | 0.6 |
| Centralized full gradient | 0.6 |

This is a transparent deterministic control-variate example, not a complete implementation of SCAFFOLD's stored stochastic controls. See [Karimireddy et al.](https://proceedings.mlr.press/v119/karimireddy20a.html) for that algorithm and its assumptions, and [Stich](https://arxiv.org/abs/1805.09767) for local-SGD theory. The native controls retain uncorrected averaging. Their strict contraction flag now checks the round correction before subtracting it from one, avoiding a false boundary classification at tiny η.

## Reproduce the calculation in Python

```python
from fractions import Fraction as Q
h=[Q(1),Q(4)]; a=[Q(-1),Q(1)]; eta=Q(1,10); E=4
local=corrected=Q(0); optimum=Q(3,5)
for round_index in range(8):
    raw=[]; fixed=[]; G=(5*corrected-3)/2
    for hi,ai in zip(h,a):
        u=local; v=corrected
        for _ in range(E):
            u-=eta*hi*(u-ai)
            v-=eta*(hi*(v-corrected)+G)
        raw.append(u); fixed.append(v)
    local=sum(raw)/2; corrected=sum(fixed)/2
    print(round_index+1,float(local),float(corrected),float(corrected-optimum))
s=sum((1-(1-eta*hi)**E)/hi for hi in h)/2
assert s==Q(1123,4000) and 1-s*Q(5,2)==Q(477,1600)
```

## Exercise and explained answer

From w₀=0 use κ=4,η=0.1,E=4 for eight rounds. The local average reaches approximately 0.433337 while centralized GD reaches approximately 0.599940. The local global gradient is approximately −0.416657. Explain why a small change between late rounds would be an inadequate stopping certificate.

Compare E=1 versus E=4 at equal total gradient work by adjusting the number of rounds. State both communication and computation budgets. In research claims about distributed learning, identify heterogeneity, synchronization, aggregation weights and the objective whose stationarity is being measured.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
