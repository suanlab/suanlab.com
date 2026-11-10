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

## Exercise and explained answer

From w₀=0 use κ=4,η=0.1,E=4 for eight rounds. The local average reaches approximately 0.433337 while centralized GD reaches approximately 0.599940. The local global gradient is approximately −0.416657. Explain why a small change between late rounds would be an inadequate stopping certificate.

Compare E=1 versus E=4 at equal total gradient work by adjusting the number of rounds. State both communication and computation budgets. In research claims about distributed learning, identify heterogeneity, synchronization, aggregation weights and the objective whose stationarity is being measured.

Further reading: [Bottou, Curtis and Nocedal](https://arxiv.org/abs/1606.04838).
