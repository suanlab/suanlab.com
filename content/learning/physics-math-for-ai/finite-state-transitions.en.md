## Specify both states and a rule

Let S={0,1,2} and fix a deterministic map T:S→S. Evolution follows sₜ₊₁=T(sₜ). The same initial state can have different futures under different maps. Under the same map, revisiting a state forces the same future.

State numbers are labels, not magnitudes or spatial coordinates. One transition is one discrete step. This model contains no probabilities, energies or continuous physical time.

## Encode a whole function

Each of three inputs has three possible next states, giving 3³=27 maps. The lab encodes them as

$$
c=T(0)+3T(1)+9T(2),\quad c\in\{0,\ldots,26\}.
$$

Code seven means T=(1,2,0), a three-cycle; code 17 means T=(2,2,1). Compare the displayed map with the arrows after changing the code. Each vertex has exactly one outgoing edge. Self-loops are allowed here and represent fixed points, unlike the earlier simple directed graph model.

## Composition and trajectories

T⁰ is the identity and T²=T∘T. Thus sₙ=Tⁿ(s₀). With code seven and initial zero, the trace is 0,1,2,0,1,2,…, so T⁵(0)=2. Row zero is the initial observation; N transitions give N+1 observations.

Associativity of composition implies Tᵐ⁺ⁿ=Tᵐ∘Tⁿ. Distinct maps generally do not commute: U∘T need not equal T∘U.

## Finiteness forces repetition

Four observations of three possible states must repeat a state by the pigeonhole principle. If sᵢ=sⱼ with i<j, applying the same function and then induction gives sᵢ₊ₖ=sⱼ₊ₖ for every k≥0.

A trajectory therefore has a finite transient followed by a cycle. Let μ be the first cycle-entry index and λ≥1 its least positive period. Then

$$
s_{t+\lambda}=s_t\quad\text{for every }t\ge\mu.
$$

Even for a small selected N, the lab follows the fully specified map until the first repeated state to compute μ and λ. It does not infer them from only the displayed prefix.

## Transients and fixed points

Code 17 from zero gives 0,2,1,2,1,…, with μ=1,λ=2. Period two does not imply repetition starts at the initial observation.

Code 26 sends every state to two. From zero the trace is 0,2,2,…, with μ=1,λ=1. From two it has μ=0,λ=1. A fixed point has least positive period one, not zero. The initial state can change the transient and which cycle is visited under the same map.

## When the assumptions change

A time-varying map or an external input omitted from the state can invalidate “same state, same future.” A stochastic transition can produce different samples on revisiting a state. Consider whether the clock, inputs or random generator must be included in an expanded state.

This perspective helps analyze agent workflows, protocols and repeated finite-precision computation. An infinite continuous neural state space does not inherit a finite-cycle conclusion from this argument. Even a finite-precision implementation needs external input, randomness and time dependence accounted for before claiming a closed deterministic finite-state model.

## Exercises and answers

1. A three-state deterministic trajectory visits at most **three distinct states**.
2. Code zero from initial two gives **2,0,0,…, μ=1,λ=1**.
3. Can changing only the initial state change μ? **Yes; code 26 illustrates it.**
4. With N=0 the state is **s₀**. Zero transitions do not mean an empty state set.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/) and [MIT Theory of Computation](https://ocw.mit.edu/courses/18-404j-theory-of-computation-fall-2020/). The trajectories are independently constructed examples.
