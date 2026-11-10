## Inside a ball versus on a sphere

Two-dimensional projections can hide high-dimensional volume and angle behavior. This chapter uses analytic laws rather than random scatterplots. One model is uniform inside the unit ball; the other consists of two independent uniform points on the unit sphere.

## Derive a radial CDF from volume

A radius-r ball occupies fraction rᵈ of the unit ball's volume. Thus R=||X|| has CDF rᵈ and density drᵈ⁻¹ on [0,1]. Radius itself is uniform only at dimension one.

$$
P(R\leq r)=r^d,\quad E[R]=\frac d{d+1},\quad
\operatorname{Var}(R)=\frac d{(d+2)(d+1)^2}.
$$

## Mass in a thin shell

An outer shell of thickness δ has probability 1−(1−δ)ᵈ. At d=32,δ=.1, this is about .966; at d=1 it is .1. Even thickness .01 holds about .724 at dimension 128.

Blue plots the current radial CDF and orange dimension one. Shell probability is one minus the CDF at 1−δ, not the CDF height itself. Zero thickness has probability zero; thickness one includes the whole ball.

## Inner products and distances on the sphere

In the separate sphere model, symmetry gives E[UᵀV]=0 and isotropy gives E[(UᵀV)²]=1/d. Cosine RMS is therefore 1/√d. Squared distance equals 2−2UᵀV, yielding:

$$
E\|U-V\|^2=2,\qquad \operatorname{Var}(\|U-V\|^2)=4/d.
$$

This does not say expected unsquared distance is exactly √2. Distinguish a squared-distance expectation from the expectation of distance. Finite dimensions retain variation; increasing dimension favors nearly orthogonal directions and distances near √2.

## Real data can have different geometry

These models offer intuition for normalized independent isotropic representations. Actual data may be clustered, correlated or structured on a smaller set. Do not conclude that every high-dimensional dataset has identical distances without distributional assumptions.

## Exercises with explanations

1. What is R at d=1? **Uniform on [0,1].**
2. What is mean squared sphere distance? **Exactly two.**
3. Do both panels use the same sampling model? **No: one uses ball interiors, the other sphere surfaces.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
