## 1. A path integral starts with an ordinary integral

Imagine a wandering sensor reading recorded between a known initial value and a known final value. There are many possible intermediate records. A path integral combines their contributions, but the phrase “sum over every path” does not specify a probability law, an integration measure or even a convergent operation.

We begin with a finite list of observation times. Every integral in this chapter is then an ordinary Gaussian integral over explicitly named variables. We can check normalization, condition on the final value, and compare different time grids before discussing a continuous path.

Prerequisites are Gaussian conditioning, covariance, multivariable integration and the distinction between a distribution and a drawing. The spatial-chain chapter provides useful matrix algebra, but here the index labels time and the endpoint convention is fixed.

## 2. Declare the time grid and physical units

Use N equal time intervals, total time τ>0, diffusion constant D>0, and fixed endpoints a,b. There are N+1 positions but only N−1 integrated variables. Write δ=τ/N, x₀=a and x_N=b.

$$
t_i=i\delta,\qquad \delta=\frac{\tau}{N},\qquad x_0=a,\quad x_N=b.
$$

D has units of position squared divided by time. Thus Dδ and every covariance have units of position squared. N counts intervals, not interior variables. The planned calculator accepts N=2…32, positive rational τ,D≤16 and rational |a|,|b|≤16, with at most 12 digits in each numerator and denominator. These finite computational limits do not define the mathematical continuum limit.

## 3. Normalize each increment first

An unconstrained step of duration δ has zero mean increment and variance 2Dδ. Its density includes a prefactor; the exponential alone is not a normalized density.

$$
g_\delta(y-x)=\frac{1}{\sqrt{4\pi D\delta}}\exp\!\left[-\frac{(y-x)^2}{4D\delta}\right].
$$

Multiplying the N step densities gives a joint density for the intermediate values and final endpoint, conditional on the starting value. Integrating only the intermediate values leaves a density in b. Its units are inverse position; it is not a dimensionless probability assigned to the exact event X(τ)=b.

## 4. Integrate intermediate positions to obtain a kernel

Gaussian convolution adds variances. Repeatedly integrating neighboring steps gives the same total-time free kernel, independently of N.

$$
K_\tau(b,a)=\int_{\mathbb R^{N-1}}\prod_{i=1}^{N}g_\delta(x_i-x_{i-1})\,dx_1\cdots dx_{N-1}
=\frac{e^{-(b-a)^2/(4D\tau)}}{\sqrt{4\pi D\tau}}.
$$

This exact equality is stronger than observing a stable plotted curve under refinement. It follows from Gaussian integration, including its normalization constants. Omitting one factor of δ changes the integral even when the most likely path looks unchanged.

## 5. Conditioning creates the bridge

Divide the joint density by Kτ(b,a). The resulting density over the N−1 interior coordinates integrates to one. This is the finite Brownian bridge law with the declared endpoints.

$$
p_N(x_1,\ldots,x_{N-1}\mid a,b)=\frac{\prod_{i=1}^{N}g_\delta(x_i-x_{i-1})}{K_\tau(b,a)}.
$$

Conditioning on a continuously distributed endpoint uses a conditional density, not division by the probability of an exact endpoint event. The kernel supplies the correct density normalization. Drawing many paths with the same ends does not specify this law unless the sampling procedure reproduces it.

## 6. Read the precision from temporal edges

The exponent is a quadratic sum of neighboring differences. Let L have diagonal2 and adjacent entries−1, with size N−1. The endpoint terms enter the linear part; they are not extra random coordinates.

$$
A_N(x)=\frac{1}{4D\delta}\sum_{i=1}^{N}(x_i-x_{i-1})^2,
\qquad Q=\frac{L}{2D\delta}.
$$

The diagonal is2 even at the first and last interior positions because each touches an endpoint edge. For N=2 the single interior variable touches both endpoints: its linear forcing contains a+b, rather than just one endpoint. L is positive definite for every N≥2. There is no constant zero mode to remove.

## 7. Derive the mean and covariance

The mean solves the discrete second-difference equation with the declared endpoints, so its increments are equal. The inverse of L is piecewise linear in either index, vanishes at the endpoints and has the required unit jump in its discrete derivative.

$$
\mu_i=a+\frac{i}{N}(b-a),\qquad
C_{ij}=2D\delta\left[\min(i,j)-\frac{ij}{N}\right],\qquad 1\le i,j<N.
$$

A quick check is to multiply Q by C: every row cancels away from the selected index and gives one at that index. Equal-time variance is zero at either fixed endpoint and largest near the middle. Endpoint values are certain; the interior usually is not.

## 8. Work out a four-interval bridge by hand

Set N=4,τ=D=1,a=0,b=1. Then δ=1/4, the mean is (1/4,1/2,3/4), and the exact covariance is

$$
Q=\begin{pmatrix}4&-2&0\\-2&4&-2\\0&-2&4\end{pmatrix},\qquad
C=\begin{pmatrix}3/8&1/4&1/8\\1/4&1/2&1/4\\1/8&1/4&3/8\end{pmatrix}.
$$

For example, the middle diagonal of QC is −2(1/4)+4(1/2)−2(1/4)=1. The middle mean is1/2 but its variance is1/2; the mean curve is not a typical deterministic trajectory. Positive correlations between interior values coexist with negatively correlated increments.

## 9. Check the determinant and density constants

The tridiagonal determinant obeys d_k=2d_{k−1}−d_{k−2}, with d₀=1,d₁=2. Consequently detL=N for the N−1 dimensional matrix. Completing the square separates the endpoint displacement from fluctuations.

$$
\det Q=\frac{N}{(2D\delta)^{N-1}},\qquad
A_N(\mu)=\frac{(b-a)^2}{4D\tau},\qquad
\int e^{-A_N(x)}dx=e^{-A_N(\mu)}\frac{(2\pi)^{(N-1)/2}}{\sqrt{\det Q}}.
$$

Multiplying by the N increment prefactors recovers Kτ. In the four-interval example detQ=32. Keeping only the exponent would miss the determinant and fail the total-time normalization. The finite derivation verifier compares positive prefactors through their rational squares; it does not claim to compute certified decimal values of π or square roots.

## 10. Coarsening is Gaussian marginalization

Refine N intervals to2N and keep the even time indices. The retained times are exactly the original times. Substituting 2i,2j and δ/2 into the covariance formula gives the original Cij, and the means agree too. Since both laws are Gaussian, these equalities identify their entire finite distributions.

One can instead eliminate the odd coordinates from the fine precision. The odd block is diagonal, and its Schur complement equals the coarse Q. This gives a separate matrix explanation of consistency. Merely deleting rows from the precision would give a conditional law with the deleted positions fixed, not the desired marginal law.

## 11. An integrated observable changes with the grid

A trapezoidal approximation to the area under a path uses all interior positions. Although old-time marginals agree exactly under refinement, this observable includes new random positions on the finer grid.

$$
B_N=\delta\left[\frac{a+b}{2}+\sum_{i=1}^{N-1}x_i\right],\qquad
\mathbb E B_N=\frac{\tau(a+b)}2,\qquad
\operatorname{Var}(B_N)=\frac{D\tau^3}{6}\left(1-\frac1{N^2}\right).
$$

At N=4,τ=D=1, the variance is 5/32. At N=8 it is 21/128, larger by 1/128. The limiting expression is 1/6; the coarse-grid variance misses 1/96. A grid-independent covariance at one fixed time does not imply that every discretized observable is already exact.

## 12. Fixed endpoints couple the increments

The increments must add to b−a. Their means are (b−a)/N. Their covariance has a negative off-diagonal term:

$$
\operatorname{Cov}(\Delta x_i,\Delta x_j)=2D\delta\left(\mathbf1_{i=j}-\frac1N\right).
$$

For the four-interval example each variance is 3/8 and each distinct-increment covariance is −1/8. Summing one row gives zero, as required by a fixed total displacement. This is another instance of a global constraint creating negative correlations without a negative local diffusion constant.

## 13. A smooth average is not a smooth random path

The polygon through μ is straight. But the fluctuations of a short increment are typically of order√δ, whereas an increment of a differentiable path would be of orderδ. A useful finite diagnostic is the expected sum of squared increments.

$$
\mathbb E\sum_{i=1}^{N}(\Delta x_i)^2
=\frac{(b-a)^2}{N}+2D\tau\left(1-\frac1N\right),\qquad
\mathbb E A_N=\frac{(b-a)^2}{4D\tau}+\frac{N-1}{2}.
$$

The squared-increment sum approaches 2Dτ, while the expected discrete action diverges with N. This does not contradict the existence of the finite normalized laws. It warns against treating a continuum Brownian path as an ordinary differentiable trajectory with a finite classical kinetic integral. A piecewise linear display is a representation of samples on a grid, not proof of smoothness.

## 14. Add an auxiliary source with consistent units

For a vector j with inverse-position units, the exponent jᵀx is dimensionless. It generates moments of the already conditioned finite law:

$$
W_N(j)=\log\mathbb E[e^{j^Tx}]=j^T\mu+\frac12j^TCj,
\qquad \nabla W_N=\mu+Cj,\qquad \nabla^2 W_N=C.
$$

The experiment uses a point source at one selected grid index, from 0 through N, with rational strength |j|≤16 and at most 12 digits in its numerator and denominator. A source at a fixed endpoint contributes only the constant ja or jb to W; its covariance response is zero.

Here j is an auxiliary tilt, not the physical energy source J of the preceding Gaussian-field chapter. A positive exponential tilt changes the mean and leaves Gaussian covariance unchanged. For a source of strength 2 at the middle of the four-interval example, W=2 and the tilted interior mean becomes (3/4,3/2,5/4). The endpoints remain 0 and 1, so the tilted mean can rise above the final endpoint without violating the conditioning.

## 15. Separate the three meanings of a limit

The exact matching of old-time marginals is a finite identity. Constructing a random continuous function with these finite distributions additionally requires a probability-space construction and a continuity argument. Convergence of an integrated observable or an approximation to a potential-weighted law is a third question.

The covariance expression suggests the continuous bridge kernel 2D[min(s,t)−st/τ]. That expression is a target consistent with the finite laws; printing it does not by itself prove all sample-path statements or arbitrary-potential integration theorems. This chapter's executable verifier establishes the stated finite identities. Later chapters must state the assumptions of every stronger theorem they use.

## 16. Connect to Euclidean quantum mechanics without changing the claim

For a free particle, choosing D=ℏ/(2m) matches the positive Euclidean-time kinetic exponent S_E/ℏ. Here τ is Euclidean time, not a temperature. A fixed-endpoint kernel is also different from the trace obtained by identifying and integrating endpoints.

This mathematical connection does not turn a positive diffusion density into a real-time quantum transition probability. Real-time evolution uses oscillating phases, and Born probabilities require amplitudes and states. Potential weighting, operator-product limits and real-time continuation will be treated separately. This free finite model does not prove their validity for every Hamiltonian.

## 17. Reproduce a small exact calculation in Python

The standard library suffices for this example. It checks a matrix identity and two distinct area variances without random sampling.

```python
from fractions import Fraction as F
N = 4
C = [[2 * F(1, N) * (min(i, j) - F(i*j, N))
      for j in range(1, N)] for i in range(1, N)]
Q = [[F(4 if i == j else -2 if abs(i-j) == 1 else 0)
      for j in range(N-1)] for i in range(N-1)]
assert all(sum(Q[i][k] * C[k][j] for k in range(N-1)) == (i == j)
           for i in range(N-1) for j in range(N-1))
area_variance = sum(map(sum, C)) / N**2
assert area_variance == F(5, 32)
fine_area_variance = (1 - F(1, 8**2)) / 6
assert fine_area_variance - area_variance == F(1, 128)
print(C[1][1], area_variance, fine_area_variance)
```

## 18. Six prediction-led comparisons

Before calculating, predict which quantities should change.

| Change | Prediction to check | Reason |
|---|---|---|
| Double D | Same mean; double covariance and area variance | Endpoints fix the mean; diffusion fixes fluctuation scale |
| Shift both endpoints by c | Translate the mean; preserve covariance | Increment energy depends on differences |
| Double τ at fixed N,D | Same indexwise mean; double position covariance; multiply area variance by 8 | δ doubles and area includes a time factor |
| Refine4→8 at fixed τ,D,a,b | Preserve values' law at old times; change area variance | Marginal consistency differs from quadrature accuracy |
| Change b at fixed a | Change mean slope; preserve covariance | Gaussian conditioning separates mean from fluctuations |
| Tilt the middle value | Shift interior mean; preserve covariance | An auxiliary linear source changes the completed square |

A future sampling display should distinguish a controlled random seed, a sample estimate and the exact population quantities. Replaying the same finite sample is not additional independent evidence.

## 19. Worked questions

1. **How many integration variables are there for N=4?** Three. The two endpoints are fixed and the four labels count intervals. Integrating both endpoints would define a different problem.

2. **Why is the first diagonal of L equal to2?** The first interior value appears in the edge from a and the edge to its next neighbor. The endpoint edge contributes to the quadratic term as well as the linear forcing.

3. **Does C₂₂=1/2 at N=4,τ=D=1 also hold at the midpoint after refinement?** Yes. At N=8 the midpoint is index 4, and the same covariance formula gives 1/2. Its array label changes while its physical time stays fixed.

4. **Can one coarsen by deleting rows and columns of Q?** That conditions on omitted variables. Marginalizing them uses a Schur complement. For the free bridge that complement is exactly the coarse precision.

5. **Why can the area variance change when the old-time marginals do not?** The fine trapezoidal area includes new coordinates and is a different observable. Its variance grows from 5/32 to 21/128 in the example.

6. **Do negative increment covariances indicate negative diffusion?** No. All increments must sum to the fixed endpoint displacement. For N=4 they have variance 3/8 and pairwise covariance −1/8, with each covariance row summing to zero.

7. **Does the increasing expected action make the finite probability law invalid?** No. Every finite Gaussian density is normalizable. The divergence warns that typical continuum paths cannot be assigned an ordinary finite differentiable kinetic action.

8. **Is this positive kernel already a real-time quantum probability?** No. It is a positive diffusion/Euclidean construction. Real-time amplitudes, their phases and the Born rule require separate definitions and checks.

## 20. Research use and further reading

Bridge laws appear in conditioned trajectory models, Gaussian process regression, smoothing and diffusion-based proposals. For AI research, first decide whether endpoint conditioning describes the data or is an imposed modeling choice. Test both the observation operator and the noise model. Consistent finite marginals do not validate an arbitrary learned drift or a biased discretization of a path functional.

For the operator-product and quantum path-integral background, see [Grosche, §II.1](https://arxiv.org/pdf/hep-th/9302097). For the relation to quadratic field fluctuations, see [Tong, Statistical Field Theory §2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2). The finite bridge formulas and examples above are derived for this chapter's stated law. Euclidean potential kernels, saddle approximations and real-time phases remain subsequent treatments.
