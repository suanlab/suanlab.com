## 1. A positive path weight still needs an interpretation

Imagine penalizing a sequence of positions for both abrupt jumps and large displacement from the origin. The jump penalty resembles the Brownian bridge from the preceding chapter. The new position penalty makes some trajectories contribute less. Integrating all intermediate positions produces a positive kernel, but its integral over the final position need not equal one.

This chapter connects that construction to the free particle and harmonic oscillator in Euclidean time. We will calculate ordinary finite Gaussian integrals first, then compare them with a known continuous-time operator kernel. Keeping these two answers separate makes discretization error visible.

Prerequisites are Gaussian integration, covariance, matrices, basic quantum states and the distinction between a kernel and a probability density. A quantum field theory course is not required. The example is one particle on the entire real line with no spatial boundary or magnetic field.

## 2. Declare the Hamiltonian and units

The target Hamiltonian contains kinetic energy and a nonnegative harmonic potential. Euclidean time replaces oscillatory evolution by an exponentially damped operator.

$$
H=T+V=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+\frac{m\omega^2x^2}{2},\qquad
K_\tau(b,a)=\langle b|e^{-\tau H/\hbar}|a\rangle.
$$

Here m>0 is mass, ℏ>0 has units of action, τ>0 has units of time and ω²≥0 has units of inverse time squared. The exponent τH/ℏ is dimensionless. A one-dimensional position kernel has units of inverse length. It is not a dimensionless probability of arriving at an exact position.

The planned exact calculator accepts rational m,ℏ,τ≤16, rational 0≤ω²≤16 and |a|,|b|≤16, with at most12 digits in each numerator and denominator. We enter ω², rather than approximating its square root before constructing the matrix. These are computational bounds, not limits on the physical theory.

## 3. Read Euclidean evolution as spectral filtering

An energy eigenstate of energy E is multiplied by exp(−τE/ℏ). Higher energies are suppressed more strongly. A superposition therefore changes its relative energy weights. This is useful for ground-state projection, but the norm of the state is generally not preserved.

Real-time evolution instead multiplies an eigenstate by a unit-modulus phase. Replacing that phase by positive damping does not preserve every physical interpretation. Positivity of a kernel also does not mean that the kernel is already a Born probability. The assumptions behind the Euclidean semigroup and its free/oscillator kernels are described in [Wipf, §6.1](https://www.tpi.uni-jena.de/qfphysics/homepage/wipf/lectures/pfad/pfad6.pdf).

## 4. Start with the free one-step kernel

For a time step δ>0, the free-particle kernel is a normalized Gaussian in the endpoint b. It agrees with diffusion after setting D=ℏ/(2m).

$$
G_\delta(b,a)=\sqrt{\frac{m}{2\pi\hbar\delta}}
\exp\!\left[-\frac{m(b-a)^2}{2\hbar\delta}\right],\qquad D=\frac{\hbar}{2m}.
$$

Successive free steps convolve by adding their variances. Thus integrating N−1 intermediate positions gives exactly Gτ for every N with Nδ=τ. Free time slicing has no discretization error in this example. That special identity will help us detect mistakes in the potential-weighted calculation.

## 5. Specify the potential quadrature before calculating

Choose N intervals, δ=τ/N, x₀=a and x_N=b. There are N−1 integration variables. Use half of each endpoint potential and one full weight for every interior potential.

$$
A_N(x)=\frac{m}{2\hbar\delta}\sum_{k=1}^{N}(x_k-x_{k-1})^2
+\frac{m\omega^2\delta}{2\hbar}\left[\frac{a^2+b^2}{2}+\sum_{k=1}^{N-1}x_k^2\right].
$$

A_N is the dimensionless Euclidean action, not the action before division by ℏ. The first term discourages jumps; the second discourages spending time far from the potential minimum. Both include their physical time factors.

This trapezoidal potential is not the same finite rule as evaluating V at the midpoint of each pair of positions. For a quadratic potential, the average of V(a),V(b) exceeds V((a+b)/2) by mω²(b−a)²/8. Naming a rule “symmetric” does not identify its full finite action.

## 6. Keep all Gaussian normalization factors

Integrate the exponential action together with the N free-step prefactors. Dropping those prefactors changes the kernel even if the minimum-action path looks unchanged.

$$
K_N(b,a)=\left(\frac{m}{2\pi\hbar\delta}\right)^{N/2}
\int_{\mathbb R^{N-1}}e^{-A_N(x)}\,dx_1\cdots dx_{N-1}.
$$

K_N denotes the exact value of this finite-dimensional integral. It is generally an approximation to Kτ for the original Hamiltonian. These statements are compatible: exact integration does not remove error in the model being integrated.

The N integration-step prefactors have units of length to the power−N, and the N−1 integrations restore N−1 powers of length. The remaining inverse-length unit is a useful normalization check.

## 7. Understand the symmetric operator product

One trapezoidal step is the position kernel of exp(−δV/(2ℏ)) exp(−δT/ℏ) exp(−δV/(2ℏ)). Its Nth power gives the integral above. Multiplying neighboring steps combines two half weights into the full interior potential weight.

Reversing a and b leaves this finite kernel unchanged. That is a finite symmetry statement, not a proof that the product equals exp(−τ(T+V)/ℏ). Kinetic and potential operators do not commute. General operator convergence needs hypotheses; this chapter establishes the oscillator comparison directly.

## 8. Assemble the interior precision and endpoint forcing

Let s=m/(ℏδ),v=ω²δ² and let L be the pinned chain matrix with diagonal2 and adjacent entries−1. The action is a quadratic polynomial in the interior vector.

$$
A_N(x)=\tfrac12x^TQx-f^Tx+c,\quad
Q=s(L+vI),\quad f=s(ae_1+be_{N-1}),\quad
c=\frac{s(a^2+b^2)}2+\frac{sv(a^2+b^2)}4.
$$

All endpoint edges remain in L. Even when ω²=0, the pinned precision is positive definite. For N=2 there is one interior coordinate, so e₁ and e_(N−1) name the same vector: add both endpoint contributions. Treating the first and last contributions as assignments would accidentally erase one of them.

## 9. Condition the path and complete the square

Dividing the finite joint weight by K_N produces a normalized Gaussian law over the interior positions. Its mean is the minimizing path and its covariance is the inverse precision.

$$
\mu=Q^{-1}f,\qquad C=Q^{-1},\qquad
A_{\min}=c-\tfrac12f^TQ^{-1}f.
$$

The normalized density is √[detQ/(2π)^(N−1)] times exp[−(x−μ)ᵀQ(x−μ)/2]. The kernel and the conditioned law answer different questions: the first retains the endpoint-dependent normalization, while the second integrates to one over the already pinned interior variables.

The mean of a Gaussian is also its minimizer. A typical trajectory is not equal to that mean. Adding a potential changes both the mean and the fluctuations, unlike the auxiliary linear source in the preceding bridge chapter.

## 10. Use a determinant recurrence

Define D_k as the determinant of the k-dimensional matrix with diagonal2+v and adjacent−1. Expanding a boundary row gives a scalar recurrence.

$$
D_0=1,\quad D_1=2+v,\quad D_k=(2+v)D_{k-1}-D_{k-2},\qquad
\det Q=s^{N-1}D_{N-1}.
$$

This recurrence avoids treating every determinant as an unrelated calculation. Its values are positive for v≥0. At v=0 it gives D_k=k+1, recovering the free pinned determinant. We independently check it by constructing temporal incidence rows and performing rational matrix elimination.

## 11. Recover covariance and mean from the same recurrence

For interior indices1≤i≤j≤N−1, the inverse has a product form. Symmetry supplies the other half of the covariance matrix.

$$
C_{ij}=\frac{D_{i-1}D_{N-1-j}}{sD_{N-1}},\qquad
\mu_i=\frac{aD_{N-1-i}+bD_{i-1}}{D_{N-1}}.
$$

The fixed endpoints have zero covariance and retain the values a,b. At ω²=0 the mean becomes the straight interpolation, and the covariance becomes the free bridge with D=ℏ/(2m). For nonzero potential, pulling the path toward the origin can bend the mean even when the endpoints stay unchanged.

The matrix formula describes population covariance. A heat map is only its display, and a curve through μ is only the mean. Neither is a random sample path.

## 12. Express the finite kernel with rational coefficients

All finite matrix information needed for the endpoint kernel reduces to three rational numbers P,g,h.

$$
P=\frac{s}{2D_{N-1}},\quad
g=\frac{s}{2}\left(1+\frac v2-\frac{D_{N-2}}{D_{N-1}}\right),\quad
h=\frac{s}{D_{N-1}},\qquad
K_N(b,a)=\sqrt{\frac P\pi}\,e^{-g(a^2+b^2)+hab}.
$$

Here P is the squared prefactor after removing its common π⁻¹. It has units of inverse length squared, as do g,h. This representation preserves exact rational coefficients without pretending that π, a square root or an exponential is rational. A decimal visualization of K_N is a separate approximation.

The final exponent equals −A_min. The fluctuation determinant supplies the prefactor. Keeping only the minimum action would discard part of the answer.

## 13. Work through the smallest nontrivial example

Set N=2,m=ℏ=τ=ω²=1,a=0,b=1 in declared units. Then δ=1/2,s=2,v=1/4 and there is exactly one interior variable.

| Quantity | Exact value | Interpretation |
|---|---|---|
| Q | 9/2 | Interior precision |
| f | 2 | Force from the final endpoint |
| μ₁ | 4/9 | Mean pulled below the free midpoint1/2 |
| C₁₁ | 2/9 | Less fluctuation than the free value1/4 |
| c | 9/8 | Includes both half endpoint potentials |
| A_min | 49/72 | c−fμ₁/2 |
| P,g,h | 4/9,49/72,8/9 | Rational endpoint coefficients |

Therefore K₂(1,0)=2exp(−49/72)/(3√π). Check the square completion directly: A₂(x)=9x²/4−2x+9/8. The coefficient of (x−4/9)² is9/4, and the remaining constant is49/72. This hand calculation tests factors of2, the endpoint half weight and the N−1 count together.

## 14. Compare against the continuous Mehler kernel

For ω>0 the target continuous-time answer is known explicitly.

$$
K_\tau(b,a)=\sqrt{\frac{m\omega}{2\pi\hbar\sinh(\omega\tau)}}
\exp\!\left[-\frac{m\omega\big((a^2+b^2)\cosh(\omega\tau)-2ab\big)}{2\hbar\sinh(\omega\tau)}\right].
$$

The free kernel is its limit as ω→0; directly substituting zero into the displayed quotient would create0/0. Evaluate that branch using the free formula. The Mehler reference and the finite recurrence use different constructions, making their comparison more informative than drawing two versions of the same code output. See [Wipf, equations6.9–6.10](https://www.tpi.uni-jena.de/qfphysics/homepage/wipf/lectures/pfad/pfad6.pdf).

## 15. Explain why the finite answer has a different frequency

For ω>0, write θ=2asinh(ωδ/2). The recurrence has a hyperbolic solution, and the finite kernel equals a Mehler kernel with modified parameters.

$$
D_{N-1}=\frac{\sinh(N\theta)}{\sinh\theta},\qquad
\omega_{\mathrm{eff}}=\frac\theta\delta,\qquad
m_{\mathrm{eff}}=m\frac{\sinh\theta}{\theta}.
$$

This is an exact identity for the declared finite rule. It says that the product propagates a different oscillator. Calling that modified oscillator the target system would hide the discretization error. In the free limit take θ→0, so m_eff→m.

Expanding θ gives θ=ωδ−ω³δ³/24+terms of orderδ⁵. Thus at fixed physical parameters the modified frequency and mass differ from their targets at orderδ². This explains the second-order pointwise kernel error for this oscillator; it is not a uniform error theorem for all potentials, endpoints or times.

## 16. A right-endpoint rule exposes a first-order boundary error

Instead put a full potential weight at every right endpoint. Interior terms are unchanged; only the two endpoint weights differ. Subtracting the finite actions gives an exact multiplicative relation.

$$
K_N^{\mathrm R}(b,a)=e^{-\delta[V(b)-V(a)]/(2\hbar)}K_N(b,a).
$$

The log correction is generally first order in δ. If a=b, or if a=−b for this even potential, V(a)=V(b) and the correction vanishes. Testing only diagonal kernels would miss the off-diagonal first-order effect. Reversing the endpoints changes the sign of this correction, while the symmetric kernel stays unchanged.

## 17. Measure refinement error with a declared observable

For the comparison m=ℏ=τ=ω²=1,a=1/3,b=4/3, compare log(K_N/Kτ) at N=4,8,…,256. Logs avoid subtracting two tiny kernels after underflow. Use the same physical endpoints, time and Hamiltonian at every refinement.

An80-digit Decimal comparison gives error-halving ratios near4 for the symmetric rule and near2 for the right rule. These are numerical observations for this particular example. They are not certified error intervals or a proof that every observable has the same leading coefficient. Special endpoints may cancel a leading term.

Only the input range through32 intervals is planned for full matrices. Rows through256 intervals in this analytic comparison use the scalar recurrence; a row at N=256 is not a full matrix calculation over its255 interior coordinates. Unlike the free bridge, the potential-weighted finite laws generally change their old-time marginals under refinement.

## 18. Close and integrate the endpoints to obtain a trace

To obtain a thermal partition function, first set a=b=x and then integrate over x. For ω²>0,2g−h>0, so one more Gaussian integral is finite.

$$
Z_N=\int_{\mathbb R}K_N(x,x)\,dx
=\sqrt{\frac{P}{2g-h}}
=\frac{1}{2\sinh(N\theta/2)}.
$$

The equality can be checked independently using a periodic chain with N integrated positions. For N=2, the two temporal edges connect the same pair of vertices and must both be added. In the small example Z₂²=16/17, so Z₂=4/√17. A single diagonal value K_N(0,0) would not give that trace.

At ω²=0,2g−h=0 and the free trace on the entire real line diverges. Its diagonal kernel is a positive constant across an infinite volume. Removing a constant mode, imposing a box or dividing by length changes the question and must be stated explicitly.

## 19. Relate Euclidean time to temperature

Inverse temperature β=1/(k_B T) has units of inverse energy. Thermal evolution uses τ=βℏ. Summing the oscillator's energies E_n=ℏω(n+1/2) gives the same continuous trace as integrating the Mehler diagonal.

$$
Z(\beta)=\operatorname{tr}e^{-\beta H}
=\sum_{n=0}^{\infty}e^{-\beta\hbar\omega(n+1/2)}
=\frac{1}{2\sinh(\beta\hbar\omega/2)}.
$$

The geometric sum independently checks normalization and the zero-point contribution. See [Wipf, §8.1](https://www.tpi.uni-jena.de/qfphysics/homepage/wipf/lectures/pfad/pfad8.pdf). The normalized thermal density operator is exp(−βH)/Z. Its position diagonal K_(βℏ)(x,x)/Z is a probability density in x; its off-diagonal entries are not endpoint transition probabilities.

For research, derivatives such as −∂β logZ require a declared fixed-parameter convention. At fixed N, the finite rule's δ=βℏ/N also changes with β. Substituting an effective frequency into a continuous energy formula while ignoring that dependence need not differentiate the finite partition function correctly.

## 20. Positivity does not imply endpoint mass conservation

Integrating the finite kernel over b instead of setting a=b yields another Gaussian integral.

$$
\int_{\mathbb R}K_N(b,a)\,db
=\sqrt{\frac P g}\exp\!\left[-a^2\left(g-\frac{h^2}{4g}\right)\right].
$$

For ω²>0 this is less than1: potential weighting suppresses total contribution. For the small example with a=0 it is√(32/49). In the free case P/g=1 and g−h²/(4g)=0, giving mass1.

Normalizing K_N(b,a) separately over b creates a conditional density, but that operation does not automatically preserve the original kernel's composition law. Likewise, the pinned interior law, a thermal position density and a Born measurement probability have different normalizations and meanings. Choose the object before interpreting a plot.

## 21. Reproduce a complete one-variable calculation

This standard-library exercise checks square completion, the free branch, the finite trace and the right-rule boundary shift. It uses no random sampling and does not approximate π or exp.

```python
from fractions import Fraction as F

def kernel_coefficients(n, tau, mass, hbar, omega2):
    delta = tau / n
    s, v = mass / (hbar * delta), omega2 * delta**2
    ds = [F(1), 2 + v]
    for k in range(2, n):
        ds.append((2 + v) * ds[-1] - ds[-2])
    p = s / (2 * ds[n-1])
    g = s / 2 * (1 + v / 2 - ds[n-2] / ds[n-1])
    h = s / ds[n-1]
    return p, g, h

p, g, h = kernel_coefficients(2, F(1), F(1), F(1), F(1))
q, force, constant = F(9, 2), F(2), F(9, 8)
mean, covariance = force / q, 1 / q
minimum = constant - force * mean / 2
assert (mean, covariance, minimum) == (F(4, 9), F(2, 9), F(49, 72))
assert (p, g, h) == (F(4, 9), F(49, 72), F(8, 9))
trace_squared = p / (2*g-h)
assert trace_squared == F(16, 17)
free_p, free_g, free_h = kernel_coefficients(2, F(1), F(1), F(1), F(0))
assert free_p == F(1, 2) and 2*free_g-free_h == 0
right_log_shift = -F(1, 2) * (F(1, 2) - F(0)) / 2
assert right_log_shift == -F(1, 8)
print(mean, covariance, minimum, trace_squared, right_log_shift)
```

## 22. Six prediction-led experiments

Predict before calculating. Keep exact finite outputs separate from continuous reference values and display approximations.

| Change | Prediction to check | Explanation |
|---|---|---|
| Set ω² to0 | Recover the free bridge and kernel; lose finite trace | Confinement disappears on the unbounded line |
| Increase ω² at fixed grid | Reduce every interior variance | Q gains a positive diagonal; compare the matrix inverse |
| Reverse a,b | Preserve symmetric kernel; reverse mean in time | The declared action is invariant under time reversal |
| Refine N at fixed τ,m,ℏ,ω²,a,b | Approach Mehler, but change finite marginals | Potential quadrature modifies the finite law |
| Compare right and symmetric rules off the diagonal | See the exact endpoint log correction | One endpoint receives an extra half potential |
| Set a=b, then integrate a | Obtain trace, not a single pinned kernel | Closing and integrating adds one continuous coordinate |

For a visualization, show eight stages: free steps, potential weights, pinned matrix, square completion, fluctuation determinant, continuous comparison, boundary-rule comparison and closed trace. A path overlay should identify its mean and pointwise uncertainty. A time slider changes the view; it is not another integration calculation.

## 23. Worked questions

1. **For N=2, how many random coordinates are integrated before taking a trace?** One. The endpoints are fixed. Taking the trace adds a common endpoint coordinate and integrates it as well.

2. **Why does a positive harmonic potential change the free kernel's total endpoint mass?** Each path receives an additional factor at most1. Summing these contributions leaves less mass. A positive integrand alone does not impose a mass-one normalization.

3. **Why is μ₁=4/9 rather than1/2 in the small example?** The free kinetic penalty favors the midpoint, but the potential also penalizes the interior position. Solving (9/2)μ₁=2 gives4/9. The endpoint half potentials affect the kernel normalization, not this interior forcing equation.

4. **Does exact Fraction arithmetic make K₂ equal to the target continuous kernel?** No. It gives the finite rule exactly. The finite rule still differs from the target Hamiltonian's propagation. Arithmetic error and discretization error are separate.

5. **Is P=4/9 the full Gaussian prefactor?** No. The prefactor is√(P/π)=2/(3√π). P is the squared prefactor with π⁻¹ removed, allowing its rational part to remain exact.

6. **Why is a=b a weak test of the right-rule convergence order?** V(a)=V(b), so its endpoint correction is exactly1. Both rules give the same diagonal kernel and trace even though their general off-diagonal errors differ.

7. **Can the free trace be repaired by silently removing a zero mode?** No. Infinite translational volume is a physical part of the stated unbounded system. Fixing a mean coordinate or imposing a box defines a different object and needs its own normalization.

8. **Does a ratio near4 after doubling N certify every plot's error?** No. It is a particular asymptotic observation for a chosen observable and parameters. Leading-term cancellation, roundoff, endpoint changes and a different observable can change what is observed.

## 24. Use the result carefully in AI research

Path-based generative models, trajectory optimization and energy-based learning often combine a kinetic regularizer with a learned position cost. The oscillator is a controlled benchmark for deciding whether an implementation preserves normalization, conditions on endpoints consistently and distinguishes a finite grid from the intended continuous model.

Record the target observable, fixed physical parameters, potential quadrature and normalization before comparing methods. Use exact small cases to test code and independent continuous references to test discretization. Do not infer correctness from an attractive trajectory, an apparently stable loss or a single diagonal kernel.

The finite derivation checker independently reconstructs pinned and periodic incidence matrices, rational inverses, determinants and direct actions. Its80-digit comparisons are numerical diagnostics, not interval certificates. General Feynman–Kac hypotheses, saddle/non-Gaussian approximations and real-time phases remain separate treatments. For background on Gaussian path integration and the distinction between finite rules, see [Grosche, §III.1–III.2](https://arxiv.org/pdf/hep-th/9302097).
