## 1. Many arrows can cancel a large sum

Two arrows of equal length pointing in opposite directions add to zero. A real-time path integral adds complex arrows whose directions depend on the action. It does not make each path small. Rapid changes of direction among neighboring paths cause cancellation. Near a stationary path, the first change of phase vanishes and a more coherent contribution can survive.

The preceding Euclidean chapter accumulated positive weight near low action. Here negative curvature also contributes a phase. “Find the most probable path” is therefore insufficient. We assume Gaussian integration, matrices, complex numbers and differentiation, and begin on a finite grid. This chapter does not prove existence of a general nonlinear path integral.

## 2. Distinguish phase, propagation and probability

Each phase has modulus one, although their sum may vanish. Apply the propagator to a prepared wavefunction before taking a modulus squared to obtain a position probability.

| Object | Meaning | Caution |
|---|---|---|
| exp(iS/ℏ) | Complex phase of a path | Not a positive probability |
| K(b,a) | Kernel transporting initial amplitudes | Its squared modulus alone is not a general transition probability |
| ψ(b) | Propagated prepared wavefunction | Initial positions contribute interfering amplitudes |
| abs(ψ(b))² | Normalized position density | Defines an actual probability covariance |
| iℏH⁻¹ | Connected quadratic oscillatory response | Not a position probability covariance |

$$
\psi_\tau(b)=\int K_\tau(b,a)\psi_0(a)\,da,\qquad
\rho_\tau(b)=|\psi_\tau(b)|^2,\qquad \int\rho_\tau(b)\,db=1.
$$

## 3. Integration by parts explains nonstationary cancellation

Start with one coordinate rather than a path. Let g be smooth and compactly supported, and let the real phase derivative never vanish on that support. Differentiating the exponential produces φ′/ℏ; divide by that factor and integrate by parts. The boundary term vanishes, giving the identity below.

If the resulting integral is finite, it bounds the answer by a constant times ℏ. With sufficient smoothness and a persistent gradient lower bound, repetition gives any fixed power ℏᵏ. The integrand has not become small; its contributions cancel. Boundaries or a vanishing phase derivative require separate treatment.

$$
I(\hbar)=\int g(x)e^{i\phi(x)/\hbar}\,dx
=-\frac{\hbar}{i}\int\left(\frac{g}{\phi'}\right)'e^{i\phi/\hbar}\,dx,
\qquad |I(\hbar)|\leq\hbar\int\left|\left(\frac{g}{\phi'}\right)'\right|dx.
$$

## 4. Quadratic phase near a stationary point

At a stationary point xⱼ the linear term vanishes. Rescaling x=xⱼ+√ℏy removes ℏ from the quadratic phase and contributes a volume factor ℏᵈᐟ². Each positive curvature direction contributes a Fresnel phase +π/4 and each negative direction contributes −π/4.

For a fixed smooth compact amplitude and phase, with finitely many separated nondegenerate stationary points on its support, the following leading formula holds. The signature σ counts positive minus negative eigenvalues. Remainder constants depend on phase derivatives and curvature bounds. They need not remain bounded when stationary points merge and a determinant vanishes. Nor does g(xⱼ)=0 make the entire integral exactly zero.

$$
\int_{\mathbb R^d}g(x)e^{i\phi(x)/\hbar}\,d^dx
=(2\pi\hbar)^{d/2}\sum_j\frac{g(x_j)}{\sqrt{|\det H_j|}}
 e^{i\phi(x_j)/\hbar+i\pi\sigma_j/4}
+O(\hbar^{d/2+1}),\qquad H_j=\nabla^2\phi(x_j).
$$

## 5. Specify the finite real-time action including endpoints

Divide time τ into N≥2 intervals, set δ=τ/N and d=N−1, and fix x₀=a and x_N=b. The parameters m,τ,ℏ are positive, while ω² is a signed real parameter. A negative value describes an inverted oscillator, not an imaginary input.

Real-time action subtracts potential energy from kinetic energy, unlike the Euclidean sum. Endpoint half weights disappear from interior derivatives but remain in the action and phase. Both S and ℏ have units of action; v=ω²δ² is dimensionless.

$$
V(x)=\frac{m\omega^2x^2}{2},\qquad
S_N=\frac{m}{2\delta}\sum_{i=0}^{N-1}(x_{i+1}-x_i)^2
-\delta\left[\frac{V(a)}2+\sum_{i=1}^{N-1}V(x_i)+\frac{V(b)}2\right].
$$

## 6. Finite-step unitarity and continuum accuracy differ

Fix the positive-time free square root with branch exp(−iπ/4). The integral has an oscillatory or Abel meaning, not the meaning of a positive Lebesgue probability integral. Each time step consists of a potential half step, free propagation and another potential half step. Real potential phases and free propagation each preserve the L² norm, so their finite product is unitary.

Unitarity does not make this product the exact time-τ evolution of the original Hamiltonian. Time discretization accuracy is separate. The kernel has units of inverse length, and a finite pointwise value need not exist at every endpoint.

$$
K_N(b,a)=\left(\frac{m}{2\pi i\hbar\delta}\right)^{N/2}
\int_{\mathbb R^d}e^{iS_N/\hbar}\,d^dq,\qquad
U_\delta=e^{-i\delta V/(2\hbar)}e^{-i\delta\hat p^2/(2m\hbar)}e^{-i\delta V/(2\hbar)}.
$$

## 7. Derive the Hessian from two neighboring edges

An interior coordinate receives one derivative from each adjacent kinetic edge. With s=m/δ and v=ω²δ², the Hessian diagonal is s(2−v) and neighboring entries are −s. Interior coordinates here are indexed 1 through d.

At N=2 there is only one interior coordinate: a and b add to the same forcing component. Including endpoints as additional unknown Hessian coordinates would solve a different problem.

$$
S_N(q)=\frac12q^THq-f^Tq+c,\qquad
H_{ii}=s(2-v),\quad H_{i,i+1}=-s,\quad f=s(ae_1+be_d),
\qquad c=\frac{s(a^2+b^2)}2-\frac{\delta m\omega^2(a^2+b^2)}4.
$$

## 8. A stationary path need not minimize the action

The equation Hq=f is the discrete equation of motion. If H is invertible the stationary path is unique, yet the action decreases along every negative eigenvalue direction. Calling this path a minimum or a most probable path is incorrect.

Writing q=q*+y cancels the linear term and separates stationary action from fluctuations. There is no Taylor remainder in this quadratic model, which is why its finite Gaussian integral is exact.

$$
q^*=H^{-1}f,\qquad S_*=c-\frac12f^TH^{-1}f,\qquad
S_N(q^*+y)=S_*+\frac12y^THy.
$$

## 9. A determinant recurrence gives every inverse entry

Expanding the last row of the tridiagonal determinant gives a continuant recurrence. The value θ₀=1 is the determinant of the empty matrix. The sequence does more than compute one determinant: combining left and right minors yields every Green response entry.

The inverse formula below uses indices 0 through d−1. It is unavailable when θ_d=0. An intermediate zero θ does not, however, imply that the final determinant vanishes.

$$
\theta_0=1,\quad\theta_1=2-v,\quad
\theta_k=(2-v)\theta_{k-1}-\theta_{k-2},\qquad
\det H=s^d\theta_d,\qquad
(H^{-1})_{ij}=\frac{\theta_{\min(i,j)}\theta_{d-1-\max(i,j)}}{s\theta_d}.
$$

## 10. Negative modes change phase rather than disappear

Orthogonally diagonalize the real symmetric H and integrate one Fresnel factor per mode. Positive and negative eigenvalues contribute the same magnitude but different phases. If ν counts negative eigenvalues, then σ=d−2ν.

Deleting modes or retaining only absolute curvature changes interference. Our mathematical check compares Sturm sign changes with a separate 1×1/2×2 congruence calculation. Failure of a nonpivoted LDL step at a zero first pivot is not a singularity test.

$$
\int_{\mathbb R^d}e^{iy^THy/(2\hbar)}\,d^dy
=\frac{(2\pi\hbar)^{d/2}}{\sqrt{|\det H|}}e^{i\pi\sigma/4},
\qquad \sigma=d-2\nu,\qquad \det H\ne0.
$$

## 11. Combine free-step normalization with the Maslov phase

The N free normalizations contribute −Nπ/4, and the d=N−1 internal Fresnel modes contribute π(d−2ν)/4. Their sum is −π/4−πν/2. Each additional negative mode supplies a phase −π/2.

Determinant magnitude determines amplitude but cannot recover ν. The formula below is the exact nonsingular finite quadratic kernel. It must use the same endpoint convention and normalization as the computed action.

$$
K_N(b,a)=\sqrt{\frac{s}{2\pi\hbar|\theta_d|}}
\exp\left(\frac{iS_*}{\hbar}-\frac{i\pi}{4}-\frac{i\pi\nu}{2}\right).
$$

## 12. Work a negative-curvature N=2 example by hand

Choose δ=m=1, N=2, a=0, b=1 and ω²=3. Then H=−1, f=1 and c=−1/4, so q*=−1 and S*=1/4. The internal stationary point is a maximum, yet the oscillatory integral has an Abel definition.

Since ν=1 and θ₁=−1, the full phase is 1/(4ℏ)−3π/4. Using determinant magnitude alone and retaining the free phase −π/4 misses π/2. Positive ω² produces negative interior curvature here because the time grid is coarse; this is different from a physical inverted oscillator.

## 13. An invertible matrix with a zero first pivot

For N=3, δ=m=1 and v=2, the Hessian has zero diagonal and off-diagonal entries −1. Its determinant is −1, its inverse is itself, and its eigenvalues are +1 and −1. The continuant sequence is 1,0,−1, with one sign change after zeros are omitted.

Taking a=1,b=0 gives f=(1,0), q*=(0,−1) and c=S*=0. Only a nonpivoted elimination algorithm has failed. The endpoint problem and kernel remain nonsingular. An algorithmic denominator is not automatically a physical singularity.

$$
H=\begin{pmatrix}0&-1\\-1&0\end{pmatrix},\qquad
H^{-1}=H,\qquad \det H=-1,\qquad \nu=1.
$$

## 14. At an exact caustic the kernel is a distribution

With N=2, δ=m=1 and v=2, H=0. If a+b=0, every interior q is stationary and its action is zero. If a+b differs from zero, no stationary path exists. Adding a small diagonal value to invent an inverse changes both cases.

At a general exact caustic this irreducible tridiagonal Hessian has nullity one. Its null vector is r=(θ₀,…,θ_{d−1}), and compatibility requires rᵀf=0. Below M denotes the classical propagation matrix and k=ν+1. The N2,v2 case has M=−I,k=1 and K=−iδ(b+a). Here δ is a Dirac distribution, distinct from the time-step notation. The formula does not return a pointwise infinite probability.

$$
\theta_d=0:\qquad M=(-1)^kI,\quad k=\nu+1,\qquad
K_N(b,a)=e^{-i\pi k/2}\,\delta\bigl(b-(-1)^ka\bigr).
$$

## 15. An Abel regulator helps define the integral but changes finite propagation

Multiplying the interior integrand by exp(−εqᵀq/(2ℏ)) makes an absolutely integrable complex Gaussian for ε>0. The real part of Cε=εI−iH is positive definite. Completing the square gives the following result. Iε excludes the free-step prefactor.

Its complex mean is −iCε⁻¹f and its connected response is ℏCε⁻¹. As ε approaches zero, they tend to q* and iℏH⁻¹. These are algebraic responses under a complex weight, not Born probability moments. Finite ε introduces damping and is not the original unitary propagation.

$$
C_\epsilon=\epsilon I-iH,\qquad
I_\epsilon=(2\pi\hbar)^{d/2}\prod_{j=1}^d\alpha_j^{-1/2}
\exp\left(\frac{ic}{\hbar}-\frac{f^TC_\epsilon^{-1}f}{2\hbar}\right),\qquad
\alpha_1=\epsilon-iH_{11},\quad
\alpha_j=\epsilon-iH_{jj}+\frac{s^2}{\alpha_{j-1}}.
$$

## 16. One principal determinant square root loses branch history

Each complex pivot αⱼ has positive real part. This holds for the first pivot and propagates because the reciprocal of a number with positive real part also has positive real part. Choose each square root with positive real part and retain their product.

The N=4 free Hessian has three positive eigenvalues. For small ε, its three pivots each have argument near −π/2. Their square roots multiply to an argument near −3π/4, whereas taking the principal argument of the determinant first gives +π/4. The answers differ by a sign. Exact action arithmetic cannot repair a discarded branch.

## 17. Derive the classical symplectic map for the same splitting

A potential half kick changes p by −mω²δq/2, and the free drift changes q by δp/m. Multiplying these operations gives the step matrix below. Its determinant is one, preserving phase-space area.

Write its Nth power with entries A₀,B₀,C₀,D₀, using subscripts to distinguish it from Hessian notation. The identity B₀=(δ/m)θ_d links the classical map caustic to the Hessian determinant. For B₀≠0, endpoint derivatives of the generating action give the initial and final momenta. The denominator B₀ must retain its sign.

$$
M_\delta=\begin{pmatrix}1-v/2&\delta/m\\-m\omega^2\delta(1-v/4)&1-v/2\end{pmatrix},\quad
M=M_\delta^N=\begin{pmatrix}A_0&B_0\\C_0&D_0\end{pmatrix},\quad
B_0=\frac{\delta}{m}\theta_d,\quad
S_* =\frac{D_0b^2+A_0a^2-2ab}{2B_0}.
$$

## 18. Refining time is a different experiment from reducing ℏ

Fix τ,m,ω² and compare the maps at N,2N,4N,8N. The free model gives [[1,τ/m],[0,1]] at every refinement. For positive ω², the continuum map has diagonal cos(ωτ), upper-right sin(ωτ)/(mω) and lower-left −mωsin(ωτ). For negative ω², setting κ=√(−ω²) gives the corresponding hyperbolic cosh/sinh map.

A coarse positive-oscillator step with v>4 can itself have hyperbolic eigenvalues. That numerical effect differs from physical ω²<0. Reducing ℏ alone does not improve the discrete map. Near a caustic, relative kernel errors are particularly sensitive, so compare map errors alongside signature and distributional classification.

## 19. A normalized Gaussian packet with chirp

A pointwise singular kernel can still act meaningfully on a normalized state. Specify initial position mean q₀, momentum mean p₀, position variance σ²>0 and real chirp χ. Chirp describes the slope of mean momentum with position. It leaves the initial position density unchanged while changing position-momentum correlation.

The prefactor below normalizes the integral of |ψ|² to one. Chirp has units of momentum per length. The standard deviation is σ; its square is the variance.

$$
\psi_0(x)=(2\pi\sigma^2)^{-1/4}
\exp\left[-\frac{(x-q_0)^2}{4\sigma^2}
+\frac{ip_0(x-q_0)}{\hbar}+\frac{i\chi(x-q_0)^2}{2\hbar}\right].
$$

## 20. Transport the mean and the full covariance

Applying the momentum operator −iℏ∂ₓ to the initial Gaussian gives a real term χ(x−q₀) and an imaginary width term. Averaging yields the symmetric covariance below; its cross entry uses symmetrized operator order.

Quadratic evolution transports both the mean and covariance with the same classical map. The determinant stays ℏ²/4, preserving minimum uncertainty area. This is a covariance of a normalized state, unlike the earlier complex response iℏH⁻¹.

$$
\Sigma_0=\begin{pmatrix}\sigma^2&\chi\sigma^2\\
\chi\sigma^2&\chi^2\sigma^2+\hbar^2/(4\sigma^2)\end{pmatrix},\qquad
\binom{q_\tau}{p_\tau}=M\binom{q_0}{p_0},\qquad
\Sigma_\tau=M\Sigma_0M^T,\qquad\det\Sigma_\tau=\frac{\hbar^2}{4}.
$$

## 21. Complex width stays finite through a caustic

The real part of Z₀ is chirp and its positive imaginary part is inverse width. The Möbius rule implies Im Zτ=Im Z₀/|A₀+B₀Z₀|²>0, so the normalized position variance stays positive. At B₀=0, A₀ remains nonzero; the packet does not inherit an invented infinite kernel value.

The center action is (qτpτ−q₀p₀)/2. One can derive it from d(qp)/dt=2L for quadratic motion or by summing split kick/drift actions. When B₀≠0, the Gaussian prefactor squares to 1/(A₀+B₀Z₀), but a square does not determine its branch. Track the pivot square roots and Maslov phase. At an exact harmonic caustic, apply parity and exp(−iπk/2) directly to the state.

$$
Z_0=\chi+\frac{i\hbar}{2\sigma^2},\qquad
Z_\tau=\frac{C_0+D_0Z_0}{A_0+B_0Z_0},\qquad
\sigma_{q,\tau}^2=\frac{\hbar}{2\operatorname{Im}Z_\tau},\quad
\chi_\tau=\operatorname{Re}Z_\tau,\quad
S_{\mathrm{center}}=\frac{q_\tau p_\tau-q_0p_0}{2}.
$$

## 22. Two small-ℏ families have different classical limits

First hold σ² and χ fixed while reducing ℏ. The initial covariance generally tends to a nonzero rank-one matrix: a classical ensemble satisfying p−p₀=χ(q−q₀). A broad distribution moves classically rather than contracting to one trajectory.

Alternatively set σ²=ℏr with fixed r>0,χ and a fixed finite M. Every covariance entry is then proportional to ℏ and tends to zero, concentrating at the phase-space point followed by the mean. The units of r are σ²/ℏ, unlike the first family's width input. An arbitrary r is not necessarily the oscillator coherent-state width. Nor does this fixed-M argument automatically cover long times with M growing as ℏ decreases.

$$
\left.\Sigma_0\right|_{\sigma^2\,\mathrm{fixed}}
\longrightarrow\sigma^2\begin{pmatrix}1&\chi\\\chi&\chi^2\end{pmatrix},
\qquad
\left.\Sigma_\tau\right|_{\sigma^2=\hbar r}
=\hbar M\begin{pmatrix}r&\chi r\\\chi r&\chi^2r+1/(4r)\end{pmatrix}M^T
\longrightarrow0.
$$

## 23. Position focusing alone does not imply a single trajectory

For free propagation with τ=m=1, qτ=q₀+p₀ and pτ=p₀. With χ=−1, initial position and momentum deviations cancel at the final position. Even with fixed σ², reducing ℏ sends final position variance to zero as shown below.

Final momentum variance nevertheless tends to σ². A position plot alone can misleadingly suggest a classical point. Inspect the full covariance and determinant. This free chirp focus has B₀=1 and also differs from the harmonic kernel caustic B₀=0.

$$
M=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad\chi=-1:
\qquad\sigma_{q,\tau}^2=\frac{\hbar^2}{4\sigma^2},\quad
\sigma_{p,\tau}^2=\sigma^2+\frac{\hbar^2}{4\sigma^2},\quad
\Sigma_{qp,\tau}=\frac{\hbar^2}{4\sigma^2}.
$$

## 24. A separate fold model describes merging stationary points

Now consider the separate one-coordinate phase φ=x³/3−ux. For u>0 there are stationary points ±√u; they merge at zero and disappear from the real axis for u<0. Rescaling x=ℏ¹ᐟ³t gives the Airy expression below. These are canonical units with cubic coefficient set to one.

Treat u and ℏ as independent physical controls. The merger region has u of order ℏ²ᐟ³. This model does not evaluate a general nonlinear path integral, nor does it replace the exact harmonic delta caustic with an Airy curve.

$$
J(u,\hbar)=\int_{\mathbb R}e^{i(x^3/3-ux)/\hbar}\,dx
=2\pi\hbar^{1/3}\operatorname{Ai}\left(-\frac{u}{\hbar^{2/3}}\right).
$$

## 25. Two stationary contributions interfere and eventually fail

At x=+√u, the action is −2u³ᐟ²/3 and curvature is +2√u. Both signs reverse at the negative stationary point. Adding the two Fresnel contributions yields the cosine below. Adding their magnitudes instead would erase interference zeros.

This is a useful leading approximation as ℏ decreases at fixed u>0, but its Gaussian magnitude diverges as u approaches zero. The exact canonical integral at u=0 is finite: 2πℏ¹ᐟ³Ai(0). For u<0 there is no real stationary point, yet Ai at a positive argument is positive and exponentially small. Absence of real stationary points does not mean an exactly zero integral.

$$
J_{\mathrm{SP}}(u,\hbar)=2\sqrt{\frac{\pi\hbar}{\sqrt{u}}}
\cos\left(\frac{2u^{3/2}}{3\hbar}-\frac\pi4\right),\qquad u>0.
$$

## 26. Airy evaluation needs both central and large-argument representations

Ai satisfies y″=zy with two initial values. Its coefficients obey a₂=0 and (n+2)(n+1)a_{n+2}=a_{n−1}, leaving two series with powers spaced by three. Once absolute term ratios decrease below one, a geometric tail bound becomes available.

For large positive z, however, two large series cancel to give a small Ai value, consuming precision. Evaluate the scaled value exp(ζ)Ai(z), with ζ=2z³ᐟ²/3, and retain the exponential scale or log magnitude separately. For large negative arguments use sine/cosine asymptotic series with their applicable remainder conditions. The Γ constants and π also require certification. A numerical implementation must also control truncation, rounding and range reduction; formulas alone do not certify a computed value.

$$
\operatorname{Ai}(z)=\sum_{n=0}^{\infty}a_nz^n,\quad
a_0=\frac1{3^{2/3}\Gamma(2/3)},\quad a_1=-\frac1{3^{1/3}\Gamma(1/3)},\quad a_2=0,\quad
(n+2)(n+1)a_{n+2}=a_{n-1}\quad(n\geq1).
$$

## 27. Observed numerical convergence is not an error certificate

Truncating a directly oscillatory integral and refining its grid does not control the infinite tail or aliasing. Numerical integration on a deformed contour also needs justified contour deformation and separate tail and rounding bounds. Small ℏ can additionally cause phase loss through subtraction of large numbers.

The mathematical foundation distinguishes 41,191 exact rational checks across 62 finite models from 13 Decimal diagnostics. The latter observe grid or branch agreement; they are not certified integration errors. The live experiment uses certified production calculations and an independent response check; the Decimal diagnostics are a separate comparison. Underflow must not be reported as exact zero, and an enclosure containing zero cannot promise relative accuracy or a finite log magnitude.

## 28. State which limit is being taken

Before comparing computations, state what is held fixed. In particular, keeping ε fixed while increasing the number of interior coordinates changes accumulated damping. We first define regulator removal at fixed finite N.

| Operation | Held fixed | What to inspect |
|---|---|---|
| Reduce ℏ | N, action and the chosen state family | Cancellation and the intended classical limit |
| Increase N | Total time τ and physical coefficients | Map, normalization and caustic classification |
| Remove ε | Finite N and endpoints | Abel branch and distributional meaning |
| Approach a caustic | Model and approach path | Pointwise kernel versus propagated state |
| Merge fold saddles | Scale u/ℏ²ᐟ³ | Uniform Airy form instead of two Gaussians |

## 29. Six prediction-first comparisons

These are comparison prompts for hand calculation, the code below and a later calculator. They do not assume that unimplemented controls or visualizations already exist.

1. **Negative modes:** Compare v=1 and3 at N2,δ=m=1,a=0,b=1. Predict stationary-point type and ν before computing H,f,c. Explain the phase missed by determinant magnitude alone.
2. **Zero pivots:** Compare N3,v=2 with N2,v=2. Both have zero first pivots; use their determinants to explain why only the first is invertible.
3. **Caustic and state:** At N2,v=2 compare b=−a with incompatible endpoints, then apply parity to a normalized Gaussian. Distinguish pointwise kernel values from state norm.
4. **Two width families:** Compare fixed σ² with σ²=ℏr under the same free map. Predict position and momentum variance changes when ℏ is halved.
5. **Chirp focus:** Set τ=m=1,σ²=2/3 and compare χ=0 with−1. Use the full covariance to assess whether final position concentration implies one trajectory.
6. **Fold merger:** Move u from positive through zero to negative. Explain the failure of separate stationary contributions and the small nonzero amplitude without real stationary points.

## 30. Check the hand calculations with the same Python code

The following code uses only standard-library Fraction arithmetic. It checks a negative-curvature stationary point, an invertible zero-pivot matrix, exact caustic compatibility and both packet families. At ℏ=1/2,1/4,1/8 the fixed-width chirp focus shrinks position variance while momentum variance approaches 2/3.

This exercise does not numerically integrate Airy values or compute the full propagated Gaussian phase. It exactly reproduces finite hand calculations and the covariance counterexample. Run it in local Python or a Colab cell.

~~~python
from fractions import Fraction as F

# N=2, delta=m=1, v=3, a=0, b=1.
H, f, c = F(-1), F(1), F(-1, 4)
q = f / H
S = c - f * q / 2
assert (q, S) == (F(-1), F(1, 4))
print(f"negative-mode q={q} S={S} nu=1")

# N=3, v=2: H=[[0,-1],[-1,0]] is its own inverse.
H2 = [[F(0), F(-1)], [F(-1), F(0)]]
identity = [[sum(H2[i][k] * H2[k][j] for k in range(2))
             for j in range(2)] for i in range(2)]
assert identity == [[1, 0], [0, 1]]
print("zero-pivot det=-1 inverse=H nu=1")

# N=2, v=2: H=0, so compatibility is a+b=0.
a = F(2, 3)
assert a + (-a) == 0 and a + F(0) != 0
print("caustic compatible=True incompatible=False")

# Free propagation at tau=m=1. Covariances are symmetrized.
def propagate(variance, chirp, hbar):
    qq = variance
    qp = chirp * variance
    pp = chirp**2 * variance + hbar**2 / (4 * variance)
    return qq + 2 * qp + pp, qp + pp, pp

for h in [F(1, 2), F(1, 4), F(1, 8)]:
    v = F(2, 3)
    qq, qp, pp = propagate(v, F(-1), h)
    assert qq == h*h / (4*v)
    assert pp == v + h*h / (4*v)
    assert qq*pp - qp*qp == h*h/4
    free_qq, _, _ = propagate(v, F(0), h)
    assert free_qq == v + h*h/(4*v)
    r = F(2, 3)
    narrow_qq, _, narrow_pp = propagate(h*r, F(0), h)
    assert narrow_qq/h == r + 1/(4*r)
    assert narrow_pp/h == 1/(4*r)
    print(f"h={h} focus_qq={qq} focus_pp={pp} "
          f"unchirped_qq={free_qq} scaled_qq={narrow_qq}")
~~~

## 31. Connect to AI optimization and uncertainty without identifying them

Laplace approximation and stationary phase both use critical points and Hessians, but their weights differ. Negative curvature at a proposed posterior mode invalidates the usual positive Gaussian minimum approximation. Real-time integration instead retains negative curvature through phase. The response iℏH⁻¹ is not Bayesian uncertainty.

For inverse problems and learned dynamics, test symplectic area preservation separately from finite-step accuracy. Even a norm-preserving model can approximate the intended dynamics poorly. Likewise, a narrow output position distribution does not establish small uncertainty throughout latent phase space, as the chirp counterexample shows.

Research must account for multiple stationary points, symmetry, degeneracy, cancellation and precision cost. The real-time sampling sign problem is related to statistical difficulty from cancellation of complex or signed contributions. This exact quadratic solution does not solve the general sign problem.

## 32. Questions with explanations

1. **Must the integral be large because every path phase has modulus one?** No. Arrows in different directions cancel. Integration by parts bounds a nonstationary integral with compact amplitude at small ℏ.
2. **Does a unique stationary path minimize the action?** No. N2,v3 has H=−1 and a unique maximum. Its negative Fresnel mode still contributes phase.
3. **Does equal absolute determinant imply equal kernels?** No. Action, forcing, negative-mode count and branch matter. One negative mode shifts the phase by −π/2.
4. **Does a zero first pivot imply no inverse?** No. The N3,v2 matrix has determinant −1. Check the full determinant and use appropriate pivot handling.
5. **Can a caustic kernel be represented by a large number?** No. The exact harmonic caustic is a delta distribution with parity. Interpret its action on a normalized state.
6. **Does every Gaussian approach one trajectory as ℏ vanishes?** No. Fixed width usually gives a classical ensemble. The σ²=ℏr family at fixed M makes the entire covariance vanish.
7. **Is vanishing final position variance sufficient?** No. The free χ=−1 counterexample retains momentum variance σ². Inspect the full phase-space covariance.
8. **Does the fold integral vanish without real stationary points?** No. For u<0 its positive-argument Ai value is positive and exponentially small. Distinguish underflow from exact zero.

## 33. Compare the four path-integral constructions

Use the same grid and endpoints to compare objects, rather than comparing pictures with different meanings. The [Brownian bridge](/book/online/physics-math-for-ai/brownian-bridge-time-slicing-measure/), [Euclidean kernel](/book/online/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing/) and [saddle approximation](/book/online/physics-math-for-ai/saddle-fluctuation-and-nongaussian-approximation/) chapters each provide an experiment. All four constructions retain the integration measure and normalization; they answer different questions.

| Construction | What is combined? | Meaning after the calculation | What must not be inferred? |
|---|---|---|---|
| Brownian bridge | Positive Gaussian increment densities, conditioned on endpoints | A normalized probability law over interior positions | The mean path is not a typical smooth trajectory |
| Euclidean kernel | Positive potential-weighted free kernels | A kernel of the declared finite operator product; conditioning gives an interior probability law | Positivity alone does not conserve endpoint mass or identify a partition function |
| Saddle approximation | Gaussian fluctuations and corrections around stationary paths | A local approximation, with a full ratio bound only under stated remainder conditions | A mode or local width is not automatically the actual mean or covariance |
| Real-time kernel | Complex amplitudes, including negative-mode and branch phases | An amplitude propagator; apply it to a normalized state before forming a Born density | A formal complex response is not a probability covariance |

**Worked comparison 1: make the free positive laws agree.** Set ω²=λ=0, use the same τ,m,ℏ,a,b,N, and choose D=ℏ/(2m) in the bridge. Here L is the pinned chain matrix and H₀=(m/δ)L. The Euclidean quadratic law and the saddle calculation at λ=0 have the same interior mean and covariance as the bridge. For real time, the stationary path agrees, but the corresponding connected response contains i:

$$
D=\frac{\hbar}{2m},\qquad Q_E=\frac{m}{\hbar\delta}L,\qquad
C_B=C_E=\hbar H_0^{-1},\qquad C_{\mathrm{osc}}=i\hbar H_0^{-1}.
$$

For N=4,τ=m=ℏ=1,a=0,b=1, both positive laws have mean (1/4,1/2,3/4) and middle variance1/4. Their free kernel is exp(−1/2)/√(2π). The real-time free kernel instead is exp(i/2−iπ/4)/√(2π), and its middle connected response is i/4. Equal magnitudes of normalization factors do not turn a complex amplitude into the same positive law. The expression i/4 is not a variance that can be estimated by unweighted Brownian samples.

**Worked comparison 2: keep the observation fixed when testing refinement.** For that bridge, midpoint variance stays1/4 when N goes from4 to8 because the old-time marginal is unchanged. The trapezoidal area is a different observable on the finer grid: its variance changes from5/64 to21/256, a difference of1/256; the limiting variance is1/12. At N=4 the mean path has dimensionless action1/2, while the expected action of the full bridge is2. A smooth minimum-action path therefore does not describe the typical fluctuations. This exact marginal consistency of the free bridge is stronger than, and different from, a convergence plot for a potential-weighted kernel.

**Worked comparison 3: identify the Hessian and its units before comparing colors.** Set N=2,τ=m=ℏ=ω²=1,a=0,b=1 and λ=0. The Euclidean precision has a plus sign for the potential; the real-action Hessian has a minus sign. Completing the square gives:

$$
Q_E=\frac92,\quad \mu_E=\frac49,\quad C_E=\frac29,\quad A_{E,\min}=\frac{49}{72};\qquad
H_R=\frac72,\quad q_R^*=\frac47,\quad i\hbar H_R^{-1}=\frac{2i}{7},\quad S_R^*=\frac{17}{56}.
$$

These are two different finite integrals, not a numerical disagreement. In general Q_E is the Hessian of S_E/ℏ, whereas H_R is the Hessian of S_R. Restoring a general ℏ changes which matrix includes the action scale. Analytic continuation requires a declared operator, contour, convergence conditions and branch prescription; changing a sign in a covariance heat map is not a proof of Wick rotation.

**Worked comparison 4: know what the non-Gaussian calculation certifies.** In the saddle chapter's N=2,τ=m=ℏ=ω²=λ=1,a=0,b=37/32 example, the stationary coordinate is1/2 and its local Hessian is39/8, giving local Gaussian variance8/39. Setting λ=0 instead gives the Gaussian mean37/72. The local variance8/39 is not the actual quartic variance, and the shifted cubic remainder prevents using the zero-saddle global ratio bound. Read the residual certificate, local covariance and ratio applicability as separate statements.

Before using any of these tools in AI research, state the object, measure, endpoints, units and observable, then state what is held fixed in a limit. A finite unitary map need not equal the exact continuous dynamics. The harmonic delta caustic and a finite canonical Airy merger also require different interpretations. These distinctions remain necessary even when all numerical checks pass.

## 34. Sources and the boundary of this manuscript

[Evans–Zworski, Lemma3.13/Theorem3.15](https://math.mit.edu/~vwg/semiclassicalEvansZworski.pdf#page=38) provides conditions for nonstationary and stationary phase. [Wipf§3.1](https://www.tpi.uni-jena.de/~wipf/lectures/pfad/pfad3.pdf) discusses finite quadratic path integrals; our symmetric endpoint convention is derived explicitly. [Horváthy§§2–3](https://arxiv.org/pdf/quant-ph/0702236#page=2) treats Maslov phases and distributional harmonic caustics.

For Airy equations and initial values, integrals, asymptotics and remainder conditions, see [DLMF9.2](https://dlmf.nist.gov/9.2), [9.5](https://dlmf.nist.gov/9.5) and [9.7](https://dlmf.nist.gov/9.7). Positivity at positive argument follows from the modified-Bessel connection in [9.6.2](https://dlmf.nist.gov/9.6#E2) and the positive integral representation in [10.32.9](https://dlmf.nist.gov/10.32#E9).

The experiments combine the declared finite models with independently checked calculations, reference diagrams and executable hand cases. Their certificates concern the specified model and numerical quantities. They do not construct a general nonlinear real-time measure or prove unrestricted continuum convergence, Wick rotation or a solution to the sign problem. When extending the model, revisit the measure, operator domain, boundary conditions, branch prescription and approximation error.
