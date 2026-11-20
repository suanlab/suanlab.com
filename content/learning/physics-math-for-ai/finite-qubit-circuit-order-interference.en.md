## 1. A circuit is an ordered computation

A circuit specifies which operation acts, on which wire, and when. Listing the gates is insufficient: an order change can change the output even when every gate is valid. This is useful preparation for quantum machine learning, where a trainable circuit is a composition of operations, and for debugging simulators, where a wire-order mistake can look like a physics mistake.

Assume complex vectors, tensor products and unitary matrices. We first use ideal pure states. Noise and measurement require the channel formulation in the following units. The finite experiment uses one to three qubits so every amplitude can be inspected. The general definitions have no three-qubit restriction.

For gates A then B, the state is B A|ψ〉. Circuit diagrams usually draw time from left to right; matrix products act from right to left. Always translate the verbal instruction before multiplying.

$$
|\psi_k\rangle=U_k|\psi_{k-1}\rangle,\qquad
|\psi_m\rangle=U_m\cdots U_2U_1|\psi_0\rangle.
$$

An analogy is successive rotations of a physical object. Each rotation is reversible, yet exchanging two rotations can change the final orientation. The analogy explains composition order; quantum amplitudes additionally carry relative phases.

## 2. Interference requires bringing paths back together

The Hadamard gate splits a basis state into two amplitudes. Applying a phase gate changes how those amplitudes recombine under another Hadamard. A probability-only description of the intermediate state misses this information.

$$
H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
H|0\rangle=\frac{|0\rangle+|1\rangle}{\sqrt2}.
$$

Follow H, Z, H on |0〉. The first H gives the plus state. Z changes it to the minus state. The last H gives |1〉. In contrast, Z, H, H returns |0〉. The same gate multiset produces different outputs. Check the identity HZH=X, not merely the final probabilities for one input.

For T=diag(1,e^{iπ/4}), HTH leaves a continuous-looking interference probability even though this particular phase is fixed:

$$
HTH|0\rangle=\frac{1+e^{i\pi/4}}2|0\rangle+
\frac{1-e^{i\pi/4}}2|1\rangle,\qquad
p(1)=\frac{2-\sqrt2}{4}.
$$

Both intermediate computational-basis probabilities after H and TH are1/2. The final probabilities differ because the last H converts relative phase into a population difference. Predict the result before calculating it. Removing the last H removes this particular readout of phase, not the phase itself.

## 3. Wire labels determine tensor factors

We use |b₀b₁b₂〉 with site0 the most significant bit. Thus |101〉 has coordinate index5, X₀=X⊗I⊗I, and X₂=I⊗I⊗X. A library with a different endian convention can be correct; its coordinates must be converted before comparison.

CX with control0 and target1 maps |10〉 to |11〉 and leaves |01〉 fixed. Reversing control and target changes both facts. CZ multiplies |11〉 by−1 and leaves the other two basis vectors fixed. SWAP exchanges two wire labels; it does not introduce the antisymmetric sign of a fermionic mode exchange.

Starting from |00〉, H on wire0 followed by CX₀→₁ produces (|00〉+|11〉)/√2. Each local computational outcome is uniform, while the two outcomes are perfectly correlated. A product of local marginals cannot reconstruct that joint state. Inspect joint amplitudes rather than interpreting two independent probability bars as the full result.

## 4. Global phase and controlled implementations

Multiplying a whole state by e^{iα} leaves its density matrix unchanged. Multiplying only one branch generally changes interference. A circuit simulator should distinguish exact amplitude equality from equality up to a common phase.

There is a further subtlety: U and e^{iα}U induce the same channel on a system, but controlled-U and controlled-(e^{iα}U) can differ on the control. For U=I and α=π, applying the latter to |+〉⊗|0〉 changes the control to |−〉. Therefore a system channel alone does not specify its coherent controlled implementation. This matters when comparing circuit transformations and proposed learning architectures.

## 5. Verification and cost

The reverse circuit applies adjoints in reverse chronological order. H, X, Y, Z, CX, CZ and SWAP are self-inverse; S and T require S† and T†. Undoing gates in their original order is generally wrong. Compare the recovered complex amplitudes as well as the norm.

The calculation kernel keeps coefficients in Q(√2,i), including H and T, and returns raw amplitudes with their squared norm. If the input is unnormalized, probabilities require division by that norm. A zero vector has no normalized state and must be rejected. Exact algebra avoids declaring tiny residuals zero, but does not avoid exponential state size: n qubits have2ⁿ amplitudes. Small simulation success is not a scalability claim.

The independent reference instead constructs full embedded matrices. Agreement of two different calculation paths, inverse recovery, and every computational basis action are stronger evidence than checking norm preservation alone: applying the wrong unitary also preserves norm.

## 6. Read probabilities, local states and joint correlations together

Start with any nonzero raw vector z. Its squared norm N determines how to interpret the entries. Raw amplitudes are not already normalized merely because they were produced by a circuit:

$$
N=z^\dagger z,\qquad p_b=\frac{|z_b|^2}{N},\qquad
\rho=\frac{zz^\dagger}{N},\qquad \rho_A=\operatorname{Tr}_{\bar A}\rho.
$$

For z=(1,i), N=2 and the off-diagonal entries of the density matrix are −i/2 and i/2. Its computational probabilities are1/2 and1/2, but it is not the maximally mixed state. The off-diagonal entries carry information that those two probabilities omit.

A single-qubit reduced state can be written in Bloch coordinates. These are expectations, not three independent probabilities:

$$
\rho_i=\frac{I+r_xX+r_yY+r_zZ}{2},\qquad
r_a=\operatorname{Tr}(\rho_i\sigma_a),\qquad
\operatorname{Tr}(\rho_i^2)=\frac{1+\|r\|^2}{2}.
$$

The z=(1,i) example has r=(0,1,0). Checking the sign of Y is a useful way to catch an incorrect complex conjugation. A point on the Bloch sphere is pure; a reduced state inside the ball can be mixed even when the full state is pure. In a Bell state, each Bloch vector is0 and local purity is1/2, while joint purity is1. Treating those local vectors as independent pure states would lose the entanglement.

Two-site expectations show some of the missing joint information. Connected correlations subtract the part explained by the individual means:

$$
C_{ab}^{ij}=\langle\sigma_i^a\sigma_j^b\rangle,
\qquad C_{ab}^{ij,\mathrm{conn}}=
C_{ab}^{ij}-\langle\sigma_i^a\rangle\langle\sigma_j^b\rangle.
$$

For the plus Bell state, XX=1, YY=−1 and ZZ=1, with zero individual means. A product input can instead have a nonzero two-site expectation and zero connected correlation. For example, |+i〉⊗|+〉 has Y₀X₁=1, entirely explained by Y₀=1 and X₁=1. Nonzero raw correlation alone is not an entanglement certificate.

Even all two-site correlations need not specify a three-qubit state. The GHZ state and the equal classical mixture of |000〉 and |111〉 share their one- and two-qubit reduced states. Their three-site XXX expectation differs:1 for GHZ,0 for the mixture. Retain the full density matrix when deciding whether two joint states agree.

Finally, specify the fidelity convention. Here the squared pure-state overlap is

$$
F_2(z,w)=\frac{|z^\dagger w|^2}{(z^\dagger z)(w^\dagger w)}.
$$

It equals1 for states differing only by a global phase, and1/2 between |0〉 and |+〉. Some texts call the square root of this quantity fidelity. Compare formulas before comparing reported numbers. Background on reduced states and the root-fidelity convention: [IBM density matrices](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/density-matrices), [IBM purifications and fidelity](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/purifications-and-fidelity).

## Research extension: differentiate a circuit without confusing gates and parameters

A trainable quantum model needs a derivative of an expectation, not just a valid forward state. Fix the input, observable and every other parameter. Write a single differentiated gate between two fixed unitaries. Differentiating the two exponentials gives a commutator; its factor i makes the expectation real for Hermitian G and O. Angles are dimensionless radians.

$$
U(\theta)=W e^{-i\theta G}V,\quad G=G^\dagger,\quad |\phi\rangle=V|\psi\rangle,\quad A=W^\dagger O W,\quad
 f(\theta)=\langle\phi|e^{i\theta G}Ae^{-i\theta G}|\phi\rangle,\quad
 f'(\theta)=i\langle\phi|e^{i\theta G}[G,A]e^{-i\theta G}|\phi\rangle.
$$

### Two distinct generator eigenvalues give one frequency

Subtracting cI from the generator changes only a global phase in this ordinary, uncontrolled expectation. The remaining generator is rP with P squared equal to the identity. Expanding exp(−i theta rP) into cosine and sine gives a constant plus a single frequency 2r. Subtract the two shifted trigonometric expressions to obtain an exact derivative. For Ry the generator is Y/2, r=1/2, the shift is pi/2 and the coefficient is 1/2. It is not a small-step finite difference.

$$
G=cI+rP,\quad P^2=I,\quad r>0:\qquad
 f(\theta)=a+b\cos(2r\theta)+d\sin(2r\theta),\quad
 f'(\theta)=r\left[f\left(\theta+\frac\pi{4r}\right)-f\left(\theta-\frac\pi{4r}\right)\right].
$$

### Differentiate an explicitly ordered entangling circuit

Starting from 00, rotate wire0 by theta, apply CX with control0, then rotate wire1 by phi. Measure Z on wire1. The first two gates give cos(theta/2)|00>+sin(theta/2)|11>; the final local rotation converts its Z expectation as below. The same gate order and tensor labels are used in the independent matrix calculation.

$$
|\psi(\theta,\varphi)\rangle=(I\otimes R_y(\varphi))\,\mathrm{CX}_{0\to1}\,(R_y(\theta)\otimes I)|00\rangle,\quad
 R_y(t)=e^{-itY/2},\quad O=I\otimes Z,\quad
 f=\cos\theta\cos\varphi,\quad
 \partial_\theta f=-\sin\theta\cos\varphi,\quad\partial_\varphi f=-\cos\theta\sin\varphi.
$$

### Sharing a parameter and processing a loss change the problem

If theta and phi are tied to the same variable t, the derivative is the sum of occurrence-wise partial derivatives. Shifting both occurrences together by pi/2 gives zero and misses the derivative. A nonlinear classical loss also needs its chain rule: even f=cos(theta) with L=f squared has L derivative −1 at pi/4, while directly shifting the two loss values gives zero. Exact rules for expectations do not automatically transfer to arbitrary functions of expectations.

$$
g(t)=f(t,t)=\cos^2t,\quad g'(t)=\partial_\theta f(t,t)+\partial_\varphi f(t,t)=-\sin(2t),\quad
 \frac{g(t+\pi/2)-g(t-\pi/2)}2=0;\qquad
 L=(f-y)^2:\quad\partial_\theta L=2(f-y)\partial_\theta f.
$$

### An exact formula can still require many measurements

For independent shots measuring a Pauli observable, each outcome is plus or minus one. Its sample mean has variance (1−f squared)/m. Independent estimates at the two shifted circuits therefore give the stated gradient variance. The allocation formula minimizes it over positive real shot counts when both variances are positive and known. Hardware counts are integers; unknown variances require pilot estimation. Zero-variance endpoints and rounding need separate treatment. Parameter-independent noise can be absorbed into fixed preparation/readout maps; parameter-dependent noise need not obey this ideal rule.

$$
\widehat{\partial_\theta f}=(\bar O_+-\bar O_-)/2,\quad O^2=I,\quad
 \operatorname{Var}(\widehat{\partial_\theta f})=\frac{1-f_+^2}{4m_+}+\frac{1-f_-^2}{4m_-},\quad
 m_++m_-=M,\quad \sigma_\pm=\sqrt{1-f_\pm^2},\quad
 m_\pm=\frac{M\sigma_\pm}{\sigma_++\sigma_-},\quad
 \operatorname{Var}_{\min}=\frac{(\sigma_++\sigma_-)^2}{4M}.
$$

### A controlled rotation has another generator spectrum

The controlled Ry gate has three distinct generator eigenvalues, hence two possible nonzero frequency magnitudes, 1/2 and 1. Prepare control plus and target zero, then read X on the control. The target-branch overlap gives cos(theta/2). Applying the ordinary Ry rule produces a factor 1/sqrt(2) where 1/2 is required. A controlled implementation is a different differentiated gate.

$$
C R_y(\theta)=e^{-i\theta G_c},\quad G_c=|1\rangle\langle1|\otimes Y/2,\quad
 \operatorname{spec}G_c=\{0,-1/2,1/2\},\quad
 |\psi_0\rangle=|+0\rangle,\quad O=X\otimes I:\qquad
 f_c(\theta)=\cos(\theta/2),\quad f_c'=-\tfrac12\sin(\theta/2),\quad
 \tfrac12[f_c(\theta+\pi/2)-f_c(\theta-\pi/2)]=-\tfrac1{\sqrt2}\sin(\theta/2).
$$

### Recover a rule from the actual frequency support

For a single occurrence of this controlled gate between fixed operations, all expectation frequencies are generator eigenvalue differences. Solve two linear equations so the difference formula differentiates both frequency 1/2 and frequency 1 exactly. This gives four circuit evaluations with signed weights. It is specific to this frequency set, not a universal prescription for every controlled gate or reused parameter.

$$
D_s f=f(\theta+s)-f(\theta-s),\quad
 f\in\operatorname{span}_{\mathbb R}\{1,\cos(\theta/2),\sin(\theta/2),\cos\theta,\sin\theta\}:\qquad
 f'(\theta)=\tfrac12D_{\pi/2}f+\frac{1-\sqrt2}{4}D_\pi f.
$$

### Use rational angles to check exact tables

Fix tan(phi/2)=1/2 and parameterize theta by t=tan(theta/2). Sines, cosines, both partial derivatives and the variance of the independent 100-plus-100-shot estimator are rational. A zero gradient is not a statement that its estimator has zero variance. At t=0 the derivative vanishes but the variance is 1/200; more shots reduce uncertainty rather than changing the ideal derivative.

$$
t=\tan(\theta/2),\quad\tan(\varphi/2)=1/2,\quad
 \cos\theta=\frac{1-t^2}{1+t^2},\quad\sin\theta=\frac{2t}{1+t^2},\quad
 \cos\varphi=3/5,\quad\sin\varphi=4/5,\quad
 f_\pm=\pm\partial_\theta f,\quad m_+=m_-=100,\quad
 \operatorname{Var}(\widehat{\partial_\theta f})=\frac{1-(\partial_\theta f)^2}{200}.
$$

```python
from fractions import Fraction as F
for t in [F(0),F(1,3),F(1,2),F(1)]:
    cos=(1-t*t)/(1+t*t)
    sin=2*t/(1+t*t)
    f=F(3,5)*cos
    dtheta=-F(3,5)*sin
    dphi=-F(4,5)*cos
    variance=(1-dtheta*dtheta)/200
    print(t,f,dtheta,dphi,variance)
```

| t | f | partial theta | partial phi | variance, M=200 |
| --- | --- | --- | --- | --- |
| 0 | 3/5 | 0 | -4/5 | 1/200 |
| 1/3 | 12/25 | -9/25 | -16/25 | 68/15625 |
| 1/2 | 9/25 | -12/25 | -12/25 | 481/125000 |
| 1 | 0 | -3/5 | 0 | 2/625 |

![Ordinary two-shift differentiation fails for controlled Ry, while a four-evaluation frequency rule agrees with the analytic derivative; shot variance decreases with measurement count.](/assets/learning/quantum-compression-gradient-research-parameter-shift-frequency.en.svg)

The upper panel concerns control-X after controlled Ry, not the ordinary rotation. The lower panel shows exact estimator variances for the stated independent, equally allocated Pauli shots at fixed phi. These are analytic benchmarks; the reader keeps its existing exact H/S/T circuit controls and does not add a continuous optimizer or hardware sampler.

#### Research check 1: The shared parameter needs two partials

At theta=phi=pi/4, each partial is −1/2, hence the tied derivative is −1. Moving both angles simultaneously by plus/minus pi/2 gives the same value 1/2 in both circuits and a false zero derivative.

#### Research check 2: Controlled is not ordinary

At theta=pi, the control-X expectation of controlled Ry has derivative −1/2. The ordinary rule returns −1/sqrt(2). The four-evaluation formula returns −1/2 because it was fitted to both allowed nonzero frequencies.

#### Research check 3: Zero gradient still has shot noise

At theta=0 with cos(phi)=3/5, the two shifted Z expectations vanish. With 100 shots each their independent means each have variance 1/100; dividing their difference by two yields variance 1/200. Exact zero expectation does not make individual Pauli outcomes deterministic.

Background: [Schuld et al., Evaluating analytic gradients on quantum hardware, §II](https://arxiv.org/pdf/1811.11184), the two-eigenvalue parameter-shift rule. The shared-variable counterexample, controlled-frequency formula and shot calculations above are explicitly derived finite examples, not a claim about training scalability or a barren-plateau diagnosis.

## 7. Exercises with explanations

**Predict:** Does Z change the computational probabilities of |+〉? No. It changes the relative sign, which an X-basis measurement can detect. An H before a computational measurement implements that readout.

**Calculate:** Apply CX with control1 and target0 to |01〉. The control is1, so the output is |11〉. With control0 and target1 it would remain |01〉. This checks both bit order and the meaning of control.

**Debug:** A circuit returns the correct norm and the wrong Bell correlations. Is normalization the likely fix? No. Check the initial tensor ordering, target/control indices and gate chronology. A wrong unitary can preserve every norm while computing the wrong state.

## 8. Connections and sources

Parameterized circuits add differentiable gates and objectives; noise changes state vectors into density operators. The next unit asks which noisy updates are physically meaningful. Read the earlier unitary and entanglement lessons before treating local statistics as a complete circuit description. Representation background: [IBM Quantum Learning](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels). The exact HTH and wire-order calculations here are local worked examples.
