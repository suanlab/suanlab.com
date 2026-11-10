## What does a negative correlation actually mean?

Consider two arrows, each pointing up or down. Across repeated preparations, the first arrow can fluctuate upward while the second tends to fluctuate downward. Their connected correlation is then negative. Yet the uncertainty of their sum, their difference, or any specified weighted combination cannot be negative. A matrix can describe both facts at once: its off-diagonal entries can be negative while every quadratic variance is nonnegative.

This chapter makes that distinction concrete using four named classical configurations. We will calculate their probabilities, subtract their means, derive the response to two independently controlled fields, and identify the corresponding Hessian. The goal is to understand a small complete example before interpreting large covariance heatmaps or a Fisher information matrix in an AI model.

You should know finite Boltzmann normalization, expectation and variance, and partial derivatives. The previous partition-function and finite Curie–Weiss chapters provide that background. [Tong's statistical physics notes](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html) give background on Ising systems and correlations. The four-state preparation, input ranges and examples below are defined explicitly for this chapter.

## Define two local fields and four states

There are two named spins, not two magnetization labels. Each spin is either −1 or +1. The probability law is

$$
p(\sigma_1,\sigma_2)=\frac{\exp(K\sigma_1\sigma_2+h_1\sigma_1+h_2\sigma_2)}{Z},\qquad Z=\sum_{\sigma_1,\sigma_2=\pm1}\exp(K\sigma_1\sigma_2+h_1\sigma_1+h_2\sigma_2).
$$

K, h1 and h2 are dimensionless natural parameters. In a positive-temperature physical realization, K=J/(kBT) and hi=Hi/(kBT), where J and Hi have energy units. A positive K favors aligned spins; a negative K favors opposing spins. A positive local field favors +1 at its own site. Varying a field here holds K and the other field fixed. Changing physical temperature with J and Hi fixed changes all three parameters together and is a different experiment.

The numerical model accepts finite binary64 K,h1,h2 in [−8,8]. These are computational bounds, not phase boundaries. It also accepts two observable coefficients a1,a2 in [−4,4]. Four states remain possible at every admitted input; none becomes an excluded state merely because its displayed probability is small. No infinite-coupling or zero-temperature endpoint is silently added.

## Inspect the state table before summing

The counting reference assigns one entry to each named configuration:

| State | Spin 1 | Spin 2 | Exponent |
| --- | --- | --- | --- |
| down, down | −1 | −1 | K−h1−h2 |
| down, up | −1 | +1 | −K−h1+h2 |
| up, down | +1 | −1 | −K+h1−h2 |
| up, up | +1 | +1 | K+h1+h2 |

At K=h1=h2=0 the four probabilities are each 1/4. The total spin has three possible values, −2,0,2, with probabilities 1/4,1/2,1/4. Assigning probability 1/3 to these three values would change the reference measure and produce a different model.

For hand calculations, summing the first and last states together is useful:

$$
Z=2e^K\cosh(h_1+h_2)+2e^{-K}\cosh(h_1-h_2).
$$

This formula explains the competing channels. Equal fields reinforce the aligned pair. Opposite fields favor one member of the opposing pair. In numerical evaluation, the implementation subtracts the largest exact exponent before exponentiating; it adds that exponent back when reporting log Z. A common exponent shift preserves probabilities but cannot simply disappear from a normalization calculation.

## Connected correlation removes the baseline

A raw product expectation is not yet a connected covariance. Define the site means and the covariance matrix by

$$
\mu_i=\mathbb E[\sigma_i],\qquad C_{ij}=\mathbb E[(\sigma_i-\mu_i)(\sigma_j-\mu_j)]=\mathbb E[\sigma_i\sigma_j]-\mu_i\mu_j,\qquad C_{ii}=1-\mu_i^2.
$$

Suppose two independent biased spins each have mean 1/2. Their raw product expectation is 1/4, although learning one spin provides no information about the other. Subtracting the product of the means removes this baseline. Conversely, an unconnected product of zero does not automatically prove independence if nonzero means are present.

For these two binary variables specifically, zero covariance is equivalent to independence. That implication uses the two-by-two probability table; it is not a general theorem for continuous variables or larger collections. Three fair bits constrained by even parity are pairwise independent while the triple is dependent. A covariance heatmap therefore cannot summarize every higher-order dependence in a dataset.

## Derive the gradient and Hessian

Differentiate the finite sum defining Z. Dividing its first derivative by Z produces the normalized mean. Differentiating that quotient gives a product moment minus the product of means:

$$
\frac{\partial\log Z}{\partial h_i}=\mu_i,\qquad \frac{\partial\mu_i}{\partial h_j}=C_{ij},\qquad \nabla_h^2\log Z=C.
$$

There is no thermodynamic-limit interchange in this derivation: only four smooth exponentials are involved. The mixed derivatives agree because the finite log partition function is smooth. The Hessian concerns the two local fields, with K held fixed. It is not the Hessian of the energy of a single configuration, and it is not a derivative with respect to the observed arrows themselves.

The diagonal says how a site's mean responds to its own field. The off-diagonal says how its mean responds when the other field changes. For negative K, increasing the second field can decrease the first mean. This negative cross response is compatible with convexity of log Z. Convexity constrains directional curvature, rather than requiring every Hessian entry to be positive.

## Why every observable variance stays nonnegative

For a specified linear observable, combine the fluctuations before taking a square:

$$
A=a_1\sigma_1+a_2\sigma_2,\qquad \operatorname{Var}(A)=a^{\mathsf T}Ca=\sum_{x<y}p_xp_y(A_x-A_y)^2\ge0.
$$

The pair sum runs over the six unordered pairs of the four named configurations. Expanding the squared differences proves the identity. This gives a useful picture: variation comes from pairs of possible outcomes that produce different observable values. Large probability at two outcomes does not increase this particular variance if their A values coincide.

Since every configuration has positive probability, the variance is strictly positive for any nonzero coefficient vector a. A nonzero linear observable cannot be constant on all four spin configurations. Thus C is strictly positive definite at admitted finite parameters. Its determinant has the positive representation

$$
\det C=16\sum_{x<y<z}p_xp_yp_z>0.
$$

A highly saturated preparation can make this determinant small and the matrix difficult to invert. That is numerical conditioning, not an actual loss of rank at a finite input. The all-zero probe is a different case: A is exactly constant and its variance is exactly zero, even though C remains positive definite.

## Follow one source direction carefully

Introduce a scalar source t by changing h to h+t a. This adds tA to each configuration's exponent. Then

$$
\frac{d\log Z(h+ta)}{dt}\bigg|_{t=0}=\mathbb E[A],\qquad \frac{d\mathbb E_t[A]}{dt}\bigg|_{t=0}=\operatorname{Var}(A).
$$

The weights a specify both the measured observable and the direction in which the preparation changes. Multiplying a by two doubles A and its mean, but multiplies its variance and its matching source response by four. Holding the observable fixed while changing only the source scale gives a different derivative. Always label both sides of a response experiment.

For a common field, both hi increase by the same amount. The response of the spin average is

$$
\frac{d\,\mathbb E[(\sigma_1+\sigma_2)/2]}{dt}\bigg|_{h\mapsto h+t(1,1)}=\frac{C_{11}+2C_{12}+C_{22}}{2}=\frac{\operatorname{Var}(\sigma_1+\sigma_2)}{2}.
$$

A local variance alone misses the two cross terms. At zero fields this response becomes 1+C12, not C11 and not the variance of the spin average. The factor of two comes from which observable is differentiated and which field coordinate is varied.

## Work through aligned and opposing preparations

Use an ideal symbolic K=log(3)/2 with both fields zero. The aligned weights are √3 and the opposing weights are 1/√3. After normalization, each aligned configuration has probability 3/8 and each opposing configuration has probability 1/8. Both means vanish by simultaneous spin reversal.

Reversing K exchanges these probabilities. The two covariance matrices are

$$
C_{+}=\begin{pmatrix}1&1/2\\1/2&1\end{pmatrix},\qquad C_{-}=\begin{pmatrix}1&-1/2\\-1/2&1\end{pmatrix},\qquad \det C_{\pm}=3/4.
$$

For positive K, the total spin has variance 3 and the difference has variance 1. For negative K, those variances exchange. Visualize an elongated cloud along the equal-spin direction in the first case and along the opposite-spin direction in the second. A negative off-diagonal rotates the long direction; it does not create a negative-width probability cloud.

The matching common-field response per spin is 3/2 in the positive case and 1/2 in the negative case. The ideal logarithm is a symbolic specification for this hand calculation. Entering the binary64 value `Math.log(3)/2` gives a nearby real K; the reference generator evaluates that actual input rather than asserting its exponential is exactly √3.

## Bias, conditional means and physical units

At K=0 the law factorizes into two independently biased spins:

$$
\mu_i=\tanh h_i,\qquad C=\operatorname{diag}(\operatorname{sech}^2h_1,\operatorname{sech}^2h_2).
$$

Bias makes a spin less variable by concentrating its probability near one orientation. For ideal h1=log(3)/2 and h2=−log(3)/2, the means are 1/2 and −1/2, the raw product mean is −1/4, and the connected cross covariance is zero. Each diagonal variance is 3/4. A negative raw product therefore does not establish negative connected correlation.

Conditioning on the other spin gives another view:

$$
\mathbb E[\sigma_1\mid\sigma_2=s]=\tanh(h_1+Ks),\qquad \frac{\partial\mu_i}{\partial H_j}\bigg|_{T,J}=\frac{C_{ij}}{k_BT}.
$$

The conditional means use different effective local fields. Their difference is an association inside a specified equilibrium law; it is not automatically an intervention or a causal effect in an observational AI dataset. The physical derivative on the right assumes Hi is an energy bias, and holds positive T and J fixed. Magnetic moments, dimensional magnetization, or a field measured in tesla introduce additional conversion factors. A dimensionless covariance by itself is not a susceptibility in SI units.

## Connect the calculation to AI information geometry

In this exponential family the field score is the centered feature vector. Its covariance is the field block of the Fisher information:

$$
\nabla_h\log p(\sigma)=\sigma-\mu,\qquad \mathcal I_h=\mathbb E[(\sigma-\mu)(\sigma-\mu)^{\mathsf T}]=C,\qquad \mathcal I_\theta=B^{\mathsf T}CB\quad(h=B\theta).
$$

The last expression assumes B is a fixed linear coordinate map and K is held fixed. If B has dependent columns, the parameterized Fisher matrix can be singular even though the two-field covariance is positive definite. Replacing h by a nonlinear coordinate function requires the correct chain rule; the ordinary Hessian of log Z then has additional second-derivative terms and must not be called the Fisher matrix without checking them.

This small example explains why feature scaling changes curvature, why off-diagonal terms matter in a natural-gradient calculation, and why nearly deterministic features can create an ill-conditioned metric. It does not prove that using a natural gradient improves every optimizer. Nor does an equilibrium feature covariance determine the error of a correlated sequence of training examples. That later question requires covariances across observation indices.

## Numerical evidence and honest display limits

The model treats each supplied finite binary64 number as an exact real input. It constructs exponents using exact rational arithmetic and evaluates shifted exponentials and logarithms using directed intervals. Means group identical exact exponents before cancellation. Probe variance uses the positive pair identity, and the determinant uses the positive triple identity. Neither is obtained by subtracting two rounded display numbers.

Cross covariance can be expressed without subtracting separately rounded raw moments:

$$
C_{12}=4(p_{--}p_{++}-p_{-+}p_{+-})=\frac{4(e^{2K}-e^{-2K})}{Z^2}.
$$

Consequently its exact sign is the sign of K, independently of the fields, and it vanishes structurally at K=0. For a nonzero subnormal K, the two terms can be extraordinarily close. The implementation increases precision from 128 through 8192 bits until each reported enclosure has the required sign and relative width at most 2 to the power −48. If a chosen ceiling prevents resolution, the value is unavailable; it is not replaced by zero. A known analytical sign may still be shown beside an unavailable magnitude.

A probe coefficient as small as the least positive binary64 number can give a positive variance below the range of the ordinary linear display. The observation retains its sign and bounds and uses a logarithmic magnitude for explanation. General logarithmic magnitude projections are approximate; they are separate from the directed enclosures and the explicitly computed log probabilities. A bar chart is an illustration of the four-state distribution, not an independent certificate of independence or matrix rank.

The registered reader connects this model to fixed concept steps, guided comparisons and editable source controls. Standalone numerical tests alone do not establish correct worker behavior, saved-state recovery or actual-reader usability; those require integration evidence.

## Connect covariance and source to the experiment

The probability bars count four named configurations. The arrow is the observable coefficient vector a=(a1,a2), which also specifies the source direction h→h+ta. It is not a covariance ellipse or a sampled spin cloud. Read the covariance matrix and Var(A) alongside this arrow.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| Positive → negative coupling, sum probe | Can C12 become negative while Var(A) stays positive? | The off-diagonal changes sign; the sum's variance shrinks without becoming negative. |
| Negative coupling, sum → difference probe | Must the spin probabilities change when the observable changes? | The law and covariance stay fixed while the source arrow and probe variance change. |
| Unequal fields → double observable and source | Does a twice-longer arrow produce twice the response? | E[A] doubles and Var(A) quadruples, with the same spin probabilities. |

Enter K,h1,h2,a1,a2 and commit with Enter or by leaving the field. The zero probe is an exactly constant observable; it does not turn the spin covariance into a zero matrix. Compare the smallest nonzero coupling with its 128-bit capped preparation to distinguish a known nonzero sign from an unresolved magnitude. The covariance cells, sign and representation table remain authoritative when shading or a tiny vector cannot be distinguished visually.

## Check your understanding with explained problems

### 1. Distinguish a raw product from a connected covariance

Two independent spins have means 1/2 and −1/2. Find their raw product expectation and C12.

**Explanation.** Independence gives a product expectation of −1/4. Subtracting the product of means gives C12=0. The negative raw product records opposing biases, not dependence. Their separate variances are each 3/4 because a spin square is always one.

### 2. Test a negative off-diagonal

At the ideal opposing preparation, calculate the variances of the total and the difference and check whether C is positive definite.

**Explanation.** With C11=C22=1 and C12=−1/2, the total variance is 1+1−1=1 and the difference variance is 1+1+1=3. The determinant is 3/4 and the first diagonal entry is positive, so the symmetric two-by-two matrix is positive definite. The negative cross entry is not a negative eigenvalue.

### 3. Identify the correct response experiment

For the same opposing preparation, both local fields increase together. What is the response of the spin average? What would a local-field response measure instead?

**Explanation.** The common-field response per spin is the total variance divided by two, hence 1/2. Differentiating the first mean with respect to its own field gives C11=1; differentiating it with respect to the other field gives C12=−1/2. These are three different derivatives with three explicitly different coordinates and observables.

### 4. Change a source scale

Replace the probe A by 2A and define its source by adding t(2A) to the exponent. How do the mean and response change?

**Explanation.** The mean doubles and the matching source response becomes four times Var(A). Two factors of two appear because both the measured observable and its source direction changed. Changing only the source while measuring the old A would instead double that derivative.

### 5. Separate two exact zeros from an unresolved magnitude

Compare K=0, a=(0,0), and a nonzero K whose cross covariance cannot be resolved at a 128-bit ceiling.

**Explanation.** K=0 proves independence and C12=0 for any fields; it does not make the diagonal variances vanish. The zero probe makes its own mean and variance exactly zero for any K. The precision-capped cross covariance is a third situation: its magnitude is unavailable, with known nonzero sign from K. Neither a small plot nor a capped calculation proves a structural zero.

### 6. Audit a reparameterized Fisher matrix

Let both fields be controlled by one scalar θ through h=(θ,θ). What is the Fisher information for θ, and does its dimension describe two separately identifiable field parameters?

**Explanation.** Here B is the column vector (1,1), so the scalar information is C11+2C12+C22, equal to the total-spin variance. There is one parameter in this preparation. Duplicating θ into two names would create a redundant parameterization, not two new independent directions. This is a coordinate-identifiability question, distinct from the positive definiteness of the original two-field covariance.

## Continue from a pair to space and repeated observations

The four-state model is complete at its stated finite inputs. It has no spatial distance larger than the site pair, no temporal dynamics and no infinite-system singularity. To study a correlation length, specify a lattice, distance, boundary and limit. The next chapter uses a periodic zero-field Ising ring and compares its two paths with direct configuration enumeration. Its distance index is not a worker animation frame or a Monte Carlo time index.

For research practice, derive the response to K as a covariance with the interaction feature, extend the natural-parameter vector to include that feature, and determine which combinations are identifiable. Separately, compare equal-time spatial covariance with the autocovariance of a sampler observable. Only the latter, with stationarity and appropriate limit conditions, enters the uncertainty of a correlated sample mean. These extensions require additional models and evidence; this two-spin Hessian alone does not finish them.
