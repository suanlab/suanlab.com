## A normalization constant can also be a measuring instrument

Suppose an experiment can prepare a finite list of labeled configurations. The Hamiltonian assigns an energy to each configuration, while the reservoir determines how those configurations are weighted. The partition function adds the weights. Its first job is ordinary normalization, but changing its control variables also reveals mean energies, fluctuations and responses.

Think of a shop with differently priced objects. Counting objects and adding their weighted prices answer different questions. A normalization sum records which objects were available and how they were weighted; it does not automatically record every label, every property of an object, or the speed at which a customer moves between them. Likewise, a partition function is a static description. It is not a transition-rate matrix or a relaxation time.

This chapter extends the previous canonical binary example to an explicit finite spectrum. We will derive the identities before using them. The basic normalization and energy-response viewpoint follows [Tong, sections 1.3.1–1.3.2](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf). The three-level calculations, source examples, counterexamples and numerical checks below use the preparations stated here.

## Specify the states, energy reference and units

There are m labeled microstates. State i has a fixed relative energy e_i and a common energy offset c. Energies are measured in joules, temperature in kelvin, and the inverse temperature has units of inverse joules:

$$
E_i=c+e_i,\qquad \beta=\frac{1}{k_BT},\qquad [E_i]=\mathrm{J},\quad [\beta]=\mathrm{J}^{-1}.
$$

We specify a canonical preparation separately; a Hamiltonian alone does not impose equilibrium or remove a constraint. During a temperature derivative, the list of states, their energies and their counting measure stay fixed. We will deliberately violate this assumption later to see why it matters. A mathematical finite list is bounded above and below. Consequently, its canonical sum can be normalized at either sign of finite nonzero temperature. That observation does not establish that an arbitrary material supports negative-temperature equilibrium.

The calculator accepts one through 32 finite relative energies, a finite common offset, and finite temperatures. This resource bound is a property of the experiment, not a theorem limiting statistical mechanics. It adds the offset and relative energies before the final numerical projection. A displayed energy can therefore lose visible digits while the calculation still retains the specified gaps. It cannot recover a gap already lost when a caller constructed a rounded flat energy list.

## Normalize microstates before grouping energy levels

Use the counting measure that assigns weight one to each listed microstate. The canonical partition function and probabilities are

$$
Z(\beta)=\sum_{i=1}^{m}e^{-\beta E_i},\qquad p_i=\frac{e^{-\beta E_i}}{Z(\beta)}.
$$

A weight is not yet a probability. If two labeled configurations have the same energy, each appears in the sum. Alternatively, one may list each distinct level once and attach its degeneracy. These are two ways of counting the same objects; using both representations at once double-counts them.

| Description | Object being counted | Contribution to Z |
| --- | --- | --- |
| One labeled state of energy E | One microstate | exp(−βE) |
| g distinct states at the same energy E | One grouped energy level | g exp(−βE) |
| A normalized prior over the state labels | A different reference measure | Its prior weights must be stated |

Dividing every weight by m leaves p unchanged but changes Z to Z/m. Its logarithm changes by −ln m. Derivatives with respect to a source may be unchanged, but the physical state-counting convention has changed. One should not silently use a probability prior in place of the counting measure and then interpret its entropy as the same thermodynamic entropy.

## Work through a three-level example

Take relative energies 0, ε and 2ε, with c=0 and ε/(kBT)=ln 2. The ideal weights and normalized law are

$$
(w_0,w_1,w_2)=(1,1/2,1/4),\qquad Z=7/4,\qquad (p_0,p_1,p_2)=(4,2,1)/7.
$$

Introduce the level index K=0,1,2 so that E=εK. Directly adding the three outcomes gives

$$
\langle K\rangle=\frac47,\qquad \operatorname{Var}(K)=\frac{26}{49},\qquad \kappa_3(K)=\frac{114}{343}.
$$

| K | Probability | Contribution to the mean | Contribution to the second raw moment |
| --- | --- | --- | --- |
| 0 | 4/7 | 0 | 0 |
| 1 | 2/7 | 2/7 | 2/7 |
| 2 | 1/7 | 2/7 | 4/7 |

Thus the second raw moment is 6/7, and subtracting the squared mean gives 26/49. The physical mean, variance and third cumulant carry factors ε, ε² and ε³. A probability spread and an energy spread have different units.

The hand calculation assumes an exact logarithm-of-two relation. A computer stores rounded ε, T and the numerical value of kB. Independent references for the actual supplied binary values therefore accompany the ideal rational example; an ideal identity is not used as an exact floating-point oracle.

## The first derivative generates mean energy

Differentiate a finite sum while keeping the Hamiltonian fixed. The energy factor brought down from the exponential produces

$$
\partial_\beta Z=-\sum_i E_i e^{-\beta E_i},\qquad -\partial_\beta\ln Z=-\frac{\partial_\beta Z}{Z}=\sum_i p_iE_i=U.
$$

The logarithm is essential. Differentiating Z gives an unnormalized weighted sum, while differentiating ln Z divides by the same normalizer that defines the probabilities. The minus sign comes from the negative energy in the Boltzmann exponent.

The unit check is informative: differentiating a dimensionless logarithm with respect to a variable in inverse joules produces joules. Replacing β by temperature without applying the chain rule would give the wrong sign and scale. In an experiment, this derivative describes how separately prepared equilibrium laws change with the reservoir setting. It is not the energy velocity along a single microscopic trajectory.

## The second derivative generates variance and heat capacity

Differentiating once more produces a difference between the second moment and the squared first moment:

$$
\partial_\beta^2\ln Z=\langle E^2\rangle-\langle E\rangle^2=\operatorname{Var}(E)\geq0.
$$

A positive curvature means that ln Z is convex in β. Intuitively, a spectrum with more available energy spread lets the mean energy change more when its bias is adjusted. A single energy shared by every state gives zero energy variance even though uncertainty about the state label may remain.

For fixed energy levels, applying dβ/dT yields the canonical heat-capacity relation:

$$
C=\left.\frac{\partial U}{\partial T}\right|_{E_i}=\frac{\operatorname{Var}(E)}{k_BT^2},\qquad \frac{C}{k_B}=\beta^2\operatorname{Var}(E).
$$

Variance has units J², while C has units J/K. These are connected quantities rather than interchangeable readouts. In this finite fixed-Hamiltonian model C is nonnegative at either nonzero temperature sign. Population inversion changes the preparation; it does not turn the square of temperature into a negative number.

For numerical work, subtracting two enormous displayed moments can obscure a small variance. An equivalent sum of positive pair contributions avoids that cancellation. The calculator encloses those contributions rather than subtracting rounded values of the two displayed moments.

## Higher derivatives generate cumulants

Bias the energy by an auxiliary source t, measured in inverse joules. Its moment-generating function is a ratio of partition functions, so its logarithm is

$$
K_E(t)=\ln\langle e^{tE}\rangle_\beta=\ln Z(\beta-t)-\ln Z(\beta).
$$

Differentiating at t=0 gives the cumulants:

$$
\kappa_r(E)=(-1)^r\partial_\beta^r\ln Z,\qquad \kappa_1=U,\quad \kappa_2=\operatorname{Var}(E),\quad \kappa_3=\langle(E-U)^3\rangle.
$$

The third cumulant is signed. It distinguishes opposite asymmetries that have the same variance. In the symmetric three-level spectrum, changing the temperature sign reverses the mean energy relative to its center and reverses the third cumulant, while retaining variance and entropy.

A number very close to zero is not automatically a symmetry result. The calculator declares a structural zero when all energies coincide. Otherwise it refines directed intervals. If the precision ceiling still prevents the requested relative accuracy, it reports an unresolved value and retains any sign already established by the bounds. This is a numerical limitation, not a proof that the cumulant vanishes.

## Add sources to measure other properties

To study an observable A, supply its value in every state and introduce a source. For several fixed classical features, the log normalizer obeys

$$
\mathcal A(\theta)=\ln\sum_i e^{\theta\cdot A_i},\qquad \nabla_\theta\mathcal A=\langle A\rangle,\qquad \nabla_\theta^2\mathcal A=\operatorname{Cov}(A).
$$

Each product of a source and its feature must be dimensionless. The covariance Hessian is positive semidefinite, but it need not be positive definite. If a linear combination of features is constant across every allowed state, its variance is zero. That null direction is a parameter-identifiability issue rather than an absence of fluctuations in every other direction.

An energy-only Z does not specify every observable. Assigning an observable the values (0,1) or (1,0) on a two-level spectrum leaves Z unchanged but changes its expectation from r to 1−r. The energy multiset alone does not identify which state carries the observable's value. Nor does a static Z specify transition rates. These source identities concern explicit classical features or a common eigenbasis shared by the Hamiltonian and all included observables; they are not a shortcut for arbitrary noncommuting quantum response.

## Move the energy reference without erasing the gaps

For a temperature-independent common shift c, the transformations are

$$
E_i\mapsto E_i+c:\quad Z\mapsto e^{-\beta c}Z,\quad \ln Z\mapsto\ln Z-\beta c,\quad U\mapsto U+c,\quad F\mapsto F+c.
$$

The normalized probabilities, entropy, variance, higher cumulants of order at least two, and heat capacity remain unchanged. This makes a reference shift a useful numerical and conceptual check. It also explains why Z can be enormous or tiny even when the probability law is perfectly ordinary.

There are two different rounding problems. First, adding a large c to a small e_i in an ordinary floating array may erase e_i before a model receives the data. Second, projecting an accurately retained energy onto a short display may hide the gap while leaving the internal calculation intact. Separate offset and relative-level inputs address the second problem and can avoid the first at input construction. They cannot reverse information already lost in a supplied flat array. Never group states merely because their displayed energies look equal.

## Check what is fixed in a response derivative

If the energy levels themselves depend on temperature, changing T also moves the Hamiltonian. Direct differentiation gives

$$
\frac{d\langle E\rangle}{dT}=\langle\partial_TE\rangle+\frac{\operatorname{Cov}(E,E-T\partial_TE)}{k_BT^2}.
$$

The usual variance formula follows when the partial derivatives of the energies vanish. Otherwise there is an explicit level-motion contribution and a modified probability-response contribution. An externally controlled device whose gap changes with temperature must state that protocol before calling a measured slope a canonical heat capacity.

At fixed T, another control parameter λ gives

$$
\partial_\lambda F=\langle\partial_\lambda E\rangle,\qquad \partial_\eta\partial_\lambda F=\langle\partial_\eta\partial_\lambda E\rangle-\beta\operatorname{Cov}(\partial_\lambda E,\partial_\eta E).
$$

The additional second-derivative term matters when the Hamiltonian depends nonlinearly on the control. A source that enters linearly and a generic material parameter do not have the same differentiation rule. Similarly, fixed chemical potential and fixed β times chemical potential are different directions in the grand-canonical parameter space.

## Factorization needs an appropriate state space

Two noninteracting parts with a full product state space satisfy

$$
Z_{X\times Y}=\sum_{x,y}e^{-\beta(E_X(x)+E_Y(y))}=Z_XZ_Y.
$$

Both assumptions matter: the energy must add and every pair of allowed component states must be allowed in the joint model. A fixed total-excitation constraint can prevent the second condition even without an interaction energy. The preceding finite-reservoir experiment already showed the resulting conditional dependence.

This distinction will be tested with coupled binary sites. Changing an interaction and changing the support are separate experiments. A failure of product factorization is not by itself a measurement of mutual information: that comparison also needs the normalized joint law and its actual marginal laws. It is better to specify the reference law than to attach the same name to several unrelated log differences.

## A normalized truncation can still miss probability mass

Suppose a full law p assigns mass δ outside a retained set A, and q is p conditioned on A. Then

$$
\frac{Z_A}{Z}=1-\delta,\quad \ln Z-\ln Z_A=-\ln(1-\delta),\quad \operatorname{TV}(p,q)=\delta,\quad D(q\Vert p)=-\ln(1-\delta).
$$

If any omitted state has positive p, the reverse divergence D(p‖q) is infinite. Renormalizing the retained states makes their probabilities add to one, but provides no bound on δ. A separately justified tail estimate is needed.

The later energy-ladder experiment supplies such an estimate through a geometric series. It also separates finite-state normalization from infinite-state normalization. A finite cutoff can conceal a divergent infinite partition function. Removing a state-space cutoff is a different limit from taking particle number and volume to infinity, and neither limit is established by drawing a smooth finite plot.

## Read the numerical states before comparing digits

At a finite nonzero temperature every listed state has positive probability. A missing linear number can mean a positive quantity outside the representable range. An unresolved interval instead means that the requested numerical accuracy was not established. A mathematical infinity in a zero-temperature limit is a third situation.

| Readout state | Interpretation | Appropriate action |
| --- | --- | --- |
| Structural zero | A stated degeneracy or limiting preparation proves zero | Check the stated assumptions |
| Interval resolved | Directed bounds establish sign and the reported relative accuracy | Read the units and compare bounds |
| Log-tail approximation | Positivity and a controlled cold omission are retained; logarithms are approximate | Compare logarithmic scale rather than inventing a zero |
| Unresolved at the ceiling | Available precision did not meet the accuracy target | Retain the bounds and revise the numerical question |

The numerical core uses the stored binary64 kB literal and exact arithmetic on the supplied binary inputs. Its certified intervals concern that specified numerical model. They do not certify a real apparatus, an uncertain energy measurement, or the suitability of a canonical preparation. The enclosure-width target is distinct from quantization of the final binary64 display, especially for subnormal numbers.

The T=0 input means a fixed-Hamiltonian limit from positive temperatures. It distributes probability uniformly over the lowest-energy states. U and F approach the minimum energy, C and the energy cumulants above first order approach zero, and the residual entropy is kB ln g0. Depending on the energy reference, Z can approach zero, a finite ground-state multiplicity, or infinity. One must take the normalized limit rather than divide two separately rounded zero weights.

## Connect the generating function to AI research

For logits z_i, the log-sum-exp gradient is the softmax vector, and its Hessian is diag(p)−ppᵀ. The all-ones direction is null because adding the same logit shift changes the normalizer but leaves probabilities invariant. This is the same normalization geometry expressed in different coordinates. A dimensionless model logit is not automatically a physical energy in joules.

The variational identity is useful for checking an approximation:

$$
\ln Z=\sup_q\{H(q)-\beta\langle E\rangle_q\},\qquad D(q\Vert p)=\ln Z-H(q)+\beta\langle E\rangle_q.
$$

The supremum ranges over normalized trial laws on the same labeled state space. Its nonnegative gap is KL divergence. At positive temperature this corresponds to minimizing trial free energy; multiplying by a negative kBT reverses that optimization direction. The log-normalizer identity itself continues to hold for a finite spectrum.

For a new research claim, state which source is varied, which features and support remain fixed, and whether an approximation changes the counting measure. Check gauge invariance, units, limiting cases and independent references. Approximate inference quality, sampling efficiency and physical equilibration are separate questions even if all three algorithms mention a partition function.

## Exercises with explained answers

### 1. Count degenerate states only once

Use three labeled states of energies (0,ε,ε) at βε=ln 2. What are Z, each state probability, and the probability of the excited energy level?

**Answer.** The weights are (1,1/2,1/2), so Z=2 and the microstate probabilities are (1/2,1/4,1/4). The grouped excited-level probability is 1/2. Multiplying its two already listed states by an extra degeneracy factor would double-count them.

### 2. Recover the third cumulant by hand

For probabilities (4,2,1)/7 on K=0,1,2, compute the third raw moment and then the third cumulant.

**Answer.** The third raw moment is (2+8)/7=10/7. Combining it with the mean 4/7 and second raw moment 6/7 gives 10/7−3(4/7)(6/7)+2(4/7)³=114/343. Multiplying by ε³ supplies the energy unit J³. This signed quantity is not the variance or the standardized skewness.

### 3. Move the reference by five gaps

Add 5ε to every energy while keeping ε and T fixed. Which readouts change?

**Answer.** U and F increase by 5ε and ln Z decreases by 5βε. All normalized probabilities, entropy, variance, third cumulant and heat capacity stay fixed. If one instead changes ε or makes the reference temperature dependent, that is a different experiment.

### 4. Can an energy-only partition function identify every observable?

On two energy levels, assign A=(0,1) in one model and A=(1,0) in another. Do identical Z values imply identical mean A?

**Answer.** No. If the excited-state probability is r, the means are r and 1−r. The spectrum fixes the same energy weights, but the observable has a different state assignment. A source-dependent normalizer must include that assignment to generate its expectation.

### 5. Find a null source direction

Let two dimensionless features be A1=K and A2=2K. Is their covariance Hessian positive definite?

**Answer.** It is Var(K) times the matrix with rows (1,2) and (2,4). It is positive semidefinite and has null direction (2,−1), because 2A1−A2 is identically zero. Redundant parameters are not made identifiable by collecting more samples from the same feature map.

### 6. Move the levels while changing temperature

Take E_i(T)=kBT a_i with fixed a=(0,ln 2). Compare the actual derivative of U with the fixed-Hamiltonian fluctuation formula.

**Answer.** The probabilities stay (2/3,1/3), so U=kBT ln 2/3 and dU/dT=kB ln 2/3. Inserting the energy variance into the fixed-level formula instead gives kB(ln 2)²(2/9), a different result. In the general derivative, the explicit level-motion term gives the actual slope and E−T∂TE is zero. The original assumption, rather than the variance calculation, was what failed. Taking T to zero along this moving-level path also differs from the calculator's fixed-Hamiltonian zero-temperature convention.

## Connect the generating identities to the experiment

Select **Three-level worked example** and read the relative-level diagram beside the probability table. Each index is a labeled microstate, even when two energies coincide. Predict (4,2,1)/7 before checking the computed law; then divide U, Var(E) and the third cumulant by ε, ε² and ε³ to compare with the hand calculation. Keep the SI units visible.

| Experiment | What to change or compare | What should remain distinct |
| --- | --- | --- |
| Large common offset, same gaps → Gaps lost before input | Compare a separate c with a flat list rounded before calculation | Hidden display digits and information already lost at input are different |
| Same levels, negative temperature | Keep the relative levels and change the sign of T | A bounded inverted law is a different preparation from the T=0 ground limit |
| Two ground states at zero temperature → All states at equal energy | Compare which labels share the minimum | Zero energy variance does not force zero label entropy |

**Add state** inserts a new labeled state with relative energy zero. To investigate an extra excited-state degeneracy, change that new state's energy to the existing excited level before comparing results. Adding or removing a state changes the counting measure and normalizer; shifting the common offset leaves the normalized law unchanged. Moving a level and moving the energy reference are different operations.

The reader evaluates one supplied spectrum at a time. The source-feature identities and moving-Hamiltonian example above are derivations, not extra source-vector or temperature-dependent-Hamiltonian controls. For a fixed-spectrum response check, keep every relative energy and c fixed while making a small change in T around a finite nonzero value. Compare the resulting change of U with C times that temperature change. A log, a probability and an energy have different units and should not be substituted for one another.
