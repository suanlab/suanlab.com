## An equilibrium law does not specify a journey

Knowing the probability of every configuration tells us what equilibrium looks like. It does not tell us how to generate samples, how long an algorithm takes to forget its starting state, or how a physical system moves in time. Several update rules can preserve the same equilibrium law while behaving differently along a trajectory.

Imagine four rooms arranged as a square. A configuration records two binary choices; changing the first choice moves along one pair of edges, and changing the second moves along the other pair. A conditional Gibbs update chooses a new value for one choice while holding the other fixed. Restrictions can remove rooms or edges, so even perfectly valid conditional updates can become trapped.

The conditional-update construction follows the [Stanford Gibbs sampling notes](https://ermongroup.github.io/cs323-notes/probabilistic/gibbs/). The two-spin examples, flow witness and contraction calculations below are derived explicitly. We separate three questions: does a rule preserve the target, does it satisfy detailed balance, and can it converge to the target from every allowed starting state?

## Declare the states and Hamiltonian

Use two named spins σ1,σ2 taking values −1 or +1. On the full product support, order them as (--),(-+),(+-),(++). Their energy and target law are

$$
E(\sigma_1,\sigma_2)=c-J\sigma_1\sigma_2-h_1\sigma_1-h_2\sigma_2,\qquad
\pi(\sigma_1,\sigma_2)=\frac{e^{-\beta E(\sigma_1,\sigma_2)}}{Z},\qquad
\beta=\frac{1}{k_BT}.
$$

J, h1, h2 and c have units of joules. The spin values are dimensionless, so each term in the energy has the same units. Positive J favors aligned spins at positive temperature; the fields can favor different signs at the two sites. The common offset c changes the partition function but cancels from the normalized target and every conditional probability.

Support is part of the preparation. We will compare the full four-state product with a fixed-zero-sum support containing only (-+) and (+-). A label outside the chosen support is absent, not a rare state that can eventually be visited. At finite nonzero temperature, every allowed state has strictly positive target probability.

## Derive one conditional update

Fix the second spin to s. The terms depending on the first spin reduce to −(Js+h1)σ1. Dividing the two conditional weights gives

$$
\frac{\pi(\sigma_1=+1\mid\sigma_2=s)}{\pi(\sigma_1=-1\mid\sigma_2=s)}
=e^{2\beta(Js+h_1)},\qquad
\pi(\sigma_1=+1\mid\sigma_2=s)
=\frac{1}{1+e^{-2\beta(Js+h_1)}}.
$$

The logistic argument is dimensionless. This is a heat-bath update: it draws a fresh spin value from the exact conditional, including the possibility of drawing the value already present. It is not a rule that must flip a spin at every step. The second-site formula exchanges the two indices and uses h2.

This logistic expression assumes both spin values are allowed while the other coordinate is fixed. Under the fixed-zero-sum constraint, only one first-spin value is available for each second-spin value. The conditional then assigns probability one to that value. Applying the unrestricted logistic formula to the constrained support would simulate a different model.

## Build source-row transition matrices

Let K1 update the first coordinate and K2 update the second. We use a row-vector distribution q and multiply transitions on the right:

$$
q_{t+1}=q_tK,\qquad
(K_1)_{(x,y),(x',y')}=\mathbf{1}_{y'=y}\pi(x'\mid y),\qquad
(K_2)_{(x,y),(x',y')}=\mathbf{1}_{x'=x}\pi(y'\mid x).
$$

A row represents the current room and a column the next room. Each row sums to one. Zero entries in K1 and K2 have a structural reason: that update cannot change the coordinate it is holding fixed.

For the ideal preparation βJ=ln 2 and h1=h2=0, the target is (2/5,1/10,1/10,2/5). The conditional probability of matching the fixed neighbor is 4/5. Hence

$$
K_1=\begin{pmatrix}
4/5&0&1/5&0\\
0&1/5&0&4/5\\
4/5&0&1/5&0\\
0&1/5&0&4/5
\end{pmatrix},\qquad
K_2=\begin{pmatrix}
4/5&1/5&0&0\\
4/5&1/5&0&0\\
0&0&1/5&4/5\\
0&0&1/5&4/5
\end{pmatrix}.
$$

Following the row for (--) makes the operation concrete. A first-site update stays at (--) with probability 4/5 and moves to (+-) with probability 1/5. It cannot reach (-+) because that would change the second coordinate.

## Prove preservation and detailed balance for a coordinate update

A first-site update preserves the target because summing over the old first spin restores the second-spin marginal:

$$
(\pi K_1)(x',y)=\sum_x\pi(x,y)\pi(x'\mid y)
=\pi(y)\pi(x'\mid y)=\pi(x',y).
$$

For two states sharing y, the equilibrium flow also has a symmetric expression:

$$
\pi(x,y)(K_1)_{(x,y),(x',y)}
=\pi(y)\pi(x\mid y)\pi(x'\mid y)
=\pi(x',y)(K_1)_{(x',y),(x,y)}.
$$

This is detailed balance. States with different y have zero flow in both directions under K1. The same argument applies to K2 and to a joint block that resamples both coordinates from π.

Preservation is weaker than convergence. K1 by itself never changes y, so it cannot forget which second-spin value was initially chosen. Both y classes are closed even though π is stationary and detailed balance holds. A chain can have many stationary distributions; proving that one particular target is stationary does not establish uniqueness.

## Compare random scan and systematic scan

An equal random scan selects a coordinate with probability 1/2 and performs its conditional update. A systematic sweep updates the first coordinate and then the second:

$$
K_{\mathrm{random}}=\frac{K_1+K_2}{2},\qquad
K_{\mathrm{systematic}}=K_1K_2,\qquad
\pi K_1K_2=(\pi K_1)K_2=\pi.
$$

The random mixture inherits detailed balance because its two symmetric equilibrium flows are added with the same fixed coefficients. The product preserves π because each factor does. A product of reversible kernels need not be reversible: reversing a trajectory reverses the order of the factors, so it generally corresponds to K2K1 instead.

For the ideal example, the systematic sweep matrix is

$$
K_{\mathrm{systematic}}=\frac{1}{25}
\begin{pmatrix}
16&4&1&4\\
4&1&4&16\\
16&4&1&4\\
4&1&4&16
\end{pmatrix}.
$$

Multiplying on the wrong side or swapping the update order changes this matrix. Start at (--) and follow the two branches through K1, then K2: the resulting distribution is (16,4,1,4)/25. This path calculation is a useful check of matrix orientation.

## Exhibit the failure of systematic detailed balance

The ideal systematic transition probabilities from (--) to (-+) and back both equal 4/25. Detailed balance compares probability-weighted flows, not bare transition entries:

$$
\pi_{--}P_{--,-+}=\frac25\frac4{25}=\frac8{125},\qquad
\pi_{-+}P_{-+,--}=\frac1{10}\frac4{25}=\frac2{125}.
$$

The target is preserved even though these flows differ. For random scan, both flows for this pair equal 1/25. Neither equal transition entries nor stationarity alone proves detailed balance.

There is an exact witness beyond this special example. Write the four positive target masses as a,b,c,d in state order. Then

$$
\pi_0P_{01}-\pi_1P_{10}
=\frac{ab(ad-bc)}{(a+b)(a+c)(b+d)},\qquad
\frac{ad}{bc}=e^{4\beta J}.
$$

Every denominator factor is positive. On the full support at finite nonzero temperature, the witness has the sign of βJ and vanishes exactly when J=0. Tiny coupling may make the numerical witness difficult to display, but it does not make the systematic chain exactly reversible. The fixed-zero-sum identity chain is a different support and must be handled separately.

## Measure convergence using the two-spin contraction

Total variation distance compares two probability laws, not two individual trajectories:

$$
d_t=\lVert q_t-\pi\rVert_{\mathrm{TV}}
=\frac12\sum_i|q_t(i)-\pi_i|.
$$

For a systematic sweep, the old first coordinate is discarded immediately. The second-coordinate marginal follows a two-state chain. Its nontrivial eigenvalue can be written using the two conditional contrasts:

$$
\lambda=\big[\pi(X=+1\mid Y=+1)-\pi(X=+1\mid Y=-1)\big]
\big[\pi(Y=+1\mid X=+1)-\pi(Y=+1\mid X=-1)\big]
=\operatorname{Corr}_{\pi}(X,Y)^2.
$$

To see the last equality, note that a conditional mean of a binary variable is an affine function of the other variable. Its centered slope is Cov(X,Y)/Var(Y); the reverse slope is Cov(X,Y)/Var(X). Composing the two conditional expectations multiplies these slopes. Strictly positive full-support masses give 0≤λ<1.

A centered distribution on two states has only one degree of freedom. Therefore its deviation from the stationary marginal is multiplied by λ each sweep. Reconstructing the joint state after the next two conditional updates is linear, so

$$
q_t-\pi=\lambda^{t-1}(q_1-\pi),\qquad d_t=\lambda^{t-1}d_1,\qquad t\geq1.
$$

For J=0, a full systematic sweep is an independent product refresh and q1=π; interpret that case directly without asking a numerical library to evaluate 0^0. In the ideal zero-field example, Corr(X,Y)=3/5, λ=9/25 and a point start at (--) gives d1=3/10. The distances at sweeps two and four are 0.108 and 0.0139968.

This identity also helps computation. When coupling is extremely small, q4 and π can display identical probabilities although d4 is positive. Multiplying the retained conditional contrasts and d1 before final projection preserves that distance. Subtracting displayed probabilities would report a false zero.

## Remove connections with an explicit support constraint

Now allow only (-+) and (+-). If h1=h2, the two allowed energies are equal and π=(1/2,1/2). A single-coordinate change would violate the zero-sum constraint. Consequently,

$$
K_1=K_2=K_{\mathrm{random}}=K_{\mathrm{systematic}}
=\begin{pmatrix}1&0\\0&1\end{pmatrix},\qquad
K_{\mathrm{blocked}}=\begin{pmatrix}1/2&1/2\\1/2&1/2\end{pmatrix}.
$$

Every mixture on the two isolated states is stationary for the identity chain. Starting at the first state gives q_t=(1,0) forever, with total variation distance 1/2 from the chosen target. Detailed balance holds, but there is no convergence to that target from every initial state.

A joint block resamples the entire pair. Every row equals π, so one step reaches the target. With unequal fields, the target on the two allowed states is biased; the coordinate kernels remain identity while the joint block uses the biased probabilities. Thus the trapping mechanism is support connectivity, not a large coupling or equal target weights.

| Property | Full-support random/systematic | Zero-sum coordinate updates | Joint block |
| --- | --- | --- | --- |
| Target preserved | Yes | Yes | Yes |
| One communicating class | Yes | No, two isolated states | Yes |
| Unique stationary law | Yes | No | Yes |
| Every initial state converges to target | Yes at finite nonzero T | No | Yes, after one step |

Positive self-transition probabilities make the finite classes aperiodic. Connectivity and these self-loops are properties of the mathematical kernel, even when some positive edges fall below the display range.

## Compare computational work without inventing a physical clock

One random-scan step resamples one coordinate. One systematic sweep resamples two. A comparison after the same displayed step number uses different coordinate-update counts. To compare two-coordinate budgets, compare two random updates with one sweep, while remembering that random selection can update the same site twice.

A joint block also resamples two coordinates, but that count does not make its runtime equal to two scalar updates: constructing and sampling a joint conditional can cost much more in a larger model. Report what the horizontal axis counts. “Steps,” “sweeps,” coordinate updates and wall-clock time are distinct measurements.

None of these kernels specifies a physical spin-flip rate, a reservoir relaxation time or a laboratory clock. An algorithm preserving a Boltzmann law is not thereby a model of the material's dynamics. A static equilibrium distribution permits many different reversible and nonreversible samplers.

## Separate a theoretical distribution from a finite random path

The calculation q_t=q_0P^t describes a distribution exactly within its stated numerical bounds. One finite seeded path produces empirical counts. Those counts fluctuate, depend on the start and can be strongly correlated. A trapped identity chain repeats one state for every sample; increasing its path length does not recover the target mixture.

Even a random-number implementation introduces another approximation. Suppose b random bits give a uniform integer R in 0,…,2^b−1 and a program selects an event when R/2^b<p. If the comparison uses the exact real p, its implemented probability is

$$
\widetilde p=\frac{\lceil2^bp\rceil}{2^b},\qquad 0\leq p\leq1.
$$

For b=8 and p=0.001, the event occurs with probability 1/256, not 0.001. Other placements of the grid points have different rounding behavior. If p was first rounded to floating-point zero, even this calculation changes and the event is never selected. A seed alone does not establish exact sampling of arbitrary real probabilities.

A path visualization should state the PRNG, bit precision, seed, initialization and number of coordinate updates. It should not silently replace theoretical rare transitions by zero. This model computes theoretical kernels and distributions. The reader provides a separate finite-bit path generator, whose quantized kernel and seeded trace must be distinguished from the theoretical law.

## Read boundary conditions and numerical evidence

The model accepts finite J, h1, h2, c and finite nonzero T. The finite support also permits a mathematically normalized law for negative T, with conditional preferences reversed accordingly. This does not establish negative-temperature equilibrium for an unbounded physical spectrum.

T=0 is explicitly rejected by this conditional model. A positive-side limiting construction must decide which ground states survive and what to do with a conditioning event whose target probability becomes zero. Substituting infinite β into a logistic formula does not define a conditional distribution on a null event. The preceding weighted finite-law chapter treats its own ground-state probability limit; that result does not remove this conditional-kernel question.

Exact binary Hamiltonian sums retain energy gaps under a large common offset. Each conditional is anchored locally before exponentiation, so a rare conditioning class is not lost because the global probabilities underflow. Directed enclosures adapt from 128 to 8192 bits. A resolved nonzero observation has a certified sign and relative interval width at most 2^-48. Nonzero values outside linear display range retain a sign and null linear value; a precision ceiling remains unresolved.

Numerical stationarity residuals are displayed as intervals. Merely containing zero is not a proof of stationarity; the conditional construction supplies that proof. Similarly, reversibility and graph connectivity use the exact Hamiltonian and support, not a tolerance or a graph reconstructed from rounded transition values. The independent Decimal references compare all five kernels, flows, finite-step laws, contraction and distances for actual binary64 inputs, including rare and weak-interaction cases.

## Exercises with explained answers

1. **Can one coordinate update sample the whole target?** Start at (--) on the full support and apply K1 repeatedly. **Answer:** The second spin stays −1. The first update samples the correct conditional given that value, and subsequent updates preserve that conditional. The chain never reaches either state with second spin +1, so it cannot converge to the strictly positive four-state target from this start.

2. **Which way does the matrix product go?** Starting at (--) in the ideal example, update the first spin and then the second. **Answer:** With source rows and row-vector laws, use K1K2. The final probabilities are (16,4,1,4)/25. K2K1 instead gives (16,1,4,4)/25 and describes a different sweep order. Both preserve π.

3. **Do equal forward and reverse entries imply reversibility?** P(--,-+)=P(-+,--)=4/25. **Answer:** Multiply by the source equilibrium masses. The flows are 8/125 and 2/125, so detailed balance fails. Stationarity still holds because it constrains the sum of incoming flows at each state, rather than each pair separately.

4. **How fast does the ideal systematic law approach π?** A point start at (--) has d1=0.3 and λ=0.36. **Answer:** d2=0.3×0.36=0.108 and d4=0.3×0.36³=0.0139968. These distances concern the full probability law after a sweep, not the empirical error of one path or a physical relaxation time.

5. **Does a valid conditional guarantee mixing under a constraint?** Restrict support to (-+) and (+-) with equal energies. **Answer:** Each single-site conditional is deterministic, so all coordinate scan kernels are identity. They preserve π but a point start stays at distance 1/2 forever. A joint block has both rows equal to π and reaches it in one step. The change that matters is the update's ability to connect allowed states.

6. **What does an eight-bit draw actually implement?** Compare R/256<0.001 for uniform integers R=0,…,255. **Answer:** Only R=0 succeeds, so the implemented probability is 1/256. It differs from the theoretical event probability despite being reproducible with a seed. Using more bits reduces this particular discretization error, but does not cure trapping, correlated samples or a wrongly specified support.

## Continue to continuous Gibbs laws

Finite support guarantees a finite partition sum at finite nonzero temperature. On an infinite continuous space, normalizability must be proved before conditionals or samplers are trusted. The next quadratic Gaussian example will specify the reference volume measure, test positive definiteness and distinguish a proper density from a finite plot that merely looks normalized.

## Connect kernels, distributions and paths to the experiment

Select **Nonreversible systematic flow example**. In the result panel choose **Systematic sweep** and inspect the transition matrix and stationary-flow table. Equal transition entries4/25 in opposite directions become unequal equilibrium flows8/125 and2/125 after multiplying by the source target masses. Change the theoretical kernel to **Random scan** and compare the detailed-balance diagnostic. Changing this inspection selector does not redefine the saved physical preparation.

| Experiment | What to inspect | What to explain |
| --- | --- | --- |
| Independent sites, one-sweep refresh | q after one systematic sweep and its TV distance | A full product refresh reaches the target when J=0 |
| Zero sum, trapped coordinate updates → Biased target, still no coordinate moves | Compare coordinate kernels with Joint block | Target preservation does not connect isolated states; a joint refresh changes the available moves |
| Nonzero coupling hidden by display rounding | Read contraction and distance status beside the probability bars | Equal-looking displayed laws do not prove a zero theoretical distance |

To compare the whole law at another update count, edit the increasing observation-step list and apply it, then use the result's observation selector. One random step updates one coordinate; one systematic sweep updates two. The graph and q_t tables describe theoretical probabilities, not empirical visits or a physical clock. Arrows are retained using theoretical positivity, even when the displayed transition number is unavailable.

The reader also already provides a separate **Generate a finite-bit example path** option. Enable it and set seed1, bits8 and path steps16. The path rule is a separate setting from the theoretical kernel inspection selector. Manual playback moves a cursor through a precomputed deterministic path; the visit counts refer to the entire generated path, including the initial state, rather than only the displayed prefix.

Three objects must stay separate: the real-probability kernel P, the quantized kernel Q assuming independent uniform grid draws, and the seeded LCG illustration. The declared recurrence is s←(1664525s+1013904223) mod2³², using high bits for the probability grid. For random scan another high bit selects the coordinate. Reusing a seed makes an implementation reproducible; it does not make its outputs independent.

The Q table uses denominator2^d. A b-bit coordinate draw has d=b; the equal random mixture has d=b+1 and the two-update systematic product has d=2b. Thus an8-bit random example displays d=9, while a systematic example displays d=16. This does not mean that the grid-bits input changed. Q can alter or remove positive transitions and need not preserve the original target exactly. Its one-step TV bound is not a general bound on stationary-law error. Compare the theoretical diagnostics and finite-bit results as different numerical questions.

Commit each scalar field with Enter or by moving focus away from it.
