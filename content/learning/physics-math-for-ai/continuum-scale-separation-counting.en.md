## Choosing how much to average

A cup of fluid can have a stable measured density while a box containing only a few molecules fluctuates strongly. Enlarging the box suppresses count noise but can erase the spatial variation we want to observe. The continuum assumption therefore concerns a useful intermediate observation scale. It does not literally put infinitely many molecules into an arbitrarily small volume. We compare count, collision and resolution requirements without treating their intersection as a universal certificate.

## Three lengths answer different questions

The mean spacing d is a geometric scale derived from number density n. The mean free path is a collision-related travel distance and is not generally equal to that spacing. The macroscopic length L describes the variation relevant to the observable. It may be a narrow gap or gradient length rather than the overall device size. A consistent physical application must choose these quantities from an appropriate material state and collision model.

$$
d=n^{-1/3},\qquad \mathrm{Kn}=\lambda/L,\qquad [n]=\mathrm{m^{-3}},\quad [d]=[\lambda]=[L]=\mathrm m.
$$

## A cubic observation and expected particle count

Use a cube of side ell, so its volume is ell cubed. Multiplying by homogeneous number density gives an expected count, which need not be an integer. An individual observation still has an integer count. Do not use the cubic formula for particles confined to a line or surface. The preceding line-density smoothing experiment and this three-dimensional counting benchmark explore related ideas while retaining different dimensions and assumptions.

$$
V_\ell=\ell^3,\qquad \mu=\mathbb E[N_\ell]=n\ell^3.
$$

## Meaning and derivation of the Poisson benchmark

For a homogeneous Poisson counting benchmark, the mean and variance both equal mu. Dividing standard deviation by mean gives the familiar inverse square-root scaling. This is a model assumption, not a law for every fluid. Excluded volume, interactions and long-range correlations can change the count variance. Use measured or independently modelled fluctuations when applying the argument to a real material; the simple formula is a transparent reference calculation.

$$
N_\ell\sim\operatorname{Poisson}(\mu),\qquad \operatorname{Var}(N_\ell)=\mu,\qquad C_V=\frac{\sqrt{\operatorname{Var}(N_\ell)}}{\mu}=\mu^{-1/2}.
$$

## From an allowed fluctuation to a minimum length

Choose a relative fluctuation tolerance epsilon for the observation task. One percent is an example choice, not a universal physical constant. Solving the count condition gives a minimum expected population and then a minimum cube side. Requiring smaller relative noise demands a larger averaging region. Refining a numerical mesh alone does not increase the physical population within a cell; the observation and numerical discretization scales must be distinguished.

$$
C_V\le\epsilon\quad\Longleftrightarrow\quad \mu\ge\epsilon^{-2}\quad\Longleftrightarrow\quad \ell\ge\ell_{\rm count}=(n\epsilon^2)^{-1/3}.
$$

## Collision length and local equilibrium

For a gas-like local fluid description, impose the additional proxy ell at least c times the mean free path. The user chooses c. Passing this condition does not prove a Maxwellian velocity distribution, sufficient thermal relaxation or slow temporal forcing. Other materials can require other microscopic scales. In particular, mathematical coarse-graining can be performed below a mean free path; this extra condition concerns the intended collisional interpretation, not the existence of an average.

$$
\ell\ge\ell_{\rm collision}=c\lambda,\qquad c\ge1.
$$

## An upper scale that preserves macroscopic structure

An averaging region comparable to the macroscopic variation length can blur the structure of interest. Use ell no greater than eta times L as a declared resolution proxy. The same window may be adequate for a slowly varying profile and inadequate near a sharp layer. This length ratio is not a rigorous bias bound for arbitrary functions. Actual averaging error also depends on field shape, the kernel and which observable is being compared.

$$
\ell\le\ell_{\max}=\eta L,\qquad 0<\eta<1.
$$

## Compute the intersection of the three requirements

Take the larger of the count and collision lower limits, then compare it with the resolution upper limit. If the lower limit exceeds the upper one, no scale meets the declared proxy requirements. Separate existence of a feasible interval from acceptability of the currently selected ell. Endpoints are included by the chosen inequalities. Equality at a boundary does not establish an asymptotically wide scale separation or validate a continuum constitutive model.

$$
\ell_{\min}=\max(\ell_{\rm count},c\lambda),\qquad \ell_{\min}\le\ell\le\eta L.
$$

## Worked example: an available intermediate window

Choose number density ten to the twenty-fourth per cubic metre, mean free path ten to the minus seventh metre and macroscopic length one millimetre. With epsilon 0.01, eta 0.1 and c equal to ten, the collision condition sets the stronger lower limit. Selecting a ten-micrometre cube gives a billion expected particles and small Poisson relative noise. These inputs define an educational scale comparison, not measured properties of a named gas.

$$
\ell_{\rm count}\approx2.154\times10^{-7}\mathrm m,\quad \ell_{\min}=10^{-6}\mathrm m,\quad \ell_{\max}=10^{-4}\mathrm m,\quad \mu=10^9,\quad C_V\approx3.162\times10^{-5}.
$$

## When a smaller device removes the window

Keep the other worked inputs fixed and reduce L to one micrometre. The resolution upper limit becomes one tenth of a micrometre, below the one-micrometre collision lower limit. The declared window disappears. This is not merely a discretization error cured by a finer numerical grid. It calls for reconsidering the observable and model. A physical application should also check material consistency, boundary effects and whether a kinetic description is needed.

$$
L=10^{-6}\mathrm m:\qquad \ell_{\max}=10^{-7}\mathrm m<10^{-6}\mathrm m=\ell_{\min}.
$$

## Correction for a fixed total particle count

In a finite reservoir with fixed total population N and independently uniform particle positions, the count in a volume fraction p is binomial. Its variance includes a factor one minus p, approaching the Poisson benchmark when p is small. Independence of particle positions does not imply independence of disjoint window counts. Fixing the total population couples those counts because a particle assigned to one window cannot simultaneously belong to another.

$$
N_\ell\sim\operatorname{Binomial}(N,p),\quad \mu=Np,\quad \operatorname{Var}(N_\ell)=Np(1-p),\quad C_V=\sqrt{\frac{1-p}{Np}}.
$$

## A finite-count example and two exceptional endpoints

For one thousand particles and volume fraction one tenth, the mean is one hundred and variance ninety. Relative standard deviation is approximately 0.09487, below the Poisson value 0.1. At p equal to one the count is exactly fixed, but spatial resolution has not improved. At p equal to zero there is no nonzero mean with which to normalize. Distinguish an undefined endpoint ratio from the growing relative noise of small positive probabilities.

$$
N=1000,\ p=0.1:\quad \mu=100,\quad \sigma^2=90,\quad C_V=\sqrt{90}/100;\qquad p=0:\ C_V\text{ undefined}.
$$

## Correlated populations and a Fano factor

For a general positive-mean count, define the Fano factor as variance divided by mean. Relative noise then depends on both population and Fano factor. Correlations can make the latter depend on observation scale, so automatically using the Poisson value one can be misleading. This experiment does not estimate a correlation function or infer a Fano factor from fluid data. The expression identifies where further statistical evidence must enter.

$$
F_\ell=\frac{\operatorname{Var}(N_\ell)}{\mu},\qquad C_V=\sqrt{F_\ell/\mu}\quad(\mu>0).
$$

## Consistency of length units and material state

When changing the length unit, scale all lengths consistently and transform number density by the inverse cube of that factor. Expected count and the dimensionless ratios should remain unchanged. Separately, this educational model permits independent number-density and mean-free-path inputs. It does not enforce a collision cross-section or equation of state. Real-gas applications must establish that the chosen inputs describe a consistent material state before using the proxy comparison.

$$
\ell^{\prime}=a\ell,\quad \lambda^{\prime}=a\lambda,\quad L^{\prime}=aL,\quad n^{\prime}=n/a^3\quad\Longrightarrow\quad \mu^{\prime}=\mu,\ \mathrm{Kn}^{\prime}=\mathrm{Kn}.
$$

## AI validation and the next experiments

First verify the worked feasible interval. Change only the selected ell to identify which condition fails at small or large widths. Then shrink L until the interval itself disappears. In the finite reservoir, vary p and compare binomial and Poisson noise. For an AI-predicted field, also document measured fluctuations, averaging scale, time scales, boundaries and constitutive assumptions. A small conservation residual does not replace these checks. The next topic develops stress and deformation.

## Exercises with answers

### Exercise 1

Under the Poisson benchmark, reduce relative noise from ten percent to one percent. How must count and cube side change?

**Answer:** Expected count increases by a factor of one hundred. At fixed number density the side length increases by its cube root, about 4.642.

### Exercise 2

For the worked inputs and a ten-micrometre averaging length, find expected count and Knudsen number.

**Answer:** The expected count is one billion and Knudsen number is 0.0001. They describe different quantities.

### Exercise 3

Reduce the worked macroscopic length to one micrometre. Does a declared averaging interval remain?

**Answer:** No. The upper limit is one tenth of a micrometre, below the one-micrometre lower limit. Changing only the selected width cannot fix this.

### Exercise 4

For binomial counting with N equal to one thousand and p equal to one tenth, find mean, variance and relative standard deviation.

**Answer:** They are 100, 90 and square root of ninety divided by one hundred, approximately 0.09487. The Poisson benchmark is 0.1.

### Exercise 5

May both p equal to zero and p equal to one be displayed as zero relative count noise?

**Answer:** No. At p zero the mean vanishes and the ratio is undefined. At p one a positive fixed total gives exact zero count noise.

### Exercise 6

Does passing the three declared length inequalities guarantee local equilibrium and a valid constitutive law?

**Answer:** No. Only the declared proxy conditions have passed. Correlations, time variation, boundaries and a consistent material description still require evidence.

## Sources and next connections

For background on continuum averaging and scale separation, see Natalia Berloff’s [Averaging in Fluid Dynamics](https://www.damtp.cam.ac.uk/user/tong/concepts/fluid2.pdf). For mean free paths and collision times, see David Tong’s [Kinetic Theory §1.2](https://www.damtp.cam.ac.uk/user/tong/kintheory/one.pdf). The compact kernel, finite particle experiments and declared proxy inequalities here are this textbook’s constructions; the sources do not certify the implementation.
