## How can smooth finite curves develop a corner?

A finite sum of positive weights can vary smoothly with a control parameter. Nevertheless, a sequence of such sums can approach a function with a corner as the number of sites grows. The interesting question is what happens to the probabilities and fluctuations while this limit is taken.

Imagine a field that is allowed to be completely empty or completely occupied. Increasing a small preference for occupancy changes which of these two alternatives dominates. A larger field amplifies the preference because it contributes once for every occupied site. At any finite size, both alternatives retain positive probability. In the size limit, a fixed preference selects one of them.

This chapter constructs that statement from an explicit two-configuration law. We derive its finite rounding, its size-dependent approaches to coexistence and its complex partition zeros. The prerequisites are finite probability sums, derivatives, logarithms and the polar representation of a complex number. The calculation is exact enough to distinguish a tiny positive correction from a displayed zero.

## Specify the two configurations before writing a partition function

There are N binary sites, but only two allowed configurations. In configuration E every site is zero; in configuration O every site is one. Each has reference weight one before the control field is applied. Mixed configurations are forbidden in this model.

```text
          E: ALL EMPTY                    O: ALL OCCUPIED
       [0] [0] [0] [0]                   [1] [1] [1] [1]
       [0] [0] [0] [0]                   [1] [1] [1] [1]

          count K=0                         count K=N
          weight 1                          weight z^N

            No intermediate configurations are allowed.
```

The diagram illustrates a two-dimensional arrangement for readability, but the law uses only the site count. It does not define a distance, a nearest-neighbor energy or a boundary interaction. The word phase here names the two global alternatives; it is not a derivation of a microscopic material.

[Tong's made-up partition-function example](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html) motivates looking at the factor1+z^N. His example also includes an independent background factor. We omit that factor and explicitly specify our own two allowed states, so its complete pressure and density are not the quantities calculated here. No general Lee–Yang theorem is assumed for other models.

## Normalize positive activity weights

Let z>0 be the activity and let g=ln z be its dimensionless log field. The field multiplies the weight of a configuration with count K by exp(gK). The normalizing sum is

$$
Z_N(z)=1+z^N=1+e^{Ng},\qquad
p_N=\mathbb P(O)=\frac{z^N}{1+z^N},\qquad
\mathbb P(E)=1-p_N.
$$

For rational z=A/B with positive integer weights, compute z^N as A^N/B^N and normalize using A^N+B^N. There are no binomial multiplicities because each allowed configuration is named exactly once.

The independent-site partition function would be(1+z)^N and would include configurations with intermediate occupancy. For N=4,z=2, our sum is17, whereas the independent sum is81. Confusing these two sums changes the state space before any numerical approximation has occurred.

The calculator accepts N from1 to512 and activity weights from1 to one million. A chosen size path can replace the fixed activity by a prescribed approach to one. Those finite input bounds do not limit the mathematical family to512 sites.

## Separate density fluctuation from total-count fluctuation

Write the density as ρ=K/N. In the two allowed states its values are zero and one. Therefore its variance does not receive an automatic factor1/N:

$$
\mathbb E[\rho]=p_N,\qquad
\operatorname{Var}(\rho)=p_N(1-p_N),\qquad
\mathbb E[K]=Np_N,\qquad
\operatorname{Var}(K)=N^2p_N(1-p_N).
$$

At z=1 both configurations are equally probable, regardless of N. The density variance is then1/4. Making the occupied configuration larger does not create independent pieces within it. The entire field still makes one binary choice.

For N=4, the following exact ledger compares a field, its reciprocal and coexistence. The response column will be derived next. The final column is the exponential of N times the pressure correction, not the pressure correction itself.

| z | ZN | pN | Var(ρ) | dpN/dg | exp(NΔN) |
| --- | --- | --- | --- | --- | --- |
| 1/2 | 17/16 | 1/17 | 16/289 | 64/289 | 17/16 |
| 1 | 2 | 1/2 | 1/4 | 1 | 2 |
| 2 | 17 | 16/17 | 16/289 | 64/289 | 17/16 |

The reciprocal change exchanges the occupied and empty probabilities. Their product, the density variance, stays unchanged. The pressure itself does change because the favored branch changes.

## Derive the response at fixed size

Define the dimensionless pressure per site as φN=ln(ZN)/N. Differentiating at fixed N gives

$$
\phi_N'(g)=p_N,\qquad
\phi_N''(g)=\frac{dp_N}{dg}=Np_N(1-p_N),\qquad
\frac{d\mathbb E[K]}{dg}=N^2p_N(1-p_N)=\operatorname{Var}(K).
$$

The last identity holds because the control was explicitly defined as an exponential tilt exp(gK) of fixed reference weights. It is not a rule that every variance equals a derivative with respect to any input knob. In the previous spatial-mixture chapter, changing p modified the preparation law in a different way.

Also distinguish differentiation in g at a fixed N from moving along a size path gN. A path changes the system size and the field simultaneously. The response displayed at each point is the fixed-size derivative, not a total derivative along that path.

## Extract a small pressure correction without subtraction

For any real g, factor out the larger of1 and exp(Ng). Set t=exp(−N|g|), which lies in(0,1]. Then

$$
\phi_N(g)=\max(0,g)+\Delta_N(g),\qquad
\Delta_N(g)=\frac{\ln(1+t)}N,\qquad t=e^{-N|g|}.
$$

This representation gives a uniform pressure error bound:

$$
0<\Delta_N(g)\le\frac{\ln2}{N},\qquad
\phi_N(g)\longrightarrow\max(0,g).
$$

Equality in the upper bound occurs at g=0. At every finite positive z and finite N, the correction is strictly positive. The limiting pressure has a corner at g=0, although every finite curve is smooth on the real field axis.

For z=2,N=4, the correction is ln(17/16)/4. Computing it by subtracting two nearly equal floating-point pressures is unnecessary. Compute the small logarithm directly. The exponential exp(NΔN)=1+t is an exact rational when z is rational, but ΔN is generally not rational.

## Measure rounding in the coordinate actually used

The density crosses from near zero to near one over a shrinking field interval. Solve pN=0.1 and pN=0.9 using the odds pN/(1−pN)=exp(Ng):

$$
g_{0.1}=-\frac{\ln9}{N},\qquad
g_{0.9}=\frac{\ln9}{N},\qquad
w_g=g_{0.9}-g_{0.1}=\frac{2\ln9}{N}.
$$

Doubling N halves this10–90% width in g. The corresponding activity endpoints are9^(−1/N) and9^(1/N); their difference is a different coordinate width. Neither width is a relaxation time.

A useful plot places g on the horizontal axis and pN on the vertical axis for several sizes. The curves pass through(0,1/2), become steeper there and retain smooth finite transitions. A second plot of φN shows their pressure curves close to the cornered limiting branch. Reading one plot does not replace inspecting the other observable.

## Coexistence keeps macroscopic uncertainty

At exactly g=0, increasing N gives

$$
p_N=\frac12,\qquad
\operatorname{Var}(\rho)=\frac14,\qquad
\phi_N''(0)=\frac N4,\qquad
\Delta_N(0)=\frac{\ln2}{N}.
$$

The pressure correction vanishes while the density variance stays fixed and the response diverges. Thus a uniformly small pressure error does not establish uniform convergence of its derivatives. It also does not establish self-averaging of the density.

For fixed g<0, pN tends to zero; for fixed g>0 it tends to one. In those cases the density law approaches one deterministic endpoint and its variance vanishes. At coexistence, its law remains a half-and-half mixture of the endpoints. Its mean1/2 is not an allowed configuration with half the sites occupied.

## Approach coexistence along a declared size path

The fixed-activity limit and a moving-activity limit answer different questions. Define0≤c≤1 and a side s=+1 or−1. The two approach families use

$$
z_N=(1+c/N)^s\quad\text{or}\quad z_N=(1+c/N^2)^s,\qquad g_N=\ln z_N.
$$

Both approach z=1, but their scaled fields differ. From ln(1+x)/x→1,

$$
Ng_N\longrightarrow sc\quad\text{on the inverse-size path},\qquad
Ng_N\longrightarrow0\quad\text{on the inverse-square path}.
$$

Writing L(x)=1/(1+exp(−x)), the limits are

| Path | Limit of NgN | Limit of pN | Limit of Var(ρ) |
| --- | --- | --- | --- |
| Fixed z<1 | −∞ | 0 | 0 |
| Fixed z=1 | 0 | 1/2 | 1/4 |
| Fixed z>1 | +∞ | 1 | 0 |
| (1+c/N)^s | sc | L(sc) | L(sc)[1−L(sc)] |
| (1+c/N²)^s | 0 | 1/2 | 1/4 |

On either moving path the pressure tends to zero. On the inverse-size path, the density limit remembers c and the side. With c=0 both paths are exactly at coexistence at every size. In the calculator, moving paths use the offset weights c=C/D and the side; fixed-activity weights are retained but unused. On the fixed path, the offset and side are unused.

## A small field can remain inside the rounding window

Take c=1/2,N=4. On the positive inverse-size path, z=9/8 and z^4=6561/4096. The occupied probability is6561/10657. On the positive inverse-square path, z=33/32 and the probability is1185921/2234497. The two finite fields are positive, yet neither law is close to a deterministic occupied endpoint.

The reason is that the relevant crossover variable is Ng. A positive gN that tends to zero need not select occupancy in the size limit. If it stays of order1/N, it stays within a rounding window of the same order. If it shrinks faster, the limiting law approaches the symmetric mixture.

This gives a concrete order-of-limits distinction:

$$
\lim_{g\to0^+}\lim_{N\to\infty}p_N(g)=1,\qquad
\lim_{N\to\infty}p_N(0)=\frac12,\qquad
\lim_{N\to\infty}p_N(g_N)=L(c)\quad(z_N=1+c/N).
$$

There is no contradiction. Each expression specifies a different experiment. A claim about a thermodynamic limit must say which controls are fixed and which are allowed to vary with size.

## Continue the polynomial into the complex plane

Positive real activity gives positive weights and a nonzero partition function. Complex activity is a mathematical continuation of the polynomial; its terms are no longer probabilities. Its zeros satisfy z^N=−1, so all roots are

$$
z_j=\exp\left[2\pi i\frac{2j+1}{2N}\right],\qquad j=0,\ldots,N-1.
$$

Multiplying each turn fraction by N gives an integer plus1/2. Therefore zj^N=−1 exactly. The roots are distinct and there are N of them, completing the roots of a degree-N polynomial. The conjugate index is N−1−j. The negative real root−1 occurs only when N is odd; no root is positive real.

| N | Exact root turns | Negative real root |
| --- | --- | --- |
| 1 | 1/2 | yes |
| 2 | 1/4, 3/4 | no |
| 3 | 1/6, 1/2, 5/6 | yes |
| 4 | 1/8, 3/8, 5/8, 7/8 | no |

For N=4, a schematic complex-plane view is

```text
                       Im z
                         ^
                  o      |      o
                         |
           -1 -----------+----------- 1 ----> Re z
                         |
                  o      |      o

           The four roots lie on the unit circle.
           No finite-size root sits at the point z=1.
```

The drawing is schematic rather than a numerical coordinate certificate. It helps locate the roots relative to the positive real axis, where the probabilities were defined.

## Track how the closest zeros approach one

The closest root angles to the positive axis are±π/N. Their distance to z=1 and the corresponding smallest log-field imaginary distance are

$$
\theta_{\min}=\frac\pi N,\qquad
|z_{\rm near}-1|=2\sin\frac{\pi}{2N},\qquad
|\operatorname{Im}g_{\rm near}|=\frac\pi N.
$$

The complex logarithm has branches differing by2πi. The last expression selects the nearest representative. At N=1 the negative real root has arguments±π, both describing the same point. It should not be counted twice.

The zeros move toward the coexistence point as size grows while never sitting there at finite N. This explicit polynomial demonstrates the relationship between a shrinking real-axis rounding window and approaching complex zeros. It does not prove a circle theorem or classify transitions in arbitrary Hamiltonians.

The calculator retains exact turn fractions and the polynomial phase identity. Sine, cosine and root distances are approximate drawing values. A small residual obtained from floating-point coordinates is a numerical check, not the proof that the root satisfies the polynomial.

## Preserve a tiny positive logarithm

At z=10^6,N=512, t=10^−3072. A floating-point representation of1+t rounds to one, but its logarithm is positive. Instead of forming that rounded sum, write

$$
q=\frac{t}{2+t},\qquad
\ln(1+t)=2q\sum_{j=0}^{\infty}\frac{q^{2j}}{2j+1},\qquad0<q\le\frac13.
$$

Keep q as an exact rational. The remaining series is bounded near one, so directed fixed-point intervals can enclose it without erasing the scale of q. After terms j=0 through72, its omitted tail obeys

$$
0<R_{72}\le\frac{9}{8(147)9^{73}}.
$$

This follows by bounding every omitted denominator below by147 and summing a geometric series with ratio at most1/9. Multiplying the resulting interval by the exact2q/N leaves a strictly positive pressure-correction lower bound, even when the decimal display cannot represent the value.

Exact finite probabilities have rational point bounds. Logarithms have nonzero-width certified intervals. Neither the display projection nor a high-precision reference decimal replaces those bounds. In particular, exp(NΔN) may display as1 while ΔN is certified positive. That combination reflects limited display resolution rather than an inconsistency.

## Connect the example to model analysis carefully

The two configurations can serve as a simple analogy for a model choosing between global alternatives. A small score preference multiplied over many sites can create a sharp change in the dominant alternative. The example suggests checking size scaling, fluctuations and parameter paths when interpreting a sharp observed curve.

It does not predict an AI system's training transition or justify fitting an arbitrary loss curve to a thermodynamic law. The state space and field coupling were explicitly chosen here. A research application would need to identify its own states, weights, observables, dependence assumptions and evidence for the relevant limit.

The three families in this topic now expose different pitfalls: energy scaling does not settle additivity, equal one-site marginals do not settle self-averaging, and small pressure errors do not settle derivative or fluctuation limits. Each conclusion depends on a defined comparison rather than a visual resemblance.

## Connect the two configurations and complex zeros to the experiment

At the default N=4,z=2, the two allowed weights are1 and16, so occupied probability is16/17. Adding intermediate occupancies changes the model. Read the two-configuration table before comparing pressure, response and roots: these describe probabilities, a derivative at fixed size and a complex polynomial respectively.

| Action | Observation to inspect | Question to explain |
| --- | --- | --- |
| Compare activity two, reciprocal activity and coexistence examples | Occupied probabilities are16/17,1/17,1/2; the first two variances both equal16/289 | Can exchanging the favored configuration preserve its fluctuation? |
| Compare positive inverse-size and inverse-square paths while increasing size | Ng, limiting density and the fixed-N field response | Does a positive field at every finite size force limiting occupied probability one? |
| Choose ‘Odd degree, negative real root’ and select j=1 | At N=3, exact turns and powered turns modulo one both equal1/2 | How does exact phase verify the root−1 without relying on rounded coordinates? |

The root slider selects a root at the requested current N without changing N. ‘Log pressure-correction display’ reveals tiny positive DeltaN. Even if exp(NDeltaN) displays as one, open the certificate to inspect the positive lower bound for DeltaN. Activity or offset inputs disabled on a path remain stored and unused by that path. Size progress is not time, and complex-root terms are not probabilities.

## Six exercises with explanations

### 1. Reconstruct the four-site probability ledger

For N=4,z=2, compute ZN,pN, the density variance, the total-count variance and the density response to g. Compare the independent-site normalizer.

**Explanation.** The two weights are1 and16, so ZN=17 and pN=16/17. The density variance is16/289; multiplying by16 gives total-count variance256/289. The density response is four times the density variance,64/289. The independent-site sum is3^4=81 because it has a different state space.

### 2. Compare shrinking pressure error with persistent fluctuation

At z=1, determine how ΔN,Var(ρ) and φN″ change when N doubles. Is a small ΔN sufficient to conclude concentration of density?

**Explanation.** ΔN=ln2/N halves. Var(ρ)=1/4 does not change, while φN″=N/4 doubles. The law still places half its probability on each global configuration. A small pressure correction alone does not imply concentration or convergence of derivatives.

### 3. Identify the path hidden in a positive small field

At c=1/2,N=4, compute the positive inverse-size and inverse-square activities and occupied probabilities. Give the limits of pN along each path.

**Explanation.** The activities are9/8 and33/32. Raising to the fourth power and normalizing gives6561/10657 and1185921/2234497. Their limits are L(1/2) and1/2, respectively. Fixed positive g would instead give limit1. Positivity of each finite field is insufficient to identify the limiting law.

### 4. Derive and label the rounding width

Solve for the fields giving pN=0.1 and0.9. Does the resulting width describe activity units or a time interval?

**Explanation.** Odds are1/9 and9, so Ng is−ln9 or+ln9. The field width is2ln9/N. In activity coordinates the width would be9^(1/N)−9^(−1/N). Neither expression describes a dynamical relaxation time because no time evolution was defined.

### 5. Verify roots without trusting the drawing

List the turn fractions for N=3 and N=4. Which size has a root at−1? How can every listed root be verified exactly?

**Explanation.** For N=3 the turns are1/6,1/2,5/6; for N=4 they are1/8,3/8,5/8,7/8. Only N=3 includes the half-turn−1. Multiplying each fraction by its N gives an integer plus1/2, proving z^N=−1. This exact phase argument does not require rounded sine or cosine coordinates.

### 6. Explain a displayed one with a positive logarithmic correction

At N=512,z=10^6, explain why displaying exp(NΔN) as1 cannot prove ΔN=0. What quantity should preserve the tiny scale before interval evaluation?

**Explanation.** Here t=10^−3072 and exp(NΔN)=1+t rounds to one in ordinary floating point. Keep q=t/(2+t) exact and enclose the bounded series multiplying2q. Scaling by1/N gives a positive interval for ΔN. The exact rational, its logarithm interval and its display projection are different representations with different contracts.

The current reader connects these families through guided experiments, concept views and assessment. Keep exact probabilities, certified intervals and approximate drawings distinct, and continue through Book → Interactive Book.
