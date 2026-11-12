## Brightness is a sequence of physical conversions

A camera does not observe a noiseless real number called brightness. Light brings a random count of quanta; a sensor collects electrons; electronics add noise and convert charge to a digital code. The same final code can result from many different physical histories.

We will build a forward model with explicit units and operation order. It is detailed enough to expose shot noise, dark current, read noise, gain, quantization and two kinds of saturation. It is still an educational model, not a characterization or compliance claim for a real camera. Distribution calculations will use probability sums rather than treating one random simulated exposure as ground truth.

## Start with a calibrated exposure

For constant monochromatic irradiance $E$ over area $A$ and time $T$,

$$
\mu_p=\frac{EAT\lambda}{hc}.
$$

Use the previous lesson's area/time integral when illumination varies. Here $\mu_p$ is the incident photon mean already integrated over the exposure. Changing $T$ in the sensor model changes the dark contribution; a physical experiment that changes exposure must also recompute $\mu_p$ from its illumination.

For $A=25\,\mu\mathrm m^2$, $T=10$ ms, $E=0.001$ W/m² and wavelength 550 nm, $\mu_p\approx 692.19$. Zero exposure requires zero integrated incident photons. It does not necessarily eliminate read noise from a readout performed without illumination.

## Declare the photon statistics

Assume independent arrivals with a fixed mean during an exposure:

$$
N_p\sim\operatorname{Poisson}(\mu_p),\qquad
P(N_p=n)=e^{-\mu_p}\frac{\mu_p^n}{n!}.
$$

Then both the mean and variance are $\mu_p$. A count is dimensionless, but naming photons and electrons prevents confusing two different conversion stages.

Poisson statistics are an assumption about the preparation, not a universal property of all light. Correlated sources, fluctuating illumination and multiplication processes can require other distributions. Do not conclude that every deviation from Poisson variance is a broken sensor implementation.

## Quantum efficiency thins the photon process

With effective quantum efficiency $0\leq q\leq 1$, each incident photon independently produces one collected electron with probability $q$. Poisson thinning gives

$$
N_e\sim\operatorname{Poisson}(\mu_e),\qquad\mu_e=q\mu_p.
$$

At $q=0.6$, the calibrated example gives approximately 415.31 photoelectrons. The expected number need not be an integer; each realized count is an integer. QE defined per total pixel area already includes collection losses associated with that definition. Applying a fill factor again would double-count losses. Alternatively one may define an active area and an intrinsic collection efficiency, but those conventions must not be mixed.

## Dark current contributes both a mean and a variance

Let dark rate $d$ have units electrons/s. In this model,

$$
N_d\sim\operatorname{Poisson}(dT),\qquad N=N_e+N_d
 \sim\operatorname{Poisson}(\mu_e+dT).
$$

Photo and dark processes are independent. Subtracting a known mean dark level removes an offset in expectation; it does not remove the random dark-electron variance. Estimating that dark level from a finite noisy calibration can add further uncertainty, which is outside the current ideal-known-parameter calculation.

The temperature dependence of dark rate is not fitted here. A fixed dark rate represents one declared operating condition.

## Add read noise in electron units

Take independent read noise $R\sim\mathcal N(0,\sigma_r^2)$. Before saturation,

$$
Z=N+R,\qquad E[Z]=\mu_e+dT,
\qquad\operatorname{Var}(Z)=\mu_e+dT+\sigma_r^2.
$$

Read noise can make this electronic measurement negative relative to a reference level. That does not mean a negative number of physical photoelectrons arrived. Keep the physical integer charge and the signed readout fluctuation conceptually separate.

For $\mu_e=6$, $dT=0.2$ and $\sigma_r=2$ electrons, the variance is 10.2 electrons squared. Adding standard deviations instead would incorrectly produce a different noise model.

## Gain changes units, not the incoming information

Let gain $K$ have units DN/electron and offset $b$ have units DN. The unsaturated analog-valued digital scale is

$$
Y=b+KZ.
$$

Its mean is $b+K(\mu_e+dT)$ and variance is $K^2(\mu_e+dT+\sigma_r^2)$. With $K=0.7$ and $b=8$, the preceding example has mean 12.34 DN and variance 4.998 DN².

Doubling gain doubles the signal and standard deviation before quantization. It does not double the photon count or the underlying signal-to-noise ratio. It can change the effect of a finite ADC step and cause earlier ADC clipping.

## Calculate a conditional signal-to-noise ratio

For a known mean dark level and no saturation or quantization, define

$$
\operatorname{SNR}_{\rm preADC}
=\frac{\mu_e}{\sqrt{\mu_e+dT+\sigma_r^2}}.
$$

In the shot-limited regime this approaches $\sqrt{\mu_e}$. Four times as many photoelectrons doubles SNR. In a read-noise dominated regime, increasing exposure initially helps more quickly because the fixed per-read variance becomes less important.

If both numerator and denominator are zero, the model reports an undefined ratio instead of calling the absence of a signal infinitely good. This pre-ADC SNR is not presented as the exact SNR of a clipped digital image.

## State where full-well saturation occurs

We explicitly choose the charge-storage model

$$
C=\min(N,F),\qquad Y=b+K(C+R),
$$

where integer $F$ is a storage cap in electrons. Dark and photo charge share this cap; read noise is added afterwards. We calculate $P(N>F)$ as the overflow probability. Being exactly at capacity is different from overflowing it.

This is a declared idealized operation order. Real devices can bloom, become nonlinear or alter read-noise behavior. The clipping model does not infer those behaviors. After charge saturation, use the calculated distribution of $C$, not the unsaturated Poisson variance formula, to describe the stored charge.

## Quantization and ADC clipping are separate operations

For a $B$-bit unsigned converter, $M=2^B-1$. The rule is

$$
D=\min\{M,\max[0,\lfloor Y+1/2\rfloor]\}.
$$

Half-code ties round upward. The code 0 bin includes all $Y<0.5$; the codeM bin includes all $Y\geq M-0.5$. Rail occupancy is therefore different from the probability that rounding actually leaves the valid code range. Lower clipping occurs for $Y<-0.5$ and upper clipping for $Y\geq M+0.5$.

A deterministic value 15 DN in a 4-bit converter occupies the upper rail with probability one but requires no out-of-range clipping. A value 15.6 DN rounds to 16 and does require clipping to 15. This distinction matters when interpreting saturation indicators.

## Compute the digital distribution directly

Conditioned on $N=n$, define $c_n=\min(n,F)$ and $m_n=b+Kc_n$. For positive read noise, an interior code has probability

$$
P(D=j\mid N=n)=
\Phi\left(\frac{j+1/2-m_n}{K\sigma_r}\right)
-\Phi\left(\frac{j-1/2-m_n}{K\sigma_r}\right).
$$

Use half-infinite bins at the rails. Sum these conditional probabilities against Poisson weights. When $\sigma_r=0$, apply the deterministic rounding rule directly instead of dividing by zero. The resulting histogram describes repeated exposures under fixed parameters, not the distribution of one saved image's arbitrary pixel values.

Mean and variance are calculated from the digital code probabilities. Full-well and ADC clipping can reduce variance while destroying information; a smaller output variance is not automatically an improved camera.

## Disclose numerical truncation and roundoff

The Poisson distribution has infinite support. The implementation retains a finite count interval around its mode, computes probabilities by recurrence and records a Chernoff bound for omitted tail mass. It does not renormalize the retained probabilities to force their sum to one.

For codes between 0 andM, omitted probability $\epsilon$ contributes at most $M\epsilon$ to the unnormalized first moment and $M^2\epsilon$ to the second moment. These are truncation bounds, not complete floating-point error certificates. The reported probability-mass residual also reveals numerical roundoff. Normal CDF evaluation uses Gaussian quadrature; the discarded tail beyond nine standard deviations is disclosed separately in the implementation.

## Understand the limits of the 1/12 rule

An often useful approximation treats quantization error as independent uniform noise over a one-DN interval, with variance 1/12 DN². It is not an identity for every input distribution.

A completely deterministic analog value 8.2 DN always rounds to 8 DN. Its exact output variance is zero, not 1/12. Inputs near clipping rails or with narrow distributions also violate the naive independent-uniform picture. The model exposes the approximation separately and uses actual bin probabilities for digital moments.

## Use this model in an AI research workflow

A denoising loss should reflect which stage its input represents: irradiance, electron counts, analog-valued readout or quantized clipped codes. A Gaussian likelihood with fixed variance can miss the signal-dependent variance of shot noise. Clipped highlights also contain one-sided information that differs from an exact linear observation.

Useful controlled experiments vary photon exposure and gain independently, compare predicted code probabilities against held-out exposures and inspect saturation before fitting a noise slope. Do not mistake a successful educational simulation for measured camera calibration. Real research may add pixel-response nonuniformity, correlated read noise, color channels and uncertainty in estimated parameters.

## Research extension: saturation reduces information even when variance shrinks

A narrow clipped histogram is not automatically a precise observation of illumination. We quantify that statement with Fisher information about a positive Poisson mean. The derivation first isolates a noiseless storage cap, then treats read noise and ADC conversion as another known observation channel. This is supplementary likelihood analysis, not a new native estimator or measured camera calibration.

### Name the parameter and the complete-data score


$$
N\sim\operatorname{Poisson}(\lambda),\quad\lambda>0,\quad p_n=e^{-\lambda}\lambda^n/n!:\qquad s_N(n)=\partial_\lambda\log p_n=n/\lambda-1,\quad J_N=\mathbb E s_N^2=1/\lambda.
$$

Lambda is the total expected collected electron count at one fixed exposure, including a known dark contribution when present. All other parameters are held fixed. The score measures sensitivity of the log likelihood to that mean; information is the expected squared score. It is not output variance or a realized number of photons. The regular formula here excludes the boundary lambda=0.

### Derive the likelihood of a capped count


$$
C=\min(N,F),\quad F\ge1\text{ integer}:\qquad P(C=n)=p_n\ (n<F),\quad P(C=F)=r_F=\Pr(N\ge F),\quad r_F\prime=p_{F-1}.
$$

Differentiate one minus the finite sum of probabilities below F; the recurrence p-prime(n)=p(n−1)−p(n) telescopes. The top bin includes N=F as well as overflow. The native overflow probability is Pr(N>F), so it is different from this top-bin probability. Confusing them gives the wrong score and information.


$$
s_C(n)=n/\lambda-1\ (n<F),\quad s_C(F)=p_{F-1}/r_F,\qquad J_F=\sum_{n=0}^{F-1}p_n(n/\lambda-1)^2+\frac{p_{F-1}^2}{r_F}.
$$

The saturated code is a censored event, not an exact report N=F. Its score is the derivative of log(rF), which remains positive. Retaining this tail likelihood uses the one-sided information that clipping leaves, without pretending the original count was observed.

### Identify the exact information that was discarded


$$
s_C(C)=\mathbb E[s_N(N)\mid C],\qquad\frac1\lambda-J_F=\mathbb E\operatorname{Var}(s_N\mid C)=\frac{r_F}{\lambda^2}\operatorname{Var}(N\mid N\ge F)>0.
$$

Differentiate the grouped probabilities to obtain the conditional-score identity, then use the law of total variance and E score=0. Only the top bin merges distinct counts. For finite F and lambda>0 its conditional variance is positive, so grouping strictly loses information about lambda. A smaller stored-count variance cannot reverse this inequality.

The table uses F=3, read sigma zero, gain one, offset zero and a two-bit ADC. Then the native digital code equals C exactly. Its upper-rail probability is r3, whereas storage overflow is r3−p3. The information efficiency lambda J3 compares with the complete Poisson observation at the same mean. Values are rounded to nine significant digits.

| mean lambda | capped information J3 | efficiency lambda J3 | upper-rail probability |
|---|---|---|---|
| 0.5 | 1.99165959 | 0.995829794 | 0.014387678 |
| 2 | 0.429594981 | 0.859189961 | 0.323323584 |
| 5 | 0.0667240463 | 0.333620232 | 0.875347981 |
| 10 | 0.00187110431 | 0.0187110431 | 0.997230604 |

```python
from math import exp

# No read noise; gain=1, offset=0, two-bit ADC, full well F=3.
for mean in [.5, 2, 5, 10]:
    cap = 3
    probabilities = [exp(-mean)]
    for n in range(1, cap):
        probabilities.append(probabilities[-1]*mean/n)
    tail = 1-sum(probabilities)  # Stable enough for these stated rows.
    information = sum(p*(n/mean-1)**2
                      for n, p in enumerate(probabilities))
    information += probabilities[-1]**2/tail
    print(mean, *(format(x, '.9g') for x in
          [information, mean*information, tail]))
```

Subtracting a finite sum is adequate for these table rows. Extremely small tail probabilities require a positive tail recurrence or another stable method. The independent verifier uses positive Poisson terms through count 256; for its largest mean 20 the omitted upper-tail Chernoff bound is below 1e−180. No retained distribution is silently renormalized.

### Treat electronics as a known observation channel


$$
Q(d\mid c)\text{ independent of }\lambda,\quad P_D(d)=\sum_cQ(d\mid c)P_C(c):\qquad s_D(d)=\mathbb E[s_C\mid D=d],\quad J_D=\sum_d\frac{[P_D\prime(d)]^2}{P_D(d)}\le J_F<1/\lambda.
$$

Read noise, known gain and deterministic rounding/clipping after the charge cap form such a channel in this model. The same conditional-score argument proves contraction. Use bins of positive probability; zero bins carry no probability mass under the model. Unknown noise, gain or offset parameters require a joint inference problem, not this one-parameter comparison. This result does not assume that quantization always adds 1/12 variance.


$$
K\text{ independent equal-parameter exposures}:\qquad J_F^{(K)}=KJ_F,\quad\frac{J_F^{(K)}}{J_N^{(K)}}=\lambda J_F.
$$

Independent scores add and have zero cross covariance. More exposures add information but do not restore the information efficiency lost per exposure. This comparison fixes the mean per exposure; changing total time, read count or photon budget is a different experimental design. No universal estimator error bar or globally unbiased finite-sample estimator is asserted here.


$$
C_1=\cdots=C_K=F:\qquad L(\lambda)=r_F(\lambda)^K,\quad L\prime(\lambda)=Kr_F^{K-1}p_{F-1}>0,\quad\sup_{\lambda>0}L(\lambda)=1.
$$

The capped family is still structurally identifiable for lambda>0: P(C=0)=exp(−lambda) is one-to-one. Reduced information does not mean identical full distributions. The following all-rail case is a finite-data likelihood issue.

For all-rail noiseless observations this likelihood increases toward its supremum as lambda tends to infinity; it has no finite maximum-likelihood estimate on the unbounded parameter space. Calling the measured cap the true mean is unjustified. A finite prior, bounded parameter range or other exposure can produce a finite inference, but each is additional information. This example concerns C or the noiseless identity ADC above, not every noisy rail event.


$$
\lambda=q\mu_p+dT,\quad q,d,T\text{ known}:\qquad J_{\mu_p}=q^2J_D,\quad q=0\ \Longrightarrow\ J_{\mu_p}=0.
$$

The chain rule separates information about incident photons from information about the collected mean. With zero QE, dark and read noise may still produce a nontrivial histogram but it carries no information about the incident mean in this stated model. The Poisson preparation, independent electronics and known calibration are assumptions; correlated light and unknown nuisance parameters need other likelihoods. The EMVA source supplies linear sensor terminology, not this censoring-information proof or a camera compliance claim.

![Complete Poisson information exceeds capped-count information and subsequent noisy-code information; shrinking capped variance at high illumination accompanies loss of information efficiency.](/assets/learning/waves-sampling-censored-information.en.svg)

### Research questions with explanations

**Does a narrow saturated histogram mean high precision about illumination?** **Answer:** No. Grouping counts reduces Fisher information by the conditional score variance, even when stored output variance becomes small.

**Is the upper-bin likelihood the overflow probability?** **Answer:** No. The cap bin includes N=F as well as N>F. Its correct probability is Pr(N≥F); using strict overflow changes the likelihood.

**Do all-rail observations give a finite maximum-likelihood estimate?** **Answer:** Not in this unbounded noiseless-cap model. Their likelihood is monotone, so a finite estimate needs additional exposure data, a bound or a stated prior.

## Explained exercises

### 1. Distinguish expected and observed counts

A mean of 415.31 electrons is allowed because it is an expectation over repeated exposures. Each photoelectron count remains integer. Rounding the mean before constructing the Poisson distribution changes the model unnecessarily.

### 2. Remove a dark offset

If the known dark mean is 0.2 electrons, subtracting 0.2 removes its expectation. Its variance 0.2 remains. A separately measured dark frame has its own uncertainty and can increase the variance of the difference.

### 3. Increase exposure or increase gain

In a shot-limited regime, increasing physical exposure fourfold raises signal fourfold and standard deviation twofold. SNR doubles. Increasing gain fourfold raises both the DN signal and its standard deviation fourfold, leaving the pre-ADC SNR unchanged.

### 4. Identify two saturation mechanisms

With mean 100 electrons, full-well 12 electrons and an 8-bit gain-one converter, stored charge is almost always 12 while the ADC remains far from 255. With full-well 8192 and a 4-bit converter, the ADC instead clips near 15 before the storage cap matters. Neither mechanism is described by one generic saturation number.

### 5. Test the uniform quantization approximation

Set photon mean, dark rate and read sigma to zero, with offset 8.2 DN. Every output is 8 DN, so variance is zero. This counterexample disproves the claim that quantization always adds exactly 1/12 DN².

### 6. Interpret a narrow histogram

A bright, heavily clipped preparation has most probability at the maximum code and a small digital variance. It has lost distinctions between many high-input states. Compare its forward likelihood and overflow/clipping probabilities rather than declaring it less noisy from output variance alone.

## Sources and scope

[EMVA 1288 Release 4.0 Linear, §§2.1 and 2.4–2.7](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) provides terminology and a linear sensor-noise framework. The explicit storage cap, deterministic ADC bins, numerical probability sums and counterexamples here are the lesson's stated educational extensions. The model does not implement the standard's measurement procedures, certify a camera, or cover all non-Poisson light and sensor architectures.
