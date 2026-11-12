## Start with a physical state and an observation map

A physical state describes quantities in the model; an observation describes what a sensor or data pipeline reports. They need not contain the same information. This lesson keeps the previous exchange-then-leak law fixed and compares two candidate initial states in kilograms: world zero starts from (8,2,0), and world one from (2,8,0).

The transfer parameters are known and identical across the candidates. The interface does not generate measured data or assign prior probabilities to the worlds. It compares their predicted, noiseless outputs over a declared set of recording times. A real dataset might support one candidate, both, or neither; equality of candidate predictions is an information-set statement rather than an automatic estimate of reality.

## Separate the boundary from the sensor

The available sensors report A, B, A+B, or A−B. They are observation functions applied to a three-component state:

$$
y_n=H x_n,\qquad x_n=(A_n,B_n,E_n)^{\mathsf T}.
$$

For example, the sum sensor uses H=(1,1,0), whereas the difference uses H=(1,−1,0). The difference can be negative. It is a derived signed measurement with kilogram units, not the nonnegative mass stored inside a physical region.

An accounting boundary determines which transfers are internal or external in a balance equation. A sensor determines which combinations of the state are recorded. Sometimes a sensor measures the mass inside the selected boundary, but the two choices remain conceptually different. Changing the sensor here does not change the physical trajectory.

## Make the recording window part of the information

The controls select an inclusive first and last recorded minute between zero and twenty. A minute-zero reading is an observation of the initial state. Starting at minute one omits that information. A prediction that would distinguish two worlds at an unrecorded time does not distinguish them using the available records.

If the first selected minute exceeds the last, the window contains no records. The interface reports the equality comparison as unavailable. It does not turn an empty comparison into a positive claim that the worlds have equivalent observations. This prevents the logical convention that every predicate holds on an empty set from becoming a misleading empirical conclusion.

Line segments help connect discrete records visually. The model specifies one operation each minute; the drawing does not assert an additional continuous-time trajectory between the records.

## Prove why a total sensor can hide internal states

With no leakage, A+B is conserved for every permitted exchange fraction. Both initial worlds have total ten kilograms, so the sum sensor reports ten at every recorded minute:

$$
H x_n^{(0)}=H x_n^{(1)}=10\quad\text{for }H=(1,1,0),\ \ell=0.
$$

The hidden A and B values can still differ. In the default example with exchange fraction 0.2, the initial imbalance D=A−B evolves by multiplying by 0.6 each step. The two worlds have opposite imbalances and the same total. Their sum records agree even while their internal states remain different.

This is a counterexample to inferring state equality from equality of a particular observation. It does not say every sensor is uninformative. Changing only the sensor to A includes an initial reading of eight versus two kilograms, immediately distinguishing these two hypotheses.

## Derive a delayed-record information loss

Without leakage the imbalance follows

$$
D_{n+1}=(1-2p)D_n,\qquad
A_n=\frac{10+D_n}{2}.
$$

At p equal to one half, the imbalance becomes zero after one operation. Both worlds then have A and B equal to five. If A is first recorded at minute one, every available A record is identical across the two initial worlds. Including minute zero distinguishes them because the initial A values differ.

In this case the modeled states themselves merge after the operation. The ambiguity is not merely a hidden difference persisting in another compartment. The declared lumped transfer map loses the initial distinction. This is a property of the coarse state model and does not establish a claim about the microscopic reversibility of every physical realization.

The separate latent-state graph shows candidate A histories, not measured or reconstructed states. It provides an explanation of the ambiguity while preserving the observer's restricted information set.

## Let leakage reveal a different distinction

Changing the physical law can change what a sensor reveals. With exchange fraction 0.2 and leakage fraction 0.1, the first world's mixed B is 3.2 and the second world's mixed B is 6.8 kilograms. Their environmental transfers are 0.32 and 0.68. Consequently A+B after the first minute is 9.68 versus 9.32.

The sum sensor now distinguishes the selected worlds at minute one, although their initial sum readings are still equal. This does not justify a general claim that leakage always identifies every state or parameter. It demonstrates that information depends jointly on the physical evolution, the observation function, and the recording times.

A different sensor can remain blind. With full leakage, B is zero after every completed operation. B-only records beginning at minute one can agree while A remains different. The choice of what is measured is part of the inference problem.

## Compare exact records before rounding their displays

The implementation represents supported transfer histories with integer numerators and known common denominators. At each recorded minute it compares observation numerators exactly, rather than comparing their rounded decimal displays.

For exchange fraction 0.49, no leakage, and a single A record at minute twenty, the two candidate outputs differ by

$$
6(0.02)^{20}\ \mathrm{kg}.
$$

This is positive but extremely small. Both floating-point trace coordinates can appear as five, while the exact integer comparison still detects the gap. The maximum-gap readout is computed from the exact numerator difference before conversion to a display number, avoiding a subtraction of already rounded values.

Exact mathematical discrimination is not a promise of practical sensor resolution. Noise, quantization, uncertain parameters, and finite arithmetic in a real instrument can remove a distinction that exists in this idealized information model. Those effects require their own observation law.

## Do not promote a two-hypothesis result into full identification

The interface asks whether these two fixed initial-state hypotheses give equal available records. A nonzero gap distinguishes them under the declared exact model. It does not prove unique reconstruction among all possible initial states, unknown exchange fractions, or alternative physical laws.

Likewise, equal records do not establish that one candidate is the true state. If actual measurements disagree with their shared predicted record, both candidates may be wrong. A general identification problem requires a specified admissible state or parameter family and a proof about that whole family, rather than a comparison of two convenient examples.

The end-state equality readout is also separate from record equality. Worlds may have different end states but equal projected records, or different initial states may merge and then have equal end states. The labels distinguish these propositions explicitly.

## Connect information limits to physics-based learning

A model trained only on a conserved total may learn that total accurately while leaving internal transfer behavior unconstrained. Many initial states, and sometimes many parameter values, can produce the same observed history. A conservation loss does not automatically supply missing state information.

Choosing an additional sensor or changing the recording schedule can be more informative than merely repeating the same uninformative output. The appropriate design depends on the model. The half-exchange example shows why recording an initial transient can matter, while the tiny-gap example shows why theoretical distinguishability must also be evaluated against noise and resolution.

## Research extension: Quantify the information lost to observation noise

### Add a probability law rather than a vague noise warning

Retain the two initial worlds (8,2,0) and (2,8,0), a known exchange fraction p, zero leakage, and the A sensor. Now suppose each selected reading has fresh independent Gaussian noise of known positive standard deviation sigma kilograms. Give the worlds equal prior probability. These are new assumptions for the following calculation; the existing laboratory still compares exact noiseless records and does not sample this noise.

At recorded integer minute n, the candidate mean gap is Delta_n=6(1−2p)^n kilograms. For any nonempty finite window, collect the means in vectors m_0 and m_1 and let Sigma=sigma squared times the identity matrix. Expanding the two Gaussian log densities yields the minimum-error decision statistic below. Choose world zero when T is positive.

$$
T=(m_0-m_1)^T\Sigma^{-1}\left(Y-{m_0+m_1\over2}\right),\qquad
d^2=(m_0-m_1)^T\Sigma^{-1}(m_0-m_1)
={36\over\sigma^2}\sum_{n=s}^{e}(1-2p)^{2n}.
$$

Under world zero, T has mean d squared over two and variance d squared; under world one its mean changes sign with the same variance. Standardizing this scalar Gaussian gives the balanced Bayes error. If d=0, the record laws agree and any best decision has error one half.

$$
R^*=\Phi(-d/2),\qquad
\text{advantage}=\tfrac12-R^*=\tfrac12\operatorname{erf}\!\left({d\over2\sqrt2}\right).
$$

The advantage measures improvement over guessing with equal priors. This formula remains numerically useful when subtracting the rounded error from one half would lose the entire difference. It concerns the selected pair and the specified Gaussian law, not identification of every state or robustness to arbitrary noise. For the general decision-rule framework see [MIT's hypothesis-testing lecture](https://ocw.mit.edu/courses/6-011-signals-systems-and-inference-spring-2018/e6055996bacfce70426cfb7a080c03aa_MIT6_011S18lec22.pdf); the mean-gap specialization is derived here.

### Compute the cost of waiting

With p=0.49 and sigma=0.01 kg, one minute-one reading has d=12. A minute-twenty reading instead has a positive mean gap of about 6.29 times ten to the minus34 kg. Exact arithmetic can distinguish the two candidate means, but the resulting classification advantage is only about 1.25 times ten to the minus32. A displayed error of 0.5 does not make the exact advantage zero.

![Classification advantage over equal-prior guessing decreases with delayed single A readings, far faster for exchange fraction 0.49 than 0.2; both use independent Gaussian noise with standard deviation 0.01 kilograms.](/assets/learning/physical-noisy-contrast-en.svg)

For m independent repetitions of the same experiment and recording schedule, keeping the true initial-world label fixed, d squared multiplies by m. To attain a target error with 0<alpha<1/2 when d is positive, the real-valued repetition threshold follows from the Gaussian quantile. This assumes actual fresh experiments; copying a stored record does not create information.

$$
m\ge\left({2\Phi^{-1}(1-\alpha)\over d}\right)^2.
$$

The minute-twenty example needs a repetition count on the order of ten to the power63 for alpha=0.05. This is a model-based resource calculation, not a feasible experimental recommendation or an exact integer count certified by floating-point quantiles. At p=1/2 with recording beginning after minute zero, d is exactly zero, so no number of such repetitions distinguishes the initial worlds. The state difference was erased before observation.

| Recording choice | Mathematical gap | Statistical implication under the stated law |
|---|---|---|
| Minute one, p=0.49 | 0.12 kg | d=12, very small positive Bayes error |
| Minute twenty, p=0.49 | Positive but about 6.29e−34 kg | Advantage about 1.25e−32 |
| Any delayed window, p=1/2 | Exactly zero | Error 1/2 for every repetition count |

### Worked research question

**Question:** Can more observations always fix an incomplete sensor record? **Answer:** No. Independent records can improve a nonzero signal-to-noise distance, but cannot resurrect an exactly erased difference. Unknown parameters, correlated noise, and quantization need their own likelihoods. The code computes the gap sum with exact fractions, evaluates Gaussian tails with the standard library, and uses erf directly to preserve tiny advantages. Decimal probabilities are approximations, not exact coverage certificates.

```python
from fractions import Fraction as F
from math import sqrt, erfc, erf, log10
from statistics import NormalDist
def separation(p,start,end,sigma=F(1,100)):
    assert 0<=p<=1 and 0<=start<=end and sigma>0
    d2=sum((6*(1-2*p)**n/sigma)**2 for n in range(start,end+1))
    d=sqrt(float(d2))
    return d2,d,erfc(d/sqrt(8))/2,erf(d/sqrt(8))/2
for start,end in [(1,1),(2,20),(20,20)]:
    d2,d,risk,advantage=separation(F(49,100),start,end)
    assert d2>0 and advantage>0
    print('window',start,end,'distance',d,'error',risk,'advantage',advantage)
assert separation(F(49,100),1,1)[0]==144
assert separation(F(1,2),1,20)==(F(0),0.0,0.5,0.0)
assert separation(F(1,2),0,0)[0]>0
d=separation(F(49,100),20,20)[1]
z=NormalDist().inv_cdf(.95)
log10_repetitions=2*(log10(2*z)-log10(d))
assert 63<log10_repetitions<64
assert separation(F(49,100),20,20)[2]==.5
assert separation(F(49,100),20,20)[3]>0
print('log10 approximate repetition threshold',log10_repetitions)
print('A rounded error of0.5 does not erase the positive erf advantage.')
```

## Exercises with worked solutions

1. **Exercise:** Under zero leakage, can sum records distinguish initial states (8,2,0) and (2,8,0)? **Solution:** No. Both sums remain ten kilograms at every time in this model.
2. **Exercise:** Which available sensor distinguishes those states at minute zero? **Solution:** A and B each do, and the signed difference does too. Their initial outputs differ across the worlds.
3. **Exercise:** Why do delayed A records fail at half exchange? **Solution:** The imbalance is multiplied by zero in the first operation. Both modeled states become the same before recording starts.
4. **Exercise:** With leakage 0.1 and exchange 0.2, compute the first sum-record gap. **Solution:** The sums are 9.68 and 9.32, giving a difference of 0.36 kilograms.
5. **Exercise:** Would rounding the minute-twenty, p=0.49 A outputs to three decimal places preserve their exact distinction? **Solution:** No. Both round to 5.000. The quantized observation map contains less information than the exact map.
6. **Exercise:** Does distinguishing the selected pair prove every unknown parameter is identifiable? **Solution:** No. It answers only a declared two-hypothesis question. General state or parameter identification requires a broader argument.

## Sources and further study

The [MIT control-volume course outline](https://web.mit.edu/2.25/www/225_sect_5.html) provides physical background for choosing system boundaries. The observation maps, imbalance recurrence, delayed-record counterexample, and exact-gap calculation above are direct derivations for this lesson. Later study can connect this finite example to observability, experimental design, inverse problems, and noisy statistical inference while retaining explicit distinctions between modeled state, available information, and actual measured data.
