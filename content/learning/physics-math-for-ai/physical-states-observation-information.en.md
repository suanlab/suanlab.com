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

## Exercises with worked solutions

1. **Exercise:** Under zero leakage, can sum records distinguish initial states (8,2,0) and (2,8,0)? **Solution:** No. Both sums remain ten kilograms at every time in this model.
2. **Exercise:** Which available sensor distinguishes those states at minute zero? **Solution:** A and B each do, and the signed difference does too. Their initial outputs differ across the worlds.
3. **Exercise:** Why do delayed A records fail at half exchange? **Solution:** The imbalance is multiplied by zero in the first operation. Both modeled states become the same before recording starts.
4. **Exercise:** With leakage 0.1 and exchange 0.2, compute the first sum-record gap. **Solution:** The sums are 9.68 and 9.32, giving a difference of 0.36 kilograms.
5. **Exercise:** Would rounding the minute-twenty, p=0.49 A outputs to three decimal places preserve their exact distinction? **Solution:** No. Both round to 5.000. The quantized observation map contains less information than the exact map.
6. **Exercise:** Does distinguishing the selected pair prove every unknown parameter is identifiable? **Solution:** No. It answers only a declared two-hypothesis question. General state or parameter identification requires a broader argument.

## Sources and further study

The [MIT control-volume course outline](https://web.mit.edu/2.25/www/225_sect_5.html) provides physical background for choosing system boundaries. The observation maps, imbalance recurrence, delayed-record counterexample, and exact-gap calculation above are direct derivations for this lesson. Later study can connect this finite example to observability, experimental design, inverse problems, and noisy statistical inference while retaining explicit distinctions between modeled state, available information, and actual measured data.
