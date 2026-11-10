## Distinguish repeated readings from the measurand

A stable instrument can produce tightly clustered readings while its readings remain shifted from the physical quantity of interest. Repeating the same measurement helps estimate the instrument's reading mean. It does not automatically identify an unknown fixed offset. This lesson separates those two questions before constructing any interval.

All quantities in the example use centimeters. The target quantity x and instrument bias b are fixed unknown parameters. The target reading model is

$$
R_i=x+b+\varepsilon_i,\qquad
\varepsilon_i\stackrel{\mathrm{iid}}\sim N(0,\sigma_R^2).
$$

The noise scale is assumed known from external information. The controls inspect sample means rather than generating random observations. The oracle-world selector supplies hypothetical true parameters for explanation; those truths are not estimates available to the observer.

## Prove the uncalibrated ambiguity

Consider two worlds: x is nine and b is one in the first; x is seven and b is three in the second. Both have reading mean ten and the same noise variance. Every finite reading vector therefore has the same joint distribution in both worlds, regardless of how many repetitions are collected.

More generally, replacing x by x+h and b by b−h preserves the entire uncalibrated observation law. This is an identifiability problem. An estimator based only on those readings cannot distinguish the worlds because its input distribution is identical. Seeing the oracle marker move does not mean that a data-based estimator has discovered a different truth.

Even an infinite sequence that revealed the reading mean exactly would reveal x+b, not x and b separately. Additional information about the instrument or an independent reference is needed to answer a different question about x itself.

## Calculate what averaging actually reduces

Under the stated independent noise model, averaging n_R positive-count readings gives

$$
E[\overline R]=x+b,\qquad
\operatorname{Var}(\overline R)=\frac{\sigma_R^2}{n_R}.
$$

The standard error sigma_R divided by the square root of n_R describes variability of the reading mean around x+b. It does not measure the magnitude of b. In the first guide, one hundred repetitions with noise standard deviation one give a standard error of 0.1 centimeters, yet x remains unidentified without calibration.

This square-root reduction requires the specified sampling assumptions. Repeatedly storing the same record, strong temporal dependence, or a changing instrument state does not create the same independent sample. Sample count is not a universal substitute for a measurement model.

## Introduce an exact reference as an explicit idealization

Suppose a reference quantity c is exactly five centimeters in this limited example. Measure it with the same fixed instrument bias using an independent stream:

$$
C_j=c+b+\eta_j,\qquad
\eta_j\stackrel{\mathrm{iid}}\sim N(0,\sigma_C^2).
$$

With positive sample counts in both streams, form the corrected estimate

$$
\widehat x=\overline R-\overline C+c.
$$

The same fixed bias cancels algebraically. Independence of the two noise streams then gives variance sigma_R squared over n_R plus sigma_C squared over n_C. Subtracting a measured correction does not erase its measurement uncertainty. A shared random noise source would require a covariance term and a different calculation.

Real references need not be exact. Their uncertainty, possible common influences, and traceability information belong in an expanded measurement model. The exact-reference assumption here isolates the calibration mechanism; it is not permission to treat every real calibration certificate as an exact number.

## Work the corrected guide numerically

Set both sample counts to one hundred and both known noise standard deviations to one centimeter. Query target mean ten and reference mean six. The corrected estimate is ten minus six plus five, or nine centimeters. The combined variance is 0.01 plus 0.01, equal to 0.02 square centimeters. Its standard error is approximately 0.141421 centimeters.

For a fixed sampling design and these known independent Gaussian noise scales, a nominal two-sided 95 percent interval has radius about 1.959964 times that standard error, or 0.277181 centimeters. The displayed endpoints are numerical approximations. They are not a formal machine-arithmetic proof of coverage and do not validate the sampling assumptions.

An individual queried interval may miss an illustrative true value. Frequentist coverage concerns the repeated procedure under the assumed law, rather than a posterior probability assigned to the interval after the means have been queried.

## Add a deterministic drift bound without inventing a distribution

A real instrument may change between target and reference measurements. Let d be the difference between their biases. Then the corrected estimator has the form

$$
\widehat x=x+d+e,\qquad e\sim N(0,s^2),\qquad
s^2=\frac{\sigma_R^2}{n_R}+\frac{\sigma_C^2}{n_C}.
$$

Assume externally that the absolute drift is at most D. On the Gaussian event where the absolute noise is at most z times s, the triangle inequality bounds the total error by z times s plus D. The expanded interval radius is therefore

$$
\text{radius}=z_{0.975}s+D.
$$

No probability distribution for d was specified. Adding D squared to the noise variance would silently invent a different model. With D equal to one centimeter, the guide's variance stays 0.02 square centimeters and the expanded radius becomes approximately 1.277181 centimeters. Selecting D with a control does not demonstrate that the actual drift satisfies the bound.

## Recognize the calibration floor

The horizontal graph varies a proposed target repetition count from one to one hundred while holding the reference count fixed. Blue shows target-mean noise variance; orange shows corrected variance; green marks the calibration variance contribution. The vertical unit is square centimeters, so the three curves are comparable variance quantities.

Increasing only the target count cannot remove the reference term. As that count grows, corrected variance approaches sigma_C squared divided by n_C. Drift adds a separate interval-radius limitation. To reduce these limits one must improve the reference information or justify tighter drift control, rather than merely collecting more target readings.

If no reference readings exist, the corrected curve and calibration floor are omitted instead of drawn at zero. Those missing values do not represent a perfect reference.

## Handle empty samples and support checks honestly

At a zero target count, no target sample mean exists. At a zero reference count, no reference mean exists. The corresponding estimates, standard errors, and corrected interval are unavailable as appropriate. A formula containing division by zero is not replaced by a fictional zero uncertainty.

With positive noise, a Gaussian sample mean has the full real line as its support. An exact queried real number still has point probability zero; support compatibility is not a positive-probability claim about that singleton. With zero noise, the law becomes a point mass and a different queried mean is incompatible with the configured oracle world. The interface flags that case while leaving the algebraic query clearly distinguished from generated data.

## Exercises with worked solutions

1. **Exercise:** Does a standard error of 0.1 centimeters bound a fixed bias by 0.1? **Solution:** No. It describes variation around the reading mean x+b. The unknown fixed offset is not identified by the standard error.
2. **Exercise:** Why does changing the truth world leave the uncalibrated reading law unchanged? **Solution:** Both worlds have the same sum x+b and noise law. Every observable reading vector has the same distribution.
3. **Exercise:** Compute corrected variance with target count one hundred, reference count four, target sigma one, and reference sigma two. **Solution:** It is 0.01 plus four divided by four, equal to 1.01 square centimeters. The reference variance dominates.
4. **Exercise:** What happens if the target count tends to infinity in that example? **Solution:** The remaining variance approaches one square centimeter; it does not vanish while the reference count stays fixed.
5. **Exercise:** Why is the drift bound added to radius rather than automatically to variance? **Solution:** It defines a deterministic error set. Combining this set with a probabilistic noise interval uses the triangle inequality, without assigning drift a distribution.
6. **Exercise:** Can the known-scale Gaussian interval be used unchanged after estimating sigma from a tiny dataset? **Solution:** No. Estimating the noise scale changes the procedure and its distributional justification; it needs an appropriate additional argument.

## Source and AI connection

The [JCGM guide to measurement uncertainty](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf) provides background on measurement models, corrections, and uncertainty components. The two-world ambiguity, exact-reference experiment, and bounded-drift interval above are direct derivations for this lesson, not an implementation of the entire guide. In sensor-based AI, collecting more biased labels or readings can tighten an estimate of the wrong target. Explicit calibration variables and a declared information set are part of deciding what a learned prediction can identify.
