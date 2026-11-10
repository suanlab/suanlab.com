## Estimate a target mean using another distribution

Target masses p=(0.2,0.3,0.5) live on states i=0,1,2, with f(i)=i and exact expectation 1.3. If direct target sampling is difficult, draw from proposal q and correct by p/q. The default proposal is uniform. Changing q(0) divides the remaining mass equally between the other two states.

$$
E_p[f]=E_q\!\left[\frac{p(X)}{q(X)}f(X)\right].
$$

## Check support first

Every positive-target state must be reachable by the proposal. At q(0)=0, target mass 0.2 is permanently missed. Even a displayed sample containing only the other states cannot justify recovery of the full target mean. The lab withholds estimates in this case and separately reports whether the displayed draw itself is possible under q.

## Fixed samples and two estimates

The screen does not generate fresh random draws. It displays three fixed count patterns for six samples imagined to come from the proposal. The ordinary importance estimate of the normalizing constant is the mean weight. The self-normalized mean divides a weighted sum by the total weight.

$$
\widehat Z=\frac1N\sum_s w_s,\qquad
\widehat\mu_{\rm SN}=\frac{\sum_s w_sf(X_s)}{\sum_sw_s}.
$$

A uniform proposal and two samples per state give weights 0.6,0.9,1.5 and estimate exactly 1.3. This is a property of these balanced counts, not zero error for every possible draw. Self-normalized ratios are generally biased at finite sample size. By contrast, an ordinary importance mean for normalized p is unbiased under the support and integrability conditions.

## What ESS can miss

Weight effective sample size is (Σw)²/Σw². It summarizes weight concentration and lies between one and six for six positive-weight samples. The third pattern repeats state two six times. All sample weights are equal, giving ESS six, yet the estimated mean is two. A high weight ESS cannot detect unvisited regions or model error.

$$
\operatorname{ESS}=\frac{(\sum_s w_s)^2}{\sum_s w_s^2}.
$$

An extremely small positive proposal probability differs from missing support. The implementation calculates ratios and ESS using relative log weights. Raw weights or normalizer estimates outside numeric range are withheld with a numeric-range warning.

## Exercises with explanations

1. Can more samples repair missing proposal support? **No: repair the proposal itself.**
2. Is the self-normalized estimate always unbiased? **No: expectation of a ratio needs care.**
3. Does ESS six imply accuracy? **No: inspect the repeated-state counterexample.**

Further reading: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
