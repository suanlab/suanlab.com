## 1. Assign each statement an evidential role

A theorem gives a conclusion under explicit assumptions. An empirical statement reports what was measured under a recorded protocol. A working hypothesis proposes an explanation to be tested. For example, “the mean observed loss is 0.18” is empirical. “A physical constraint improves out-of-distribution behavior” is a hypothesis until its scope and evidence are specified.

## 2. State a finite-sample theorem

For a fixed model, suppose $m$ losses are independent, identically distributed and in $[0,1]$. Let $R$ be their population mean and $\widehat R$ their sample mean. Hoeffding's inequality gives

$$
\Pr(|\widehat R-R|\ge\varepsilon)\le2e^{-2m\varepsilon^2}.
$$

Setting the right side to $\delta$ yields $\varepsilon=\sqrt{\ln(2/\delta)/(2m)}$. The theorem is conditional on the stated boundedness and independence assumptions. [Hoeffding, Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830)

## 3. Account for choosing among fixed candidates

If $M$ candidate models are specified independently of this evaluation sample, a union bound gives simultaneous control with $\varepsilon=\sqrt{\ln(2M/\delta)/(2m)}$. The candidates' errors can be correlated with each other. Independence is required across the sampled observations for each Hoeffding bound. Adaptive construction of candidates from the evaluation sample requires a different argument.

## 4. Work through the reported numbers

With $m=100$ and $\delta=0.05$, one fixed model has radius about 0.136. If the observed mean is 0.18, the corresponding clipped interval is about $[0.044,0.316]$. For 100 fixed candidates the simultaneous radius is about 0.204. The same observed number leads to different guarantees because the claim changed.

## 5. Check the radius and an exact special case

```python
from math import comb, exp, log, sqrt

m, delta = 100, .05
radius = sqrt(log(2/delta)/(2*m))
simultaneous = sqrt(log(200/delta)/(2*m))
assert radius < simultaneous
n, epsilon = 20, .3
tail = sum(comb(n, k) for k in range(n+1) if abs(k/n-.5) >= epsilon-1e-12)/2**n
bound = 2*exp(-2*n*epsilon**2)
assert tail <= bound
print(radius, simultaneous, "exact fair-sign tail:", tail, "bound:", bound)
```

## 6. Keep explanation and guarantee connected to their assumptions

A small measured error supports the observation under the evaluated conditions. The theorem extends a claim through its assumptions. A proposed physical explanation needs an intervention or other discriminating evidence. Record all three so that a reader can identify which conclusion follows from proof, which from measurements, and which remains open.
