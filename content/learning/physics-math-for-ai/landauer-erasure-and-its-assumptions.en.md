## 1. Specify what a reset does

A reset maps either input bit to the target state 0. The output alone cannot identify the prior bit. Consider two equally energetic memory states with identical internal entropy, a heat bath initially at temperature $T$, and a controller returned to its starting condition. Assume the bath is initially uncorrelated with the memory, and the process uses no external record of the input. These assumptions identify where the lost memory entropy can go. [Landauer, Irreversibility and Heat Generation in the Computing Process](https://doi.org/10.1147/rd.53.0183)

## 2. Obtain the bound from entropy balance

If the initial probability of state 1 is $p$, define binary entropy in nats by $h(p)=-p\ln p-(1-p)\ln(1-p)$, with $0\ln 0=0$. A perfect reset leaves $h(0)=0$. With the memory's other entropy contributions held fixed, its entropy decreases by $k_B h(p)$. In the single-bath idealization, the second law requires

$$
Q_{\rm bath}\ge k_B T h(p),
$$

where heat entering the bath is positive. Equality describes a thermodynamically reversible limit of the protocol. With a fair input, $p=1/2$ and $h(p)=\ln 2$.

## 3. Change one assumption at a time

If the initial bit is already certain, its binary entropy is zero. The informational contribution to the heat lower bound then vanishes. If the reset leaves probability $\varepsilon$ in state 1, the corresponding entropy decrease is $k_B[h(p)-h(\varepsilon)]$ in the same model. For a fair initial bit and $\varepsilon=0.1$, the dimensionless decrease is about $0.368$. The size of the allowed error belongs in the statement of the task.

## 4. Compare the physical limit and a measurement

At 300 K the perfect reset of an unbiased bit has a lower bound of about $2.87\times10^{-21}$ J of average heat to the bath. In a colloidal experiment a particle in a double-well potential represents the bit. A controlled protocol changes the wells to favor one final state. Many repetitions estimate average dissipation, and the slow-protocol limit is compared with $k_B T\ln 2$. A finite-rate protocol, measurement circuitry, and reset control can add energy costs. [Bérut et al., experimental verification of Landauer’s principle](https://www.nature.com/articles/nature10872)

## 5. Check fair, biased and imperfect resets

```python
from math import isclose, log

k_B, temperature = 1.380649e-23, 300

def entropy(p):
    return -sum(x * log(x) for x in (p, 1 - p) if x > 0)

def heat_floor(p, error):
    return k_B * temperature * (entropy(p) - entropy(error))

fair = heat_floor(.5, 0)
assert isclose(fair, k_B * temperature * log(2), rel_tol=1e-12)
assert 0 < heat_floor(.1, 0) < fair
assert 0 < heat_floor(.5, .1) < fair
assert heat_floor(0, 0) == 0
for p, error in [(.5, 0), (.1, 0), (.5, .1)]:
    print(f"p={p:.1f}, error={error:.1f}, floor={heat_floor(p, error):.3e} J")
```

## 6. Specify what a computing claim counts

The argument counts an entropy decrease in a defined physical memory operation. To apply it to a computing claim, identify which states are merged, how their probabilities are determined, what records survive, and which control degrees of freedom return to their initial conditions. Compare a real system by measuring energy for its stated workload and boundary. Even logically reversible updates can dissipate heat under a practical implementation.
