## 1. Combine a changing model with arriving observations

Data assimilation updates a state estimate as observations arrive. Use a scalar state with dynamics $x_{n+1}=a x_n+w_n$ and observation $y_{n+1}=x_{n+1}+v_{n+1}$. Independent zero-mean Gaussian process and measurement noises have variances $q$ and $r$. With a Gaussian starting state, the scalar Kalman filter gives the exact conditional means and variances under this model. [Kalman, A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552)

## 2. Forecast the state and uncertainty

Given the previous analysis mean $m_n$ and variance $P_n$, the next forecast is

$$
m^-=a m_n,\qquad P^-=a^2P_n+q.
$$

Multiplying the state by $a$ scales the variance by $a^2$. Independent process noise contributes $q$. The forecast step can be executed even when the next observation is unavailable.

## 3. Update using the innovation

The innovation $y-m^-$ measures the gap between the new reading and the forecast. Its model variance is $P^-+r$. The analysis uses the gain $K=P^-/(P^-+r)$:

$$
m^+=m^-+K(y-m^-),\qquad P^+=(1-K)P^-.
$$

A noisier observation gives a smaller gain. When an observation is missing, carry $m^-,P^-$ forward without the update. Process noise can increase the variance across gaps. With strong contraction and small process noise, a forecast variance can decrease instead.

## 4. Work through one forecast and analysis

Use $m_0=0$, $P_0=1$, $a=0.9$, $q=0.2$, and $r=0.25$. The first forecast has $P^-=1.01$. For $y_1=1$ the innovation is 1 and $K=1.01/1.26\approx0.8016$. The analysis mean is about 0.8016 and the analysis variance is about 0.2004. If the next observation is missing, forecast from those values and skip the analysis step.

## 5. Run a filter with a missing reading

```python
from math import isclose

a, q, r = .9, .2, .25
m, p = 0., 1.
observations = [1., None, .2]
for t, y in enumerate(observations, 1):
    m_pred, p_pred = a*m, a*a*p+q
    if y is None:
        m, p = m_pred, p_pred
        gain = 0.
    else:
        gain = p_pred/(p_pred+r)
        m = m_pred+gain*(y-m_pred)
        p = p_pred*r/(p_pred+r)
        assert p < p_pred
    if t == 1:
        assert isclose(m, 1.01/1.26)
    print(t, y, f"gain={gain:.4f}, mean={m:.4f}, variance={p:.4f}")
```

## 6. Diagnose a mismatch with the data

Record innovations and their predicted variances when readings arrive. Persistent signed innovations suggest bias or an inadequate state model. Compare normalized innovations across observation conditions and investigate systematic patterns. A filter conditions each estimate on observations available through the current time. A smoother can also use later observations to revise an earlier state.
