## 1. 잡음이 있는 관측으로 분포 추정하기

미지 스칼라 $x$의 사전 분포를 $x\sim\mathcal N(\mu_0,v_0)$로 둡니다. 센서는 $y=sx+\eta$를 관측하며, $s$는 알려진 관측 강도이고 독립 잡음은 $\eta\sim\mathcal N(0,r)$입니다. 베이즈 규칙은 사전 밀도와 우도를 곱하고 정규화합니다. 사후 분포는 관측값과 가정한 모형에 따른 불확실성을 나타냅니다. [Stuart, 베이즈 관점의 역문제](https://doi.org/10.1017/S0962492910000061)

## 2. 완전제곱으로 사후 평균과 분산 구하기

상수항을 제외한 음의 로그 사후 밀도는 다음과 같습니다.

$$
\frac{(x-\mu_0)^2}{2v_0}+\frac{(y-sx)^2}{2r}.
$$

$x^2$의 계수에서 정밀도, $x$의 계수에서 사후 평균을 구합니다.

$$
v=\left(v_0^{-1}+s^2/r\right)^{-1},\qquad
\mu=v\left(\mu_0/v_0+sy/r\right).
$$

사전 분포와 관측이 모두 정밀도에 기여합니다. 이 가우스 모형에서는 최대 사후 확률 추정값이 평균과 같습니다. 사전 평균이 0이면 같은 최적점이 $(y-sx)^2+(r/v_0)x^2$를 최소화하므로 티호노프 벌점에 확률적 해석이 생깁니다.

## 3. 정보가 많은 관측과 약한 관측 비교하기

$\mu_0=0$, $v_0=1$, $s=0.5$, $r=0.25$, $y=1$이면 사후 평균은 1, 분산은 0.5입니다. $s=0$이면 우도가 $x$에 대해 상수여서 사후 분포가 사전 분포와 같습니다. 센서가 0이 아닐 때 $r$이 작으면 관측이 제공하는 정보가 늘어납니다.

## 4. 새 관측 예측하기

잡음 분산이 $r_*$인 독립 후속 관측 $y_*=s_*x+\eta_*$의 예측 분포는 다음과 같습니다.

$$
y_*\mid y\sim\mathcal N(s_*\mu,\ s_*^2v+r_*).
$$

분산의 첫 항은 매개변수 불확실성, 둘째 항은 새 관측의 잡음입니다. $x$의 신용구간과 $y_*$의 예측구간은 다른 양의 범위를 나타냅니다.

## 5. 수치 적분으로 사후 분포 검산하기

```python
import numpy as np

x = np.linspace(-8, 8, 16001)
prior_mean, prior_variance = 0., 1.
strength, noise_variance, observation = .5, .25, 1.
weights = np.exp(-.5 * ((x-prior_mean)**2 / prior_variance
                        + (observation-strength*x)**2 / noise_variance))
weights /= weights.sum()
mean = np.dot(weights, x)
variance = np.dot(weights, (x-mean)**2)
assert np.isclose(mean, 1.)
assert np.isclose(variance, .5)
print(mean, variance, "predictive variance:", strength**2 * variance + noise_variance)
```

## 6. 가정에 따라 추론이 얼마나 바뀌는지 확인하기

가능한 사전 분포의 척도와 잡음 크기를 바꾸며 추정값을 비교합니다. 관측 연산자가 약하면 사전 분포의 영향이 커집니다. 적합에 사용하지 않은 관측으로 예측 잔차를 확인합니다. 사후 분포가 좁다는 판단도 가정한 모형에 의존하며, 순방향 물리나 잡음 분포가 틀리면 불확실성이 과소평가될 수 있습니다.
