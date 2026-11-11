## 1. 온도계가 읽을 값을 예측하기

물체 온도를 $T(t)$, 측정한 주변 온도를 $T_\infty$, 냉각률을 $k>0$이라고 합니다. 물체 내부의 온도가 거의 균일하고 주변 온도가 일정하며 열 손실이 온도 차이에 비례하면 초과 온도는 다음 식을 따릅니다.

$$
\dot y=-ky,\qquad y=T-T_\infty,\qquad y(t)=Ae^{-kt}.
$$

초기의 초과 온도는 $A>0$입니다. 관측 시각 $t_1,\ldots,t_m$을 정하면 순방향 사상은 $(A,k)$를 그 시각의 예측 온도 벡터에 대응시킵니다. 무엇을 언제 재는지도 사상의 일부입니다. [Tarantola, 이산 역문제](https://epubs.siam.org/doi/10.1137/1.9780898717921.ch1)

## 2. 관측에서 매개변수 찾기

역문제는 측정값에서 시작해 그 값을 낼 수 있는 매개변수를 찾습니다. 시각 $t_1$의 양의 관측 $y_1$ 하나는 다음 제약을 줍니다.

$$
A=y_1 e^{kt_1}.
$$

가능한 각 $k$마다 대응하는 $A$가 있습니다. 곡선들이 모두 같은 측정점을 지나지만 다른 시각의 예측은 달라질 수 있습니다. 초기 온도를 별도로 알면 제약이 추가되어 역문제가 달라집니다.

## 3. 잡음이 없는 관측 두 개로 풀기

$t_2>t_1$이고 $y_1>y_2>0$일 때 두 측정값을 나누어 $A$를 없앱니다.

$$
k=\frac{\ln(y_1/y_2)}{t_2-t_1},\qquad A=y_1 e^{kt_1}.
$$

$A=40$ K, $k=0.2\,\mathrm{min}^{-1}$라면 2분과 6분의 초과 온도는 약 26.81 K와 12.05 K입니다. 반올림 전 수치를 넣으면 원래 매개변수를 복원합니다. 온도 차이는 섭씨와 켈빈에서 크기가 같습니다.

## 4. 관측 시점에 따른 오차 살펴보기

두 관측 시각이 너무 가까우면 분모가 작습니다. 너무 늦게 재면 초과 온도가 0에 가까워 로그를 취할 때 상대 오차가 커질 수 있습니다. 관측 시각이 고정되어 있다면 일차 오차는 다음과 같습니다.

$$
\delta k\approx\frac{\delta y_1/y_1-\delta y_2/y_2}{t_2-t_1}.
$$

잡음 때문에 관측값이 0 이하가 되거나 온도가 증가한 것처럼 보이면 양의 냉각률 모형을 정확하게 만족하는 해는 없습니다. 잡음이 있는 관측이 많다면 오차 모형을 명시하고 순방향 모형을 맞춘 뒤 별도의 측정값으로 검증합니다.

## 5. Python으로 복원과 예측 검산하기

```python
from math import exp, isclose, log

def forward(amplitude, rate, time):
    return amplitude * exp(-rate * time)

def inverse(y1, y2, t1, t2):
    rate = (log(y1) - log(y2)) / (t2 - t1)
    return y1 * exp(rate * t1), rate

t1, t2 = 2, 6
y1, y2 = [forward(40, .2, time) for time in (t1, t2)]
for rate in (.05, .2, .4):
    assert isclose(forward(y1 * exp(rate * t1), rate, t1), y1)
amplitude, rate = inverse(y1, y2, t1, t2)
assert isclose(amplitude, 40)
assert isclose(rate, .2)
print(f"A={amplitude:.2f} K, k={rate:.3f}/min")
print("perturbed late observation:", inverse(y1, y2 + .5, t1, t2))
```

## 6. 새로운 관측으로 모형 검증하기

추정한 매개변수로 세 번째 시각의 온도를 예측합니다. 추정에 쓰지 않은 측정값과 비교합니다. 차이가 크다면 센서 잡음, 주변 온도의 오차, 시간에 따라 변하는 열전달처럼 모형에서 빠진 과정이 원인일 수 있습니다. 물리 가정, 관측 연산자, 매개변수 추정 과정을 각각 기록하면 원인을 따로 검증할 수 있습니다.
