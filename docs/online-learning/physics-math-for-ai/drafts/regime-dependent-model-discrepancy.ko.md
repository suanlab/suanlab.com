## 1. 모형이 생략한 과정 밝히기

관측을 $y(t)=m(t,\theta)+\delta(t)+\epsilon(t)$로 씁니다. $m$은 시뮬레이터, $\delta$는 체계적인 모형 불일치, $\epsilon$은 측정 오차입니다. 반복해서 나타나는 차이는 누락된 과정, 치우친 경계 조건 또는 부정확한 근사에서 생길 수 있습니다. 측정 정밀도를 높이면 이 차이가 더 뚜렷해질 수 있습니다. [Brynjarsdóttir와 O’Hagan의 모형 불일치 연구](https://doi.org/10.1088/0266-5611/30/11/114007)

## 2. 평형값을 생략한 감쇠 모형 적합하기

$0\le c<1$, $k>0$에서 정확한 합성 데이터를 $y(t)=c+(1-c)e^{-kt}$로 만듭니다. 초기값은 1, 평형값은 $c$입니다. 간단한 모형 $m(t,\widehat k)=e^{-\widehat k t}$는 평형값을 0으로 고정합니다. 양의 시각 $t_c$에서 관측 하나를 맞추면

$$
\widehat k=-\frac{\ln y(t_c)}{t_c}
$$

입니다. $c>0$이면 $y(t_c)>e^{-kt_c}$이므로 추정한 감쇠율이 실제보다 작습니다. 감쇠율이 생략된 평형값의 영향을 일부 흡수합니다.

## 3. 교정 구간 밖에서 모형 검사하기

$t_c=1$, $c=0.25$, $k=1$이면 관측은 약 0.526, 추정 감쇠율은 약 0.643입니다. $t=3$에서 간단한 모형은 $y(1)^3\approx0.145$를 예측하고 실제 데이터를 만든 값은 약 0.287입니다. 초기값과 교정 관측을 정확히 맞추어도 이후 차이가 생깁니다.

## 4. 모형 수정과 추가 관측 비교하기

평형값을 독립적으로 재면 $c$를 정하고 $(y(t_c)-c)/(1-c)$에서 $k$를 복원할 수 있습니다. 유연한 불일치 함수로 교정 데이터를 맞출 수도 있지만, 서로 다른 매개변수를 서로 다른 불일치 함수가 보상할 수 있습니다. 가능한 설명들의 예측이 달라지는 조건에서 관측을 추가합니다.

## 5. Python으로 교정과 검증 재현하기

```python
from math import exp, isclose, log

c, rate, calibration_time = .25, 1., 1.
truth = lambda t: c + (1-c)*exp(-rate*t)
fit = -log(truth(calibration_time))/calibration_time
simple = lambda t: exp(-fit*t)
assert isclose(simple(0), truth(0))
assert isclose(simple(calibration_time), truth(calibration_time))
assert fit < rate and truth(3)-simple(3) > .14
corrected = -log((truth(calibration_time)-c)/(1-c))/calibration_time
assert isclose(corrected, rate)
print(fit, truth(3)-simple(3), corrected)
```

## 6. 모형을 적용할 수 있는 조건 기록하기

검증용 관측의 오차를 시각이나 물리 조건에 따라 그립니다. 가능하면 측정 잡음, 수치 이산화, 구조적 불일치를 서로 다른 실험으로 비교합니다. 매개변수를 식별할 수 있게 한 가정을 기록하고, 시뮬레이터의 적합값과 독립적으로 측정한 물리량을 구분합니다.
