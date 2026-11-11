## 1. 다른 매개변수가 같은 관측을 예측하는지 확인하기

정해진 실험에서 서로 다른 매개변수가 서로 다른 이상적 관측을 주면 구조적으로 식별할 수 있습니다. 실제 식별 가능성은 잡음이 있는 유한한 관측이 매개변수의 범위를 얼마나 좁히는지를 다룹니다. 두 성질은 물리 모형, 관측 연산자, 실험 조건에 따라 달라집니다. [Raue 외, 구조적·실제 식별 가능성 분석](https://doi.org/10.1093/bioinformatics/btp358)

## 2. 섞여 관측된 두 성분 구분하기

미지 진폭 $a,b$를 측정해 $y_1=a+b$, $y_2=a+rb$를 얻으며 센서 계수 $r$은 안다고 가정합니다. $r=1$이면 두 센서는 같은 식을 제공합니다. 직선 $a+b=y_1$ 위의 모든 조합이 이상적 관측과 맞습니다. 같은 측정을 반복하면 합의 잡음을 줄일 수 있지만 두 진폭 각각은 여전히 여러 값이 가능합니다.

## 3. 다른 방향의 관측을 고르기

$r\ne1$이면 두 식을 빼서 다음을 얻습니다.

$$
b=\frac{y_1-y_2}{1-r},\qquad a=y_1-b.
$$

$(a,b)=(2,3)$이고 $r=-1$이면 측정값은 5와 −1입니다. 두 관측의 합은 $a$의 두 배, 차는 $b$의 두 배입니다. $r=0.9$에서는 둘째 관측이 4.7입니다. 해는 유일하지만 두 관측의 차이를 $1-r=0.1$로 나누어야 하므로 잡음도 커집니다.

## 4. 센서 설계에서 추정 불확실성 계산하기

각 센서의 오차가 독립이고 평균 0, 공통 분산 $\sigma^2$이면 차이의 분산은 $2\sigma^2$입니다. 따라서

$$
\operatorname{Var}(\widehat b)=\frac{2\sigma^2}{(1-r)^2}.
$$

일반적으로 민감도 행렬 $S$가 열 완전 계수이고 관측 잡음 공분산 $\Sigma$가 양의 정부호이면, 일반화 최소제곱의 매개변수 공분산은 $(S^T\Sigma^{-1}S)^{-1}$입니다. 불확실성이 큰 방향은 센서 응답과 잡음 양쪽에 달려 있습니다. 비선형 모형에서 야코비안은 국소 근사이며, 멀리 떨어진 매개변수에서 같은 출력이 나오는지 따로 확인해야 합니다.

## 5. NumPy로 측정 방향 검산하기

```python
import numpy as np

theta = np.array([2.0, 3.0])
sigma = 0.1
for r in (-1, 0, .9):
    sensors = np.array([[1, 1], [1, r]], dtype=float)
    observations = sensors @ theta
    estimate = np.linalg.solve(sensors, observations)
    covariance = sigma**2 * np.linalg.inv(sensors.T @ sensors)
    assert np.allclose(estimate, theta)
    assert np.isclose(covariance[1, 1], 2 * sigma**2 / (1 - r)**2)
    print(r, observations, np.sqrt(covariance[1, 1]))

repeated = np.ones((2, 2))
assert np.linalg.matrix_rank(repeated) == 1
assert np.allclose(repeated @ [2, 3], repeated @ [1, 4])
```

## 6. 정보를 더 주는 후속 실험 설계하기

약하게 관측되는 매개변수 조합에 다르게 반응하는 센서나 자극을 고릅니다. 정밀도나 표본 수를 늘리기 전에 예상 잡음을 계산합니다. 후보 매개변수들의 예측을 시뮬레이션하고, 잡음에 비해 예측 차이가 큰 관측을 고릅니다. 추정에 사용하지 않은 측정값을 남겨 모형을 검증합니다.
