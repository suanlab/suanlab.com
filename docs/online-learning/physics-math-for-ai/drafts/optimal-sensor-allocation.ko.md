## 1. 정해진 관측 횟수 배분하기

같은 단위의 두 매개변수 $a,b$를 추정합니다. 센서 A는 $a$를 읽고 평균 0, 표준편차 $\sigma_a$의 잡음이 있으며, 센서 B는 $b$를 표준편차 $\sigma_b$의 잡음과 함께 읽습니다. 각 관측의 잡음은 독립입니다. 전체 횟수 $N$에서 두 센서에 적어도 한 번씩 $n_a+n_b=N$이 되도록 배분합니다. 표본평균의 분산은 각각 $\sigma_a^2/n_a$와 $\sigma_b^2/n_b$입니다. [Boyd와 Vandenberghe, 볼록 최적화 7.5절](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

## 2. 전체 분산 최소화하기

A-최적 설계는 추정량 공분산의 대각합을 최소화합니다.

$$
J_A=\frac{\sigma_a^2}{n_a}+\frac{\sigma_b^2}{n_b}.
$$

연속값으로 배분한다고 가정하고 $n_b=N-n_a$를 대입해 미분하면 $n_a/n_b=\sigma_a/\sigma_b$입니다. 잡음이 큰 매개변수에 더 많은 관측을 배분합니다. 횟수가 정수여야 하면 연속해 주변의 가능한 배분을 비교하거나 예산이 작을 때 모든 배분을 확인합니다.

## 3. 불확실성 타원의 넓이 최소화하기

D-최적 설계는 행렬식 $J_D=\sigma_a^2\sigma_b^2/(n_a n_b)$를 최소화합니다. 이 대각 모형에서는 전체 횟수를 고정하면 균등 배분에서 $n_a n_b$가 최대입니다. 가우스 잡음에서 신뢰 수준을 고정하면 행렬식의 제곱근이 불확실성 타원의 넓이에 비례합니다. 대각합과 행렬식은 서로 다른 목표를 표현합니다.

## 4. 관측 12회를 배분해 보기

$\sigma_a=1$, $\sigma_b=3$, $N=12$라면 A-최적 배분은 $(3,9)$이며 대각합은 $4/3$, 행렬식은 $1/3$입니다. D-최적 배분은 $(6,6)$이고 대각합은 $5/3$, 행렬식은 $1/4$입니다. 선택한 목적에 따라 최적 배분이 바뀝니다. 작업에서 한 매개변수가 더 중요하면 해당 분산에 가중치를 줄 수 있습니다.

## 5. Python으로 가능한 배분 모두 비교하기

```python
from math import isclose

def criteria(total, first, sigma_a, sigma_b):
    va = sigma_a**2 / first
    vb = sigma_b**2 / (total - first)
    return va + vb, va * vb

plans = [(i, *criteria(12, i, 1, 3)) for i in range(1, 12)]
by_trace = min(plans, key=lambda plan: plan[1])
by_det = min(plans, key=lambda plan: plan[2])
assert by_trace[0] == 3 and isclose(by_trace[1], 4/3)
assert by_det[0] == 6 and isclose(by_det[2], .25)
print("A-optimal", by_trace)
print("D-optimal", by_det)
```

## 6. 관측 전에 설계 가정 검증하기

예비 실험으로 센서 잡음을 추정하고 관측 1회의 비용을 기록합니다. 관측 사이에 상관이 있거나 센서별 비용이 다르면 최적화 문제가 달라집니다. 후속 연구에서 중요한 양을 기준으로 목적함수를 고르고, 따로 남겨 둔 관측으로 잡음 가정과 실제 정밀도를 확인합니다.
