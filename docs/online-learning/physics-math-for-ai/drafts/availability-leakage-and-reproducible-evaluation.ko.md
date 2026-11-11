## 1. 판단 시점에 사용할 정보 정하기

시각 $t$의 예측에 사용하는 각 특성이 언제 공개되는지 기록합니다. 보고 지연과 이후 수정도 포함합니다. 날짜가 $t$인 값이라도 판단 시점 뒤에 공개될 수 있습니다. 목표 예측 구간과 학습 정답을 사용할 수 있었던 시각도 정합니다. [Kapoor와 Narayanan의 데이터 누출 연구](https://doi.org/10.1016/j.patter.2023.100804)

## 2. 예측 가능성이 제한된 과정 만들기

서로 독립이며 평균 0, 분산 1인 변수를 $X,U,N$이라 합니다. 미래 신호를 $F=\rho X+\sqrt{1-\rho^2}U$, 목표를 $Y=0.8F+0.6N$으로 둡니다. 판단 시점에는 $X$만 있습니다. 이 정보의 최적 선형 예측은 $0.8\rho X$이고 평균제곱오차는 $0.64(1-\rho^2)+0.36$입니다.

## 3. 미래 특성 누출의 효과 재기

평가 과정에서 $F$를 노출하면 $0.8F$로 예측해 MSE 0.36을 얻습니다. 목표 자체를 노출하면 오차가 0입니다. $\rho=0$에서 정당한 정보의 오차는 1이므로 좋아 보이는 결과가 사용할 수 없던 정보에서 올 수 있습니다. $\rho$가 커지면 현재 특성에 실제 예측 정보가 생깁니다.

## 4. 연구 질문에 맞게 분할하기

미래 예측에서는 각 학습과 시험 판단 시점에 무엇을 사용할 수 있었는지 재현합니다. 배포 시스템이 이전 관측을 쓸 수 있다면 시험 특성에서도 과거 관측을 사용할 수 있습니다. 정규화와 특성 선택은 허용된 학습 정보로 적합합니다. 새로운 장치에 대한 주장이면 장치 단위로 분할하고 식별자나 유사 중복 자료가 시험 정답을 알려 주는지도 확인합니다.

## 5. 합성 사례의 누출을 정확히 검산하기

```python
from itertools import product
from math import isclose, sqrt

rho = .5
squared = {"available": [], "future": [], "target": []}
for x, u, n in product([-1, 1], repeat=3):
    future = rho*x+sqrt(1-rho**2)*u
    y = .8*future+.6*n
    predictions = {"available": .8*rho*x, "future": .8*future, "target": y}
    for name, prediction in predictions.items():
        squared[name].append((y-prediction)**2)
mse = {name: sum(values)/len(values) for name, values in squared.items()}
assert isclose(mse["available"], .64*(1-rho**2)+.36)
assert isclose(mse["future"], .36) and mse["target"] == 0
print(mse)
```

## 6. 누출 검사를 재현할 근거 보존하기

특성 정의, 가용 시각 규칙, 분할 식별자, 코드 버전, 전처리 상태, 시드를 저장합니다. 기록한 자료 상태에서 평가를 재실행합니다. 같은 누출이 있으면 재실행에서도 같은 낙관적인 결과가 나올 수 있으므로 재현성과 정보 조건의 타당성을 각각 확인합니다.
