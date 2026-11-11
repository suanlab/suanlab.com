## 1. 어떤 원인을 바꾸는 실험인지 밝히기

제거 실험은 구성 요소를 빼거나 바꿉니다. 음성 대조는 주장하는 작동 원리를 깨뜨리면서 선택한 다른 특성은 유지합니다. 가중치를 고정하는지 다시 학습하는지, 어느 분포를 보존하는지, 설명이 맞을 때 어떤 결과를 기대하는지 명시합니다. 무작위화 실험으로 학습 매개변수나 학습 정보에 반응하지 않는 설명 방법을 찾은 사례가 있습니다. [Adebayo 외, 현저성 지도의 검산](https://research.google/pubs/sanity-checks-for-saliency-maps/)

## 2. 모두 확인할 수 있는 예측 계 만들기

독립 부호 $X,Z,N$이 −1과 +1을 같은 확률로 취한다고 가정합니다. 목표를 $Y=0.7X+0.3Z+0.1N$으로 둡니다. 전체 예측기는 $0.7X+0.3Z$, 둘째 특성을 뺀 예측기는 $0.7X$입니다. 독립적으로 빠진 항의 분산이 더해지므로 평균제곱오차는 각각 0.01과 0.10입니다.

## 3. 고정된 예측기에서 특성 무작위화하기

추론할 때 $Z$를 독립 부호 $Z'$로 바꾸면 고정된 전체 예측은 $0.7X+0.3Z'$입니다. 잔차는 $0.3(Z-Z')+0.1N$이어서 오차는 $0.09+0.09+0.01=0.19$입니다. 무작위 특성은 원래 특성과 같은 주변 분포를 가지지만 목표와의 관계는 사라집니다.

## 4. 학습을 바꾸는 실험은 별도로 정의하기

독립 $Z'$를 사용해 모집단에서 다시 적합하면 그 특성의 최적 계수는 0입니다. 예측기는 $0.7X$가 되고 오차는 0.10입니다. 고정 예측기의 무작위화는 배포된 예측기가 입력에 얼마나 의존하는지 검사합니다. 재학습은 입력 관계가 바뀐 상황에서 학습 과정이 어떤 정보를 사용하는지 검사합니다.

## 5. 모든 경우 열거하기

```python
from itertools import product
from math import isclose

errors = {"full": [], "removed": [], "frozen_random": [], "refitted_random": []}
for x, z, z_random, n in product([-1, 1], repeat=4):
    y = .7*x+.3*z+.1*n
    predictions = {
        "full": .7*x+.3*z,
        "removed": .7*x,
        "frozen_random": .7*x+.3*z_random,
        "refitted_random": .7*x,
    }
    for name, prediction in predictions.items():
        errors[name].append((y-prediction)**2)
means = {name: sum(values)/len(values) for name, values in errors.items()}
for name, expected in [("full", .01), ("removed", .10), ("frozen_random", .19), ("refitted_random", .10)]:
    assert isclose(means[name], expected)
print(means)
```

## 6. 실험 가정 안에서 대조 해석하기

고정된 모형의 무작위화로 성능이 낮아지면 그 실험에서 바꾼 입력에 의존했다는 근거가 됩니다. 현실에 없는 입력 조합으로도 성능이 낮아질 수 있습니다. 분포 변화, 교정 손상, 구현 오류를 위한 추가 대조를 둡니다. 개입 방법과 재학습 여부를 기록하면 어떤 원인을 검사했는지 판단할 수 있습니다.
