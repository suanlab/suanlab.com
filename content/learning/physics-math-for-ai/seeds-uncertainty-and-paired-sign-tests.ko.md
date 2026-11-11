## 1. 반복마다 무엇이 바뀌는지 밝히기

학습 시드를 바꾸면 고른 데이터와 방법 아래에서 초기화, 순서, 확률적 최적화의 변동을 검사합니다. 시험 사건을 다시 뽑으면 다른 불확실성을 봅니다. 같은 시계열의 많은 시점은 서로 상관되어 있습니다. 구간이나 검정을 계산하기 전에 대상 집단과 독립 단위를 정합니다.

## 2. 주장한 효과에 맞는 검정 고르기

쌍별 부호 검정은 독립 쌍에서 후보가 기준선보다 좋은 경우가 더 많은지 살펴볼 수 있습니다. 쌍별 효과에서 미리 정한 실용 기준을 빼고, 정확히 같은 경우는 제외한 뒤 나머지 $n$개 중 양수인 $k$개를 셉니다. 연속적인 차이 분포에서 귀무가설은 각 부호의 확률을 절반으로 둡니다. 이 검정은 부호 확률 또는 중앙값을 다루며, 평균 효과에는 그에 맞는 분석이 필요합니다. [NIST 부호 검정](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm)

## 3. 정확한 단측 꼬리확률 유도하기

독립인 공정 부호에서 양수의 개수는 이항 분포입니다. 양의 효과를 대립가설로 두면

$$
p_{\rm tail}=\sum_{j=k}^{n}\binom nj2^{-n}.
$$

여섯 개가 모두 양수이면 $1/64=0.015625$이고, 여섯 개 중 다섯 개가 양수이면 $7/64=0.109375$입니다. 부호만 세므로 개선의 크기는 별도로 보고합니다.

## 4. 분석 계획과 결과 선택 구분하기

주요 결과, 실용 기준, 근거가 있는 단측 방향을 미리 정합니다. 독립인 여러 귀무 검정을 각각 유의수준 0.05로 수행하면 $m$개 중 적어도 하나의 거짓 양성이 나올 확률은 $1-0.95^m$입니다. 검정이 의존하면 이 식도 달라집니다. 다중검정 방법을 정하거나 선택된 후속 주장을 새 데이터로 확인합니다.

## 5. 귀무 부호를 모두 열거하기

```python
from itertools import product
from math import comb, isclose

def tail(n, k):
    return sum(comb(n, j) for j in range(k, n+1)) / 2**n

sign_patterns = list(product([-1, 1], repeat=6))
for k in (5, 6):
    enumerated = sum(sum(s > 0 for s in pattern) >= k for pattern in sign_patterns)/len(sign_patterns)
    assert isclose(tail(6, k), enumerated)
assert isclose(tail(6, 6), .015625)
assert isclose(tail(6, 5), .109375)
print("six positives:", tail(6, 6), "five positives:", tail(6, 5))
```

## 6. 추정값과 실험 단위 함께 보고하기

쌍별 효과값, 효과 크기 요약, 시드 식별자, 독립 단위 수, 검정 가정을 공개합니다. 실제로 바꾼 조건에 맞춰 데이터와 학습 무작위성의 불확실성을 보고합니다. 작은 p값은 가정 아래의 귀무가설에 반하는 근거이며, 운영 판단에는 효과 크기와 비용도 필요합니다.
