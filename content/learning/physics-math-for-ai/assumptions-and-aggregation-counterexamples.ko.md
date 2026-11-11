## 1. 지표가 대표하는 대상을 밝히기

재시작 사건 $i$에 평가 시점이 $n_i>0$개 있고, 그 사건 안에서 평균한 쌍별 개선량이 $d_i$라고 가정합니다. 사건 가중 평균은 각 재시작에 같은 가중치를 줍니다. 시점 가중 평균에서는 더 길게 관측한 재시작에 가중치가 큽니다. 관측 횟수가 다르면 두 지표가 겨냥하는 대상도 다릅니다.

## 2. 가정과 함께 정리 적기

사건 $N$개에 대해 윗줄은 사건별 산술평균을 뜻합니다. $\operatorname{Cov}_N(n,d)=N^{-1}\sum_i(n_i-\bar n)(d_i-\bar d)$로 정의합니다. $\bar n>0$이면

$$
\frac{\sum_i n_i d_i}{\sum_i n_i}=\bar d+\frac{\operatorname{Cov}_N(n,d)}{\bar n}.
$$

관측 횟수가 모두 같으면 두 평균이 같습니다. 일반적으로 두 평균이 같을 필요충분조건은 이 공분산이 0인 것입니다.

## 3. 항등식 증명하기

평균을 뺀 두 항의 곱을 전개합니다. 평균을 뺀 값들의 합은 0이므로 교차항을 정리하면 $\operatorname{Cov}_N(n,d)=\overline{nd}-\bar n\bar d$입니다. $\bar n$으로 나누어 정리합니다. 음의 개선량이 있더라도 주어진 관측 횟수 조건을 만족하는 모든 유한 목록에 증명이 적용됩니다.

## 4. 반례 구성하기

두 사건의 횟수가 $(1,9)$이고 효과가 $(0.4,-0.1)$ K라고 합시다. 사건 가중 효과는 $0.15$ K지만 시점 가중 효과는 $-0.05$ K입니다. 효과가 나쁜 긴 사건에 더 큰 가중치가 실립니다. 사건 가중 개선이 양수이면 시점 가중 개선도 양수라는 주장에 대한 반례입니다.

## 5. Python으로 계산과 항등식 검산하기

```python
from math import isclose
from statistics import mean

counts = [1, 9]
effects = [.4, -.1]
macro = mean(effects)
micro = sum(n*d for n, d in zip(counts, effects)) / sum(counts)
covariance = mean((n-mean(counts))*(d-macro) for n, d in zip(counts, effects))
assert isclose(micro, macro + covariance/mean(counts))
assert isclose(macro, .15) and isclose(micro, -.05)
print("episode mean:", macro, "point mean:", micro)
```

수치 예는 계산을 확인하고 반례를 제공합니다. 일반 항등식은 대수적 증명으로 성립합니다.

## 6. 결과가 뒷받침하는 주장 기록하기

평가 전에 연구 질문에 맞춰 가중치를 고릅니다. 전형적인 사건과 전형적인 관측 시점은 서로 다른 대상입니다. 관측 횟수, 가중치 규칙, 다른 타당한 선택에 대한 민감도를 보고합니다. 대수적 항등식, 가정 아래 성립하는 성질, 경험적 관측을 구분합니다.
