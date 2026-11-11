## Count 경로와 Count 분포는 다르다

동차 Poisson 과정은 일정 intensity lambda와 겹치지 않는 구간의 독립 증분을 가집니다. N(T)의 평균·분산은 lambda T입니다. 한 count가 그 값과 같을 필요는 없습니다. 동전 한 번의 결과가 앞면 확률 자체가 아닌 것과 같습니다.

위 그림은 seed를 고정한 도착 경로와 이론선 lambda t입니다. 파란 jump는 실제 생성한 사건이고 주황선은 여러 실현에 대한 기대값입니다. Rate가 0이면 도착이 없고 관측 시간이 0이면 count도 0입니다.

## 대기시간으로 도착을 만든다

양의 intensity에서 독립 지수 대기시간은 memoryless 성질을 가집니다. 열린 구간 uniform U로 -log(U)/lambda를 만들고 관측창을 벗어나기 전까지 누적합니다. Seed는 재현성을 주지만 독립 증분의 정리를 표본으로 증명하지는 않습니다.

$$
\Pr(N(T)=k)=e^{-m}\frac{m^k}{k!},\qquad m=\lambda T,
\qquad p_{k+1}=p_k\frac m{k+1}.
$$

이 재귀로 count 확률을 직접 계산합니다. 보이는 도착 개수를 불확실성이 없는 이론값으로 바꾸어 읽지 마세요. 시뮬레이션의 안전 cap에 닿았는지도 따로 표시합니다.

## 잘린 그림은 빠진 질량을 남겨야 한다

파란 stem은 p0부터 pK입니다. K+1 위치의 주황 stem은 단일 count가 아니라 N(T)>K 전체 사건입니다. Cutoff를 확률 모형 변경과 혼동하지 않도록 bucket 의미를 명시합니다.

남은 양의 recurrence 항들을 더하며 파랑을 억지로 합 1로 만들지 않습니다. 재정규화하면 N(T)<=K라는 조건부 법칙입니다. 의도했을 때만 별도 모형으로 사용할 수 있습니다. Cutoff만 바꾸어도 도착 경로 자체는 바뀌지 않습니다.

## Intensity 2를 손계산하기

lambda=2, T=1이면 처음 세 확률은 exp(-2), 2exp(-2), 2exp(-2)입니다. 합은 약 0.676676, 꼬리는 약 0.323324입니다. 한 경로의 도착 수가 무엇이든 이론적 count 분산은 2입니다.

## 총 Count를 조건화하면 의존 구조가 바뀐다

0<s<T에서 독립 증분으로 N(s)∼Poisson(λs), N(T)−N(s)∼Poisson(λ(T−s))다. Joint 확률을 전체 Poisson 확률로 나누면 λ>0, 관측 총수 n에서

$$
\Pr(N(s)=k\mid N(T)=n)=\binom nk(s/T)^k(1-s/T)^{n-k}.
$$

총수를 조건화하면 한 구간의 count가 다른 것을 결정해 음의 의존성이 생긴다. 무조건 공분산은 0이지만 조건부 공분산은 −n(s/T)(1−s/T)다. 도착 n개를 조건화한 순서 도착시각의 밀도는 ordered simplex에서 n!/Tⁿ이며 독립 uniform 시각 n개의 순서통계량과 같다. 잘린 Poisson 그림의 재정규화와는 다른 조건화다.

n=4, s/T=1/4이면 조건부 평균은 1, 분산은 3/4이다.

| k | 조건부 확률 |
|---|---:|
| 0 | 81/256 |
| 1 | 108/256 |
| 2 | 54/256 |

```python
from fractions import Fraction as F
from math import comb
n, p = 4, F(1, 4)
law = [comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
mean = sum(k*w for k,w in enumerate(law))
variance = sum((k-mean)**2*w for k,w in enumerate(law))
covariance = sum((k-mean)*((n-k)-(n-mean))*w for k,w in enumerate(law))
assert sum(law)==1 and mean==1 and variance==F(3,4)
assert covariance == -variance
print(law, mean, variance, covariance)
```

유리수로 전체 이항 법칙과 음의 공분산을 확인한다. 무조건 count의 로그 우도는 상수를 제외하면 n logλ−λT이고 내부 최댓값은 n/T, λ>0의 Fisher 정보는 T/λ다. n=0에서는 비음수 rate MLE가 경계에 있으므로 내부 정규 근사는 별도 정당화가 필요하다. Count 우도·조건부 bridge·무조건 seed 경로는 다른 통계 질문이다.

## 직접 풀기: 두 창과 두 Cutoff

인접한 단위 구간의 count는 독립 Poisson(lambda)이고 합은 Poisson(2lambda)입니다. 이 분포 명제와 한 2단위 경로를 세는 작업을 구별하세요. lambda T=2에서 K를 2→10으로 늘리면 왜 꼬리만 줄고 도착 시간·평균은 같은지도 설명하세요. 재정규화한 세 stem이 뜻하는 조건부 법칙도 적으세요.

더 읽기: [Lawler 저자의 확률미적분 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
