## 유한 경로 그림은 무한차원 법칙의 표본이다

표준 Brownian motion은 0에서 시작하고 연속 경로와 독립 Gaussian 증분을 가지며 증분 분산은 시간 길이입니다. 실험은 128-step 미세 grid의 seed 고정 증분을 만들고, 거친 grid는 같은 증분을 묶습니다. 세분할 때 공유 끝점을 유지하며 다른 경로를 새로 만들지 않습니다.

파랑의 직선 연결은 렌더링 선택입니다. Brownian 경로가 고전적으로 미분 가능한 조각 직선이라는 주장이 아닙니다. 주황 ±sqrt(t)는 주변 분포의 1 표준편차이며 전체 경로를 동시에 둘러싸는 신뢰구간이나 결정론적 상한이 아닙니다.

## 공분산은 공유 증분에서 나온다

s<=t에서 W_t=W_s+(W_t-W_s)입니다. 뒤 증분은 W_s와 독립이고 평균 0이므로 다음입니다.

$$
E[W_t]=0,\qquad \operatorname{Var}(W_t)=t,\qquad
\operatorname{Cov}(W_s,W_t)=\min(s,t).
$$

Control은 T에 대한 시간 비율입니다. 표시값은 한 경로에서 추정한 공분산이 아니라 Gaussian 과정 법칙의 성질입니다. 서로 다른 시각의 값이 상관되어도 겹치지 않는 증분이 독립이라는 사실과 모순되지 않습니다.

## 제곱 증분 합은 다르게 움직인다

N개의 균등 구간에서 QV는 증분 제곱의 합입니다. 각 증분의 분산은 T/N이고 제곱의 평균·분산은 T/N, 2(T/N)²입니다.

$$
E[QV]=T,\qquad \operatorname{Var}(QV)=2T^2/N.
$$

실제 QV는 보이는 경로에서 계산하며 보통 T와 다릅니다. 세분화는 이론적인 흔들림 규모를 줄이지만 관측값이 단조롭게 T에 접근할 필요는 없습니다. 한 좋은 그림이 정리를 증명하지 않습니다.

## 시간 배율과 표본을 구별하기

N=16,T=1이면 QV 분산은 0.125, N=64이면 0.03125입니다. 이 coupling에서는 끝점이 같습니다. 같은 seed에서 T를 두 배로 하면 경로는 sqrt(2)배, QV는 2배, 그 이론 분산은 4배입니다.

## 유한 Chi-square 법칙에서 경로별 결론까지

균등 Brownian 격자에서 QV_N=(T/N)ΣZᵢ²이며 Zᵢ는 독립 표준 Gaussian이다. 따라서 N QV_N/T는 자유도 N인 chi-square다. 적률생성함수 (1−2s)^(−N/2)에 양·음의 지수 인자를 최적화하면 0<δ<1에서

$$
\Pr(|QV_N/T-1|\ge\delta)\le
e^{-N(\delta-\log(1+\delta))/2}+
e^{-N(-\delta-\log(1-\delta))/2}.
$$

별도로 E[(QV_N−T)²]=2T²/N이 L² 수렴을 증명한다. 중첩 dyadic 격자 N=2ᵏ에서는 고정 양의 오차 문턱의 Chebyshev 확률합이 유한하다. Borel–Cantelli와 가산 문턱열로 거의 확실한 dyadic 수렴을 얻는다. 격자 사이 독립성은 필요 없다. 이는 무한 Brownian 구성에 대한 명제이며 실습의 유한 128-step 표시로 증명한 것이 아니다.

![Quadratic variation의 상대 오차에 대한 두 이론적 확률 상계이며 표본 오류 빈도와 구별하는 그림.](/assets/learning/stochastic-variation-ko.svg)

| N, T=1 | 정확한 QV 평균제곱오차 |
|---|---:|
| 16 | 1/8 |
| 64 | 1/32 |
| 256 | 1/128 |

```python
from fractions import Fraction as F
from math import exp, log1p
delta = .5
for N in (16, 64, 256):
    mse = F(2, N)
    chernoff = min(1., exp(-N*(delta-log1p(delta))/2)+exp(-N*(-delta-log1p(-delta))/2))
    chebyshev = min(1., float(mse)/(delta*delta))
    assert 0 < chernoff <= 1 and 0 < chebyshev <= 1
    print(N, mse, chernoff, chebyshev)
# Sum Var(QV_2^k) from k=4 to infinity, T=1.
assert 2*F(1,16)/(1-F(1,2)) == F(1,4)
```

연속·유한 total variation 경로에서는 Σ(Δx)²≤max|Δx|×total variation이 격자 축소로 0에 간다. Brownian dyadic QV는 T>0으로 가므로 total variation이 거의 확실히 무한하다. 코드는 moment 배율과 상계 상수를 확인하며 시뮬레이션으로 경로 정리를 증명하지 않는다.

## 직접 풀기: 공유 구간 설명

T=2에서 1/4·3/4 시각의 공분산은 1/2입니다. 두 합에 함께 들어가는 증분으로 유도하세요. 거친·미세 grid의 끝점이 같아도 QV가 다른 이유는 합의 제곱에 cross term이 생기기 때문입니다. 주황 표준편차 선을 벗어난 것이 왜 모형 실패가 아닌지도 설명하세요.

더 읽기: [Lawler의 Brownian·Variation 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
