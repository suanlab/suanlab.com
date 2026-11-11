## 측정 수보다 먼저 독립 단위 정하기

평균의 불확실성은 측정 개수뿐 아니라 측정 사이 공분산에 의존합니다. 같은 개체·데이터셋·학습 실행에서 반복한 값들을 서로 독립이라고 세면 표준오차를 잘못 계산할 수 있습니다. 실습은 알려진 단위 분산과 상관을 가정한 세 설계를 비교합니다. 상관은 실제 데이터에서 추정한 값이 아니라 모형 입력입니다. 첫 두 모드는 평균 차이, 군집 모드는 한 모집단 평균이라는 서로 다른 추정 대상을 갖습니다.

$$
\operatorname{Var}\!\left(\frac1N\sum_iY_i\right)=\frac1{N^2}\sum_{i,j}\operatorname{Cov}(Y_i,Y_j).
$$

## 쌍 비교에서는 공통 잡음이 상쇄된다

n개 독립 쌍 안에서 두 측정의 분산이 각각 1이고 상관이 ρ이면 차이의 분산은 2(1−ρ)입니다. n=10,ρ=0.8이면 평균 차이 표준오차는 0.2입니다. 서로 독립인 두 그룹에서 각각 n개를 측정하면 표준오차는 √(2/n)≈0.447입니다. 두 설계 모두 측정은 2n개이지만 공분산 구조가 다릅니다. 이 모드들에서는 반복 수 k 조절기가 사용되지 않습니다.

$$
\operatorname{Var}(\overline{Y_B-Y_A})=\frac{2(1-\rho)}n,\qquad\operatorname{Var}(\bar Y_B-\bar Y_A)=\frac2n\ \text{for independent groups}.
$$

## 군집 반복은 독립 표본을 늘리지 않는다

n개 독립 군집마다 k개 측정이 있고 군집 안 공통 상관이 ρ≥0이면 평균 분산은 [1+(k−1)ρ]/(nk)입니다. 독립으로 가정한 분산 1/(nk)에 비해 설계 효과 DE만큼 큽니다. n=10,k=4,ρ=0.8이면 DE=3.4, 실제 SE≈0.292, naive SE≈0.158입니다. 표의 유효 표본 수는 이 단위 분산 평균 모형에 대한 등가 수이며 모든 추정량에 공통인 숫자는 아닙니다.

$$
\mathrm{DE}=1+(k-1)\rho,\qquad n_{\mathrm{eff}}=\frac{nk}{\mathrm{DE}}.
$$

## 랜덤화·블록·평가 분할의 연결

ρ=1인 군집은 반복 잡음이 완전히 같아 유효 표본 수가 n입니다. 쌍 차이의 오차가 0이 되는 완전 상관 경계도 모형의 강한 가정을 반영하며 실제 무오차를 보장하지 않습니다. 실제 실험은 무작위 배정으로 교란을 줄이고 블록이나 대응으로 변동을 줄이는 원리를 구분해야 합니다. 무작위 배정과 모집단에서의 무작위 표집도 다릅니다. AI 평가에서는 재사용한 데이터와 seed, 선택한 지표, held-out 분할, 비교 계획을 함께 기록해야 합니다.

## 공통 잡음과 개별 잡음으로 군집 공식을 유도하기

0≤ρ≤1이고 독립 군집 C개에 각각 m번 측정하는 균형 설계에서 Y_ji=μ+U_j+ε_ji로 씁니다. 평균 0인 U_j의 분산은 ρ, 평균 0인 ε_ji의 분산은 1−ρ이고 모든 군집 효과와 개별 오차는 서로 독립이라고 가정합니다. 각 측정 분산은 1이고 같은 군집의 서로 다른 측정 사이 공분산은 ρ입니다. 전체 평균은 C개 공통 효과와 Cm개 개별 오차를 따로 평균하므로 다음과 같습니다.

$$
\operatorname{Var}(\bar Y)=\frac{\rho}{C}+\frac{1-\rho}{Cm}
=\frac{\rho+(1-\rho)/m}{C}.
$$

이 분산 계산에는 Gaussian 잡음이 필요하지 않습니다. 공식만으로 신뢰구간 포함률이 보장되지는 않습니다. 추정 대상은 명시한 독립 군집 모형의 공통 모집단 평균입니다.

## 반복 수 선택 전에 비용 모형 추가하기

군집을 하나 시작하는 비용은 a>0, 한 번 측정하는 비용은 b>0, 총예산은 T라고 합시다. 정수 C≥2,m≥1과 C(a+bm)≤T를 만족하는 균형 설계로 범위를 정합니다. 예산을 모두 쓰는 연속 완화에서는 C=T/(a+bm)입니다. 0<ρ<1에서 분산은 다음과 같습니다.

$$
V(m)=\frac{(a+bm)(\rho+(1-\rho)/m)}{T},\qquad
m_* = \sqrt{\frac{a(1-\rho)}{b\rho}}.
$$

m*를 구하려면 TV(m)=aρ+b(1−ρ)+bρm+a(1−ρ)/m으로 전개합니다. 커지는 bρm과 작아지는 a(1−ρ)/m의 도함수가 상쇄되는 곳이 해입니다. 이차 미분은 양수이므로 제약 없는 연속 최소는 유일합니다. 이후 m≥1,C≥2 같은 경계도 적용해야 합니다. 시작 비용이 크면 반복을 늘리는 쪽, 공통 잡음이 크면 독립 군집을 늘리는 쪽이 유리해집니다.

## 정수 예산이 최적 배분을 바꿀 수 있다

a=9,b=1,T=120,ρ=1/4로 둡니다. 완화한 반복 최적값은 √27≈5.196이지만 이를 5로 반올림한 설계가 가장 좋은 정수 균형 설계는 아닙니다. m을 고정하면 C가 클수록 분산이 작으므로 C=floor(120/(9+m))를 사용합니다. C≥2 조건에서 가능한 반복 수 전체는 1부터 51입니다.

| m | C | 비용 | 평균 분산 |
|---|---|---|---|
| 1 | 12 | 120 | $1/12$ |
| 3 | 10 | 120 | $1/20$ |
| 5 | 8 | 112 | $1/20$ |
| 6 | 8 | 120 | $3/64$ |

이 전체 범위를 열거하면 m=6,C=8이 유일한 최적해이고 분산은 3/64입니다. m=5에서는 아홉 번째 군집을 시작하면 예산을 넘기므로 비용 8을 남깁니다. 정수 비용 제약 때문에 단순 반올림이 실패합니다. 지정한 균형 설계 안의 최적해이며 군집 크기 불균형, 다른 비용, 정보성 군집 표집은 별도 문제입니다.

## 배분과 공분산 계산을 정확하게 검산하기

다음 코드는 가능한 모든 정수 균형 설계를 분수로 열거하고 표의 선택에서는 전체 공분산 행렬을 독립적으로 합산합니다. 계획 상관에 대한 민감도도 보여줍니다. ρ를 추정했다면 그 불확실성도 설계 민감도 분석에 포함해야 합니다. 비용 모형은 별도 실행 확장이며 기존 조절기는 고정 개수의 대응·독립·군집 설계를 비교하고 예산을 최적화하지는 않습니다.

```python
from fractions import Fraction as Q
from math import sqrt

budget, start_cost, repeat_cost = 120, 9, 1
rho = Q(1, 4)

def designs(rho):
    out = []
    for m in range(1, 52):  # Exactly the feasible repeat range when C >= 2.
        C = budget//(start_cost+repeat_cost*m)
        assert C >= 2
        cost = C*(start_cost+repeat_cost*m)
        variance = (rho+(1-rho)/m)/C
        out.append((m, C, cost, variance))
    return out

rows = designs(rho)
best = min(row[3] for row in rows)
assert [row for row in rows if row[3] == best] == [(6, 8, 120, Q(3, 64))]
continuous = sqrt(start_cost*(1-float(rho))/(repeat_cost*float(rho)))
assert round(continuous) == 5
for m, C, cost, variance in rows:
    if m not in [1, 3, 5, 6]:
        continue
    N = C*m
    covariance_sum = sum(Q(1) if i == j else rho if i//m == j//m else Q(0)
                         for i in range(N) for j in range(N))
    assert covariance_sum/(N*N) == variance
    print("repeats, clusters, cost, mean variance:", m, C, cost, variance)
for alternative in [Q(1, 10), Q(1, 4), Q(1, 2)]:
    candidate = designs(alternative)
    value = min(row[3] for row in candidate)
    print("rho:", alternative, "optimal balanced designs:",
          [row for row in candidate if row[3] == value])
print("Continuous repeat optimum:", continuous)
```

## 추가 문제와 연구 연결

1. 측정을 늘리는 것이 군집을 늘리는 것보다 덜 유용할 수 있나요? **반복에도 U_j의 공통 기여가 남으므로 가능합니다.**
2. 연속 최적해를 구하면 정수 제약은 없어지나요? **표의 최적 배분이 m* 반올림과 다르듯 그렇지 않습니다.**
3. 연구의 예산 비교에는 무엇을 보고해야 하나요? **추정 대상, 독립 단위, 공분산 가정, 비용 모형과 허용 설계 집합입니다.**

배분 공식은 명시한 랜덤 효과·선형 비용 모형에서 직접 유도했습니다. 반복 측정을 독립 증거로 읽기 전에 [기대·분산 항등식](../expectation-variance/)과 [대응 연구 예제](../research-methods/)를 함께 비교하세요.

## 확인 문제와 해설

1. ρ=0이면 군집 설계 효과는? **1입니다.**
2. k=1이면 반복 상관이 분산을 바꾸나요? **아니요. 군집당 측정이 하나입니다.**
3. 같은 테스트셋에서 seed만 바꾼 값을 모든 모집단 변동으로 읽어도 되나요? **아니요. 어떤 변동을 반복했는지 구분해야 합니다.**

추가 읽기: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).
