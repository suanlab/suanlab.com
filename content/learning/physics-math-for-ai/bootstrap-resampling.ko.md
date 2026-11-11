## 표집분포를 모르는 상황

통계량의 표집분포는 모집단 법칙에 의존하지만 실제로 그 법칙을 모르는 경우가 많습니다. 비모수 Bootstrap은 관측 경험분포를 대신 넣어 재표본을 만듭니다. 현재 데이터 (0,1,2)는 각 관측 행에 질량 1/3을 배정합니다. 재표본의 크기도 3이고 복원추출하므로 같은 관측을 여러 번 사용할 수 있습니다.

$$
\widehat P_3=\frac13\sum_{i=1}^3\delta_{x_i},\qquad X_1^*,X_2^*,X_3^*\mid x\overset{\mathrm{iid}}\sim\widehat P_3.
$$

## 작은 경우의 완전 열거

각 위치에서 세 행 중 하나를 고르므로 순서 재표본은 27개입니다. 실습은 무작위 일부만 뽑지 않고 전부 열거합니다. 중복 값이 있어도 행 선택은 구별되며 같은 통계량 결과를 합산합니다. 평균 모드에서 가능한 값은 0부터 2까지 1/3 간격이고 빈도는 (1,3,6,7,6,3,1)입니다. 이는 관측 데이터에 조건부인 정확한 분포이지 모집단에서의 정확한 표집분포가 아닙니다.

$$
P^*(T^*=t\mid x)=\frac{\#\{\text{ordered resamples giving }t\}}{27}.
$$

## 편향·표준오차와 비선형 통계량

평균은 Bootstrap 조건부 기대가 원래 평균과 같아 여기서 편향 추정이 0입니다. 원래 (0,1,2)의 평균 표준오차는 √(2/9)입니다. (0,0,3)의 중앙값으로 바꾸면 원래 중앙값은 0이지만 재표본 중앙값이 3일 확률은 7/27입니다. 따라서 조건부 평균은 7/9이고 편향 추정도 약 0.778입니다. 모수의 참 편향을 정확히 알아낸 것이 아니라 경험분포에서 계산한 값입니다.

$$
\widehat{\mathrm{bias}}^*=E^*[T^*]-T(x),\qquad\widehat{\mathrm{SE}}^*=\sqrt{\operatorname{Var}^*(T^*)}.
$$

## 백분위 구간의 한계

구간 끝점은 재표본 통계량의 경험 CDF 역함수로 정합니다. 정렬한 27개 값에서 누적비율이 q 이상이 되는 첫 값을 택하며 보간하지 않습니다. 80%나 95%는 선택한 백분위 수준이지 이 작은 데이터의 참 포함률 보장이 아닙니다. 상수 데이터에서는 재표집 표준오차가 0이지만 원래 모집단이 상수라는 증거는 아닙니다. 시계열·군집 자료라면 iid 행 재표집 대신 의존성을 반영하는 단위를 검토해야 합니다.

$$
[\,\widehat F_*^{-1}(\alpha/2),\;\widehat F_*^{-1}(1-\alpha/2)\,].
$$

## 통상적인 bootstrap이 재현하지 못하는 통계량

θ>0에서 iid Uniform(0,θ) 표본과 최댓값 M_n을 생각합시다. 0≤t≤n이면 독립성으로 끝점 추정 pivot의 정확한 생존확률을 구할 수 있습니다.

$$
R_n=\frac{n(\theta-M_n)}\theta,\qquad
P_\theta(R_n>t)=(1-t/n)^n\longrightarrow e^{-t}.
$$

따라서 R_n은 표준 지수분포로 수렴하며 그 CDF는 연속이고 원점에 점 질량이 없습니다. 유한 n에서도 거의 확실히 M_n<θ이므로 P(R_n=0)=0입니다. 표본평균의 CLT가 아니라 경계 통계량의 문제입니다.

## 경험분포 재표집에서 사라지지 않는 질량 찾기

거의 확실히 관측 최댓값은 유일합니다. 크기 n의 경험분포 bootstrap에서는 매번 그 행을 고를 확률이 1/n입니다. 한 번이라도 그 행을 고르면 재표본 최댓값 M_n*는 M_n과 같습니다. 따라서 bootstrap pivot에는 큰 점 질량이 생깁니다.

$$
R_n^*=\frac{n(M_n-M_n^*)}{M_n},\qquad
P^*(R_n^*=0\mid X)=1-(1-1/n)^n\longrightarrow1-e^{-1}.
$$

표의 두 확률 열은 원점의 점 질량입니다.

| n | Bootstrap | 모집단 |
|---|---|---|
| 3 | 0.703704 | 0 |
| 10 | 0.651322 | 0 |
| 100 | 0.633968 | 0 |

pivot이 비음수이므로 원점 CDF는 이 점 질량입니다. 모집단 극한 CDF는 그 연속점에서 0입니다. 따라서 조건부 경험 bootstrap 법칙은 이 pivot의 극한 법칙을 일치적으로 근사할 수 없습니다. bootstrap 반복 수를 늘려도 잘못된 조건부 분포 주변의 시뮬레이션 오차만 줄어들고 점 질량은 사라지지 않습니다. 아래 완전 열거에는 Monte Carlo 오차가 없지만 같은 문제가 남습니다.

## 모형 기반 재표본은 이 pivot에서 성공할 수 있다

Uniform(0,θ) 모형이 정당하다면 대신 Uniform(0,M_n)에서 새 독립 표본을 뽑습니다. 이를 iid Uniform(0,1) 기초 변수 U_i*에 M_n을 곱한 값으로 쓰면 정규화 pivot은 n(1−max U_i*)입니다. 따라서 M_n>0인 모든 관측에서 유한 n의 R_n과도 법칙이 정확히 같습니다.

$$
P^*\!\left(\frac{n(M_n-M_{n,\mathrm{param}}^*)}{M_n}>t\mid X\right)=(1-t/n)^n\quad(0\le t\le n).
$$

성공의 조건은 가정한 균등분포 모형과 scale-free pivot입니다. 모집단 모형이 틀려도 모수적 재표집이 타당하다는 뜻은 아닙니다. 재표집 기제는 반복 횟수만의 문제가 아니라 통계 모형의 선택입니다.

## 관측 세 점의 경험 예제를 완전 열거하기

관측값 (1/5,1/2,4/5)의 27개 순서 경험 재표본은 pivot 값 0,9/8,9/4를 각각 19,7,1번 만듭니다. 다음 코드는 정확한 분수와 모형 기반 scale 항등식을 확인합니다. 고정된 기초 값은 그 항등식의 예시이며 무작위 추출이 아닙니다. 이 최댓값 예제는 기존 세 행의 평균·중앙값 조절기와 별도로 실행합니다.

```python
from fractions import Fraction as Q
from itertools import product
from collections import Counter
from math import exp

sample = (Q(1, 5), Q(1, 2), Q(4, 5))
n, observed_max = len(sample), max(sample)
counts = Counter(n*(observed_max-max(draw))/observed_max
                 for draw in product(sample, repeat=n))
assert counts == Counter({Q(0): 19, Q(9, 8): 7, Q(9, 4): 1})
for value, count in sorted(counts.items()):
    print("empirical-bootstrap pivot, conditional mass:", value, Q(count, n**n))
for n in [3, 10, 100]:
    atom = 1-(1-Q(1, n))**n
    population_survival_at_one = (1-Q(1, n))**n
    assert atom+population_survival_at_one == 1
    print("n, empirical atom at zero, population CDF at zero:", n, float(atom), 0)
print("limiting empirical atom:", 1-exp(-1))
# Fixed base values illustrate the scale identity; these are not random draws.
u = (Q(1, 7), Q(2, 3), Q(4, 5))
for M in [Q(1, 10), Q(4, 5), Q(3)]:
    parametric_sample = tuple(M*x for x in u)
    pivot = len(u)*(M-max(parametric_sample))/M
    assert pivot == len(u)*(1-max(u))
print("Parametric normalized pivot is independent of the positive plug-in endpoint.")
```

## 추가 문제와 연구 연결

1. 경험 bootstrap을 정확히 열거하면 참 표집법칙이 되나요? **시뮬레이션 오차를 제거할 뿐 경험 법칙과 모집단의 차이는 남습니다.**
2. 경험 재표집의 최댓값에는 왜 점 질량이 있나요? **관측 최댓값을 한 번이라도 선택하면 그 값이 유지되기 때문입니다.**
3. bootstrap 보고서에는 무엇을 명시해야 하나요? **통계량, 재표집 단위·법칙, 원하는 근사를 뒷받침하는 가정입니다.**

최댓값 반례와 모형 기반 비교는 고전적인 결과입니다. [Bickel–Freedman (1981), §6, Counter-example 2, p.1210](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/BickelFreedman81.pdf), [출판 DOI](https://doi.org/10.1214/aos/1176345637)를 참고하세요. 세 점의 유한 열거는 여기서 직접 계산했습니다.

## 확인 문제와 해설

1. 27개 중 같은 숫자 표본도 따로 세나요? **행을 고른 순서가 다르면 따로 세고 결과 통계량에서 합칩니다.**
2. 모든 데이터를 한 번씩만 섞는 것은 같은가요? **아니요. 여기서는 복원추출입니다.**
3. 작은 n의 percentile 구간이 항상 명목 포함률인가요? **아니요. 추가 이론과 진단이 필요합니다.**

추가 읽기: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
