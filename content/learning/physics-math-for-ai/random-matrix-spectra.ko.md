## 관측 행에서 공분산 만들기

m×d 행렬 A의 행이 관측이고 열이 변수라고 합시다. S=AᵀA/m은 비중심화한 표본 이차 모멘트입니다. 모형 평균이 0임을 알고 있으므로 여기서는 표본평균을 빼거나 m−1로 나누지 않습니다. 이 구분은 실측 데이터의 covariance 전처리와 중요하게 연결됩니다.

$$
S=\frac1m A^\top A,\qquad v^\top Sv=\|Av\|^2/m\geq0.
$$

## 작은 행렬의 실제 고유값

실습은 고정 seed로 만든 유사 Gaussian 행렬과 Jacobi 대각화를 사용합니다. 파랑 점은 정렬한 유한 표본 고유값입니다. 연결선은 번호별 비교를 위한 것이며 스펙트럼의 확률밀도가 아닙니다. trace는 고유값 합과 같아 검산에 사용할 수 있습니다.

대각화에는 부동소수점 오차가 있습니다. 최대 고유값 규모와 1 중 큰 값의 10⁻¹⁰배 이하 절댓값은 수치 0으로 표시하며 그 허용오차를 문장에 기록합니다. 수치 랭크가 임의의 자료에서 기호적 랭크를 증명하는 것은 아닙니다.

## Marchenko–Pastur의 점근 기준

이상적 iid 중심 0·분산 1 원소에서 m,d가 함께 커지고 d/m→γ>0인 극한을 생각합니다. 적절한 가정 아래 nonzero bulk의 경계는 (1±√γ)²이고, γ>1에서는 0에 질량 1−1/γ가 생깁니다.

$$
\lambda_-=(1-\sqrt\gamma)^2,\quad\lambda_+=(1+\sqrt\gamma)^2,\quad
\text{zero atom mass}=\max(0,1-1/\gamma).
$$

주황 선은 이 점근 경계입니다. 현재처럼 최대 d=8인 작은 행렬의 신뢰구간이나 모든 고유값의 확정 범위가 아닙니다. 그 밖의 값이 나와도 극한 정리의 반례로 볼 수 없습니다.

## 표본 수가 변수 수보다 적으면

rank(AᵀA)≤min(m,d)이므로 d=8,m=3이면 영 공간 차원은 적어도 5입니다. nonzero bulk의 하한이 양수여도 별도의 영 원자를 빼먹으면 안 됩니다. 이상적 Gaussian의 랭크는 거의 확실히 min(m,d)이지만, 현재 화면은 정해진 유사난수 실현과 수치 허용오차로 검사합니다.

## AI에서 읽는 스펙트럼

표현·gradient·공분산의 스펙트럼은 조건수와 저차원 구조를 이해하는 도구입니다. 그러나 실제 학습 데이터의 의존·비등방성·spike 구조는 iid 기준과 다릅니다. normalization, centering, 표본/변수 축, 유한 표본과 극한을 먼저 명시한 뒤 비교해야 합니다.

## 작은 표본에서도 정확한 Gaussian 모멘트

이제 A의 모든 원소가 이상적인 독립 N(0,1)인 모형을 가정합니다. m,d는 양의 정수이며 평균 0을 알고 있으므로 S=AᵀA/m을 그대로 사용합니다. 표본평균을 빼는 추정량의 공식과 섞지 마세요. Gaussian 적분의 부분적분으로 E[G²]=1, E[G⁴]=3을 얻습니다.

대각 S_ii는 m개 G²의 평균이므로 E[S_ii]=1, Var(S_ii)=2/m입니다. 서로 다른 열 i≠j에서는 S_ij가 평균 0이고, 행 사이 독립성과 같은 행의 열 독립성으로 E[S_ij²]=1/m입니다. 대칭성으로 tr(S²)=Σ_i,j S_ij²이므로

$$
E\operatorname{tr}S=d,\qquad
E\operatorname{tr}(S^2)=d+\frac{d^2+d}{m},\qquad
\operatorname{Var}\!\left(\frac{\operatorname{tr}S}{d}\right)=\frac{2}{dm}.
$$

tr(S²)와 (tr S)²는 다릅니다. 후자의 기대값은 d²+2d/m입니다. 위 두 번째 모멘트는 off-diagonal 항까지 포함한 고유값 제곱의 합입니다.

## trace 집중과 스펙트럼 제어는 다른 주장이다

Chebyshev로 t>0에서 P(|tr S/d−1|≥t)≤min(1,2/(dmt²))를 얻습니다. 평균 고유값은 집중하지만 이것만으로 각 고유값이나 조건수를 제어하지는 못합니다. 결정적인 PSD 행렬 I_3과 diag(3,0,0)은 둘 다 trace 3이지만 연산자 노름은 1과 3, tr(S²)는 3과 9입니다. 이들은 비교용 행렬이며 Gaussian 표본이라고 주장하는 예가 아닙니다.

![trace 3인 두 PSD 행렬의 고유값 1,1,1과 3,0,0을 같은 축으로 비교합니다. 제곱합과 연산자 노름은 다릅니다.](/assets/learning/high-dimensional-trace-spectra-ko.svg)

m<d이면 영 고유값이 적어도 d−m개이고, 이상적 Gaussian에서는 이 수가 거의 확실히 정확합니다. 비영 고유값 전체가 1 근처라는 뜻은 아닙니다. 표본·차원 증가에서 d/m→γ이면 E[tr(S²)/d]=1+(d+1)/m은 1+γ로 갑니다. 이는 MP의 모멘트와 연결되는 계산이지, 그 자체로 스펙트럼 분포 수렴이나 극단 고유값 정리를 증명한 것은 아닙니다.

아래 Moment는 E[tr(S²)/d], Variance는 Var(tr S/d)입니다.

| (d,m) | Moment | Variance |
|---|---|---|
| (2,1) | 4 | 1 |
| (2,2) | $5/2$ | $1/2$ |
| (3,2) | 3 | $1/3$ |

## Gaussian을 표집하지 않고 다항식 적분 검산하기

각 독립 Gaussian 좌표에 세 절점 (−√3,0,√3)과 가중치 (1/6,2/3,1/6)를 사용하면 4차까지의 모멘트가 정확히 맞습니다. E[G]=E[G³]=0, E[G²]=1, E[G⁴]=3을 직접 확인할 수 있습니다. 위 trace 통계는 각 좌표에 대해 4차 이하인 다항식이므로 절점의 곱으로 정확히 적분됩니다.

코드는 절점을 √3·z, z∈{−1,0,1}로 써 Gram 원소를 분수 (3/m)Σz_ki z_kj로 계산합니다. 이 세 점 분포는 Gaussian 법칙 자체가 아닙니다. 예를 들어 모두 0인 행렬에도 양의 가중치가 있어 Gaussian의 랭크 확률을 재현하지 않습니다. 검산하는 대상은 지정한 다항식 모멘트뿐입니다. 화면의 유사 Gaussian seed 예제와도 별도입니다.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod

nodes, weights = [-1, 0, 1], {-1: Q(1, 6), 0: Q(2, 3), 1: Q(1, 6)}
for d, m in [(1, 2), (2, 1), (2, 2), (3, 2), (2, 3)]:
    total = mean_trace = mean_trace_square = mean_squared_trace = Q(0)
    for flat in product(nodes, repeat=m*d):
        weight = prod(weights[z] for z in flat)
        A = [flat[k*d:(k+1)*d] for k in range(m)]
        S = [[Q(3, m)*sum(A[k][i]*A[k][j] for k in range(m))
              for j in range(d)] for i in range(d)]
        trace = sum(S[i][i] for i in range(d))
        trace_square = sum(v*v for row in S for v in row)
        total += weight
        mean_trace += weight*trace
        mean_trace_square += weight*trace_square
        mean_squared_trace += weight*trace*trace
    assert total == 1 and mean_trace == d
    assert mean_trace_square == d+Q(d*d+d, m)
    assert mean_squared_trace == d*d+Q(2*d, m)
    variance = (mean_squared_trace-mean_trace**2)/(d*d)
    assert variance == Q(2, d*m)
    print("d, m, normalized second moment, normalized trace variance:",
          d, m, mean_trace_square/d, variance)
```

## 연구에 적용하기 전에 묻기

1. tr(S²)와 (tr S)²를 바꾸어 써도 되나요? **아니요. 다른 모멘트입니다.**
2. 평균 고유값이 1이면 작은 고유값도 안전한가요? **아니요. trace만으로 영 공간과 조건수를 알 수 없습니다.**
3. 세 점 적분으로 무엇을 검산했나요? **지정한 저차 다항식 모멘트이며 전체 스펙트럼 법칙이나 랭크 확률은 아닙니다.**

Gaussian 행렬과 점근·유한 표본 구분의 배경은 [Vershynin, §5.3.1·5.4.3](https://arxiv.org/html/1011.3027)을 참고하세요. 위 모멘트와 세 절점 분수 적분은 명시한 모형에서 직접 유도했습니다.

## 확인 문제와 해설

1. d>m일 때 반드시 생기는 것은? **영 공간입니다.**
2. 주황 경계 밖 값은 정리 위반인가요? **아니요. 점근 기준입니다.**
3. S는 왜 양의 준정부호인가요? **모든 v에서 vᵀSv가 제곱 노름이기 때문입니다.**

추가 읽기: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).

[Vershynin, Introduction to the Non-Asymptotic Analysis of Random Matrices](https://arxiv.org/abs/1011.3027).
