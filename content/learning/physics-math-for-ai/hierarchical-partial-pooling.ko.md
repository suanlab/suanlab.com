## 작은 그룹을 따로 추정할 때 생기는 문제

여러 병원·도시·데이터셋의 평균을 각각 추정한다고 합시다. 표본이 적은 그룹의 관측 평균은 크게 흔들립니다. 모든 그룹을 하나의 평균으로 합치면 그룹 차이를 지우고, 완전히 별개로 추정하면 공통 정보를 버립니다. 계층모형은 그룹 모수가 공통 분포에서 왔다고 가정해 두 극단 사이의 부분 풀링을 만듭니다.

$$
\theta_j\sim N(0,\tau^2),\qquad \bar Y_j\mid\theta_j\sim N(\theta_j,1/n).
$$

## 정밀도를 더해서 사후 계산하기

개별 관측 분산은 알려진 1이고 그룹당 관측 수는 n입니다. 사전 정밀도는 1/τ², 관측 평균의 정밀도는 n입니다. 제곱 완성으로 사후 분산과 평균을 구합니다.

$$
v=(n+\tau^{-2})^{-1},\qquad m_j=v n\bar y_j=w\bar y_j,\qquad w=\frac{n\tau^2}{1+n\tau^2}.
$$

n=4, τ=1이고 두 관측 평균이 2와 −1이면 w=0.8, 사후평균은 1.6과 −0.8, 각 사후 분산은 0.2입니다. 파랑은 관측 평균, 주황은 사후평균입니다. 점의 높이가 평균이고 분산은 별도 수치와 표에서 읽습니다.

## 당기는 힘을 바꾸기

τ가 작으면 그룹 차이가 작다는 강한 사전이므로 공통 중심 0으로 더 많이 당깁니다. τ=0.25이면 w=0.2로 첫 평균 2가 0.4까지 줄어듭니다. 반대로 n이 커지면 그룹 데이터가 강해져 w가 1에 가까워집니다. 이 값은 알고리즘이 임의로 고른 혼합 비율이 아니라 명시한 확률 모형의 결과입니다.

## 이 실습이 조건부 계산인 이유

공통 중심 0과 τ는 고정 입력입니다. 초모수의 사후를 추정하거나 적분하지 않습니다. 따라서 이 모형에서 두 그룹의 조건부 사후 공분산은 0입니다. 공통 중심을 미지수로 두어 함께 추론하면 그룹들이 사후에서 의존할 수 있습니다. 데이터로 τ를 추정해 고정하는 empirical Bayes와 초모수 사후까지 적분하는 완전 Bayes도 구분하세요.

새 그룹의 관측이 없다면 그 그룹 모수 분산은 τ²입니다. 기존 그룹의 조건부 사후 분산 v를 새 그룹의 불확실성으로 대신 쓰면 안 됩니다. AI에서는 task별 성능·지역별 효과처럼 공유 구조와 고유 차이가 공존할 때 이 구분이 중요합니다.

## 공통 중심을 모르면 그룹들이 사후에서 연결된다

위 화면은 중심을 0으로 고정합니다. 이번 별도 모형에서는 중심 μ에도 사전을 둡니다. 그룹 j=1,…,J에서 n_j≥1이고 개별 관측 분산 1, s²>0과 τ²>0은 알려져 있다고 합시다. μ를 주면 그룹 모수들이 독립이고, 모수들을 주면 관측 평균들도 독립입니다.

$$
\mu\sim N(0,s^2),\qquad \theta_j\mid\mu\sim N(\mu,\tau^2),\qquad
\bar Y_j\mid\theta_j\sim N(\theta_j,v_j),\quad v_j=1/n_j.
$$

θ_j를 먼저 적분하면 관측 평균은 μ를 중심으로 분산 h_j=τ²+v_j인 정규분포입니다. 정밀도를 모으고 제곱 완성하면 공통 중심의 사후도 정규분포가 됩니다.

$$
\mu\mid y\sim N(m_\mu,S),\qquad
S=\left(s^{-2}+\sum_jh_j^{-1}\right)^{-1},\qquad
m_\mu=S\sum_j\bar y_j/h_j.
$$

관측 오차가 큰 그룹은 1/h_j가 작아 공통 중심을 덜 움직입니다. 반대로 한 그룹을 바꾸면 m_μ가 달라져 다른 그룹의 추정도 바뀝니다. 이것이 이 모형에서 정보를 공유하는 구체적인 경로입니다.

## 조건부 독립을 적분한 뒤에도 유지할 수는 없다

μ와 관측을 주었을 때 그룹별 사후평균은 w_j·관측 평균+(1−w_j)μ이고 분산은 r_j입니다. 여기에 전기대·전공분산 법칙을 적용합니다.

$$
w_j=\frac{\tau^2}{h_j},\quad r_j=\frac{\tau^2v_j}{h_j},\qquad
E[\theta_j\mid y]=w_j\bar y_j+(1-w_j)m_\mu,
$$

$$
\operatorname{Cov}(\theta_j,\theta_k\mid y)
=\mathbf1_{\{j=k\}}r_j+(1-w_j)(1-w_k)S.
$$

첫 항은 중심이 알려져도 남는 그룹별 불확실성입니다. 둘째 항은 중심이 함께 움직여 생기는 항입니다. j≠k의 조건부 공분산은 0이지만, 조건부 평균들의 공분산은 양수입니다. μ를 사후평균 하나로 대체하면 두 번째 항을 놓칩니다. 단, 모든 plug-in 방식이 같은 오차를 갖는다는 일반 주장은 아닙니다.

## 두 그룹 손계산과 새 그룹 예측

n_1=n_2=4, τ²=s²=1, 관측 평균 (2,−1)이면 S=5/13, m_μ=4/13입니다. 아래 비교의 ‘고정’은 원래 화면처럼 μ=0을 고정한 계산입니다. ‘적분’은 새 모형에서 μ의 사후를 적분한 계산입니다.

| 양 | 고정 | 적분 |
|---|---|---|
| 첫 평균 | $8/5$ | $108/65$ |
| 각 분산 | $1/5$ | $14/65$ |
| 그룹 공분산 | 0 | $1/65$ |
| 새 모수 분산 | 1 | $18/13$ |

아직 관측하지 않은 새 그룹의 모수 θ_new는 μ를 공유하므로 사후예측 평균 m_μ, 분산 τ²+S입니다. 기존 그룹의 작은 사후분산으로 대신하면 안 됩니다. 새 그룹에서 n_new개 관측의 평균을 예측한다면 관측 잡음 1/n_new도 더합니다. 여기서는 공통 중심만 추론하며, τ 자체의 불확실성은 여전히 포함하지 않습니다.

## 결합 정밀도 행렬로 독립 검산하기

코드는 (μ,θ_1,θ_2)의 결합 로그 사후를 제곱식에서 직접 구성한 정밀도 행렬로 풀어 위 분해와 비교합니다. 분수 Gauss–Jordan 소거를 쓰므로 Monte Carlo 오차가 없습니다. 화면의 고정 중심 조절기와는 별도 실행 예제입니다.

```python
from fractions import Fraction as Q

def inverse(matrix):
    n = len(matrix)
    a = [[Q(v) for v in row]+[Q(i == j) for j in range(n)]
         for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = a[j][j]
        assert pivot != 0
        a[j] = [v/pivot for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [v-factor*w for v, w in zip(a[i], a[j])]
    return [row[n:] for row in a]

# Prior exponent: mu^2 + (theta1-mu)^2 + (theta2-mu)^2.
# Likelihood exponent: 4*(2-theta1)^2 + 4*(-1-theta2)^2.
precision = [[3, -1, -1], [-1, 5, 0], [-1, 0, 5]]
cov = inverse(precision)
linear = [0, 8, -4]
means = [sum(cov[i][j]*linear[j] for j in range(3)) for i in range(3)]
S, center, w, r = Q(5, 13), Q(4, 13), Q(4, 5), Q(1, 5)
assert cov[0][0] == S and means[0] == center
assert means[1:] == [Q(108, 65), Q(-48, 65)]
for j in [1, 2]:
    for k in [1, 2]:
        assert cov[j][k] == (r if j == k else 0)+(1-w)**2*S
assert means[1:] == [w*y+(1-w)*center for y in [2, -1]]
assert 1+S == Q(18, 13)
print("Posterior means (center, group 1, group 2):", means)
print("Group covariance matrix:", [row[1:] for row in cov[1:]])
print("New-group parameter variance:", 1+S)
```

## 연구에 적용하기 전에 묻기

1. 조건부 사후가 독립이면 μ를 적분한 사후도 독립인가요? **아니요. 공통 중심의 불확실성이 의존성을 만듭니다.**
2. μ=m_μ를 대입하면 무엇을 놓치나요? **여기서는 (1−w_j)(1−w_k)S 항입니다.**
3. 새 지역의 예측 오차를 보고할 때 무엇을 구분하나요? **기존 그룹·새 그룹, 모수·새 관측, 고정·추론한 초모수를 구분합니다.**

계층 구조의 배경은 [Stan User’s Guide, Hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression)을 참고하세요. 위 Gaussian 공분산과 분수 예제는 명시한 모형에서 직접 유도했습니다.

## 확인 문제와 해설

1. n=4,τ=1일 때 데이터 가중치는? **0.8입니다.**
2. τ를 작게 하면 관측 정보를 더 믿나요? **아니요. 공통 중심으로 더 강하게 당깁니다.**
3. 이 그림만으로 초모수 불확실성까지 반영했나요? **아니요. 초모수를 고정한 조건부 모형입니다.**

추가 읽기: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
