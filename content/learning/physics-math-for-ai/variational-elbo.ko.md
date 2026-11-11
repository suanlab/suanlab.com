## 어려운 사후를 최적화 문제로 바꾸기

잠재변수 모형의 사후가 복잡하면 계산하기 쉬운 분포족 q를 골라 사후를 근사합니다. 여기서는 원리를 숨기지 않도록 잠재 z가 0 또는 1인 작은 모형을 사용합니다. 관측 x에 대한 결합 질량은 p(x,z=1)=0.4π, p(x,z=0)=0.4(1−π)이고 따라서 evidence는 0.4, 정확 사후확률은 π입니다. π 조절은 모형의 결합 질량도 함께 바꿉니다.

## ELBO를 결합과 엔트로피로 읽기

변분 후보는 q(z=1)=r입니다. 결합 로그확률의 q 평균에 엔트로피를 더하면 evidence lower bound를 얻습니다. 직접 식을 전개하면 evidence와 ELBO 사이 간격이 KL임을 확인할 수 있습니다.

$$
\mathcal L(q)=\sum_zq(z)\log p(x,z)-\sum_zq(z)\log q(z),
\qquad\log p(x)=\mathcal L(q)+\operatorname{KL}(q\Vert p(z\mid x)).
$$

KL은 음수가 아니므로 ELBO가 하한입니다. 고정 데이터·모형에서는 log evidence가 r에 따라 바뀌지 않으므로 ELBO 최대화는 이 방향의 KL 최소화와 같습니다. 모형 모수도 함께 학습하는 상황은 이 고정 모형 실습과 구분하세요.

## 기본 예제를 숫자로 확인하기

π=0.7,r=0.5이면 KL≈0.08718, log evidence≈−0.91629, ELBO≈−1.00347입니다. 두 음수를 보고 크기를 혼동하지 마세요. −1.00347이 더 작은 하한입니다. 파랑은 r에 따른 ELBO이고 주황 수평선은 log evidence입니다. 선택한 r은 점으로 표시됩니다.

$$
\operatorname{KL}=r\log\frac r\pi+(1-r)\log\frac{1-r}{1-\pi}.
$$

## 근사족의 제한과 KL 방향

이 실습의 모든 Bernoulli q 중에는 정확한 사후도 포함되어 r=π에서 간격 0이 됩니다. 실제 mean-field 변분족처럼 변수 사이 의존성을 제거하면 정확 사후를 포함하지 않을 수 있습니다. 최적화가 잘 되어도 근사족의 오차가 남습니다.

r=1에서는 q가 상태 0을 무시하지만 KL(q||posterior)=log(1/π)는 유한합니다. 0 log 0은 연속 확장으로 0으로 계산합니다. 반대 방향 KL(posterior||q)은 상태 0에 양의 사후 질량이 있는데 q가 0이므로 무한합니다. 방향을 바꾸면 같은 값이 아닙니다. 이 작은 예제만으로 고차원에서 근사의 품질을 보장할 수는 없습니다.

## 오차 0에 도달할 수 없는 근사족도 있다

위 Bernoulli 변분족은 정확 사후를 포함합니다. 의존성을 제거하는 효과를 분리해 보려면 이미 정규화된 이변량 Gaussian 목표 p를 생각하세요. 실제 사후를 이 형태로 알고 있다고 가정한 별도 예제입니다. |ρ|<1이면 공분산 Σ가 양의 정부호입니다. 후보 q는 평균 m과 양의 대각 분산 d_1,d_2만 조절합니다.

$$
p=N(0,\Sigma),\quad \Sigma=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix},\qquad
q=N(m,D),\quad D=\operatorname{diag}(d_1,d_2).
$$

두 로그 밀도를 빼서 q 아래 평균하면 Gaussian KL을 직접 얻습니다. Λ=Σ^{-1}를 정밀도라 쓰겠습니다.

$$
\operatorname{KL}(q\Vert p)=\frac12\left[
\operatorname{tr}(\Lambda D)+m^\top\Lambda m-2+
\log\frac{\det\Sigma}{d_1d_2}\right].
$$

Λ가 양의 정부호이므로 평균의 이차식은 m=0에서 유일하게 최소입니다. 분산 d_i에 의존하는 항은 (Λ_ii d_i−log d_i)/2입니다. 미분하면 (Λ_ii−1/d_i)/2, 이차 미분은 1/(2d_i²)>0입니다. 따라서 이 대각 Gaussian 족의 전역 최적 분산은 정밀도 대각의 역수입니다.

$$
d_i^*=\Lambda_{ii}^{-1}=1-\rho^2,\qquad
\min_q\operatorname{KL}(q\Vert p)=-\frac12\log(1-\rho^2).
$$

## 기울어진 타원과 작은 원을 구별하기

정확 목표의 각 주변분산은 1인데 최적 q의 각 분산은 1−ρ²입니다. ρ≠0이면 정확 주변분산을 맞추지 않습니다. 이 값은 오히려 이 Gaussian 목표의 조건부 분산 Var(Z_1|Z_2)와 같습니다. 최적화가 실패해서가 아니라 정한 족과 KL 방향의 결과입니다.

ρ>0일 때 목표 등밀도선은 대각선 방향으로 긴 타원입니다. 독립인 q의 원을 주변 표준편차 1까지 키우면 목표가 희박한 반대 대각선 쪽에도 질량을 놓습니다. KL(q||p)를 줄이는 원은 작아집니다. 이는 이 지정한 Gaussian 족의 설명이며 모든 변분법이 언제나 분산을 줄인다는 정리는 아닙니다.

![상관 0.9에서 정확한 파란 타원, 작은 주황 KL(q||p) 원, 점선 KL(p||q) 원의 공분산이 다릅니다.](/assets/learning/bayes-mean-field-ko.svg)

그림은 ρ=0.9를 고정합니다. 각 곡선은 자기 분포에서 Mahalanobis 반지름 2인 등밀도선이며, 95% 포함률을 주장하는 그림이 아닙니다.

| ρ | Var(q) | KL |
|---|---|---|
| 0 | 1 | 0 |
| $1/2$ | $3/4$ | 0.14384 |
| $9/10$ | $19/100$ | 0.83037 |

ρ의 부호를 바꾸면 타원의 기울기는 반대가 되지만 이 분산과 최소 KL은 같습니다. |ρ|=1에서는 목표가 특이해지므로 위 밀도·역행렬 식을 그대로 쓸 수 없습니다.

## 방향을 바꾸면 무엇이 달라지는가

KL(p||q)를 같은 족에서 최소화하면 q의 평균은 0, 분산은 각각 1입니다. 직접 전개하면 d_i 관련 항이 (1/d_i+log d_i)/2이고 d_i=1에서 최소입니다. 이 방향은 정확한 주변 모멘트를 맞추지만 공분산은 여전히 0입니다. 이 2차원 예제에서는 두 방향의 최소 KL 숫자가 우연히 같아도 최적 분포는 다릅니다. 방향의 차이를 숫자 하나만으로 판단하지 마세요.

코드는 양 방향의 Gaussian 로그밀도 평균식을 계산하고 평균·분산을 바꾼 후보와 비교합니다. 유한 후보 검사는 대수적 전역 최적 증명을 대신하지 않습니다. 화면의 Bernoulli 조절기는 이 Gaussian 예제를 조절하지 않습니다.

```python
from fractions import Fraction as Q
from math import log, isclose
from itertools import product

def forward(rho, mean, d):
    det = 1-rho*rho
    quad = (mean[0]**2-2*rho*mean[0]*mean[1]+mean[1]**2)/det
    return .5*((sum(d)/det)+quad-2+log(float(det/(d[0]*d[1]))))

def reverse(rho, mean, d):
    det = 1-rho*rho
    return .5*(sum((1+m*m)/v for m, v in zip(mean, d))-2
               +log(float(d[0]*d[1]/det)))

for rho in [Q(0), Q(1, 2), Q(9, 10), Q(-9, 10)]:
    variance = 1-rho*rho
    gap = -.5*log(float(variance))
    assert isclose(forward(rho, [0, 0], [variance]*2), gap, abs_tol=1e-12)
    assert isclose(reverse(rho, [0, 0], [Q(1)]*2), gap, abs_tol=1e-12)
    for m1, m2 in product([Q(-1, 2), Q(0), Q(1, 2)], repeat=2):
        for a, b in product([Q(1, 2), Q(1), Q(2)], repeat=2):
            assert forward(rho, [m1, m2], [a*variance, b*variance]) >= gap-1e-12
            assert reverse(rho, [m1, m2], [a, b]) >= gap-1e-12
    print("rho, forward-optimal variance, reverse-optimal variance, gap:",
          rho, variance, 1, gap)
```

## 연구에 적용하기 전에 묻기

1. ELBO 최적화가 끝나면 사후 오차가 0인가요? **족이 정확 사후를 포함하지 않으면 아닙니다.**
2. 이 q의 분산을 정확 사후의 주변분산으로 보고해도 되나요? **아니요. 여기서는 1−ρ²와 1로 다릅니다.**
3. 정확도 비교에서 무엇을 고정하나요? **목표·좌표·족·KL 방향·최적화 오차를 구분합니다.**

배경은 [Blei 등, §2.3과 Figure 1](https://arxiv.org/html/1601.00670v9#S2.SS3)을 참고하세요. 위 단위 주변분산 사례의 최적값과 역방향 비교는 로그밀도로 직접 계산했습니다.

## 확인 문제와 해설

1. r=π이면 ELBO는? **log evidence와 같습니다.**
2. q를 바꾸면 고정 모형의 evidence도 커지나요? **아니요. 하한과 간격이 변합니다.**
3. KL은 대칭 거리인가요? **아니요. 방향과 지지 조건이 중요합니다.**

추가 읽기: [Blei, Kucukelbir & McAuliffe, Variational Inference](https://arxiv.org/abs/1601.00670).
