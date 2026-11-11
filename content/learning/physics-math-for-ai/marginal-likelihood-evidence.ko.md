## 모수 하나의 최적값과 모형 전체의 점수

MLE는 우도를 최대화하는 모수를 찾습니다. Evidence 또는 주변우도는 사전에 따라 모든 모수의 우도를 평균합니다. 최적 모수가 데이터를 잘 설명해도 그 주변 사전 질량이 작으면 평균은 낮을 수 있습니다. 두 연산은 다른 질문에 답합니다.

$$
Z=p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
$$

## Beta 모형에서 정확한 값 구하기

관측 수 n=8과 성공 수 k를 사용합니다. 사전 Beta(α,β) 아래 한 순서 표본의 evidence는 Beta 함수 비율입니다. k=6,α=β=1이면 Z=1/252≈0.003968, 자연로그 값은 약 −5.529입니다.

$$
Z=\frac{B(\alpha+k,\beta+n-k)}{B(\alpha,\beta)}.
$$

같은 성공 수를 갖는 모든 순서를 합한 개수 사건의 확률은 조합계수 C(8,k)를 곱해야 합니다. 기본 예제에서 그 값은 1/9입니다. 그림과 수치에는 일관되게 하나의 순서 표본을 사용합니다.

## 순차 예측의 곱으로 다시 계산하기

관측을 하나씩 읽으며 그 직전 사후의 다음 관측 예측확률을 곱해도 같은 evidence를 얻습니다. 현재까지 성공 s, 실패 f를 봤다면 다음 성공확률은 (α+s)/(α+β+s+f)입니다. 다음 실패에는 반대쪽 분자를 사용합니다.

$$
p(x)=\prod_{i=1}^{n}p(x_i\mid x_1,\ldots,x_{i-1}),\qquad \log p(x)=\sum_i\log p(x_i\mid x_{<i}).
$$

표는 성공들을 먼저 놓은 고정 순서를 사용합니다. 교환가능 모형이므로 중간 경로는 순서에 따라 달라도 마지막 곱은 같습니다. 파랑은 이 누적 로그 예측이고 주황은 전체 데이터의 MLE를 처음부터 고정해 같은 관측에 적용한 누적 로그우도입니다. 주황 기준은 미래를 미리 본 비교값이며 온라인 예측 절차가 아닙니다.

## 사전 민감도와 한계

사전 모수를 바꿔 evidence가 어떻게 달라지는지 보세요. Evidence는 모수에 대한 사전 평균이므로 사전 선택에 민감할 수 있습니다. 적절히 정규화되지 않은 부적절 사전은 임의의 상수를 남겨 일반적인 evidence 비교를 정의하지 못할 수 있습니다. 이 실습은 정규화된 적절한 Beta 사전만 사용합니다.

작은 값의 곱은 수치적으로 사라질 수 있어 로그합을 쓰는 이유도 이해해야 합니다. 여기서는 작은 정수 예제를 정확한 함수 비율과 순차 분해로 검산합니다.

## 같은 z값이어도 evidence 비율은 달라진다

이제 이산 질량 대신 연속 데이터의 밀도를 비교하는 별도 예제를 봅시다. 알려진 관측 분산 1에서 X_i|θ가 iid N(θ,1)이고 n≥1입니다. M0는 θ=0, M1은 적절한 사전 θ~N(0,τ²), τ²>0을 둡니다. 두 모형은 같은 관측과 같은 기준 측도를 사용합니다. 밀도값은 관측 한 점의 확률이 아닙니다.

표본평균은 M0 아래 N(0,1/n), M1 아래 N(0,τ²+1/n)입니다. Gaussian 표본의 평균을 뺀 잔차 부분은 두 모형에서 같은 밀도 인자로 분리되어 비율에서 상쇄됩니다. 따라서 전체 표본의 Bayes factor BF_10은 표본평균 밀도 비율로 구할 수 있습니다. z=√n·표본평균, r=nτ²로 쓰면

$$
\operatorname{BF}_{10}
=\frac1{\sqrt{1+r}}\exp\left(\frac{z^2r}{2(1+r)}\right),\qquad
\log\operatorname{BF}_{10}=-\frac12\log(1+r)+\frac{z^2r}{2(1+r)}.
$$

첫 인자는 넓어진 M1 예측 밀도의 정규화에서 나옵니다. 둘째 인자는 관측 평균에 더 잘 맞는 효과입니다. 최적 적합만 보고 첫 인자를 버리면 evidence를 계산한 것이 아닙니다.

## 고정 통계량의 극한과 고정 효과의 극한을 구분하기

τ²=1, z=2를 유지하며 n을 늘립시다. 그러려면 관측 평균은 2/√n으로 작아집니다. M0의 양측 z 검정 p값은 항상 약 0.04550이지만 Bayes factor는 다음처럼 달라집니다.

| n | 관측 평균 | BF_10 |
|---|---|---|
| 1 | 2 | 1.92212 |
| 100 | 0.2 | 0.72082 |
| 10000 | 0.02 | 0.07387 |

고정된 유한 z와 τ²>0에서 지수항은 exp(z²/2)로 수렴하지만 앞의 인자는 0으로 갑니다. 따라서 BF_10→0입니다. 이는 p값을 사후 모형 확률로 바꿀 수 없다는 계산 예입니다. 사후확률에는 모형 사전 odds도 필요합니다. 두 절차의 확률 정의와 대립 가정이 달라 숫자가 다르게 움직입니다.

반대로 관측 평균을 비영 상수 c로 유지하는 경로에서는 z²=nc²이고 log BF_10/n→c²/2>0입니다. 이 경로에서는 M1 쪽 evidence가 커집니다. 고정 z의 결론을 ‘표본을 늘리면 어떤 신호에서도 M0가 이긴다’로 일반화하면 안 됩니다.

## 사전 폭도 데이터 분석의 명시적 입력이다

n과 z를 고정하고 τ²→∞로 보내도 BF_10→0입니다. 사전이 관측과 맞는 국소 영역에 주는 질량이 줄기 때문입니다. 이것은 적절한 사전들의 극한입니다. 정규화 상수를 제거한 부적절한 평탄 사전을 M1의 evidence에 그대로 넣어도 된다는 뜻이 아닙니다. 적절한 사후를 얻는 문제와 모형 사이의 evidence를 정의하는 문제는 구분해야 합니다.

실제 연구에서는 자료를 보기 전에 과학적으로 가능한 효과 척도와 사전 민감도 범위를 명시하세요. 이 Gaussian 예제는 선택한 두 모형의 상대 비교를 설명하며 후보들이 현실에 적합하다는 보장은 하지 않습니다. 기존 Beta 조절기는 이 Gaussian 분석과 독립입니다.

## 밀도 적분으로 공식 검산하기

아래 코드는 표본평균의 우도 밀도와 사전 밀도를 직접 곱해 수치 적분한 값을 닫힌 식과 비교합니다. 적분 변수만 사후 중심·척도로 바꾸어, n이 커질 때 좁아진 봉우리를 놓치지 않게 합니다. 유한 구간 Simpson 적분의 근사 검산이며 위 대수적 극한 증명을 대신하지 않습니다.

```python
from math import exp, sqrt, pi, log1p, erfc, isclose

def normal(x, mean, sd):
    return exp(-.5*((x-mean)/sd)**2)/(sqrt(2*pi)*sd)

def integrated_bf(n, prior_sd, z):
    y = z/sqrt(n)
    variance = 1/(n+1/prior_sd**2)
    center = variance*n*y
    sd = sqrt(variance)
    def integrand(u):
        theta = center+sd*u
        return normal(y, theta, 1/sqrt(n))*normal(theta, 0, prior_sd)*sd
    steps, lo, hi = 4000, -10, 10
    h = (hi-lo)/steps
    total = integrand(lo)+integrand(hi)
    total += sum((4 if i % 2 else 2)*integrand(lo+i*h)
                 for i in range(1, steps))
    return total*h/3/normal(y, 0, 1/sqrt(n))

z = 2
print("Fixed-z two-sided p-value:", erfc(abs(z)/sqrt(2)))
for n in [1, 100, 10000]:
    for prior_sd in [.5, 1, 2]:
        r = n*prior_sd**2
        exact = exp(-.5*log1p(r)+z*z*r/(2*(1+r)))
        quadrature = integrated_bf(n, prior_sd, z)
        assert isclose(quadrature, exact, rel_tol=1e-10, abs_tol=1e-12)
        print("n, prior sd, observed mean, BF:", n, prior_sd, z/sqrt(n), exact)
print("The integration is approximate; the limiting arguments are in the text.")
```

## 연구에 적용하기 전에 묻기

1. 같은 p값이면 같은 Bayes factor인가요? **아니요. n과 대립 사전 척도도 관여합니다.**
2. 고정 z 경로에서 관측 평균도 고정인가요? **아니요. z/√n으로 변합니다.**
3. 무엇을 재현 기록에 남기나요? **관측 모형·분산·사전 척도·모형 odds·비교한 자료 표현을 남깁니다.**

모형 의존성과 사전 민감도의 배경은 [PyMC, Bayes Factors and Marginal Likelihood](https://www.pymc.io/projects/examples/en/2022.01.0/diagnostics_and_criticism/Bayes_factor.html)를 참고하세요. 위 Gaussian 식·두 극한·표는 명시한 모형에서 직접 계산했습니다.

## 확인 문제와 해설

1. evidence와 최대 우도 중 어느 것이 더 클 수 있나요? **적절한 사전의 평균은 최대 우도를 넘지 않습니다.**
2. 관측 순서를 바꾸면 마지막 evidence가 달라지나요? **이 교환가능 모형에서는 같습니다.**
3. evidence가 모수의 확률인가요? **아니요. 모수를 적분한 데이터 사건의 확률입니다.**

추가 읽기: [Kevin P. Murphy, Probabilistic Machine Learning — author’s book page](https://probml.github.io/pml-book/book1.html).
