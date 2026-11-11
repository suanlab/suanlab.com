## 표본평균과 알려진 잡음

Xi가 독립적으로 N(μ,σ²)를 따르고 σ가 알려져 있다고 가정합니다. 표본평균의 표준오차는 σ/√n입니다. 이 경우 정규 표준화는 유한 n에서도 정확합니다. 실습은 실제 자료에서 σ를 추정하는 t 절차를 실행하지 않습니다. 관측 평균과 n을 바꿔 구간과 H0: μ=0의 양측 검정을 비교합니다.

$$
Z=\frac{\bar X-\mu}{\sigma/\sqrt n}\sim N(0,1).
$$

## 신뢰수준은 절차의 포함률

95% 구간은 반복 표본에서 구성된 무작위 구간이 고정 μ를 포함하는 장기 비율이 95%라는 뜻입니다. 이미 계산한 한 구간이 μ를 포함할 사후확률이 95%라는 의미는 아닙니다. 기준 n=16,관측평균 0.5,σ=1에서는 표준오차 0.25이고 구간은 약 (0.010,0.990)입니다. 알려진 σ라는 가정이 실제로 맞는지 확인해야 합니다.

$$
C(X)=\left[\bar X-z_{1-\alpha/2}\frac\sigma{\sqrt n},\;\bar X+z_{1-\alpha/2}\frac\sigma{\sqrt n}\right],\quad P_\mu(\mu\in C(X))=1-\alpha.
$$

## 양측 검정과 구간의 관계

귀무 평균이 0이면 관측 Z는 2입니다. 표준정규에서 이보다 절댓값이 큰 값의 확률이 약 0.0455입니다. 5% 양측 검정에서는 임계값 약 1.96을 넘어 기각하고, 같은 95% 구간은 0을 제외합니다. p값은 귀무가설 아래의 자료 극단성이지 H0가 참일 확률이 아닙니다. 기각하지 못했다는 것이 H0를 증명했다는 뜻도 아닙니다.

$$
p_{\mathrm{two}}=2\Phi(-|Z_{\mathrm{obs}}|),\qquad\text{reject if }|Z_{\mathrm{obs}}|>z_{1-\alpha/2}.
$$

## 그림의 밀도와 적용 범위

그림의 주황 꼬리는 기각 영역이며 면적이 명목 유의수준입니다. 파랑 점은 관측 Z가 표시 구간 [−4,4] 안에 있을 때만 그립니다. 수치 표의 Z는 잘리지 않습니다. 90% 구간은 임계값이 작아 더 좁습니다. 실제 미지 분산, 비정규성, 선택된 표본, 반복 검사 또는 자료를 본 뒤 가설을 고르는 경우에는 해당 절차의 보장을 새로 검토해야 합니다.

## 다른 표집 모형에서 포함률을 직접 점검하기

앞의 정확한 Gaussian 표준화는 알려진 σ와 Gaussian 표집을 전제로 합니다. Bernoulli 비율에서는 표준오차를 추정해 같은 모양에 넣어도 정확한 구간이 자동으로 생기지 않습니다. 고정 p에서 K∼Binomial(n,p), p̂=K/n이라고 합시다. Wald 구간은 p̂±z√(p̂(1−p̂)/n)이며 [0,1]로 자를 수도 있습니다. K=0이면 구간이 점 0으로 붕괴하여 모든 양의 p를 놓칩니다. 끝점을 모수 범위로 자르는 것으로 해결되지 않습니다.

구체적으로 n=10,p=1/20,z=49/25=1.96을 택합니다. 두 정규 기반 규칙에는 통상적인 약 95% 임계값을 사용합니다. 한 구간의 모양 대신 이항 표집법칙 전체에서 포함률을 계산합니다.

$$
\operatorname{Coverage}_p(C)=\sum_{k=0}^n\mathbf1_{\{p\in C(k)\}}{n\choose k}p^k(1-p)^{n-k}.
$$

## 관측 분산을 대입하는 대신 score 조건을 뒤집기

내부 후보 p의 score 검정은 귀무 분산 p(1−p)를 사용하고 Wald 규칙은 관측 p̂(1−p̂)를 사용합니다. score 수용 조건을 p에 대해 풀면 Wilson 구간을 얻습니다. 정의하는 이차 부등식은 다음과 같습니다.

$$
n(\hat p-p)^2\le z^2p(1-p),\qquad
(n+z^2)p^2-(2n\hat p+z^2)p+n\hat p^2\le0.
$$

이차식은 p=0 또는 1도 분산 0으로 나누지 않고 평가합니다. K=0에서는 끝점이 0과 z²/(n+z²)이므로 Wald처럼 한 점으로 붕괴하지 않습니다. 그러나 여전히 정규 기반 근사이며 모든 유한 n,p에서 95% 이상을 포함한다는 보장은 아닙니다.

## 정확한 이산 꼬리를 뒤집고 포함률 상계를 증명하기

α=1/20으로 둡니다. Clopper–Pearson이라고 부르는 동등 꼬리 정확 이항 구간은 두 포함 꼬리확률이 모두 α/2 이상인 p를 남깁니다.

$$
C_{\mathrm{exact}}(k)=\{p:\ P_p(K'\le k)\ge\alpha/2,\ P_p(K'\ge k)\ge\alpha/2\}.
$$

p를 고정하면 하측 꼬리확률이 α/2보다 작은 사건은 가능한 개수의 앞부분이며 그 전체 확률은 α/2 이하입니다. 상측 꼬리 기각 사건에도 같은 상계가 적용됩니다. 합집합 상계로 참 p를 제외할 확률이 α 이하이므로 포함률은 **적어도** 1−α입니다. 이산성 때문에 더 클 수 있습니다. k=0의 하한은 0, k=n의 상한은 1을 사용합니다.

고정한 n=10,p=1/20에서 포함하는 개수와 포함률은 다음과 같습니다.

| 규칙 | 포함하는 k | 포함률 |
|---|---|---|
| Wald | 1,2,3 | 0.400235 |
| Score | 0,1 | 0.913862 |
| 정확 꼬리 | 0,1,2 | 0.988496 |

첫 규칙은 발생 확률이 큰 영 횟수를 놓칩니다. score 규칙은 여기서 그 문제를 개선하지만 여전히 명목 포함률보다 낮습니다. 정확 꼬리는 이 예제에서 보수적입니다. 지정한 규칙과 모수에서의 결과이며 한 표만으로 구간 폭이나 성능의 보편적인 순위를 정할 수는 없습니다.

## 정확한 분수로 포함률을 완전 열거하기

다음 코드는 정의 부등식으로 구간 포함 여부를 검사하므로 수치 근 탐색이 필요 없습니다. 유한 포함률 합은 정확한 분수이고 표는 반올림한 소수입니다. 모수 101점 검사는 정확 꼬리 보장의 예시이며 일반 증명은 앞의 꼬리 논증입니다. 이 Bernoulli 확장은 기존 알려진 σ의 Gaussian 조절기와 별도입니다.

```python
from fractions import Fraction as Q
from math import comb

n, p, z, alpha = 10, Q(1, 20), Q(49, 25), Q(1, 20)

def binomial_law(p):
    mass = [Q(comb(n, k))*p**k*(1-p)**(n-k) for k in range(n+1)]
    assert sum(mass) == 1
    return mass

def exact_tail_accepts(k, mass):
    return sum(mass[:k+1]) >= alpha/2 and sum(mass[k:]) >= alpha/2

mass = binomial_law(p)
accepted = {name: [] for name in ['Wald', 'Score', 'Exact tails']}
for k in range(n+1):
    phat = Q(k, n)
    gap = n*(phat-p)**2
    if gap <= z*z*phat*(1-phat):
        accepted['Wald'].append(k)
    if gap <= z*z*p*(1-p):
        accepted['Score'].append(k)
    if exact_tail_accepts(k, mass):
        accepted['Exact tails'].append(k)
assert accepted == {'Wald': [1, 2, 3], 'Score': [0, 1], 'Exact tails': [0, 1, 2]}
for name, ks in accepted.items():
    coverage = sum(mass[k] for k in ks)
    print(name, "accepted counts:", ks, "coverage:", coverage, float(coverage))
for j in range(101):
    law = binomial_law(Q(j, 100))
    coverage = sum(law[k] for k in range(n+1) if exact_tail_accepts(k, law))
    assert coverage >= 1-alpha
print("Exact-tail coverage checked on 101 parameter values; general proof is in the text.")
```

## 추가 문제와 연구 연결

1. 계산한 구간 하나로 절차의 포함률을 알 수 있나요? **고정 모수에서 데이터를 반복하는 확률이므로 아닙니다.**
2. “정확”이면 포함률이 반드시 95%와 같나요? **이산 보장은 보수적일 수 있으므로 아닙니다.**
3. 벤치마크의 오차 구간에는 무엇을 명시해야 하나요? **표집 법칙, 독립 단위, 구간 규칙과 모형 선택·반복 검정 여부입니다.**

Wilson 역검정과 정확 이항 꼬리는 [NIST, §7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)을 참고하세요. 위 유한 비교와 포함률 상계는 지정한 이항법칙에서 명시적으로 계산·증명했습니다.

## 확인 문제와 해설

1. 관측평균이 0이면 p값은? **1입니다.**
2. n을 네 배로 하면 구간 폭은? **σ가 같으면 절반입니다.**
3. 95% 구간을 사후구간이라고 불러도 되나요? **아니요. 해석과 구성 원리가 다릅니다.**

추가 읽기: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[NIST, Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) — 반복 포함률 해석과 미지 σ의 t 구간 비교 자료입니다.
