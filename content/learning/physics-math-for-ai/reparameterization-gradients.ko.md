## 분포 매개변수와 잡음을 분리하기

Gaussian 변수 z를 직접 불투명한 표본으로 다루는 대신 μ+σε로 씁니다. ε의 분포는 매개변수와 무관한 표준정규입니다. 미분할 때는 같은 ε를 고정하여 μ와 σ가 출력에 미치는 영향을 추적합니다. 실습 조절기는 ε를 새로 추출하지 않고 지정한 고정값으로 둡니다. 따라서 그림과 표는 무작위 표본 실험이 아닙니다.

$$
z=\mu+\sigma\epsilon,\quad\epsilon\sim N(0,1),\qquad\ell(z)=\tfrac12(z-1)^2.
$$

## 한 경로의 경사를 손으로 구하기

연쇄법칙으로 μ에 대한 도함수는 z−1이고 σ에 대해서는 (z−1)ε입니다. 기준 μ=0,σ=1,ε=0에서는 z=0, 손실 0.5, 경사는 (−1,0)입니다. ε=1로 바꾸면 z가 목표 1과 같아 표본 손실과 표본 경사가 모두 0이 됩니다. 그러나 이것만으로 분포 전체의 목적함수가 최소라고 결론낼 수 없습니다.

$$
\partial_\mu\ell=z-1,\qquad\partial_\sigma\ell=(z-1)\epsilon.
$$

## 기대 경사와 비교하기

Eε=0, Eε²=1을 이용하면 이차 손실의 기대값을 정확하게 계산할 수 있습니다. 기준 기대 손실은 1이고 기대 경사는 (−1,1)입니다. 중앙 잡음 ε=0에서의 σ 경사 0과 다릅니다. 표의 ε=−1과 1을 같은 가중치로 평균하면 첫 두 모멘트가 맞아 이 이차 예제의 기대 손실과 경사를 정확히 재현합니다. 이 두 점은 Gaussian 표본도, 일반 함수에 정확한 적분법도 아닙니다.

$$
E\ell=\tfrac12[(\mu-1)^2+\sigma^2],\qquad\nabla E\ell=(\mu-1,\sigma).
$$

## 기대와 미분을 교환할 조건

재매개화만으로 기대와 미분의 교환이 자동으로 정당화되지는 않습니다. 적절한 미분 가능성과 적분 가능한 지배 조건 등을 확인해야 합니다. 이 예제는 다항식 손실과 Gaussian의 유한 모멘트로 교환을 정당화할 수 있습니다. σ=0이면 법칙이 한 점으로 붕괴하지만 경로 식과 이차 기대 손실은 여전히 계산됩니다. 척도를 음수로 확장한 매끄러운 식의 도함수로 읽거나 허용 영역의 한쪽 미분으로 해석할 수 있습니다.

$$
\nabla_\theta E_\epsilon[\ell(g_\theta(\epsilon))]=E_\epsilon[\nabla_\theta\ell(g_\theta(\epsilon))]\quad\text{under suitable conditions}.
$$

## gradient 추정량의 분산을 유도하기

불편 gradient도 크게 변동할 수 있습니다. m=μ−1로 두고 표준정규 잡음 법칙을 유지합니다. 두 경로 gradient는 gμ=m+σε, gσ=mε+σε²입니다. 제곱을 전개하고 Eε=Eε³=0, Eε²=1, Eε⁴=3을 사용하면 다음과 같습니다.

$$
E[g_\mu]=m,\quad E[g_\sigma]=\sigma,\qquad
\operatorname{Var}(g_\mu)=\sigma^2,\quad
\operatorname{Var}(g_\sigma)=m^2+2\sigma^2.
$$

기대 gradient가 맞는다고 gradient 불확실성까지 맞는 것은 아닙니다. 결정적 노드 ε=−1,+1의 사차 모멘트는 3이 아닌 1입니다. μ=0,σ=1에서 이 두 점의 scale gradient는 2와 0이며 평균 1, 분산 1입니다. 실제 Gaussian scale gradient의 분산은 3입니다. 기존 두 노드 검사는 이차 기대 손실과 기대 gradient에는 정확하지만 Gaussian 분산 추정으로 쓸 수 없습니다.

## 같은 평가 횟수로 antithetic 쌍 비교하기

하나의 무작위 Gaussian ε를 사용하여 ε와 −ε에서 모두 평가합니다. 평균하면 홀수항이 상쇄됩니다.

$$
\bar g^{\mathrm{anti}}_\mu=m,\qquad
\bar g^{\mathrm{anti}}_\sigma=\sigma\epsilon^2,\qquad
\operatorname{Var}(\bar g^{\mathrm{anti}}_\sigma)=2\sigma^2.
$$

손실 gradient 평가는 두 번입니다. 같은 비용의 비교 대상은 독립 잡음 경로 두 개의 평균이며 그 분산은 σ²/2, (m²+2σ²)/2입니다. antithetic 쌍은 위치 gradient의 분산을 제거하지만 scale gradient의 분산은 m²>2σ²일 때만 개선합니다. 잡음의 대칭성만으로 모든 성분의 개선이 보장되지는 않습니다.

각 결과는 (위치 gradient 분산, scale gradient 분산) 순서입니다.

| $(\mu,\sigma)$ | 독립 경로 2개 | ε, −ε 쌍 |
|---|---|---|
| $(0,1)$ | $(1/2,3/2)$ | $(0,2)$ |
| $(-2,1)$ | $(1/2,11/2)$ | $(0,2)$ |
| $(0,0)$ | $(0,1/2)$ | $(0,0)$ |

σ=0에서 기대 scale 미분은 0이지만 단일 경로의 미분 mε는 여전히 변할 수 있습니다. 앞에서처럼 매끄러운 signed-scale 확장 또는 한쪽 미분으로 경계를 해석합니다. 표시된 잡음 조절기는 결정적 값이므로 이 분산들은 표의 세 행에서 추정한 경험적 값이 아닙니다.

## 문턱 함수에서는 거의 모든 경로의 미분만으로 부족하다

U가 [0,1]의 균등분포이고 0<θ<1에서 Bθ=1{U<θ}라고 합시다. E[Bθ]=θ이므로 미분은 1입니다. 그러나 거의 모든 고정 U에 대해 Bθ는 θ 근방에서 상수라 경로 미분이 0입니다. 기대와 미분을 교환하면 틀린 결과를 얻습니다.

$$
\frac{d}{d\theta}E[B_\theta]=1,\qquad
E\!\left[\frac{d}{d\theta}B_\theta\right]=0.
$$

양수 h<1−θ에서 전진 차분은 폭 h인 U 구간에서 높이 1/h이고 나머지는 0입니다. 기대값은 계속 1이지만 거의 모든 점에서의 극한은 0입니다. 좁아지는 구간과 커지는 높이 때문에 필요한 지배 조건을 충족하지 못합니다. 이산 표본을 매개변수와 독립인 잡음의 함수로 썼다는 사실만으로 단순 경로 미분이 정당화되지는 않습니다. 이산 score 추정량이나 다른 방법은 별도로 유도해야 합니다.

## 구적법을 표집으로 오해하지 않고 모멘트 계산하기

다음 코드는 Gaussian 다항식 모멘트를 정확한 분수로 계산하여 같은 평가 횟수를 비교하고, 문턱 함수의 직사각형 차분을 정확하게 적분합니다. 무작위 경로를 뽑거나 잠재변수 모형을 학습하지 않습니다. 이차 예제의 분산 분석을 확장하며 현재 조절기와 세 잡음 노드는 그대로 사용합니다.

```python
from fractions import Fraction as Q
from math import prod

def gaussian_moment(k):
    return Q(0) if k % 2 else Q(prod(range(1, k, 2)))

def expected(poly):
    return sum(c*gaussian_moment(k) for k, c in enumerate(poly))

def multiply(a, b):
    out = [Q(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out

def variance(poly):
    return expected(multiply(poly, poly)) - expected(poly)**2

for mu, sigma in [(Q(0), Q(1)), (Q(-2), Q(1)), (Q(0), Q(0))]:
    m = mu-1
    g_mu, g_sigma = [m, sigma], [Q(0), m, sigma]
    raw = (variance(g_mu), variance(g_sigma))
    iid_two = tuple(v/2 for v in raw)
    antithetic = (variance([m]), variance([Q(0), Q(0), sigma]))
    assert (expected(g_mu), expected(g_sigma)) == (m, sigma)
    assert raw == (sigma**2, m*m+2*sigma**2)
    assert antithetic == (0, 2*sigma**2)
    print("mu, sigma:", mu, sigma, "raw:", raw,
          "two independent:", iid_two, "one antithetic pair:", antithetic)
assert variance([Q(0), Q(-1), Q(1)]) == 3
node_gradients = [Q(2), Q(0)]  # epsilon = -1, +1 at mu=0, sigma=1.
node_mean = sum(node_gradients)/2
node_variance = sum((g-node_mean)**2 for g in node_gradients)/2
assert node_mean == 1 and node_variance == 1  # Not Gaussian variance 3.
for h in [Q(1, 4), Q(1, 16), Q(1, 64)]:
    interval_width, difference_quotient = h, 1/h
    assert interval_width*difference_quotient == 1
    print("threshold difference quotient:", h, difference_quotient,
          "integrated value:", interval_width*difference_quotient)
```

## 추가 문제와 연구 연결

1. 쌍의 위치 gradient 분산이 0이면 두 성분 모두 개선됐나요? **첫 행의 scale gradient 분산은 독립 경로 두 개보다 오히려 큽니다.**
2. 왜 첫 두 잡음 모멘트 검사만으로 gradient 분산을 알 수 없나요? **scale gradient를 제곱하면 사차 잡음 모멘트가 필요합니다.**
3. 추정량을 비교할 때 무엇을 고정해야 하나요? **목적함수, 매개변수 값, 불편성 가정과 모형 평가 횟수입니다. 비용이 다르면 다른 비교가 됩니다.**

미분 가능한 잡음 구성은 [Kingma–Welling, §§2.3–2.4](https://arxiv.org/html/1312.6114v11)를 참고하세요. 위 이차 분산·동일 비용 비교·문턱 반례는 직접 유도했으며 논문의 학습 실험 재현은 아닙니다.

## AI 연결과 확인 문제

VAE 같은 연속 잠재변수 모델은 이런 경로 미분을 이용해 기대 목적함수의 경사를 추정합니다. 이 절은 이차 모형의 원리를 설명하며 VAE 학습이나 ELBO 전체를 실행하지 않습니다.

1. ε=1에서 손실 0이면 기대 손실은? **기준 μ=0,σ=1에서 여전히 1입니다.**
2. 미분 중 잡음을 바꿔도 되나요? **같은 경로를 미분할 때는 고정해야 합니다.**
3. 두 노드 검산이 모든 함수에 정확한가요? **아니요. 사용한 모멘트 범위에 한정됩니다.**

추가 읽기: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).

[Kingma–Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
