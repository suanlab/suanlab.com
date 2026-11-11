## gradient를 다시 미분하기

매끄러운 스칼라 함수 f:Rⁿ→R의 gradient는 벡터이고 이를 다시 미분한 Hessian H는 n×n 행렬입니다. 특정 방향 v에 대한 gradient의 변화는 Hv입니다.

$$
H_{ij}=\partial_i\partial_j f,\qquad H v=D(\nabla f)[v].
$$

충분한 연속 혼합 미분 조건에서 H는 대칭입니다. 그러나 항상 PSD이거나 PD인 것은 아닙니다. 자동 미분이 도함수를 계산했다는 사실이 손실의 볼록성을 주지는 않습니다.

## 비선형 예제의 손계산

실습 함수는 f(x,y)=½x²y+sin y입니다.

$$
\nabla f=(xy,\tfrac12x^2+\cos y),\qquad
H=\begin{pmatrix}y&x\\x&-\sin y\end{pmatrix}.
$$

기준점 (1,0)에서 gradient=(0,1.5), H=[[0,1],[1,0]]입니다. 고유값은 ±1이므로 부정부호입니다. v=(1,1)이면 Hv=(1,1)이고 vᵀHv=2입니다. v가 단위 벡터가 아니므로 이 2를 최대 고유값이라고 읽으면 안 됩니다.

## 방향 곡률과 이차 모형

한 변수 경로 g(t)=f(z+tv)를 만들면

$$
g'(0)=\nabla f(z)^Tv,\qquad g''(0)=v^TH(z)v.
$$

따라서 이차 근사는 f(z)+t∇f·v+½t²vᵀHv입니다. 기준점과 방향에서 1.5t+t²를 얻습니다. 실제 경로는 ½(1+t)²t+sin t이므로 고차 나머지가 남습니다. 그림은 실제 경로와 이차 모형을 비교합니다.

v=(1,0)에서는 이 기준점의 경로 함수가 0으로 일정하고 곡률도 0입니다. v=0이면 어떤 점에서도 경로가 일정합니다. 0 방향은 입력 공간이 없어진다는 뜻이 아닙니다.

## Hessian 전체 없이 곱 계산하기

고차원에서 H를 명시적으로 만들면 n²개 원소가 필요합니다. 자동 미분에서 gradient 계산에 방향 미분을 적용하면 Hv를 전체 H 없이 계산할 수 있습니다. 또는 스칼라 ∇f·v를 v를 상수로 두고 미분하면 Hᵀv이고, 매끄러운 스칼라 함수에서는 H 대칭성으로 Hv와 같습니다.

이 실습의 엔진은 해석 식을 계산하며 실제 자동 미분 라이브러리를 실행하는 것은 아닙니다. JVP·VJP·중첩 미분으로 구현할 때는 형상, 계산 그래프 보존, v에 대한 미분 여부를 명시해야 합니다.

## 차분 검산과 AI 연구

$$
Hv\approx\frac{\nabla f(z+hv)-\nabla f(z-hv)}{2h}.
$$

이는 별도 gradient 식에서 중앙차분을 만들어 독립 검산하는 방법입니다. h=0은 불가하고 작은 h의 반올림·소거 오차도 있습니다. 실습의 회귀 검사는 이 차분으로 해석 Hv와 방향 곡률을 확인합니다.

Newton-CG, 곡률 진단, 영향함수 등에서 HVP가 쓰입니다. 실제 손실의 비매끄러운 연산·확률 표본·배치 선택이 바뀌면 고전적 미분 가정과 검산 조건을 다시 확인해야 합니다.

## 행렬 곱만으로 trace 추정하기

매개변수·데이터·모형의 난수를 포함해 실수 대칭 Hessian H를 고정하세요. 각 좌표가 독립적으로 −1과 1을 같은 확률로 갖는 Rademacher 벡터 v를 뽑습니다. E[vᵢvⱼ]=δᵢⱼ이므로

$$
\mathbb E[v^THv]=\operatorname{tr}H,\qquad
v^THv=\operatorname{tr}H+2\sum_{i<j}H_{ij}v_i v_j.
$$

vᵢ²=1이어서 대각 성분의 기여는 정확합니다. 서로 다른 두 비순서 쌍의 교차 기댓값은 0입니다. 인덱스가 겹치지 않으면 독립인 평균 0 인자가 남고, 하나를 공유하면 제곱이 소거된 뒤 서로 다른 평균 0 인자 두 개가 남습니다. 따라서

$$
\operatorname{Var}(v^THv)=4\sum_{i<j}H_{ij}^2,\qquad
\operatorname{Var}\left(\frac1M\sum_{m=1}^M v_m^THv_m\right)
=\frac4M\sum_{i<j}H_{ij}^2
$$

입니다. 두 번째 식은 M개 벡터가 독립일 때 성립합니다. 같은 무작위 벡터를 M번 재사용하면 단일 표본의 분산이 그대로 남습니다. 위 식은 Rademacher 분포에 대한 식이며 다른 분포에서는 4차 모멘트를 다시 계산해야 합니다.

## 기준점의 모든 표본과 곡률 해석

기존 실습의 기준점 (1,0)에서는 Hv=(v₂,v₁)입니다. 확률이 같은 네 벡터를 전부 나열하면 다음과 같습니다.

| v | Hv | vᵀHv |
|---|---|---|
| (1,1) | (1,1) | 2 |
| (1,−1) | (−1,1) | −2 |
| (−1,1) | (1,−1) | −2 |
| (−1,−1) | (−1,−1) | 2 |

평균은 0이고 단일 표본의 분산은 4입니다. 독립 표본 M=1,2,8개의 평균은 각각 분산 4,2,1/2을 갖습니다. 양수 관측의 개수 K는 Binomial(M,1/2)이고 추정량은 2(2K−M)/M입니다.

Trace는 부호 있는 곡률의 합입니다. H=diag(2,−1)이면 모든 Rademacher 표본이 정확히 trace 1을 주지만 음의 고유값이 남습니다. 볼록성을 판단하려면 trace 이외의 적절한 곡률 조건이 필요합니다. 실습의 기준 Hessian처럼 반대 부호 곡률이 상쇄되면 trace 0에서도 곡률이 존재합니다.

## Python으로 모든 표본과 배치 모멘트 검산

다음 코드는 gradient의 중앙차분과 해석 HVP를 대조한 뒤 정확한 유리수로 표본·이항 모멘트를 열거합니다. 차분은 수치 검산이고 정확한 열거는 유한 분포의 모멘트 검산입니다. 대화형 엔진은 기존 해석 계산을 유지합니다.

```python
from fractions import Fraction as Q
from itertools import product
from math import comb, cos

def gradient(x, y):
    return (x * y, 0.5 * x * x + cos(y))

def hvp(v):
    return (v[1], v[0])  # at (x, y) = (1, 0)

values = []
for v in product([-1, 1], repeat=2):
    h = 1e-4
    plus = gradient(1 + h * v[0], h * v[1])
    minus = gradient(1 - h * v[0], -h * v[1])
    difference = tuple((a - b) / (2 * h) for a, b in zip(plus, minus))
    assert max(abs(a - b) for a, b in zip(difference, hvp(v))) < 1e-8
    values.append(sum(a * b for a, b in zip(v, hvp(v))))
assert sum(values) == 0
assert Q(sum(q * q for q in values), 4) == 4

for m in [1, 2, 8]:
    mean, second = Q(0), Q(0)
    for k in range(m + 1):
        probability = Q(comb(m, k), 2**m)
        estimate = Q(2 * (2 * k - m), m)
        mean += probability * estimate
        second += probability * estimate**2
    assert mean == 0
    assert second - mean**2 == Q(4, m)
    print(m, mean, second - mean**2)
```

비대칭 동역학 Jacobian J에서는 S=(J+Jᵀ)/2에 대해 vᵀJv=vᵀSv, tr S=tr J입니다. 기댓값 항등식은 유지되지만 분산 식에는 S를 사용합니다. 벡터장의 발산은 tr J이고 스칼라 손실의 곡률 합은 tr H입니다. 실험에서 무엇을 추정하는지 먼저 정하세요.

출처 연결: [FFJORD v3](https://arxiv.org/abs/1810.01367v3)의 §3.1, 식 (7)은 동역학 Jacobian에 확률적 trace 항등식을 적용합니다. 여기의 Hessian 열거와 분산 유도는 직접 구성했습니다. 논문의 flow 학습과 수치 적분 재현은 별도 과제입니다. 이 고정 행렬 예제에서는 같은 HVP 예산으로 독립 표본과 한 표본 재사용을 비교해 보세요.

## 확인 문제와 답

1. (1,0),v=(1,−1)에서 Hv와 vᵀHv는? **(−1,1),−2**입니다.
2. H가 대칭이면 PSD인가요? **아니요. 기준 H가 반례입니다.**
3. Hv는 스칼라인가요? **벡터이며 vᵀHv가 스칼라**입니다.

추가 읽기: [MIT 공식 강좌 안내](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). 강좌 범위를 확인해 보충 자료로 연결하며, 아래의 작은 예제와 논증은 직접 구성했습니다.
