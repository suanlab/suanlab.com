## 벌점을 실제로 포함하는 한 걸음

경사하강법은 미분 가능한 함수를 현재 점에서 선형화한다. 근접 연산자는 벌점 자체를 포함하는 작은 최소화 문제를 푼다. proper closed convex 함수 f와 η>0에 대해

$$
P(y)=\operatorname{prox}_{\eta f}(y)
=\arg\min_x\left[f(x)+\frac{(x-y)^2}{2\eta}\right].
$$

proper는 함수가 어디서나 무한대가 아니며 음의 무한대를 취하지 않는다는 조건이다. closed는 하반연속성을 뜻한다. 이 조건에서 이차항이 강한 볼록성을 주므로 모서리나 무한대 지시함수가 있어도 해가 하나로 정해진다. y를 향해 당기는 용수철과 벌점이 균형을 이루는 그림으로 이해하자. η가 커지면 벌점에 비해 용수철이 약해진다.

## 부분경사로 확인하는 균형

최적성 조건은 (y−P(y))/η∈∂f(P(y))다. 실습은 필요한 부분경사, 실제 부분미분 구간, 두 값 사이의 거리를 보여준다. 목적값이 작다는 관찰과 최적성 조건을 만족한다는 판단은 별개다.

f(x)=λ|x|에서 원점의 부분미분은 [−λ,λ]다. |y|≤ηλ이면 용수철 힘을 이 구간 안에서 상쇄하여 정확히 0에 멈출 수 있다. 그 밖에서는 일정한 벌점 기울기만큼 이동한다.

$$
P(y)=\operatorname{sign}(y)\max(|y|-\eta\lambda,0).
$$

y=2, λ=0.5, η=1이면 근접점은 1.5이고 전체 목적값은 0.875다. 문턱은 λ가 아니라 ηλ다. 단계 크기를 바꿀 때 이 차이를 빠뜨리면 다른 알고리즘을 구현하게 된다.

## 투영과 ridge 수축

구간 [−λ,λ]의 지시함수는 안에서 0, 밖에서 무한대다. 근접점은 y를 구간으로 잘라 넣는 투영이며 η와 무관하다. 이 모드의 λ는 벌점 강도가 아니라 구간 반지름이다. 밖의 무한대는 유한한 그래프 높이로 그리지 않는다.

f(x)=λx²/2이면 해는 y/(1+ηλ)다. y=2, λ=1, η=0.5에서 4/3을 얻는다. 명시적 경사하강 한 걸음은 1이므로 결과가 다르다. 근접 단계는 새 점의 기울기를 사용하는 암시적 갱신으로도 읽을 수 있다.

## Moreau envelope의 미분

Eηf(y)를 근접 최소화 문제의 최솟값으로 정의하자. 이 값 함수는 유한 차원의 proper closed convex f에 대해 미분 가능하다.

$$
\nabla E_\eta f(y)=\frac{y-P(y)}{\eta}.
$$

이는 입력 y에 대한 미분이다. 최적화 변수 x에 대한 미분과 혼동하지 말자. 화면의 전체 근접 목적값이 envelope 값이고, 필요한 부분경사 값이 이 도함수와 일치한다. 구간 지시함수에서는 구간까지의 거리 제곱을 2η로 나눈 값이다.

두 근접점의 부분경사 단조성을 합치면 ‖P(y)−P(z)‖²≤(P(y)−P(z))(y−z)를 얻는다. 이를 firm nonexpansiveness라고 하며 Lipschitz 상수 1을 준다. 항상 엄격한 수축이라는 뜻은 아니다. 항등함수인 경우 두 점 사이 거리가 그대로다.

## Moreau 분해로 원문제 단계와 쌍대 투영 연결하기

![Soft threshold와 배율을 반영한 쌍대 clipping 투영의 합이 입력임을 확인하고, 절댓값 벌점의 모서리를 매끄럽게 만드는 Moreau envelope를 비교합니다.](/assets/learning/convex-moreau-ko.svg)

p=proxηf(y), s=(y−p)/η로 두면 최적성에서 s∈∂f(p)입니다. Fenchel 등호에 따라 p∈∂f*(s)와 동치이며, 이를 정리하면 쌍대 proximal 문제의 최적성 조건이 됩니다.

$$
\operatorname{prox}_{\eta f}(y)+\eta\operatorname{prox}_{f^*/\eta}(y/\eta)=y.
$$

f=λ|x|의 conjugate는 [−λ,λ]의 indicator입니다. 따라서 soft threshold와 배율을 반영한 clipping 투영의 합이 입력입니다. 문턱을 외우는 데 그치지 않고 두 배율을 독립 검산할 수 있습니다.

강볼록성은 비팽창성을 강화합니다. f가 μ-강볼록이면 p,q에서 ∂f의 강단조성으로 (p−q)·(y−z)≥(1+ημ)‖p−q‖²입니다. Cauchy–Schwarz를 적용하면 수축률 1/(1+ημ)를 얻습니다. 절댓값은 강볼록이 아니며 같은 바깥 구간의 두 입력 사이에서는 proximal map이 거리를 유지합니다.

| λ=4/5, η=3/4에서 y | Primal prox p | 배율을 반영한 쌍대 항 ηs |
|---:|---:|---:|
| -2 | -7/5 | -3/5 |
| -3/5 | 0 | -3/5 |
| 0 | 0 | 0 |
| 3/5 | 0 | 3/5 |
| 2 | 7/5 | 3/5 |

Envelope 도함수는 이제 각 proximal 해석 공식으로 계산합니다. Ridge에서 y=1,η=1,λ=10⁻²⁰이면 점은 1로 반올림되지만 도함수는 약 10⁻²⁰입니다. 그 반올림 점을 y에서 빼면 잘못된 0을 얻습니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
lam,eta=F(4,5),F(3,4)
def clip(y): return max(-lam,min(lam,y))
for y in [F(-2),-F(3,5),F(0),F(3,5),F(2)]:
    p=(1 if y>=0 else -1)*max(abs(y)-eta*lam,0)
    s=clip(y/eta)
    assert p+eta*s==y and lam*abs(p)==s*p
    print(y,p,eta*s)
tiny=F(1,10**20); p=1/(1+tiny); derivative=tiny*p
assert 1-p==derivative and derivative>0
print('rounded ridge point:',float(p),'analytic derivative:',float(derivative))
```

## 직접 풀기

각 모드에서 λ=0으로 두자. 절댓값과 ridge 벌점은 사라져 항등사상이 된다. 지시함수는 {0}으로의 투영이 된다. 이 singleton의 법선 원뿔은 실수 전체이므로 임의의 용수철 힘을 상쇄한다. 같은 숫자 매개변수가 모드마다 무엇을 뜻하는지 먼저 읽어야 한다.

문턱 양쪽에서 y를 조금 바꾸어 envelope 값의 중앙차분과 표시된 도함수를 비교하자. 근접점의 공식은 바뀌어도 envelope의 도함수는 연속이다. 비매끄러운 벌점을 매끄러운 값 함수와 연결하는 핵심이다.

더 읽기: [Parikh와 Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
