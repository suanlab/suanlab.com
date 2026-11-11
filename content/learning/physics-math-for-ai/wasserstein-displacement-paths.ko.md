## 분포 혼합과 입자 이동은 다르다

선형 mixture μ_mix,t=(1−t)μ+tν는 기존 위치에서 질량 비율을 바꾼다. Displacement 보간은 coupling γ를 정하고 각 질량의 위치를 옮긴다.

$$
\mu_t=\bigl((x,y)\mapsto(1-t)x+ty\bigr)_\#\gamma.
$$

평균은 둘 다 (1−t)EμX+tEνY지만 분산·지지집합·수송 거리는 다를 수 있다. 위 그림은 가로가 위치, 세로가 시간이고 주황 점이 현재 입자다.

## 최적 coupling과 Wasserstein geodesic

제곱 유클리드 비용의 유한 이차 모멘트 최적 coupling에서는

$$
W_2(\mu,\mu_t)=tW_2(\mu,\nu),\qquad
W_2(\mu_t,\nu)=(1-t)W_2(\mu,\nu).
$$

실습은 오른쪽 값을 그대로 거리로 대입하지 않고 quantile 겹침으로 실제 거리를 다시 계산한다. 이를 통해 항등식을 별도로 확인한다.

Optimal 모드는 단조 coupling, product 모드는 독립 출발·도착 추출이다. Product도 주변 분포는 맞지만 일반적으로 비용 최적은 아니다. 위치 보간은 합법적인 측도 경로지만 자동으로 최단 일정 속도 경로가 되지는 않는다.

## 원자 하나의 중간점

μ=δ₋₁,ν=δ₁에서 t=0.5라면 mixture는 양 끝에 절반씩 남아 분산 1이다. Displacement는 모두 0으로 이동하여 분산 0이다. 평균은 둘 다 0이다.

끝점 W₂는 2다. 이동 중간점은 출발에서 거리 1이지만 mixture 중간점은 √2다. 보간과 중간점이라는 말에 어떤 기하를 쓰는지 명시해야 한다.

## 같은 끝점의 비최적 경로

μ=ν=(δ₋₁+δ₁)/2에서 product coupling은 네 쌍에 각각 1/4을 둔다. t=0.5의 법칙은 0.25δ₋₁+0.5δ₀+0.25δ₁로 원래 분포와 다르다. 끝점 거리는 0인데 선택 coupling 비용은 2다.

중간 분포와 출발 거리는 √0.5로 t·0이 아니다. 주변 질량을 맞추고 입자를 직선으로 움직여도 최적 coupling 없이는 geodesic을 보장하지 못한다.

## 전체 일정 속도 geodesic 항등식 증명하기

최적 coupling의 X,Y에서 E|X−Y|²=D²라고 하자. 같은 무작위 끝점으로 Xₛ=(1−s)X+sY와 Xₜ=(1−t)X+tY를 연결하면 0≤s≤t≤1에서 W₂(μₛ,μₜ)≤(t−s)D다. 끝점 삼각부등식으로 반대쪽 상계를 얻는다.

$$
D\le W_2(\mu_0,\mu_s)+W_2(\mu_s,\mu_t)+W_2(\mu_t,\mu_1)
\le sD+W_2(\mu_s,\mu_t)+(1-t)D.
$$

따라서 W₂(μₛ,μₜ)=(t−s)D다. D를 진짜 끝점 거리로 식별하려면 끝점 coupling이 최적이어야 한다. 더 큰 선택 비용으로는 같은 역방향 상계를 얻을 수 없다.

최적인지와 관계없이 모든 coupling에서 이차 모멘트를 전개하고 같은 평균 제곱을 빼면 다음 진단을 얻는다.

$$
\operatorname{Var}(\mu_{mix,t})-\operatorname{Var}(\mu_t)
=t(1-t)\mathbb E|X-Y|^2.
$$

이는 선택한 coupling 비용을 측정하며 그 비용의 최적성까지 판정하지는 않는다. 같은 대칭 ±1 끝점의 최적 대각 coupling 비용은 0, product 비용은 2다. 두 경우 모두 mixture 분산은 1이다.

| t | 최적 분산 | Product 분산 | Product 차이 |
|---|---:|---:|---:|
| 1/4 | 1 | 5/8 | 3/8 |
| 1/2 | 1 | 1/2 | 1/2 |
| 3/4 | 1 | 5/8 | 3/8 |

```python
from fractions import Fraction as F
atoms = (F(-1), F(1))
for t in (F(1, 4), F(1, 2), F(3, 4)):
    positions = [(1-t)*x+t*y for x in atoms for y in atoms]
    mean = sum(positions)/4
    variance = sum((z-mean)**2 for z in positions)/4
    coupling_cost = sum((x-y)**2 for x in atoms for y in atoms)/4
    deficit = 1-variance
    assert mean == 0 and coupling_cost == 2
    assert deficit == t*(1-t)*coupling_cost
    print(t, F(1), variance, deficit)
```

Product로 연결한 원자를 직접 전개한다. 비geodesic 경로에서도 올바른 분산 항등식은 성립하므로, 이 검사는 수송 최적성 인증을 보완하지만 대체하지 않는다.

## 직접 풀기와 표현 방식

두 중간점 예제의 질량을 직접 계산하자. t=0과 1에서는 겹친 원자가 합쳐져 끝점 법칙을 되찾아야 한다. 구현은 같은 위치를 합하고 0 질량을 제외하여 하나의 원자를 여러 확률 줄기로 잘못 표시하지 않는다.

질량이 다른 최적 예제에서 출발·도착까지 거리의 합이 끝점 거리와 같은지 확인한다. 위 그림은 개별 coupling 경로를 남기고 아래 표는 같은 위치의 입자를 합한다. 같은 측도를 다른 세부 수준으로 보는 것이다.

생성·분포 AI 모델에서 경로를 transport geodesic이라고 부르려면 coupling·비용·보간 방식을 먼저 기록해야 한다.

더 읽기: [Peyré와 Cuturi의 최적수송 교재](https://optimaltransport.github.io/book/).
