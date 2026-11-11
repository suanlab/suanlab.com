## 확률 제약에 맞는 기하 선택

확률 벡터는 Δ₃={q:qᵢ≥0,Σqᵢ=1}에 속한다. 일반 GD는 밖으로 나갈 수 있어 projected GD는 유클리드 거리로 되돌린다. Mirror descent는 볼록 생성 함수로 이동 비용을 정한다.

실습은 q₀=(0.2,0.3,0.5)에서 선형 비용 c·q를 최소화한다. 정확한 최솟값은 min cᵢ다. 최저 비용 동률이 있으면 하나의 꼭짓점 대신 최적 face가 생긴다.

## Entropy가 주는 곱셈 갱신

음의 entropy 생성 함수의 simplex Bregman divergence는 KL(q||q_old)다.

$$
q^+=\arg\min_{q\in\Delta_3}
\left[\eta c^Tq+D_{KL}(q\|q_{old})\right],
\qquad
q_i^+=\frac{q_{old,i}e^{-\eta c_i}}{\sum_jq_{old,j}e^{-\eta c_j}}.
$$

새 변수가 KL의 첫 인자라는 순서가 중요하다. 정규화 승수를 넣어 미분하면 지수 공식을 얻는다. 유클리드 clipping 후 정규화와는 다른 근접 문제다.

비용이 일정하면 qₖ,ᵢ∝q₀,ᵢe⁻ᵏηᶜⁱ다. 구현은 최대 로그 가중치를 뺀 뒤 지수화한다. 제어 범위 안에서 양의 지지집합과 정규화를 유지한다.

## 유클리드 투영과 비교

파란 경로는 q−ηc를 매번 유클리드 투영하고 주황 경로는 entropic 갱신을 반복한다. 유클리드 투영은 합이 1이 되도록 공통 문턱 τ를 찾아 qᵢ⁺=max(vᵢ−τ,0)로 만든다. 유한 보폭에서 정확한 0이 생길 수 있다. Entropic 양의 가중치는 유한 보폭에서 양수이고 극한에서 0으로 접근할 수 있다.

삼각형 좌표는 (q₂+q₃/2,√3q₃/2)다. 동일 축척은 이 유클리드 표시를 보존할 뿐 Fisher 거리나 KL 기하를 그대로 보존하지 않는다. 실제 확률은 표로 읽자.

## 다른 비용과 동률 예제

c=(0,1,2),η=1,4회이면 mirror 확률은 약 (0.972467,0.026717,0.000816), 목적 오차는 약 0.028348이다. 투영 GD는 (1,0,0)에 도달해 오차 0이다. 이 유한 예제가 한 방법의 보편적 우위를 뜻하지는 않는다.

c=(0,0,1)이면 첫 두 mirror 확률의 초기 비율 2:3이 유지된다. 셋째 질량이 사라지며 (0.4,0.6,0)으로 접근한다. 기하와 초기화가 최적 face의 특정 점을 선택하는 셈이다. 유클리드 투영의 선택은 다를 수 있다.

## 변하는 유계 비용에 대한 유한 시간 보장

매 round 다른 전체 비용 벡터 cₜ∈[0,1]³을 허용하자. 그 round의 벡터를 보기 전에 qₜ로 예측하고, 관측 후 qₜ₊₁,ᵢ∝qₜ,ᵢexp(−ηcₜ,ᵢ)로 갱신한다. Zₜ=Σᵢq₀,ᵢexp(−ηΣₛ₌₀ᵗ⁻¹cₛ,ᵢ)라 쓰면 로그 정규화량 증가가 −ηqₜ·cₜ+η²/8 이하이다. 상수의 근거는 중심화 로그 적률생성함수의 이계 미분이 기울어진 분포의 분산이며, [0,1] 값의 분산은 1/4 이하라는 점이다. 두 번 적분하면 나머지 η²/8을 얻는다.

증가량을 합하고 Z_T를 비교 성분 하나의 항으로 아래에서 제한하면 모든 고정 성분 j와 η>0에 대해

$$
\sum_{t=0}^{T-1}q_t^Tc_t-\sum_{t=0}^{T-1}c_{t,j}
\le\frac{\log(1/q_{0,j})}{\eta}+\frac{\eta T}{8}.
$$

q₀=(0.2,0.3,0.5)이므로 모든 비교항을 log5로 제한할 수 있다. η=√(8log5/T)를 택하면 평균 regret 상계는 √(log5/(2T))다. 전체 비용을 관측하고 최선의 고정 성분과 비교하는 온라인 보장이지, 매 round의 우위나 bandit 보장이 아니다. [−2,2]의 상수 비용에는 재척도화하거나 비용 범위 제곱을 반영해야 한다.

| T | 평균 regret 상계 |
|---|---:|
| 4 | 0.448531 |
| 16 | 0.224265 |
| 64 | 0.112133 |

```python
from math import exp, log, sqrt, fsum
initial = (0.2, 0.3, 0.5)
cycle = ((0., 1., .5), (1., .5, 0.), (.5, 0., 1.))
for T in (4, 16, 64):
    eta = sqrt(8*log(5)/T)
    cumulative, loss = [0., 0., 0.], 0.
    for t in range(T):
        logits = [log(q)-eta*c for q, c in zip(initial, cumulative)]
        largest = max(logits)
        weights = [exp(x-largest) for x in logits]
        q = [w/fsum(weights) for w in weights]
        cost = cycle[t % 3]
        loss += fsum(p*c for p, c in zip(q, cost))
        cumulative = [s+c for s, c in zip(cumulative, cost)]
    regret = loss-min(cumulative)
    bound = log(5)/eta+eta*T/8
    assert regret <= bound+1e-12
    print(T, "regret:", round(regret, 6), "average bound:", round(bound/T, 6))
```

별도 프로그램은 변하는 비용으로 확률을 실제 갱신한다. 갱신 전에 발생 손실을 계산하고 누적 손실을 비교하여 증명에 필요한 관측 순서를 지킨다. 위 인터랙티브 실습은 기존의 상수 비용 실험을 유지한다.

![시간 길이별로 보폭을 따로 정한 전체 정보 실험에서 실제 평균 regret과 증명한 상계를 비교합니다.](/assets/learning/variational-regret-ko.svg)

## 직접 풀기와 한계

비용을 모두 같게 하면 모든 점이 최적이고 mirror 분포는 초기값이다. 모든 비용에 같은 상수를 더했을 때 정규화된 지수식에서 상수가 사라짐을 증명하자. 처음부터 0인 성분은 곱셈 갱신으로 되살릴 수 없으므로 여기서는 엄격히 양의 초기화를 사용한다.

일반 수렴률에는 볼록성·기울기 제한·생성 함수 기하·보폭 일정이 필요하다. 세 성분 선형 예제는 실제 갱신과 지지집합 거동을 설명하며 그 일반 정리의 증명을 대신하지 않는다.

더 읽기: [Nemirovski의 Mirror Descent 강의](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf).
