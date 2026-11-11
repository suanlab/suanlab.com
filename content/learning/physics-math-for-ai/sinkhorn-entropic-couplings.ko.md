## 정규화는 문제 자체를 바꾼다

양의 주변 질량에서 실습의 목적은

$$
J_\epsilon(\gamma)=\langle C,\gamma\rangle+
\epsilon\sum_{ij}\gamma_{ij}(\log\gamma_{ij}-1),
\qquad\gamma\in\Pi(a,b).
$$

Entropy 부호와 상수 관례를 확인해야 한다. 값이 음수일 수 있고 Wasserstein metric이 아니다. 질량 분산을 보상하므로 수송 비용 항은 원래 최솟값보다 커질 수 있다. ε를 줄이는 것과 각 정규화 문제를 정확히 푸는 것은 별개다.

## 로그 좌표의 행·열 갱신

Entropy conjugate 쌍대는

$$
D(f,g)=a^Tf+b^Tg-\epsilon\sum_{ij}
\exp((f_i+g_j-C_{ij})/\epsilon).
$$

유한 potential에는 원래 LP의 fᵢ+gⱼ≤Cᵢⱼ를 그대로 걸지 않는다. 지수 conjugate 항이 올바른 벌점을 제공한다.

f 갱신은 행, g 갱신은 열 주변 질량을 맞춘다. 작은 kernel을 직접 나누는 대신 log-sum-exp를 사용한다. 표는 완전한 행·열 cycle을 기록한다. 열 갱신 뒤 열 잔차가 작아도 행 잔차는 남을 수 있다.

## 유한 반복은 가능 상한이 아닐 수 있다

Raw 계획은 exp((f+g−C)/ε)다. 두 주변 제약을 만족하기 전에는 비용이나 정규화 목적값이 원문제 상한이 아니다. 반면 어떤 유한 potential의 쌍대값도 정규화 문제의 하한이다.

이 2×2 문제는 양의 기준 해를 독립적으로 구할 수 있다.

$$
\frac{\gamma_{00}\gamma_{11}}{\gamma_{01}\gamma_{10}}
=\exp(2L/\epsilon).
$$

작은 off-diagonal 질량을 거의 포화된 대각값에서 빼지 않고 안정적인 양의 이차근으로 직접 구한다. 균형 L=1의 경우 그 질량은 1/[2(1+exp(1/ε))]다.

## 안정적인 Fenchel 잔차

가능 기준 γ*의 primal–dual gap은 εΣ[γ*log(γ*/γ_raw)−γ*+γ_raw]다. 두 질량이 가까우면 안정적인 국소 나머지를 사용하여 작은 양의 잔차를 보존한다. 거의 같은 두 목적값을 직접 빼는 방식과 구별하자.

a=b=0.5,L=1,ε=0.5에서는 대칭성 때문에 한 cycle로 맞는다. 기준 수송 비용은 약 0.119203, 정규화 목적값은 약 −0.910038이다. 어느 것도 원래 W₂라고 부르면 안 된다.

## 정규화 해의 인증과 수송 비용 편향 구별하기

유한 비용·엄격히 양의 주변 질량·ε>0에서는 엔트로피 최적해가 유일하다. 가능집합은 compact하고 entropy 항은 0에서도 연속 확장된다. 엄격 볼록성으로 유일성이 따른다. 최적해에 0 원소가 있다면 양의 곱 coupling 방향으로 혼합할 때 그 원소의 entropy 방향 미분이 −∞여서 유한 비용 미분보다 우세하므로 최적일 수 없다.

내부 최적점 P*의 ∇Jε(P*)는 행·열 potential의 합이다. 가능한 확률 계획 P와의 차이 P−P*에 대한 내적은 주변 보존으로 0이다. 목적을 전개하면

$$
J_\epsilon(P)-J_\epsilon(P^*)
=\epsilon\sum_{ij}P_{ij}\log(P_{ij}/P^*_{ij}),\qquad
\|P-P^*\|_1\le\sqrt{2G/\epsilon},
$$

여기서 G는 P의 유효한 정규화 원·쌍대 gap이다. 두 번째 상계는 확률 행렬을 펼친 Pinsker 부등식과 Jε(P)−Jε(P*)≤G를 사용한다. 총질량이 1이 아닌 불가능 raw 행렬에 확률 Pinsker를 그대로 적용한 식은 아니다.

절반 질량·L=1·ε=1/2에서 최적 off-diagonal 질량을 s=1/[2(1+exp2)]라 하자. Off-diagonal이 r인 대칭 가능 계획의 L1 오차는 4|r−s|다.

| r | L1 오차 |
|---|---:|
| 0 | 4s |
| s | 0 |
| 1/4 | 1−4s |

```python
from decimal import Decimal as D, localcontext
with localcontext() as ctx:
    ctx.prec = 80
    eps = D('0.5')
    s = 1/(2*(1+D(2).exp()))
    reference = (D('0.5')-s, s, s, D('0.5')-s)
    def objective(P):
        return P[1]+P[2]+eps*sum(x*(x.ln()-1) if x else D(0) for x in P)
    for r in (D(0), s, D('0.25')):
        P = (D('0.5')-r, r, r, D('0.5')-r)
        gap = objective(P)-objective(reference)
        kl = sum(x*(x/y).ln() if x else D(0) for x, y in zip(P, reference))
        error = sum(abs(x-y) for x, y in zip(P, reference))
        assert abs(gap-eps*kl) < D('1e-75')
        assert error*error <= 2*kl+D('1e-75')
        print("r,L1,gap:", r, error, gap)
```

별도의 작은 양인 기준 해의 비정규화 수송 초과 비용은 이 두 원자 가족에서 정확히 2Ls다. 실습은 이 곱을 직접 계산하여 L=3,ε=0.05의 약 2.63×10⁻²⁶을 보존한다. 반올림된 두 절대 비용을 빼면 0일 수 있어 절대 비용은 별도 진단으로 남긴다. 작은 최적화 gap도 이 특수 공식도 보편적인 entropy 편향 추정량은 아니다.

## 직접 풀기와 풀이

같은 대칭 예에서 ε=0.05이면 작은 기준 질량은 약 1.030577×10⁻⁹로 여전히 양수다. 다음 비대칭 초기 kernel을 선택하자. Raw 목적값이 진짜 최솟값보다 낮을 수 있는 이유는 불가능성이다. 먼저 행·열 오차를 확인한다.

작은 주변 잔차, 정규화 gap, 원래 수송에 대한 작은 편향은 서로 다른 정확도 조건이다. 이 유한 예제가 고정 반복 수로 모든 수송 문제가 풀린다는 보장은 아니다.

더 읽기: [Peyré와 Cuturi의 최적수송 교재](https://optimaltransport.github.io/book/).
