## 계획과 별도로 검사할 수 있는 하한

출발·도착 주변 질량을 고정한 비음수 coupling은 선형계획의 가능집합이다. 쌍대는 출발 potential uᵢ와 도착 potential vⱼ를 둔다.

$$
\max_{u,v}\sum_i a_i u_i+\sum_j b_j v_j,
\qquad u_i+v_j\le C_{ij}.
$$

Potential은 음수여도 된다. 총질량이 같을 때 u에 상수를 더하고 v에서 빼도 부등식과 목적값은 같다. 실습은 u₀=0인 gauge를 사용한다.

## 주변 질량으로 약쌍대성 유도

계획과 potential이 모두 가능하면

$$
\sum_{ij}\gamma_{ij}C_{ij}
-\left(\sum_i a_i u_i+\sum_j b_j v_j\right)
=\sum_{ij}\gamma_{ij}(C_{ij}-u_i-v_j)\ge0.
$$

각 항이 질량과 비음수 slack의 곱이다. 최적 쌍의 양의 흐름은 tight edge만 사용한다. 반대로 질량 0인 간선이 반드시 양의 slack을 갖는 것은 아니다. 사용하지 않는 tight edge도 가능하다.

인증 gap은 실제 최솟값을 몰라도 현재 원문제 오차를 위에서 잡는다. 여기서는 정확한 최적값도 계산하여 인증과 실제 오차를 비교한다.

## 완전한 원·쌍대 쌍 손계산

출발 (0,1), 도착 (0,L)의 제곱 비용은 [[0,L²],[1,(L−1)²]]다. a≤b이면 u=(0,1),v=(0,L²−2L), a>b이면 u=(0,1−2L),v=(0,L²)가 최적 선택이다. 단조 계획의 양의 간선이 tight하다.

a=0.4,b=0.6,L=2에서는 계획 [[0.4,0],[0.2,0.4]], potential u=(0,1),v=(0,0)다. 쌍대값 0.6이 비용과 같고, 사용하지 않는 0→L 간선 slack은 4다.

## Potential을 바꾸기 전에 가능성 검사

δ는 v₁에 더하는 값이다. 음수 δ는 부등식을 보존하지만 쌍대값을 |δ|(1−b) 낮춘다. δ=−1이면 하한 0.2, gap 0.4다. 계획이 이미 최적이어도 승수가 부정확하면 gap이 남는다.

양수 δ는 tight 부등식을 어긴다. δ=1이면 대수적 primal-minus-dual은 −0.4다. 쌍대 불가능성이 원인이므로 약쌍대성의 반례가 아니다. 화면은 부호 있는 값을 남기되 인증 상한은 제공하지 않는다.

## 쌍대 가능성을 복구하고 slack 간선의 질량 제한하기

임의 유한 potential에서 최대 양의 위반량을 δ=max(0,maxᵢⱼ(uᵢ+vⱼ−Cᵢⱼ))라 하자. 모든 출발 potential에서 δ를 빼면 모든 부등식이 가능해진다. 총질량 1이면 쌍대값은 δ만큼 낮아진다. 실제로 가능한 후보 계획 P에 대해

$$
0\le\langle C,P\rangle-OT
\le\langle C,P\rangle-D(u,v)+\delta.
$$

총질량이 M이면 보정량은 Mδ다. 수치 위반량도 모든 부등식의 상계여야 하며 일부 간선 검사만으로 충분하지 않다. Potential 복구가 주변 질량이 틀린 후보까지 복구하지는 않는다.

가능한 원·쌍대 쌍의 gap이 G라면 slack이 τ>0 이상인 간선의 총질량은 G/τ 이하다. G=ΣPᵢⱼslackᵢⱼ에서 나머지 비음수 항을 버리면 된다. 작은 양의 흐름을 보고 정확히 tight하다고 판단하는 대신 상보성을 정량적으로 읽을 수 있다.

P=[[1/5,1/5],[2/5,1/5]], C=[[0,4],[1,1]], u=(0,1),v=(0,d)를 쓰면 실제 후보 오차는 4/5다.

| d | 원래 쌍대 | 복구 쌍대 | Gap |
|---|---:|---:|---:|
| 0 | 3/5 | 3/5 | 4/5 |
| 1/10 | 16/25 | 27/50 | 43/50 |
| 1 | 1 | 0 | 7/5 |

```python
from fractions import Fraction as F
P = ((F(1, 5), F(1, 5)), (F(2, 5), F(1, 5)))
C, a, b = ((0, 4), (1, 1)), (F(2, 5), F(3, 5)), (F(3, 5), F(2, 5))
primal = sum(P[i][j]*C[i][j] for i in range(2) for j in range(2))
for d in (F(0), F(1, 10), F(1)):
    u, v = (F(0), F(1)), (F(0), d)
    violation = max(F(0), max(u[i]+v[j]-C[i][j] for i in range(2) for j in range(2)))
    repaired = tuple(x-violation for x in u)
    raw_dual = sum(x*y for x, y in zip(a, u))+sum(x*y for x, y in zip(b, v))
    dual = raw_dual-violation
    slack = [[C[i][j]-repaired[i]-v[j] for j in range(2)] for i in range(2)]
    gap = sum(P[i][j]*slack[i][j] for i in range(2) for j in range(2))
    assert min(x for row in slack for x in row) >= 0
    assert gap == primal-dual and gap >= primal-F(3, 5)
    assert P[0][1] <= gap/4
    print(d, raw_dual, dual, gap)
```

프로그램은 별도로 복구를 수행한다. 기존 실습은 수정하지 않은 불가능 potential의 인증을 계속 제공하지 않으며, 이는 올바른 동작이다.

## 직접 풀기와 수치 해석

δ=0에서 계획 혼합을 최적 끝점에서 멀리 옮기자. 가중 slack 합과 비용 증가가 같은지 확인한다. 이어 v₁을 낮추어 원문제 오차와 쌍대 하한의 느슨함을 구별하자.

구현은 거의 같은 목적값의 차감 대신 상보성 항을 직접 계산하고 원래 차감도 함께 표시한다. 큰 해결기에서도 주변 잔차와 모든 potential 부등식을 검사해야 한다. 불가능 계획의 낮은 비용을 원문제 상한으로 사용하면 안 된다.

더 읽기: [Peyré와 Cuturi의 최적수송 교재](https://optimaltransport.github.io/book/).
