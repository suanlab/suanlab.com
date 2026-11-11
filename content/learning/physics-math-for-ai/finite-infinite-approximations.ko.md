## 고정 대상의 오차와 모든 입력의 오차

ℓ2에서 Pₙ은 첫 N개 좌표를 남기는 직교 투영입니다. 고정한 x에 대해 제곱 꼬리가 0으로 가므로 Pₙx→x입니다. 이를 strong 수렴이라고 합니다. 하지만 연산자 노름은 입력을 먼저 고정하지 않고 모든 단위 입력의 최악 오차를 구합니다.

$$
\|(I-P_N)x\|_2^2=\sum_{k>N}|x_k|^2,\qquad \|I-P_N\|=1.
$$

각 N마다 eₙ₊₁을 택하면 전체를 잘라 오차가 1입니다. 반례 입력이 N과 함께 바뀌므로 각 고정 x에 대한 수렴과 모순되지 않습니다. supremum과 극한의 순서를 바꾸는 데에는 추가 조건이 필요합니다.

## 정확한 꼬리 계산

실습의 고정 대상은 xₖ=1/k입니다. Basel 합을 사용하면 전체 제곱 노름은 π²/6이고 절단 오차는 다음과 같습니다.

$$
\|x-P_Nx\|_2^2=\frac{\pi^2}{6}-\sum_{k=1}^{N}\frac1{k^2}.
$$

적분 비교로 이 값은 1/(N+1) 이상, 1/N 이하입니다. 따라서 고정 대상 오차는 줄지만 파란 곡선 위의 모든 점이 주황의 연산자 노름 1을 낮추지는 않습니다. N=4에서 e5를 선택하면 witness 오차 1, e2를 선택하면 0입니다. 선택 witness 하나가 0이어도 supremum이 0은 아닙니다.

## Compact 사상의 근사는 더 강할 수 있다

T=diag(1/k)는 compact이고 ‖T−PₙT‖=1/(N+1)로 노름 수렴합니다. 유한 rank Pₙ이 strong 수렴한다는 사실만으로 모든 연산자에 같은 노름 수렴이 생기는 것은 아니며, 이 예제에서는 T의 감쇠 구조가 중요합니다.

$$
\|T-T_N\|=1/(N+1),\qquad \|T_N^{-1}\|=N.
$$

Tₙ의 역은 첫 N개 좌표 공간에서 정의합니다. 모두 invertible이어도 역 노름이 커져 작은 고주파 잡음이 증폭됩니다. 원래 T는 ℓ2 전체로의 bounded 역을 갖지 않습니다. 해상도를 올리는 것과 역문제의 안정성을 높이는 것은 같은 일이 아닙니다.

## Galerkin 수렴에는 근사와 안정성이 함께 필요하다

실수 Hilbert 공간 V의 bounded coercive bilinear form a에 대해 |a(u,v)|≤M‖u‖‖v‖, a(v,v)≥α‖v‖², α>0을 가정하면 conforming Galerkin 근사의 Céa 경계는 다음과 같습니다.

$$
\|u-u_N\|\le\frac M\alpha\inf_{v_N\in V_N}\|u-v_N\|.
$$

현재 실습은 이 PDE 해법을 구현한 것이 아니라 수렴 방식의 차이를 계산합니다. 연구에서 작은 이산 잔차만으로 연속 문제의 정확도·안정성을 주장하지 말고 일관성, 공간의 조밀성, 경계조건과 해상도 독립 상수를 확인하세요.

## Coercivity에서 존재·안정성까지 연결하기

실수 Hilbert 공간 V의 bounded bilinear form a와 연속 범함수 ℓ을 생각합니다. Riesz 표현으로 ⟨Au,v⟩=a(u,v), ℓ(v)=⟨g,v⟩인 유계 A와 g를 얻습니다. Coercivity a(u,u)≥α‖u‖²와 Cauchy–Schwarz는 ‖Au‖≥α‖u‖를 줍니다. 따라서 A는 단사이고 range가 닫혔습니다. 실제로 Au_n이 Cauchy이면 u_n도 Cauchy이고 완비성과 A의 연속성이 극한을 보존합니다.

Range에 직교하는 y는 모든 u에 대해 a(u,y)=0이므로 u=y를 대입하면 y=0입니다. 닫힌 range가 조밀하기도 해서 V 전체이고, Au=g의 유일해가 존재하며 ‖u‖≤‖ℓ‖/α입니다. 이것이 이 설정의 Lax–Milgram 논증입니다. 대칭성은 필요하지 않습니다. 닫힌 conforming 부분공간에서도 같은 조건이 성립해 Galerkin 해를 얻습니다.

Galerkin 직교성 a(e,v_N)=0, e=u−u_N에서 임의 v_N에 대해

$$
\alpha\|e\|^2\le a(e,e)=a(e,u-v_N)
\le M\|e\|\|u-v_N\|.
$$

를 얻고 inf를 취하면 앞 Céa 경계가 나옵니다. 예를 들어 A=[[2,1],[1,2]], 우변 (1,0), V_N=span(e₁)이면 정확해는 (2/3,−1/3), Galerkin은 (1/2,0)입니다. Euclidean 직교 투영 (2/3,0)과 다릅니다. Galerkin의 Euclidean 오차 √5/6≈.372678은 최적 Euclidean 오차 1/3보다 크지만 M/α=3 경계를 만족합니다. 대칭 양의 이 예제에서는 에너지 노름에서 Galerkin이 최적입니다.

## 역문제에서 N을 크게만 하면 안 되는 이유

T=diag(1/k), 참 해 x_k=1/k이면 자료 y_k=1/k²입니다. 잡음 ‖η‖₂≤δ가 섞인 자료를 첫 N개에서 역산하면 x̂_k=k(y_k+η_k), 나머지는 0입니다. 절단 꼬리와 잡음 성분의 지지가 직교하므로

$$
\sup_{\|\eta\|_2\le\delta}\|x-\widehat x_N\|_2^2
=\sum_{k>N}\frac1{k^2}+N^2\delta^2.
$$

상한의 잡음 항은 η=δe_N이 달성합니다. 각 N에서 최악 잡음이 달라지므로 표와 그림은 동일한 고정 잡음열의 관측 곡선이 아니라 허용 잡음 집합의 worst-case envelope입니다.

![절단 rank가 커질 때 1/k 해의 꼬리 오차는 줄지만 잡음 증폭 Nδ는 증가하여 최악 전체 오차가 다시 커지는 비교 그래프. δ=.05이며 rank축은 두 배 간격의 log2 척도이다.](/assets/learning/functional-inverse-rank-noise-ko.svg)

| N, δ=.05 | 절단 bias | 최대 잡음 Nδ | 최악 전체 오차 |
|---|---|---|---|
| 2 | .628438 | .100000 | .636344 |
| 4 | .470450 | .200000 | .511198 |
| 8 | .342800 | .400000 | .526794 |
| 16 | .246145 | .800000 | .837011 |
| 32 | .175405 | 1.600000 | 1.609586 |

코드는 분수 2×2 Galerkin과 구성한 최악 잡음의 역산을 검산합니다. N=32 등은 별도 코드 예제이고 화면의 rank 제어는 16까지입니다. 이 envelope에서 1~64를 비교하면 최소 rank는 6입니다. δ→0에서 N→∞이면서 Nδ→0이면 이 특정 해에 수렴합니다. 꼬리 제곱≈1/N과 잡음 제곱=N²δ²를 균형 잡으면 N의 규모는 δ⁻²ᐟ³이고 오차 규모는 δ¹ᐟ³입니다. 모든 역문제의 보편 수렴률이라는 뜻은 아닙니다.

**직접 풀기:** 매 N의 이산 선형계 잔차가 0이면 참 해도 더 정확해지나요? **해설:** 보장하지 않습니다. 잡음까지 정확히 맞추면 Nδ가 커질 수 있습니다. 잔차·근사·안정성·잡음 수준을 함께 확인해야 합니다.

```python
from fractions import Fraction as F
from math import sqrt,pi

A=[[F(2),F(1)],[F(1),F(2)]]
u=[F(2,3),F(-1,3)]
galerkin=[F(1,2),F(0)]
euclidean=[u[0],F(0)]
def energy(p,q):return sum(p[i]*A[i][j]*q[j] for i in range(2) for j in range(2))
e=[u[i]-galerkin[i] for i in range(2)]
e_best=[u[i]-euclidean[i] for i in range(2)]
assert energy(e,[F(1),F(0)])==0
assert sum(x*x for x in e)==F(5,36)
assert energy(e,e)==F(1,6)<energy(e_best,e_best)==F(2,9)
assert sqrt(float(sum(x*x for x in e)))<=3*sqrt(float(sum(x*x for x in e_best)))
print('Galerkin Euclidean error, best Euclidean error:',sqrt(5)/6,1/3)

noise=F(1,20)
def errors(N):
    tail2=pi*pi/6-sum(1/(k*k) for k in range(1,N+1))
    # Different worst-case noise eta=delta*e_N is allowed for each N.
    data=[F(1,k*k)+(noise if k==N else 0) for k in range(1,N+1)]
    recovered=[k*y for k,y in enumerate(data,1)]
    noise2=sum((recovered[k-1]-F(1,k))**2 for k in range(1,N+1))
    assert noise2==(N*noise)**2
    return sqrt(tail2),float(N*noise),sqrt(tail2+float(noise2))
for N in [2,4,8,16,32]:
    print('rank, bias, maximum noise amplification, exact worst-case error:',
          N,*(round(x,6) for x in errors(N)))
best=min(range(1,65),key=lambda N:errors(N)[2])
print('best rank among 1..64 for this envelope:',best)
```


## 확인 문제와 해설

1. N=16에서 witness e17의 오차는? **1입니다.** 여전히 완전히 생략합니다.
2. xₖ=1/k의 오차가 줄면 모든 단위 벡터에 같은 N이 충분한가요? **아니요.** N 뒤의 기저가 반례입니다.
3. Tₙ의 작은 전방 오차가 역의 안정성도 보장하나요? **아니요.** 전방 꼬리 노름은 줄지만 역 노름은 N으로 증가합니다.

추가 읽기: [Imperial Finite Element Course, Convergence](https://finite-element.github.io/L5_convergence.html).
