## 점을 더 모으는 것과 경계를 바꾸는 것은 다르다

비균일 격자는 변화가 빠른 곳에 계산 위치를 집중시키는 방법입니다. 이 페이지에서는 xᵢ=(i/N)^r로 격자를 만들고 각 요소의 실제 길이를 사용합니다. 모든 간격에 하나의 평균 h를 대입하면 다른 연산자를 풀게 됩니다. 위 그림은 실제 mesh이고 점이 조밀한 위치를 직접 확인할 수 있습니다.

기준 함수는 u=x³+ax이며 source는 -6x입니다. Delta를 더하면 source가 -6x+δ가 됩니다. 두 Dirichlet, 왼쪽 Dirichlet·오른쪽 Neumann, 양쪽 Neumann을 나누어 실험합니다. 경계가 바뀌면 같은 source라도 문제가 달라집니다.

## 요소 행렬과 경계 제거

선형 FEM의 한 요소 길이를 hₑ라 하면 다음을 조립합니다.

$$
K_e=\frac1{h_e}\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad
K_{II}u_I=b_I-K_{IB}u_B.
$$

Dirichlet 값을 고정할 때 해당 행만 바꾸는 것으로 끝내지 않고 이웃 우변도 수정합니다. 자연 경계인 Neumann 값은 outward 도함수입니다. 왼쪽은 g_L=-u′(0)=-a, 오른쪽은 g_R=u′(1)=3+a입니다. 같은 좌표 방향의 미분을 양쪽에 그대로 넣으면 왼쪽 부호가 틀립니다.

Load는 요소별 두 점 Gauss로 적분하며 현재의 일차 source와 선형 hat 함수의 곱은 정확히 적분됩니다. 1차원 상수 계수 문제에서는 이 정확한 weak load로 선형 FEM의 절점 값이 정확해질 수 있습니다. 그렇다고 요소 내부의 선형 보간이 3차 곡선과 같지는 않습니다.

## 순수 Neumann의 존재와 유일성

$$
\int_0^1 f(x)\,dx+g_L+g_R=0,\qquad
K\mathbf1=0.
$$

첫 식은 호환성, 둘째 식은 상수를 더해도 같은 미분을 갖는 자유도입니다. 현재 source에서 첫 식의 값은 δ입니다. δ≠0이면 양쪽 flux와 내부 source가 맞지 않아 해가 없습니다. 실험은 이때 계산 해를 만들지 않고 파랑 곡선을 생략합니다. 주황은 원래 기준 함수이며 비호환 문제의 해라고 해석하면 안 됩니다.

호환성이 맞으면 u(0)=0이라는 gauge로 하나의 대표를 고릅니다. Gauge를 고정하는 것은 비호환 데이터를 고치는 일이 아닙니다. 원래 unconstrained stiffness의 nullity와 수정 후 선형 시스템을 구별해 읽으세요.

## Gauge가 비호환 Load를 해로 바꾸지는 못한다

연결된 1D mesh의 대칭 Neumann stiffness는 K1=0이고 range가 상수에 직교한다. 따라서 Σbᵢ=0이 필요하고 nullity 1이면 충분하다. 이 조건 이후에만 절점 하나를 고정해 대표를 선택한다. 비호환 load를 그대로 pin하면 버린 식이 빠진 수지를 떠안는다.

실제 source 섭동의 해석적 호환성 합은 δ다. 반올림한 구적 load가 이를 숨기거나 일반 tolerance가 허용해도 지정한 δ가 0이 아니면 거부하도록 수정했다. 독립 조립한 부동소수점 합은 계산 API에 남기지만, 알려진 이 데이터의 존재 기준으로 쓰지 않는다.

균일 source 보정 δ의 절점 load 가중치는 wᵢ=∫φᵢ다. 비균일 mesh에서 균일 벡터가 아니다. 절점 (0,1/4,1)에서는

| 절점 | 상수 Source의 Load 가중치 |
|---|---:|
| 0 | 1/8 |
| 1/4 | 1/2 |
| 1 | 3/8 |

```python
from fractions import Fraction as F
xs=[F(0),F(1,4),F(1)]
K=[[F(0) for _ in xs] for _ in xs];weights=[F(0)]*3
for j in range(2):
    h=xs[j+1]-xs[j]
    K[j][j]+=1/h;K[j+1][j+1]+=1/h
    K[j][j+1]-=1/h;K[j+1][j]-=1/h
    weights[j]+=h/2;weights[j+1]+=h/2
base=[x**3 for x in xs];delta=F(1,10**20)
rhs=[sum(a*b for a,b in zip(row,base))+delta*w for row,w in zip(K,weights)]
assert sum(rhs)==delta and weights==[F(1,8),F(1,2),F(3,8)]
a,b,c=K[1][1],K[1][2],K[2][2]
det=a*c-b*b
u=[F(0),(c*rhs[1]-b*rhs[2])/det,(a*rhs[2]-b*rhs[1])/det]
residual=[sum(x*y for x,y in zip(row,u))-r for row,r in zip(K,rhs)]
assert residual==[-delta,0,0]
print(weights,residual)
```

정확한 pinned 풀이는 생략한 식에 −δ 잔차를 남긴다. δw를 빼면 균일 source bias가 제거된다. 반면 Euclidean load 사영 b−δ1/3은 합을 0으로 만들더라도 다른 보정이다. 해 존재·gauge 선택·어느 물리 데이터를 고칠지는 서로 다른 결정이다.

## 손계산과 직접 풀기

a=1/2, δ=0이면 source 적분은 -3이고 outward 값은 -1/2와 7/2입니다. 합은 0이므로 pure Neumann 해가 존재하며 상수만큼 비유일합니다. u(0)=0을 선택하면 기준 함수 자체가 대표입니다.

직접 풀기: 같은 경계에 δ=0.2를 더하면 어떻게 되나요? 풀이: source 적분이 -2.8로 바뀌어 전체 합은 0.2입니다. 상수 이동은 미분이나 flux를 바꾸지 못하므로 gauge만으로 해결할 수 없습니다. Source나 경계 데이터 중 물리적으로 잘못된 것을 수정해야 합니다.

## AI 모델의 경계와 Gauge

PINN, mesh 기반 neural operator, 압력·퍼텐셜 추정에서는 경계 부호와 상수 자유도가 손실·식별 가능성에 영향을 줍니다. 관측 loss가 작아도 적분 호환성이 어긋날 수 있습니다. 연산자 잔차, 경계 weak 잔차와 실제 데이터의 존재 조건을 따로 검증하세요. 이 1D 결과를 고차원·가변 계수의 절점 정확성으로 일반화하지 마세요.

참고: [MIT 수치 PDE 강의](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
