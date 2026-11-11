## 함수 공간에서도 최선 근사를 찾는다

실수 Hilbert 공간 H의 닫힌 선형 부분공간 V에 대해 모든 f는 유일하게 f=v+r, v∈V, r⊥V로 분해됩니다. v=Pᵥf가 직교 투영이며 f에 가장 가까운 V의 원소입니다. 존재 보장에는 완비성과 부분공간의 닫힘이 필요합니다.

$$
\|f-w\|^2=\|f-P_Vf\|^2+\|P_Vf-w\|^2,\quad w\in V.
$$

우변 두 번째 항이 비음수이므로 최적성·유일성이 동시에 나옵니다. 이는 유한 차원 최소제곱의 함수 공간 버전이지만 노름과 내적을 먼저 정해야 합니다.

## Legendre 기저로 적분을 손으로 계산하기

L2[−1,1]에서 P₀=1, P₁=x, P₂=(3x²−1)/2는 직교하며 제곱 노름은 각각 2, 2/3, 2/5입니다. 정규직교 기저가 아니므로 계수를 구할 때 분모를 빼먹으면 안 됩니다.

$$
c_k=\frac{\langle f,P_k\rangle}{\|P_k\|^2},\qquad f=x^2+a x=\frac13P_0+aP_1+\frac23P_2.
$$

차수 1까지 투영하면 1/3+a x, 잔차는 x²−1/3입니다. a=1일 때 대상 제곱 노름은 16/15, 투영은 8/9, 잔차는 8/45이며 합이 정확히 맞습니다. 상수만 보존하면 선형 성분의 2/3도 잔차에 남아 오차가 38/45가 됩니다.

## 점별 일치보다 직교 조건을 보기

실습의 파랑은 대상, 주황은 선택한 부분공간으로의 투영입니다. 표의 잔차 내적은 보존한 모든 기저에서 0입니다. 보존하지 않은 기저에는 0이 아닐 수 있습니다. 질의점 하나에서 두 함수가 우연히 같아도 전체 L2 오차가 0이라는 뜻은 아닙니다.

차수 2까지 보존하면 대상 자체를 포함하므로 오차는 0입니다. 이 예제에서 계수 절단이 직교 투영이 되는 이유는 기저가 직교하기 때문입니다. 일반 비직교 기저에서는 Gram 행렬을 포함한 연립방정식이 필요합니다.

## 닫힘과 연구 적용

다항식 전체는 L2에 조밀하지만 닫히지 않았습니다. 비다항 L2 함수는 다항식으로 오차를 0에 가깝게 할 수 있어도 그 집합 안에서 최솟값을 달성하지 못할 수 있습니다. 반면 현재 유한 차수 공간은 유한 차원이어서 닫혀 있습니다.

Galerkin과 reduced-order 모형은 비슷한 부분공간 아이디어를 사용합니다. PDE의 에너지 내적이나 비대칭 bilinear form을 쓸 때는 현재 L2 투영과 같은 결론이 자동으로 성립하지 않으므로 안정성 조건을 따로 확인합니다.

## 최소화 열이 실제 투영으로 수렴하는 이유

닫힌 선형 부분공간 V와 d=infv∈V‖f−v‖를 두고, ‖f−v_n‖→d인 열을 고릅니다. 중점도 V에 있으므로 평행사변형 법칙은 다음 Cauchy 추정을 줍니다.

$$
\|v_n-v_m\|^2
=2\|f-v_n\|^2+2\|f-v_m\|^2
-4\left\|f-\frac{v_n+v_m}{2}\right\|^2
\le2\|f-v_n\|^2+2\|f-v_m\|^2-4d^2\to0.
$$

H의 완비성과 V의 닫힘으로 극한 v가 V 안에 존재하고 거리 d를 달성합니다. 임의 w∈V에 대해 ‖f−v−tw‖²가 t=0에서 최소이므로 그 일차항이 0, 즉 ⟨f−v,w⟩=0입니다. 이 직교성에서 앞 Pythagoras 식과 유일성을 얻습니다. 단순히 잔차가 어떤 점에서 0인지로 존재·최적성을 정한 것이 아닙니다.

## 기저를 바꿔도 부분공간의 최적 함수는 같다

같은 V=span{1,x}를 비직교 기저 φ₀=1, φ₁=1+x로 표현해 봅시다. 미지 계수 c는 Gc=b를 풀어야 하며 Gᵢⱼ=⟨φⱼ,φᵢ⟩, bᵢ=⟨f,φᵢ⟩입니다. 현재 f=x²+a x에서는

$$
G=\begin{pmatrix}2&2\\2&8/3\end{pmatrix},\quad
b=\begin{pmatrix}2/3\\2/3+2a/3\end{pmatrix},\quad
c=\begin{pmatrix}1/3-a\\a\end{pmatrix}.
$$

따라서 c₀+c₁(1+x)=1/3+a x로 Legendre 결과와 같습니다. 반대로 내적 b를 그대로 계수로 쓰면 Gram의 상관과 척도를 무시합니다.

| 방법, a=1 | 계수 | L² 잔차 제곱 |
|---|---|---|
| 직교 {1,x} | (1/3,1) | 8/45 |
| 비직교 {1,1+x}, Gc=b | (−2/3,1) | 8/45 |
| 비직교, b를 그대로 사용 | (2/3,4/3) | 784/135 |

코드는 다항식 곱을 정확분수로 적분해 Gram·직교 잔차·오차를 대조합니다. 기저의 가역 변경은 최적 함수를 바꾸지 않지만 수치 조건수는 바꿀 수 있습니다. 실제 고차 근사에서는 Gram을 무시하는 것뿐 아니라 부정확한 선형계 풀이도 별도 오차 원인입니다.

**직접 풀기:** 조밀하지만 닫히지 않은 V에도 최소화 열의 극한이 V 안에 있나요? **해설:** 보장하지 않습니다. 완비 공간 H 안에서 극한을 얻더라도 V의 닫힘이 빠지면 밖으로 나갈 수 있습니다.

```python
from fractions import Fraction as F

def inner(p,q):
    return sum(a*b*F(1-(-1)**(i+j+1),i+j+1)
               for i,a in enumerate(p) for j,b in enumerate(q))

basis=[[F(1),F(0),F(0)],[F(1),F(1),F(0)]]
f=[F(0),F(1),F(1)]  # x^2+x
G=[[inner(p,q) for q in basis] for p in basis]
rhs=[inner(f,p) for p in basis]
det=G[0][0]*G[1][1]-G[0][1]*G[1][0]
coeff=[(rhs[0]*G[1][1]-rhs[1]*G[0][1])/det,
       (G[0][0]*rhs[1]-G[1][0]*rhs[0])/det]
assert coeff==[F(-2,3),F(1)]
def residual(c):
    return [f[j]-sum(c[i]*basis[i][j] for i in range(2)) for j in range(3)]
r=residual(coeff)
assert all(inner(r,p)==0 for p in basis)
assert inner(r,r)==F(8,45)
assert inner(residual(rhs),residual(rhs))==F(784,135)
print('Gram matrix:',G)
print('correct coefficients, error squared:',coeff,inner(r,r))
print('raw-inner-product coefficients, error squared:',rhs,inner(residual(rhs),residual(rhs)))
```


## 확인 문제와 해설

1. a=0, 상수 투영의 오차는? **8/45입니다.** 없어진 선형 성분이 처음부터 0입니다.
2. 차수 1 투영의 잔차는 P₂에 직교한가요? **아니요.** 내적은 4/15입니다.
3. 정규직교가 아닌 기저에서 내적만 계수로 써도 되나요? **아니요.** 제곱 노름 또는 Gram 행렬을 반영해야 합니다.

추가 읽기: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
