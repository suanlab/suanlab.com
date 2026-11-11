## 대칭은 허용 선형 사상의 제약이 됩니다

입력·출력 벡터 작용이 같으면 모든 군 원소에 대해 Wg=gW여야 선형 층이 등변입니다. 이 행렬들의 선형 부분공간을 표현의 commutant라 합니다. 차원은 군 원소 개수만이 아니라 작용과 체에 의존합니다.

유한 군 Reynolds 연산자는 켤레 행렬의 평균입니다.

$$
P_G(W)=\frac1{|G|}\sum_{g\in G}g^{-1}Wg.
$$

유한 합의 index를 바꾸면 모든 h에서 h⁻¹P_G(W)h=P_G(W)입니다. 이미 허용된 W이면 각 항이 W이므로 P_G(W)=W이고 두 번 적용해도 같습니다. 반복 학습이나 근사 penalty가 아니라 대수적 사영입니다.

## 여기서 직교 사영인 이유

군 행렬이 직교라 켤레 작용이 Frobenius 내적을 보존합니다. 이를 평균하면 공통 고정 부분공간으로 직교 사영하므로 W−P_G(W)는 모든 허용 등변 행렬에 직교합니다. 따라서 Frobenius 거리로 가장 가까운 등변 행렬입니다.

실수·유한 군·직교 표현이라는 가정이 있습니다. compact 군에는 정규화 Haar 평균이 있지만 일반 noncompact 군에 같은 확률 측도를 가정할 수 없습니다. 어떤 군과 측도로 평균하는지 확인해야 합니다.

## 회전과 반사의 제약 차이

평면 C4에서 WJ=JW를 풀면 W=aI+bJ인 2차원 실수 공간입니다. 수평축 반사 S와도 교환하려면 b=0이라 D4에서는 scalar 1차원만 남습니다. 실수 C4 평면이 불가약인 사실과 모순되지 않습니다. Schur lemma를 단순히 scalar 항등이라고 적용하려면 계수 체 등의 가정을 확인해야 합니다.

## Shear 손계산

W=[[1,2],[0,1]]을 실제 군 평균하면 다음과 같습니다.

$$
P_{C_4}(W)=\begin{pmatrix}1&1\\-1&1\end{pmatrix},\qquad
P_{D_4}(W)=I.
$$

제곱 Frobenius 거리는 2와 4입니다. 파랑은 원래 단위원의 상, 주황은 사영 후 상입니다. 표에는 행렬·모든 원소의 최대 교환자 norm·멱등 오차가 있습니다. C4에서 D4로 바꾸는 것은 수치 허용 오차 조정이 아니라 실제 모형 제약 추가입니다.

## Bias와 다른 Feature 종류까지 제약 확장하기

입력·출력 표현이 다르면 선형 조건은 Wρᵢₙ(g)=ρₒᵤₜ(g)W입니다. ρₒᵤₜ(g)⁻¹Wρᵢₙ(g)를 평균하면 이 intertwiner 공간에 사영됩니다. 같은 index 변경 증명을 적용합니다. 보통 Frobenius metric에서의 직교성에는 직교 표현이 추가로 필요하며 일반 표현에서는 호환되는 불변 내적을 사용해야 합니다.

Affine 층 f(x)=Wx+b에는 가중치만 사영할 때 놓치기 쉬운 두 번째 조건이 있습니다.

$$
W\rho_{\rm in}(g)=\rho_{\rm out}(g)W,\qquad
\rho_{\rm out}(g)b=b.
$$

평면 C4 또는 SO2 벡터 출력에는 모든 회전에 고정되는 비영 bias가 없어 허용 bias가 0입니다. 반면 trivial scalar 출력은 어떤 scalar bias도 유지할 수 있습니다. b에 출력 군 작용을 평균하면 고정 벡터 사영을 얻습니다.

정확 계산 코드는 D4에서 shear 행렬과 비영 벡터 bias를 함께 사영한 뒤 모든 원소에서 affine 사상을 확인합니다. W를 완벽하게 사영해도 원래 bias를 유지하면 조건이 깨집니다. 뒤에 호환되는 방사형 비선형을 붙일 수 있지만 임의의 좌표별 비선형은 여전히 별도 검사해야 합니다.

| 출력 표현 | 고정 Bias 공간 | 예 |
|---|---:|---|
| Trivial scalar | 1차원 | 임의 scalar 상수 |
| 평면 C4 벡터 | 0차원 | b=0 |
| 평면 D4 벡터 | 0차원 | b=0 |

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
I=[[1,0],[0,1]]; R=[[0,-1],[1,0]]; S=[[1,0],[0,-1]]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def mv(A,x): return [sum(A[i][j]*x[j] for j in range(2)) for i in range(2)]
def tr(A): return list(map(list,zip(*A)))
rotations=[I]
for _ in range(3): rotations.append(mm(rotations[-1],R))
group=rotations+[mm(M,S) for M in rotations]
W=[[1,2],[0,1]]; b=[1,-2]
conjugates=[mm(mm(tr(g),W),g) for g in group]
projected=[[sum(F(M[i][j],8) for M in conjugates) for j in range(2)] for i in range(2)]
bias=[sum(F(mv(g,b)[i],8) for g in group) for i in range(2)]
assert projected==I and bias==[0,0]
for g in group:
    assert mm(projected,g)==mm(g,projected) and mv(g,bias)==bias
assert mv(R,b)!=b
print('projected weight:',projected,'projected bias:',bias)
```

## 직접 풀기와 풀이

W=diag(1,−1)은 trace 0이므로 D4 사영도 0입니다. 주황 상이 원점으로 줄고 모든 교환자가 0인지 확인하세요. 이어 (1,−1)을 π/2 회전해 ReLU 전후를 비교하면 다릅니다. 선형 층이 등변이어도 임의의 pointwise 비선형이 자동으로 SO2 등변은 아닙니다.

## 연구와의 연결

Steerable 층은 표현별 kernel·feature 결합 제약을 사용합니다. Reynolds 평균은 그 투명한 유한 예제입니다. 비선형·pooling·경계·이산화는 별도 검사해야 합니다. 행렬 하나의 사영이 네트워크 전체나 과제 대칭의 적합성을 인증하지 않습니다.

추가 읽기: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
