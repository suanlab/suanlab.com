## 위상도는 가능한 여러 초기 상태를 정리한다

이차원 자율계 z'=Az의 위상도는 상태 평면의 벡터장과 궤적을 보여줍니다. 성분을 시간에 대해 그리는 것과 다릅니다. 위 그림은 가로 x·세로 y의 물리 축척을 같게 하고, 아래 그림은 x(t)·y(t)를 따로 표시합니다.

파란 화살표는 읽기 쉽도록 정규화한 실제 장의 방향이며 길이는 속도가 아닙니다. 주황 궤적은 정확한 행렬 흐름이고 점은 선택한 시간의 위치입니다. 방향·기하 경로·시간 매개화를 구별해서 읽으세요.

## 회전과 팽창은 역할이 다르다

행렬의 두 행이 (alpha,-beta), (beta,alpha)이면 다음입니다.

$$
z(t)=e^{\alpha t}
\begin{pmatrix}\cos\beta t&-\sin\beta t\\
\sin\beta t&\cos\beta t\end{pmatrix}z(0),
\qquad \|z(t)\|=e^{\alpha t}\|z(0)\|.
$$

음의 alpha는 반지름을 줄이고 양수는 키웁니다. Beta는 회전 속도를 바꿉니다. Beta가 0이 아닐 때 alpha=0은 center이며, 가까운 궤적이 가까이 남아도 원점으로 수렴하지 않습니다. Lyapunov 안정성과 점근 안정성은 다릅니다. Beta=0이면 나선이 아니라 방사 방향의 repeated node가 됩니다.

초기 반지름 1, alpha=-0.3이면 t=1의 반지름은 exp(-0.3)입니다. Alpha=0이면 1로 유지됩니다. 닫힌 궤적을 수렴으로 오해하지 않도록 비교하세요.

## 안정 방향 하나와 안정 고정점은 다르다

두 번째 모형은 diag(alpha,-beta)입니다. 두 계수가 양수이면 x는 팽창하고 y는 수축하는 saddle입니다.

$$
x(t)=x_0e^{\alpha t},\qquad y(t)=y_0e^{-\beta t}.
$$

x0=0에 정확히 놓인 궤적은 원점으로 수렴합니다. 하지만 임의로 작은 x 섭동은 결국 커지므로 고정점 전체는 안정하지 않습니다. 안정성은 선택한 시뮬레이션 궤적 하나가 아니라 초기점의 근방 전체를 정량화하는 개념입니다.

## 평형이 고립되었는지도 확인하기

고정점은 Az=0의 모든 해입니다. Determinant가 0이 아니면 원점 하나뿐입니다. 영 고유값이 있으면 평형 직선이 생길 수 있고 A=0이면 모든 점이 평형입니다. 수치는 비고립 상태를 sink·source·center·saddle과 구별합니다. 비선형 계에서는 실수부 0의 고유값 때문에 고차항 분석이 필요할 수 있습니다.

## 안정한 고유값과 과도 증폭은 양립한다

실습의 회전·팽창 행렬은 normal이므로 Euclidean 반지름이 eᵅᵗ를 따른다. 모든 안정 행렬이 이렇게 단순하지는 않다. A=[[-1,K],[0,−1]], K≥0의 두 고유값은 −1이지만 초기값 (0,1)의 해는 e⁻ᵗ(Kt,1)이다. K=4이면 t=1의 노름이 초기 노름보다 크다. 점근 감쇠와 Euclidean 노름의 단조 감소는 다른 질문이다.

적절한 이차 에너지를 찾으면 모순이 사라진다.

$$
P=\begin{pmatrix}1/2&K/4\\K/4&1/2+K^2/4\end{pmatrix},
\quad A^TP+PA=-I,\quad \frac d{dt}(z^TPz)=-\|z\|_2^2.
$$

첫 leading minor가 1/2, determinant가 1/4+K²/16이므로 P는 양의 정부호다. Euclidean 반지름이 커질 때도 이 에너지는 줄어든다. 선형계에서 고유값 부호는 점근 안정성을 정하고, 노름과 고유벡터 조건성은 과도 증폭·강건성에 영향을 준다.

![동일 시각에서 비교한 안정 비정규 선형계 세 개의 Euclidean 반지름과 과도 증폭.](/assets/learning/differential-transient-ko.svg)

| K | 시간 1의 반지름 | 점근 안정 |
|---|---|---|
| 0 | e⁻¹ | 성립 |
| 2 | √5/e | 성립 |
| 4 | √17/e | 성립 |

```python
from fractions import Fraction as F
from math import exp, hypot
def multiply(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
for K in (0, 2, 4):
    A = [[F(-1), F(K)], [F(0), F(-1)]]
    P = [[F(1, 2), F(K, 4)], [F(K, 4), F(1, 2)+F(K*K, 4)]]
    AT = list(map(list, zip(*A)))
    lhs, rhs = multiply(AT, P), multiply(P, A)
    assert [[lhs[i][j]+rhs[i][j] for j in range(2)] for i in range(2)] == [[-1, 0], [0, -1]]
    radius = exp(-1)*hypot(K, 1)
    if K == 4: assert radius > 1
    print(K, radius, 'det(P):', P[0][0]*P[1][1]-P[0][1]**2)
```

유리수 행렬 계산으로 수치 미분 없이 Lyapunov 항등식을 검산한다. 실제 실습의 구조화된 두 행렬에서는 계수 자체로 고립성을 판정하도록 수정했다. 작은 비영 계수의 determinant가 underflow로 0이 되어도 반올림 진단값으로만 남기며, 실제 고립 원점을 평형점 연속체로 바꾸지 않는다.

## 직접 풀기: 안정축을 조금 벗어나기

대각 모형 alpha=0.5, beta=1, 초기 (0,1)은 t=1에서 반지름 e^-1로 작아집니다. x0를 0.01로 바꾸면 x가 1이 되는 시간은 log(100)/0.5입니다. 그 전에 끝나는 짧은 시뮬레이션이 불안정성을 숨길 수 있는 이유를 설명하세요. 두 계수를 모두 0으로 바꿔 전체 평형 집합도 구하세요.

더 읽기: [MIT 일차 시스템·위상도](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/).
