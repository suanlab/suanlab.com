## 좌표 성분 자체가 길이를 정하지는 않습니다

같은 물리 벡터도 기저에 따라 다른 성분을 가집니다. 정규직교 Cartesian 좌표에서는 성분 제곱합이 길이 제곱이지만 늘이거나 shear한 좌표에서는 metric 행렬이 필요합니다. 성분 변화와 벡터 자체의 변화를 혼동하면 기하 알고리즘이 틀어집니다.

행렬식 1의 가역 선형 좌표 x=A u를 사용합니다.

$$
A=\begin{pmatrix}s&k\\0&1/s\end{pmatrix},\qquad
v^u=A^{-1}v^x,\qquad g_u=A^TA
=\begin{pmatrix}s^2&sk\\sk&k^2+s^{-2}\end{pmatrix}.
$$

s>0은 배율, k는 shear입니다. 그림은 항상 물리 x 축입니다. 파란 선분은 물리 공간에 나타낸 u 기저인 A의 열이며 주황은 vˣ=(a,b)입니다. A를 바꾸면 기저가 바뀌고 a,b를 고정하면 실제 벡터는 유지됩니다.

## 비정규직교 예제를 손으로 계산하기

s=2,k=1,vˣ=(1,1)이면 vᵘ=(−0.5,2), g=[[4,2],[2,1.25]]입니다. 새 성분의 제곱합은 4.25이지만 올바른 노름은 다음과 같습니다.

$$
(v^u)^Tg_uv^u=(v^x)^Tv^x=2.
$$

g vᵘ=(2,1.5)를 구해 (−0.5,2)와 내적하면 −1+3=2입니다. 수치 적합의 우연이 아니라 좌표 일관성입니다. 비영 w에 대해 wᵀg w=||Aw||²>0이므로 metric은 양의 정부호입니다. 이 행렬족의 det g=1이지만 행렬식 1만으로 기저가 정규직교인 것은 아닙니다.

## 미분과 Gradient는 성분 변환이 다릅니다

f(x)=x₁+2x₂의 물리 covector 성분은 αₓ=(1,2)입니다. 연쇄법칙으로 αᵤ=Aᵀαₓ이지만 gradient 벡터에는 inverse metric이 필요합니다.

$$
\alpha_u=A^T\alpha_x,\qquad
(\operatorname{grad} f)^u=g_u^{-1}\alpha_u
=A^{-1}(1,2)^T.
$$

앞 예제에서 αᵤ=(2,2), gradient 성분은 (−1.5,4)입니다. 서로 연관되어도 tensor 종류가 다릅니다. αᵤ·vᵘ=3은 물리 좌표의 a+2b와 같습니다. αᵤ를 그대로 벡터로 쓰면 잘못된 metric을 사용한 셈입니다.

## 평평한 평면도 비선형 좌표에서는 Connection이 0이 아닐 수 있습니다

평면 전체의 가역 매끄러운 좌표 x=u, y=v+u²를 택하세요. Jacobian과 유도 metric은 다음과 같습니다.

$$
J=\begin{pmatrix}1&0\\2u&1\end{pmatrix},\qquad
g=J^TJ=\begin{pmatrix}1+4u^2&2u\\2u&1\end{pmatrix},\qquad
\Gamma^v_{uu}=2.
$$

나머지 Christoffel 기호는 모두 0입니다. 계수는 좌표 기저의 변화를 나타내므로 평면은 여전히 평평합니다. 실제 직선 (x,y)=(t,0)은 새 좌표에서 (u,v)=(t,−t²)입니다. 좌표 가속도가 0이 아니어도 v''+2(u')²=0으로 geodesic 방정식을 만족합니다. Connection 계수 자체는 tensor가 아니지만 curvature는 tensor입니다.

f=x+2y에서 df 성분은 (1+4u,2), metric으로 올린 gradient는 (1,2−2u)입니다. 후자에 J를 곱하면 물리 gradient (1,2)를 복구합니다. 정확분수 코드는 세 점에서 이 항등식과 길이 불변성을 검사합니다. 상수 Jacobian 실습의 개념을 확장한 별도 예제이며 제어 항목에 비선형 chart를 추가했다고 해석하면 안 됩니다.

| u | df 성분 | Gradient 성분 |
|---:|---|---|
| 0 | (1,2) | (1,2) |
| 1 | (5,2) | (1,0) |
| 2 | (9,2) | (1,-2) |

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
for u in map(F,[0,1,2]):
    g = [[1+4*u*u,2*u],[2*u,1]]
    alpha, grad = [1+4*u,2], [1,2-2*u]
    assert [sum(g[i][j]*grad[j] for j in range(2)) for i in range(2)] == alpha
    assert [grad[0],2*u*grad[0]+grad[1]] == [1,2]
    v = [F(1),1-2*u]
    assert sum(v[i]*g[i][j]*v[j] for i in range(2) for j in range(2)) == 2
    print(u, tuple(alpha), tuple(grad))
```

## 직접 풀기와 풀이

s=1,k=1,vˣ=(1,0)에서 vᵘ=(1,0), αᵤ=(1,3), gradient 성분 (−1,2)를 구하세요. 마지막 값에 A를 곱하면 물리 gradient (1,2)입니다. pairing과 벡터 제곱 노름은 모두 1입니다. shear만 바꿔 반복해도 물리 스칼라는 유지됩니다.

## 불변 학습과의 연결

기하 특징·latent 거리·최적화 갱신에는 tensor 종류와 변환 규칙을 밝혀야 합니다. 비선형 chart에서는 A 대신 국소 Jacobian을 사용하고 위치에 따라 metric도 달라집니다. 이 유한 상수 Jacobian 예제는 대수적 일관성을 확인할 뿐 임의의 학습 표현이 metric이나 위상을 보존한다고 증명하지 않습니다. 좌표 독립성은 이름을 붙이면 생기는 성질이 아니라 유도하고 검사할 성질입니다.

추가 읽기: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
