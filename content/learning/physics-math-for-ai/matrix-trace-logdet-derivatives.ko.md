## 행렬 방향의 변화는 선형 함수다

스칼라 손실을 행렬 A에 대해 미분할 때 dA는 행렬 모양의 방향이고 dL은 스칼라입니다. Frobenius 내적을 사용하면 gradient의 규약은

$$
dL=\langle\nabla_A L,dA\rangle_F=\operatorname{tr}((\nabla_A L)^T dA).
$$

형상을 먼저 쓰면 전치가 빠지는 오류를 줄일 수 있습니다. 같은 숫자 배열이라도 벡터화 순서와 내적 규약에 따라 표기 방식이 달라질 수 있습니다.

## Trace 이차 손실 미분

L(A)=tr(AᵀA)=ΣᵢⱼAᵢⱼ²를 미분하면

$$
dL=2\operatorname{tr}(A^T dA),\qquad\nabla_A L=2A.
$$

실습은 A=[[a,b],[b,c]]입니다. 자유 행렬의 gradient 원소는 각각 2Aᵢⱼ지만 b는 두 원소에 동시에 등장하므로 공유 매개변수 b의 도함수는 4b입니다. 대칭 행렬 공간의 미분과 자유 원소 공간의 미분을 같은 입력 형상으로 취급하지 마세요.

## Log-determinant의 정의역과 유도

여기서는 실수 SPD 행렬에서 log det A를 다룹니다. A+tE=A(I+tA⁻¹E)를 쓰고 det(I+tM)=1+t tr M+O(t²)를 적용하면

$$
D\log\det A[E]=\operatorname{tr}(A^{-1}E),\qquad\nabla_A\log\det A=A^{-T}.
$$

A가 대칭이면 A⁻ᵀ=A⁻¹입니다. 이 항등식의 일반 가역 행렬 방향 미분과 실수 logdet의 정의역을 구분해야 합니다. 실습은 SPD로 범위를 고정하며 log|det|나 복소 log의 가지 문제를 섞지 않습니다.

## 2×2 기준 계산

A=[[2,1],[1,2]]의 determinant는 3, 역행렬은 (1/3)[[2,−1],[−1,2]]입니다. E₀₀ 방향의 미분은 역행렬 첫 대각인 2/3입니다.

$$
\frac{\partial}{\partial a}\log(ac-b^2)=\frac c{ac-b^2}.
$$

중앙차분은 a를 ±h만큼 바꾼 두 행렬을 사용합니다. 양쪽이 모두 SPD일 때만 계산합니다. 특이 경계에서는 logdet와 도함수를 —로 표시합니다. 한쪽만 정의된 차분을 중앙차분처럼 쓰지 않습니다.

## 그림과 수치 안정성

그림의 가로축은 A₀₀에 더하는 t이고 세로축은 정의역 안의 logdet입니다. 정의되지 않은 점을 0으로 이어 그리지 않습니다. h=0의 차분도 —입니다. h를 줄이면 절단 오차가 줄어들다가 실제 계산에서는 반올림 오차가 지배할 수 있습니다.

특이점에 가까워지면 역행렬과 도함수의 크기가 커질 수 있습니다. 큰 값이 코드의 오류인지 문제의 조건수 때문인지 구분하세요. 큰 행렬에서는 Cholesky 분해·선형계 풀이 등 안정적인 구현을 사용하고 정의역 확인을 유지해야 합니다.

## 공분산 인자를 통한 미분

A=LLᵀ에서 L을 실수 d×d 정방 가역 행렬로 두겠습니다. 미분 가능한 스칼라 손실 ℓ(A)의 gradient를 G=∇_Aℓ로 놓으면 dA=dL Lᵀ+L dLᵀ입니다. Trace의 순환 성질로

$$
d\ell=\operatorname{tr}(G^T dL L^T)+\operatorname{tr}(G^T L dL^T)
=\operatorname{tr}\left(((G+G^T)L)^T dL\right).
$$

따라서 ∇_Lℓ=(G+Gᵀ)L입니다. SPD 정의역에서 logdet의 G=A⁻¹이므로

$$
\nabla_L\log\det(LL^T)=2A^{-1}L=2L^{-T}.
$$

L을 바꾸면 A의 두 L이 함께 변합니다. A에 대한 gradient를 L의 gradient로 그대로 복사하면 이 변화 경로를 놓치게 됩니다.

## Cholesky의 실제 자유 매개변수 구분

L이 양의 대각을 갖는 하삼각 행렬이면 logdet(LLᵀ)=2Σᵢlog Lᵢᵢ입니다. 따라서 자유 하삼각 성분의 도함수는 대각에서 2/Lᵢᵢ, 엄격한 하삼각 부분에서 0입니다. 앞의 전체 행렬 gradient에는 삼각 매개변수 공간 밖의 방향도 포함됩니다.

L=[[2,0],[1,3]]이면 A=[[4,2],[2,10]], det A=36이고

$$
2L^{-T}=\begin{pmatrix}1&-1/3\\0&2/3\end{pmatrix}.
$$

| 자유 매개변수 | 바뀌는 A의 성분 | logdet 도함수 |
|---|---|---|
| L₁₁ | A₁₁과 두 비대각 성분 | 1 |
| L₂₁ | 두 비대각 성분과 A₂₂ | 0 |
| L₂₂ | A₂₂ | 2/3 |

L₂₁을 바꾸면 공분산은 변하지만 대각 인자를 고정한 determinant는 그대로입니다. 전체 gradient의 오른쪽 위 −1/3은 dL₁₂에 곱해지는 값이고 삼각 방향에서는 dL₁₂=0입니다.

## Python으로 인자 도함수 재현

다음 코드는 A의 성분과 2×2 determinant를 직접 구성한 뒤 세 자유 인자 좌표의 중앙차분을 유도한 gradient와 비교합니다. 변형한 대각이 모두 양수이므로 SPD 정의역이 유지됩니다. 이 작은 검산에서는 직접 determinant를 계산하며 큰 시스템에서는 Cholesky 대각으로 logdet를 평가할 수 있습니다.

```python
from math import log

def logdet_from_factor(l11, l21, l22):
    assert l11 > 0 and l22 > 0
    a, b, c = l11**2, l11 * l21, l21**2 + l22**2
    return log(a * c - b * b)

point = [2.0, 1.0, 3.0]
assert abs(logdet_from_factor(*point) - 2 * log(6)) < 1e-12
h = 1e-5
for i, expected in enumerate([1.0, 0.0, 2 / 3]):
    plus, minus = point.copy(), point.copy()
    plus[i] += h
    minus[i] -= h
    difference = (logdet_from_factor(*plus) - logdet_from_factor(*minus)) / (2 * h)
    assert abs(difference - expected) < 1e-8
    print(i, difference, expected)
```

연습: L₁₁=exp(s), L₂₂=exp(t)로 매개화하면 두 logdet 도함수는 스칼라 연쇄법칙에 따라 모두 2입니다. 공분산 우도에는 다른 항도 있습니다. L₂₁의 logdet 도함수가 0이라는 사실만으로 전체 손실에서 그 매개변수가 중요하지 않다고 결론 내릴 수는 없습니다.

## AI 연결과 확인 문제

가우시안 우도, 공분산 추정, 엔트로피와 logdet 장벽에 이 미분이 나타납니다. 매개화 A=LLᵀ이면 L의 gradient는 연쇄법칙으로 별도로 구해야 하며 A의 gradient를 그대로 복사할 수 없습니다.

1. A=2I에서 ∂logdet/∂a는? **1/2**입니다.
2. b=1에서 공유 b의 d tr(AᵀA)/db는? **4**입니다.
3. SPD 경계를 넘는 중앙차분으로 정칙 도함수를 검산할 수 있나요? **아니요. 차분 정의역부터 실패합니다.**

추가 읽기: [MIT 공식 강좌 안내](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/). 강좌 범위를 확인해 보충 자료로 연결하며, 아래의 작은 예제와 논증은 직접 구성했습니다.
