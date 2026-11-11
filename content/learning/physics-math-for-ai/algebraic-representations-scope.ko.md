## 대칭을 행렬로 표현하기

군 표현은 군 원소를 가역 선형사상으로 보내며 합성을 행렬곱으로 보존합니다. 여기서는 실수 평면의 90도 회전 J와 x축 반사 S를 사용합니다. 숫자로 표현한다는 의미의 일반 임베딩보다 강한 조건입니다. J의 네제곱이 항등행렬이고 SJS가 J의 역행렬인지 확인하면 정사각형 군 관계를 볼 수 있습니다.

$$
\rho(gh)=\rho(g)\rho(h),\qquad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad S=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

## 선형층의 등변 조건

같은 표현을 입력과 출력에 사용하면 선형층 M은 MJ=JM이어야 회전 등변입니다. 일반적인 입력·출력 표현이 다를 때는 Mρin(g)=ρout(g)M입니다. 이 실습은 같은 2차원 표현만 다룹니다. 기준 M의 행은 (2,1),(0,1)이며 회전 교환자의 Frobenius 노름은 2입니다. 0이 아닌 잔차는 모든 입력에서 등변이라는 주장이 실패함을 뜻합니다.

$$
[M,J]=MJ-JM,\qquad \|[M,J]\|_F=0\iff MJ=JM.
$$

## 평균으로 제약을 만족시키기

네 회전의 켤레 작용을 평균하면 회전과 교환하는 행렬이 됩니다. 프로그램은 실제 네 행렬곱을 수행해 평균합니다. M=[[a,b],[c,d]]일 때 결과는 대각이 (a+d)/2이고 오른쪽 위가 (b−c)/2, 왼쪽 아래가 그 음수입니다. 기준 결과는 (1.5,0.5),(−0.5,1.5)입니다. 원래 행렬과 사영 결과를 원소별 표에서 비교하세요.

$$
P_{C_4}(M)=\frac14\sum_{k=0}^3J^{-k}MJ^k.
$$

## 추가 대칭과 연구 주장의 범위

회전에 등변인 행렬이 반사와도 교환하는 것은 아닙니다. 반사 조건을 추가한 D4 사영은 (a+d)/2배 항등행렬입니다. 두 사영의 차이는 선택한 군이 모델의 허용 자유도를 바꾼다는 것을 보여줍니다. 행렬 하나의 제약을 만족했다고 비선형 활성함수나 전체 네트워크가 자동으로 등변이 되지는 않습니다. 연속 회전군·다른 표현·유한체의 특성은 별도 검토가 필요합니다.

$$
P_{D_4}(M)=\frac{a+d}{2}I.
$$

## 평균이 가장 가까운 등변 행렬인 이유

NJ=JN을 성분별로 풀면 C4와 교환하는 행렬은 N=[[α,β],[−β,α]]입니다. NS=SN까지 요구하면 β=0이므로 D4의 경우 αI만 남습니다. 두 집합은 실수 행렬공간의 선형 부분공간입니다. 현재 사용하는 직교 표현에서 켤레 작용은 Frobenius 내적을 보존합니다. g의 켤레 작용의 수반은 g⁻¹의 켤레 작용이고, 역원은 군 원소를 순열하므로 군 평균은 자기수반입니다. 합의 순서를 바꾸면 평균 결과가 군과 교환함을 알 수 있고, 이미 교환하는 행렬을 다시 평균하면 그대로입니다. 따라서 평균 연산자는 멱등이고 자기수반이며, 교환하는 행렬공간으로의 직교 사영입니다.

등변 행렬 N에 대해 잔차 M−P(M)는 P(M)−N과 수직입니다. 거리의 제곱을 전개하면 가장 가까운 행렬이라는 성질과 유일성을 얻습니다.

$$
\|M-N\|_F^2=\|M-P(M)\|_F^2+\|P(M)-N\|_F^2.
$$

| 요구하는 군 | 기준 M의 사영 | M에서의 제곱 거리 |
|---|---|---|
| C4 | (3/2,1/2), (−1/2,3/2) | 1 |
| D4 | (3/2,0), (0,3/2) | 3/2 |

대칭을 추가하면 허용하는 행렬 집합과 필요한 최소 수정량이 함께 바뀝니다. 비직교 표현에서도 켤레 평균이 대수적 사영일 수는 있지만 통상적인 Frobenius 거리를 최소화한다고 보장되지는 않습니다. 유한군 평균에서 |G|로 나눌 수 있어야 한다는 조건은 [모듈의 표수 반례](../modules-linear-actions/)와 연결됩니다.

## 가중치가 교환해도 활성함수는 별도 조건이 필요하다

모든 작용과 교환하는 항등 가중치 뒤에 좌표별 ReLU를 적용합시다. v=(1,−1)에서 J를 먼저 적용한 뒤 ReLU를 취하면 (1,1)입니다. ReLU를 먼저 적용한 뒤 J를 취하면 (0,1)입니다. 따라서 이 비선형 층은 C4 등변이 아닙니다. 학습 오차나 수치 근사 없이도 실패가 드러납니다.

$$
\operatorname{ReLU}(Jv)=(1,1)\ne(0,1)=J\operatorname{ReLU}(v).
$$

아핀 층 Mv+b에는 편향이 출력 작용에 고정되어야 한다는 조건도 추가됩니다. 현재 자연스러운 2차원 C4 작용에서 고정 벡터는 영벡터뿐이므로 b=0이어야 합니다. 다른 표현에는 영이 아닌 고정 벡터가 있을 수 있어 모든 표현에서 편향이 0이어야 한다는 뜻은 아닙니다. 합성된 층의 등변성도 각 구성 사상이 지정한 입력·출력 작용을 보존할 때 따라옵니다.

## 불변 스칼라 gate로 비선형 등변 사상 만들기

불변 스칼라로 등변 벡터의 크기를 조절할 수 있습니다. P를 C4 행렬 사영으로 두고 다음의 유계 방사형 인자를 사용합니다.

$$
F(v)=\frac{P(M)v}{1+\|v\|^2},\qquad
F(gv)=\frac{P(M)gv}{1+\|gv\|^2}=gF(v)\quad(g\in C_4).
$$

증명에는 P(M)의 교환 성질과 g의 직교성이 모두 필요합니다. 기준 행렬과 v=(1,2)에서는 출력이 (5/12,5/12)입니다. 이 구성은 C4 등변이지만 반사까지 보존할 필요는 없습니다. D4 사영을 쓰면 같은 구성도 D4 등변입니다. 스칼라 gate는 불변인 제곱 노름의 다른 함수로 바꿀 수 있습니다. 이는 구체적인 층 설계 원리이며 특정 학습 모형의 일반화 성능 근거는 아닙니다.

## 정확한 분수로 사영과 비선형 사상 검산하기

다음 표준 라이브러리 코드는 네 개 또는 여덟 개 켤레 작용을 독립적으로 평균하고, 거리 항등식과 ReLU의 실패를 확인한 뒤 세 입력에 대해 모든 군 원소에서 방사형 구성을 검사합니다. 현재 행렬 실습은 기존 사영을 계산하며 아래 비선형 예제는 별도로 실행하는 연습입니다.

```python
from fractions import Fraction as Q

I, J, S = (1, 0, 0, 1), (0, -1, 1, 0), (1, 0, 0, -1)
M = (2, 1, 0, 1)

def mul(a, b):
    return tuple(sum(a[2*i+k]*b[2*k+j] for k in range(2))
                 for i in range(2) for j in range(2))

def transpose(a):
    return (a[0], a[2], a[1], a[3])

def apply(a, v):
    return tuple(sum(a[2*i+j]*v[j] for j in range(2)) for i in range(2))

def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))

def norm2(a):
    return sum(x*x for x in a)

def project(a, group):
    terms = [mul(mul(transpose(g), a), g) for g in group]
    return tuple(Q(sum(t[i] for t in terms), len(group)) for i in range(4))

rotations = [I]
for _ in range(3):
    rotations.append(mul(rotations[-1], J))
dihedral = rotations + [mul(g, S) for g in rotations]
C, D = project(M, rotations), project(M, dihedral)
assert C == (Q(3, 2), Q(1, 2), Q(-1, 2), Q(3, 2))
assert D == (Q(3, 2), 0, 0, Q(3, 2))
assert norm2(sub(M, C)) == 1 and norm2(sub(M, D)) == Q(3, 2)
for group, P in [(rotations, C), (dihedral, D)]:
    assert project(P, group) == P
    assert all(mul(P, g) == mul(g, P) for g in group)
    assert norm2(M) == norm2(P) + norm2(sub(M, P))
for alpha, beta in [(0, 0), (2, 1), (Q(1, 3), Q(-2, 5))]:
    N = (alpha, beta, -beta, alpha)
    assert norm2(sub(M, N)) == norm2(sub(M, C)) + norm2(sub(C, N))
relu = lambda v: tuple(max(0, x) for x in v)
v = (1, -1)
assert relu(apply(J, v)) == (1, 1)
assert apply(J, relu(v)) == (0, 1)

def radial(a, v):
    return tuple(Q(x) / (1+norm2(v)) for x in apply(a, v))

for group, P in [(rotations, C), (dihedral, D)]:
    for v in [(1, 2), (1, -1), (0, 0)]:
        for g in group:
            assert radial(P, apply(g, v)) == apply(g, radial(P, v))
v = (1, 2)
assert radial(C, v) == (Q(5, 12), Q(5, 12))
assert radial(C, apply(S, v)) != apply(S, radial(C, v))
print("projections:", C, D, "squared distances:", 1, Q(3, 2))
print("radial output:", radial(C, v), "ReLU counterexample:", (1, 1), (0, 1))
```

## 추가 문제와 연구 연결

1. 왜 P(M)가 유일하게 가장 가깝나요? **거리 항등식의 둘째 항은 음이 아니고 N=P(M)일 때만 0이기 때문입니다.**
2. 좌표별 ReLU는 언제나 등변성을 깨나요? **좌표 순열과는 교환하는 등 반례가 있으므로 활성함수 이름 대신 실제 작용을 검사해야 합니다.**
3. 비직교 행렬에도 방사형 gate의 증명이 그대로 적용되나요? **사용하는 스칼라 특징이 그 작용에 불변이지 않으면 노름을 이용한 논증이 실패합니다.**

군 표현과 유한군 평균은 [Etingof 외, §§3.1·3.6](https://math.mit.edu/~etingof/replect.pdf)을 참고하세요. 위의 거리 및 비선형 예제는 이 페이지의 실수 정사각형 작용에 대해 직접 유도했습니다.

## 확인 문제와 해설

1. 항등행렬은 회전과 반사에 등변인가요? **예. 둘과 교환합니다.**
2. 회전 사영 뒤 반사 잔차가 비영일 수 있나요? **예. 기준 예제가 그렇습니다.**
3. 0행렬은 등변인가요? **예. 하지만 유용한 예측 능력을 보장하지 않습니다.**

추가 읽기: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
