## 무엇을 근사하는지 먼저 정하기

저랭크 근사는 적은 수의 성분으로 원본을 얼마나 잘 표현하는지 묻습니다. 오차를 논하려면 랭크의 종류와 노름을 정해야 합니다. 이 실습은 서로 직교하는 두 좌표 바깥곱을 사용하고, 원소별 제곱합으로 정의되는 Frobenius 노름을 씁니다.

$$
X=a\,e_0\otimes e_0\otimes e_0+b\,e_1\otimes e_1\otimes e_1.
$$

기준 a=3,b=1에서는 첫 성분이 더 큽니다. 막대와 표는 원본의 모든 여덟 값 및 선택한 성분으로 복원한 값을 보여줍니다. 축 좌표를 연속 공간의 곡선처럼 보간하지 않습니다.

## 직교 성분의 오차를 계산하기

두 좌표 성분은 내적이 0입니다. 따라서 버린 성분의 제곱 가중치를 더하면 오차 제곱이 됩니다. 첫 성분만 남기면 오차는 1입니다.

$$
\|X\|_F^2=a^2+b^2,\qquad \|X-\widehat X\|_F^2=\sum_{s\text{ discarded}}\lambda_s^2.
$$

선택은 부호가 아닌 절댓값 크기 순입니다. 음수 성분도 큰 정보를 담을 수 있습니다. 절댓값이 같은 경우 실습은 좌표 순으로 고르지만, 이것이 유일한 최적 표현이라는 뜻은 아닙니다.

## 보존 에너지와 영 배열

$$
E_{\mathrm{retained}}=\frac{\sum_{s\text{ retained}}\lambda_s^2}{a^2+b^2}.
$$

기준에서 한 성분을 보존하면 9/10=0.9, 둘 다 보존하면 1입니다. 둘 다 버리면 원본 노름이 오차가 됩니다. 원본이 영 배열이면 분모도 0이므로 비율은 정의되지 않습니다. 실습은 이 경우 숫자 1을 꾸며 넣지 않고 비율을 비워 표시합니다. 절대 오차는 여전히 0입니다.

## 행렬 정리를 적용할 수 있는 범위

일반 텐서의 CP 근사는 행렬 SVD와 다릅니다. 이 예제는 펼침 행렬의 특잇값이 |a|,|b|이므로 행렬의 랭크 1 근사 하한과 실제 구성의 오차가 일치합니다. 그래서 여기서는 큰 성분을 남기는 선택이 최적입니다. 이 근거를 임의의 텐서에 옮겨서는 안 됩니다.

일반 차수 3 이상 텐서는 고정 CP 랭크 이하 집합이 닫혀 있지 않아 최적 근사가 존재하지 않는 경우도 있습니다. Tucker나 TT의 절단은 각 형식에 맞는 별도의 오차 분석이 필요합니다. 실제 모델 압축에서는 출력 오차가 작은지와 작업 성능이 유지되는지도 추가로 평가해야 합니다.

## 랭크 2 수열이 랭크 3으로 수렴하는 예

제약 없는 실수 CP 랭크를 사용합니다. e₀,e₁을 R²의 좌표 기저로 두고 2×2×2 배열의 Frobenius 노름을 사용하겠습니다. 실수 ε≠0에 대해 다음을 정의하세요.

$$
W=e_1\otimes e_0\otimes e_0+e_0\otimes e_1\otimes e_0+e_0\otimes e_0\otimes e_1,
\qquad
T_\varepsilon=\frac{(e_0+\varepsilon e_1)^{\otimes3}-e_0^{\otimes3}}{\varepsilon},\quad\varepsilon\ne0.
$$

두 번째 식은 CP 항 두 개를 사용하므로 CP 랭크가 2 이하입니다. k=0 단면은 [[0,1],[1,ε]]이고 determinant가 −1이므로 랭크는 정확히 2입니다. 곱을 전개하면 원하는 세 항과, e₁이 두 개인 계수 ε의 세 항, 계수 ε²의 마지막 항을 얻습니다. 좌표 항의 직교성으로

$$
\|T_\varepsilon-W\|_F^2=3\varepsilon^2+\varepsilon^4\longrightarrow0.
$$

입니다. 최적 근사 실패를 증명하려면 W 자체의 CP 랭크가 2일 수 없다는 점도 보여야 합니다.

## 두 단면으로 극한의 CP 랭크 증명

셋째 지표를 고정하면 W의 단면은 S₀=[[0,1],[1,0]], S₁=[[1,0],[0,0]]입니다. W에 두 항 CP 표현이 있다면 두 단면을 2×2 인자 U,V와 대각 행렬 D_k로 S_k=U D_k Vᵀ라 쓸 수 있습니다. S₀가 가역이므로 U,V,D₀도 가역입니다. 따라서

$$
S_1S_0^{-1}=U D_1D_0^{-1}U^{-1}
\quad\text{would be diagonalizable},\qquad
S_1S_0^{-1}=\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$

이어야 합니다. 실제 행렬은 0이 아니지만 제곱이 0입니다. 고유값이 모두 0이고 대각화까지 가능하면 영 행렬이어야 하므로 모순입니다. 따라서 W의 랭크는 2보다 크고 이미 세 항 표현이 있으므로 정확히 3입니다.

T_ε가 W로 가므로 랭크 2 이하 근사 오차의 하한은 0입니다. 하지만 이 하한을 달성하는 허용 텐서는 없습니다. 특정 optimizer의 실패가 아니라 이 목표의 최적 근사가 존재하지 않는다는 증명입니다. CP 랭크를 제한한 집합은 닫혀 있지 않습니다. 랭크 2 폐포에는 W가 있습니다. Border rank도 정확히 2입니다. S₀ 단면의 determinant는 −1인데, CP 랭크 1 이하인 수열의 단면 determinant는 0이고 극한에서도 보존되므로 border rank 1은 불가능합니다.

## 답은 유계인데 개별 항은 커지는 현상

두 CP 항의 노름은 각각 (1+ε²)³ᐟ²/|ε|, 1/|ε|입니다. 차이는 노름 √3인 텐서로 수렴하지만 각 항은 커집니다. 인자 내부의 스케일을 바꿔도 완전한 랭크 1 항의 노름 증가가 사라지지는 않습니다.

| ε | 근사 오차의 제곱 | 두 번째 CP 항의 노름 |
|---|---|---|
| 1/2 | 13/16 | 2 |
| 1/8 | 193/4096 | 8 |
| 1/32 | 3073/1048576 | 32 |

수치 소거 예제로 모든 모드에 H=[[1,1],[1,−1]]을 적용하세요. 가역 변환이므로 CP 랭크는 유지됩니다. 극한 성분은 3−2(i+j+k)입니다. ε=2⁻⁵⁶에서 binary64는 각 인자 1±ε를 모두 1로 반올림하므로 두 바깥곱을 직접 빼면 영 배열이 됩니다. 정확한 산술에서는 여전히 (3,1,1,−1,1,−1,−1,−3)에 가까운 성분을 얻습니다.

## Python으로 수열과 소거 재현

다음 코드는 정확한 오차, 멱영 단면 곱, 변환한 성분 식을 대조합니다. 이어 정확한 분수와 명시적으로 계산한 binary64 인자를 비교합니다. 대화형 실습은 랭크 1 최적해가 존재하는 앞의 직교 두 꼭짓점 예제를 유지합니다.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod

indices = list(product(range(2), repeat=3))
w = [Q(sum(index) == 1) for index in indices]

def outer(v):
    return [prod(v[i] for i in index) for index in indices]

for epsilon, expected in [(Q(1,2), Q(13,16)),
                          (Q(1,8), Q(193,4096)),
                          (Q(1,32), Q(3073,1048576))]:
    approximation = [(a-b)/epsilon for a, b in zip(outer([1,epsilon]), outer([1,0]))]
    squared_error = sum((a-b)**2 for a, b in zip(approximation, w))
    assert squared_error == expected == 3*epsilon**2 + epsilon**4
    print(epsilon, squared_error, 1/epsilon)

s0 = [[0,1],[1,0]]
s1 = [[1,0],[0,0]]
def multiply(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
j = multiply(s1,s0)  # s0 is its own inverse
assert j == [[0,1],[0,0]]
assert multiply(j,j) == [[0,0],[0,0]]

epsilon = Q(1,2**56)
exact, raw_float, limit = [], [], []
for index in indices:
    signs = [1 if i == 0 else -1 for i in index]
    factors = [1+epsilon*s for s in signs]
    exact.append((prod(factors)-1)/epsilon)
    raw_float.append((prod(float(a) for a in factors)-1)/float(epsilon))
    limit.append(sum(signs))
assert raw_float == [0.0]*8
assert all(abs(float(a)-b) < 1e-12 for a,b in zip(exact,limit))
print('raw binary64', raw_float)
print('transformed limiting tensor', limit)
```

출처 연결: [de Silva와 Lim의 arXiv v2](https://arxiv.org/abs/math/0607647v2) 명제 4.6과 §4.3은 랭크 3 극한 구성과 발산 계수를 다룹니다. 위 단면 곱 증명과 binary64 계산은 교재의 직접 계산입니다. 연구에서는 복원 오차와 함께 항의 노름·초기값도 기록하세요. 모든 CP 문제에 해가 없다는 주장은 아닙니다.

## 확인 문제

1. a=−3,b=1이면 무엇을 남기나요? **절댓값이 큰 −3 성분입니다.**
2. 원본이 0이면 보존 비율은? **0/0으로 정의되지 않습니다.**
3. 두 성분을 모두 남기면? **이 예제의 복원 오차는 0입니다.**

추가 읽기: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf). 본문의 작은 배열·복원·비용 예제는 직접 구성했습니다.
