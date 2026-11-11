## 남는 지표와 합산하는 지표

수축은 지정한 지표를 따라 성분을 곱하고 합하는 연산입니다. X의 shape가 (2,2,2)이고 v가 길이 2이면 마지막 축과의 수축은

$$
Y_{ij}=\sum_{k=0}^1X_{ijk}v_k.
$$

i,j는 자유 지표로 출력에 남고 k는 합산 지표로 사라집니다. 출력 shape는 (2,2)입니다. 원소별 곱이라면 합산축이 남기 때문에 다른 연산입니다. Einstein 표기에서는 반복 지표로 합을 생략할 수 있지만 자유 지표가 식의 양쪽에서 맞는지 확인해야 합니다.

## 고정된 예제의 손계산

실습은 Xᵢⱼₖ=1+4i+2j+k, v=(1,2)를 사용합니다. 즉 원본 평탄 값은 1부터 8입니다.

$$
Y_{00}=1\cdot1+2\cdot2=5,\qquad
Y_{01}=3\cdot1+4\cdot2=11.
$$

나머지는 17,23입니다. 표에서 각 자유 좌표와 값을 비교하세요. 출력 네 원소마다 곱셈 두 번과 덧셈 한 번이 필요하므로 직접 구현 기준 곱셈 8회·덧셈 4회입니다. 이 작은 연산 수는 CPU·GPU 실행 시간 측정이 아닙니다.

## 어느 축을 합하는지가 중요하다

첫 축 i를 합하면 남는 순서는 (j,k)이고 출력은 (11,14,17,20)입니다. shape는 여전히 (2,2)이지만 숫자는 달라집니다. 축 크기가 우연히 같으면 잘못된 축을 수축해도 shape 오류가 나지 않아 더 위험합니다.

$$
Z_{jk}=\sum_{i=0}^1v_iX_{ijk}.
$$

실습은 합산하지 않는 축을 원래 오름차순으로 남깁니다. 다른 라이브러리가 출력 축을 다른 순서로 배치하는지 명시적 출력 지표로 확인하세요.

## 기저와 쌍대의 역할

추상적으로 자연스러운 수축은 벡터와 공변벡터, 즉 쌍대공간 사이의 pairing입니다. 기저 변환 때 둘의 성분이 적절히 반대로 변해야 스칼라 또는 남은 텐서가 일관되게 변환됩니다. Euclidean 직교 기저에서는 내적으로 벡터를 공변벡터와 식별해 익숙한 dot product처럼 계산할 수 있습니다.

$$
\langle v^*,x\rangle=\sum_i v_i^*x_i.
$$

이 단원은 고정된 실수 Euclidean 기저를 사용합니다. 복소 공간의 Hermitian pairing에는 켤레가 필요하고, 임의 곡선 좌표에서는 metric 성분까지 확인해야 합니다. 단순 배열의 repeated index를 모든 좌표계에서 자동으로 불변인 식이라고 해석하지 마세요.

## 기저를 바꿔도 스칼라가 유지되는 이유

이전 좌표가 x=(2,1)ᵀ인 실수 벡터와 행 좌표 α=(3,4)인 쌍대벡터를 생각하면 수축 αx는 10입니다. 새 기저의 열이 가역 S에 대해 E′=ES이면 같은 벡터와 선형 함수의 새 좌표는

$$
x'=S^{-1}x,\qquad \alpha'=\alpha S,\qquad
\alpha'x'=\alpha SS^{-1}x=\alpha x.
$$

입니다. 쌍대벡터는 벡터에 작용하므로 성분이 반대 방식으로 변합니다. 전단 S=[[1,1],[0,1]]을 선택하면 새 기저는 직교정규가 아닙니다.

| 양 | 이전 좌표 | 새 좌표 |
|---|---|---|
| 벡터 x | (2,1)ᵀ | (1,1)ᵀ |
| 쌍대벡터 α | (3,4) | (3,7) |
| 유클리드 gradient 벡터 g | (3,4)ᵀ | (−1,4)ᵀ |
| 계량 행렬 | I | [[1,1],[1,2]] |
| 자연스러운 수축 | 10 | 10 |

이전 행 α를 x′에 그대로 사용하면 7입니다. 새 gradient 벡터를 보통 좌표 내적으로 x′와 곱하면 3입니다. 둘 다 불변 수축이 아닙니다.

## Gradient 벡터와 쌍대벡터 구분하기

스칼라 함수 f(x)=3x₁+4x₂의 α=df는 미분입니다. 이전 유클리드 gradient는 g=(3,4)ᵀ입니다. 새 기저에서는 계량과 gradient가

$$
M'=S^TS,\qquad g'=S^{-1}g=(M')^{-1}(\alpha')^T,
\qquad (g')^TM'x'=\alpha'x'=10.
$$

을 만족합니다. 실제로 (g′)ᵀM′=gᵀS=αS입니다. x′의 좌표 제곱합은 2이지만 물리적 길이 제곱은 여전히 5입니다. 직교정규가 아닌 좌표에서 벡터와 쌍대벡터를 식별할 때 계량이 필요한 이유입니다. 앞선 [gradient 단원](/book/online/physics-math-for-ai/gradients/)에서는 최급강하로 같은 계량 관계를 유도했습니다.

## Python으로 좌표 변환 검산

정확한 분수로 역행렬·새 쌍대벡터·계량 gradient·pairing을 별도의 행렬 곱으로 대조합니다. 이 예제는 2차원 공간의 좌표를 바꾸며 위의 기존 3축 수축 실습은 기저를 고정합니다.

```python
from fractions import Fraction as Q

def transpose(a):
    return [list(row) for row in zip(*a)]

def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def inverse(a):
    x, y = a[0]
    z, w = a[1]
    determinant = x*w-y*z
    assert determinant != 0
    return [[w/determinant, -y/determinant], [-z/determinant, x/determinant]]

s = [[Q(1), Q(1)], [Q(0), Q(1)]]
x = [[Q(2)], [Q(1)]]
alpha = [[Q(3), Q(4)]]
x_new = multiply(inverse(s), x)
alpha_new = multiply(alpha, s)
metric = multiply(transpose(s), s)
g_new = multiply(inverse(metric), transpose(alpha_new))
assert x_new == [[1], [1]] and alpha_new == [[3, 7]]
assert g_new == [[-1], [4]]
assert multiply(alpha, x) == multiply(alpha_new, x_new) == [[10]]
assert multiply(transpose(g_new), multiply(metric, x_new)) == [[10]]
assert multiply(transpose(g_new), x_new) == [[3]]
assert multiply(alpha, x_new) == [[7]]
assert multiply(transpose(x_new), multiply(metric, x_new)) == [[5]]
print(x_new, alpha_new, g_new, metric)
```

연구 과제: 전단 대신 다른 가역 행렬을 선택하여 pairing의 불변성을 확인하세요. 고차 수축에서는 자유 슬롯과 합산 슬롯을 각각의 공간·쌍대공간 규약에 따라 변환해야 합니다. 배열 shape의 일치만으로 변환 법칙을 보장할 수는 없습니다.

## AI에서의 연산식

attention에서 QᵢₐKⱼₐ를 a에 대해 합하면 점수 행렬 Sᵢⱼ가 됩니다. batch나 head가 추가되면 그 축을 남기는지, 서로 섞어 합하는지 명시해야 합니다. 같은 einsum 문자열이라도 입력 축의 이름을 잘못 대응하면 의도와 다른 모델이 됩니다.

그림은 이산 출력 원소 값이며 표는 남는 좌표를 적습니다. v=0이면 출력이 영배열이 되지만 shape가 사라지는 것은 아닙니다.

## 확인 문제와 답

1. X가 (3,4,5),v가 (5,)이고 마지막 축을 수축하면? **(3,4)**입니다.
2. v=(0,0)이면? **(2,2) 영배열**입니다.
3. 모든 크기가 2일 때 축 선택 오류를 shape만으로 잡나요? **아니요. 실제 원소와 지표 의미를 검산해야 합니다.**

추가 읽기: [Kolda–Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf). 본문의 작은 배열·복원·비용 예제는 직접 구성했습니다.
