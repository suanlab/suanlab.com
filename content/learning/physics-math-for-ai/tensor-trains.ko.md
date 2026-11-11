## 여러 축을 사슬로 연결하기

차수 d인 배열은 축마다 n개의 좌표가 있으면 n의 d제곱 개 값을 가집니다. Tensor Train(TT)은 인접한 작은 코어 사이에 내부 지표를 두어 이 배열을 표현합니다. 원소 하나를 구할 때 물리 축의 좌표는 고정하고 내부 지표를 합산합니다.

$$
X_{i_1\cdots i_d}=\sum_{a_1,\ldots,a_{d-1}}G^{(1)}_{1i_1a_1}\cdots G^{(d)}_{a_{d-1}i_d1}.
$$

끝의 내부 차원은 1입니다. 각 내부 차원 r_k를 bond 크기라고 부릅니다. 물리 축 i_k의 크기와 혼동하지 마세요. 도식에서 바깥 다리는 물리 축이고 코어 사이 선이 bond입니다.

## 세 코어를 직접 수축하기

실습의 목표 배열은 X=e000+t e111입니다. bond를 2로 설정하면 두 경로가 두 꼭짓점을 각각 복원합니다. 중간 코어가 다른 경로끼리 섞이지 않도록 막습니다. bond를 1로 줄인 구성은 첫 꼭짓점만 표현합니다.

$$
X_{ijk}=\sum_{a,b}G^{(1)}_{1ia}G^{(2)}_{ajb}G^{(3)}_{bk1}.
$$

기준 t=1에서 bond 2는 오차 0, bond 1은 Frobenius 오차 1입니다. 이는 이 예제의 구성 결과이며, 모든 데이터를 임의로 자르면 같은 오차가 나온다는 규칙은 아닙니다.

## 게이지: 다른 코어, 같은 배열

첫 코어의 한 경로를 g배 하고 다음 코어의 같은 경로를 1/g배 하면 곱에서 상쇄됩니다. 실습에서 g를 1에서 2로 바꿔도 복원 표는 동일합니다. 코어 값 자체가 유일한 설명이라는 가정은 틀립니다.

$$
(AG)(G^{-1}B)=AB,\qquad G\text{ invertible}.
$$

현재 조절기는 이 관계의 대각 스케일 사례입니다. g는 양수이고 0을 허용하지 않습니다. 임의의 부동소수점 값에서는 미세한 반올림 잔차가 생길 수 있습니다.

## 설정한 크기와 최소 랭크

최소 TT bond 랭크는 앞의 k개 축을 행으로, 나머지를 열로 펼친 행렬의 랭크와 같습니다. 축의 순서에 따라 펼침이 달라집니다. t=0이면 두 내부 최소 랭크는 1이지만 코어를 크기 2로 할당해 둘 수 있습니다. 이때 저장량은 여전히 16개입니다. 최소 랭크와 할당 예산을 구분해야 합니다.

실습은 명시적으로 만든 코어를 수축하며, 일반 TT-SVD 알고리즘을 실행하지 않습니다. 실제 압축에서는 분해 오차와 축 순서, 코어 계산 비용을 함께 검토합니다.

## 각 절단의 최소 랭크를 증명하기

영 배열이 아닌 유한 X의 축 순서를 고정하세요. k번째 절단의 왼쪽 코어들을 L_k로, 오른쪽 코어들을 R_k로 수축하면 펼침 행렬은

$$
X^{\langle k\rangle}=L_kR_k,\qquad
\operatorname{rank}X^{\langle k\rangle}\le r_k.
$$

입니다. 모든 정확한 TT 표현에 필요한 하한입니다. 이 하한은 모든 절단에서 동시에 달성할 수 있습니다. 첫 펼침을 정확한 행렬 랭크로 U₁V₁로 분해하되 U₁의 열들이 독립이게 하고, U₁을 첫 코어로 재배열하세요. V₁의 행 지표를 남긴 bond와 둘째 물리 지표로 묶어 다시 행렬 분해합니다.

원래 둘째 펼침은 U₁⊗I와 재배열한 나머지의 곱입니다. U₁⊗I의 열들이 독립이므로 나머지의 랭크가 보존됩니다. 따라서 둘째 유지 랭크가 원래 둘째 펼침의 랭크와 정확히 같습니다. 이 논증을 반복하면 모든 코어를 얻고 각 절단의 등호를 증명합니다. 영 배열은 양의 bond 크기를 사용하는 별도 규약이 필요하며, 현재 실습의 목표는 항상 첫 꼭짓점이 1인 비영 배열입니다.

이 랭크는 표현한 배열 자체에 대한 값입니다. t≠0인 정확한 목표는 두 내부 절단에서 2가 필요하지만 bond 1로 구성한 근사 배열은 다른 랭크 1 배열입니다.

## TT 랭크 제한의 최적 근사가 존재하는 이유

양의 정수 랭크 상한 r_k를 고정하세요. 각 펼침 랭크가 r_k 이하인 배열의 집합은 닫혀 있습니다. 더 큰 랭크를 만드는 모든 행렬 소행렬식이 연속 다항식이고 0이어야 하기 때문입니다. 유한 절단들의 교집합도 닫혀 있습니다. 앞선 정확 분해 논증으로 이 집합을 해당 TT 랭크 제한과 식별합니다.

이 집합에는 영 배열이 포함됩니다. Frobenius 오차를 최소화하는 수열은 오차가 유계이게 선택할 수 있고 삼각부등식으로 배열 노름도 유계입니다. 유한 차원에서 수렴 부분수열이 존재합니다. 닫힘으로 극한도 허용 배열이고 노름의 연속성으로 하한을 달성합니다. 따라서 최적 근사가 존재합니다. 이는 표현 배열에 대한 결과이며 코어 매개변수 optimizer의 수렴이나 게이지에 따른 코어 값의 유계성을 보장하지 않습니다. [CP 최적 근사 부재 예제](/book/online/physics-math-for-ai/tensor-low-rank-approximation/)와 비교하세요.

## 축 순서를 바꾸어 랭크 계산하기

네 이진 지표에서 X_ijkl=δ_ik δ_jl로 두세요. 순서 (i,j,k,l)의 가운데 펼침은 4×4 항등행렬이므로 랭크 4입니다. 순서 (i,k,j,l)에서는 (1,0,0,1)ᵀ(1,0,0,1)의 바깥곱이어서 랭크 1입니다.

| 물리 축 순서 | 최소 내부 랭크 | 저장할 코어 스칼라 |
|---|---|---|
| (i,j,k,l) | (2,4,2) | 40 |
| (i,k,j,l) | (2,1,2) | 16 |

개수는 경계 bond가 1인 Σₖ nₖrₖ₋₁rₖ입니다. 밀집 배열은 16개이므로 이 표는 축 순서의 이득을 보여주지만 밀집 크기보다 항상 압축된다고 말하지 않습니다.

재배열한 경우의 명시적 코어는 G¹_{1ia}=δ_ia, G²_{ak1}=δ_ak, G³_{1jb}=δ_jb, G⁴_{bl1}=δ_bl입니다. a,b를 합하면 바로 δ_ikδ_jl을 얻습니다. 서로 연결된 지표를 함께 두므로 가운데 절단에 미해결 지표 쌍 둘을 모두 전달하지 않아도 됩니다.

## Python으로 모든 절단 재현

정확한 유리수 Gaussian 소거로 두 순서의 세 펼침 랭크를 계산합니다. 별도의 수축으로 제시한 네 코어를 16개 원소에서 전부 확인합니다. 이 코드는 별도의 4축 예제이며 대화형 실습은 기존 세 코어를 수축하고 일반 TT-SVD를 실행하지 않습니다.

```python
from fractions import Fraction as Q
from itertools import product

def matrix_rank(matrix):
    a = [[Q(x) for x in row] for row in matrix]
    pivot_row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(pivot_row,len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
        scale = a[pivot_row][col]
        a[pivot_row] = [x/scale for x in a[pivot_row]]
        for i in range(len(a)):
            if i != pivot_row:
                scale = a[i][col]
                a[i] = [x-scale*y for x,y in zip(a[i],a[pivot_row])]
        pivot_row += 1
        if pivot_row == len(a):
            break
    return pivot_row

def target(index):
    i,j,k,l = index
    return int(i == k and j == l)

def unfolding(order, cut):
    matrix = []
    for row in product(range(2),repeat=cut):
        values = []
        for col in product(range(2),repeat=4-cut):
            index = [0]*4
            for axis,value in zip(order,row+col):
                index[axis] = value
            values.append(target(index))
        matrix.append(values)
    return matrix

for order, expected, count in [((0,1,2,3), [2,4,2], 40),
                                ((0,2,1,3), [2,1,2], 16)]:
    ranks = [matrix_rank(unfolding(order,k)) for k in [1,2,3]]
    assert ranks == expected
    bonds = [1]+ranks+[1]
    assert sum(2*a*b for a,b in zip(bonds,bonds[1:])) == count
    print(order, ranks, count)

# Explicit reordered cores: delta(i,a), delta(a,k), delta(j,b), delta(b,l).
for index in product(range(2),repeat=4):
    i,j,k,l = index
    reconstructed = sum(int(i==a)*int(a==k)*int(j==b)*int(b==l)
                        for a,b in product(range(2),repeat=2))
    assert reconstructed == target(index)
```

출처 연결: [Oseledets의 Tensor-Train Decomposition](https://doi.org/10.1137/090752286) 정리 2.1과 따름정리 2.4에서 펼침 랭크 표현과 최적 근사의 존재를 확인했습니다. 이진 지표의 축 순서 예제와 정확 검산은 직접 구성했습니다. TT-SVD 절단 상한과 데이터에 따른 랭크 선택은 후속 심화입니다.

## 확인 문제

1. g를 바꾸면 목표 배열이 바뀌나요? **아니요. 인접 코어에서 상쇄됩니다.**
2. bond 1의 저장량은? **2+2+2=6개입니다.**
3. 내부 크기 2이면 최소 랭크도 항상 2인가요? **아니요. t=0이 반례입니다.**

추가 읽기: [Oseledets, Tensor-Train Decomposition (2011)](https://doi.org/10.1137/090752286). 본문의 작은 배열·복원·비용 예제는 직접 구성했습니다.
