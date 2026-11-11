## 분산된 두 표현이 같은가

Alice와 Bob이 각각 8비트 표현 x,y를 가지고 있다고 합시다. Alice는 한 번만 메시지를 보내고 Bob이 두 값이 같은지 결정합니다. 입력 분포 가정은 없고 공개 프로토콜은 사전에 고정합니다. 계산 시간보다 **전송하는 비트 수**를 묻는 문제입니다.

원문을 모두 보내면 8비트로 정확히 판정합니다. k<8비트의 결정론 메시지로 항상 정확할 수는 없습니다. 256개 입력을 2ᵏ개 메시지로 보내면 같은 메시지를 내는 서로 다른 x,x′가 있습니다. Bob의 입력이 x일 때 두 경우의 메시지는 같은데 필요한 판정은 달라 모순입니다.

$$
m\ge\log_2 256=8\quad\text{for exact deterministic one-way equality}.
$$

## 단순 절단과 난수 지문

낮은 k비트만 보내면 x=0,y=128은 k=3에서 모두 000입니다. 이를 같다고 판정하면 거짓 양성입니다. 이웃 거리나 의미 유사성을 보존하는 표현 문제와도 다릅니다. 현재 과제는 비트열의 정확한 동일성 하나입니다.

공개된 균등 8비트 행 r와 x의 GF(2) 내적을 전송해 봅시다. 차이 δ=x XOR y가 0이 아니면 r·δ는 공정한 비트입니다. δ에서 1인 한 위치를 택하면 그 r 비트를 뒤집는 일대일 대응이 내적 0과 1인 행을 짝짓기 때문입니다.

$$
P_r[r\cdot\delta=0]=1/2\quad(\delta\ne0).
$$

## 독립한 행과 오류 확률

공유한 k개의 독립 균등 행으로 k비트 지문을 만듭니다. 서로 다른 고정 입력 쌍이 모든 행에서 충돌할 확률은 2⁻ᵏ입니다. 같은 입력은 항상 같은 지문이므로 거짓 음성은 없습니다. 이 보장은 난수를 보기 전에 고정한 입력 쌍과 독립 공개 난수에 대한 것이며, 행을 본 뒤 공격자가 입력을 고르는 다른 모형에는 그대로 적용할 수 없습니다.

$$
P[H(x)=H(y)\mid x\ne y]=2^{-k}.
$$

k=3이면 이상 충돌 확률은 1/8입니다. k=0이면 아무 메시지도 보내지 않고 항상 같다고 하므로 서로 다른 쌍의 오류가 1입니다. k=8의 무작위 행도 선형 종속일 수 있어 충돌 확률 1/256이지만, 8비트 원문 전송은 정확합니다.

## 실습의 고정 seed가 뜻하는 것

실습의 파랑은 이상적인 공개 난수 모형의 고정 쌍 충돌 확률, 주황은 결정론 절단의 최악 경우 오류입니다. 서로 다른 평가 기준이므로 같은 표본의 두 오류율로 읽지 마세요. 같은 입력을 선택하면 파란 곡선은 0이지만, 절단의 전체 입력 쌍에 대한 최악 경우 곡선은 그대로입니다. 표의 공개 mask는 재현 가능한 작은 생성 규칙으로 만든 **한 실행 예시**입니다. 이 고정 seed의 행들이 독립 균등하다는 보장은 붙이지 않습니다.

공유 난수 모형에서는 mask 전송 비용을 따로 세지 않습니다. 실제 분산 시스템에서 mask를 공유·갱신해야 하면 그 통신도 포함해야 합니다. 학습한 임베딩 차원 수만으로 네트워크 전송량이나 정확 동일성 보장을 추론할 수 없습니다.

## 고정 행렬을 본 뒤에는 null space가 중요하다

공개 행을 쌓은 H∈GF(2)ᵏˣⁿ의 rank를 r라고 합시다. 지문은 Hx이고, 충돌 차이는 Hδ=0인 null space입니다. Rank-nullity로 그 차원은 n−r이고 원소 수는 2ⁿ⁻ʳ입니다. 각 지문 fiber도 비어 있지 않으면 이 크기를 가집니다. 따라서 k<n이면 항상 비영 충돌 차이가 있습니다. 행을 공개한 뒤 공격자가 입력 쌍을 고르는 경우에는 이 차이를 골라 충돌을 만들 수 있습니다.

$$
\#\ker H=2^{n-r},\qquad
P_H[H\delta=0]=2^{-k}\text{ for each fixed }\delta\ne0.
$$

두 번째 식의 입력 차이는 난수 H와 독립적으로 먼저 고정해야 합니다. 첫 번째 식은 H 하나를 고정한 뒤의 성질입니다. 두 양화 순서를 바꾸어 “작은 k로 모든 쌍을 동시에 정확히 구분한다”고 결론내리지 않습니다.

## 2×2 행렬 16개를 전수조사하기

n=k=2로 줄이면 두 공개 행은 각각 00,01,10,11 중 균등하게 골라져 전체 행렬은 16개입니다. 고정된 비영 δ마다 충돌 행렬은 4개입니다. 반면 어떤 비영 δ든 충돌하는 행렬은 rank<2인 10개입니다.

| rank | 행렬 수 | 행렬 하나의 kernel 크기 |
|---|---|---|
| 0 | 1 | 4 |
| 1 | 9 | 2 |
| 2 | 6 | 1 |

![16개의 이진 2×2 공개 행렬에서 고정 차이 δ=01의 충돌 4개와 어떤 비영 차이라도 충돌하는 특이 행렬 10개를 별도 격자로 비교하는 그림.](/assets/learning/coding-fingerprint-quantifiers-ko.svg)

예를 들어 두 행이 모두 01이면 x=00과 y=10이 충돌합니다. 코드가 GF(2) 행 소거와 모든 비영 차이를 직접 검사합니다. 이는 화면의 8비트 고정 seed와 별개인, 이상적 균등 난수의 완전 열거입니다.

n×n 독립 균등 이진 행렬이 full rank일 확률은 행을 차례로 고를 때 이전 span을 피할 확률의 곱, ∏j=0ⁿ⁻¹(1−2ʲ⁻ⁿ)입니다. n=8에서는 약 .289919에 불과합니다. 고정 쌍의 오류 1/256과 모든 쌍에서 정확할 확률 .289919는 다른 주장입니다. Identity 행렬을 공유하고 8비트를 보내면 모든 쌍을 정확히 구분할 수 있지만 무작위 행렬은 항상 identity처럼 가역적이지 않습니다.

**직접 풀기:** k=3,n=8인데 H가 rank 3이면 kernel에는 몇 차이가 있나요? **해설:** 2⁵=32개이고 그중 31개가 비영입니다. 특정 차이의 평균적 충돌 확률이 작다는 사실과 양립합니다.

배경: [Stanford CS369E Lecture 3, §1의 public-coin equality](https://theory.stanford.edu/~tim/w15/l/l3.pdf). 위 rank 분포와 격자는 직접 열거한 것입니다.

```python
from itertools import product
from collections import Counter
from math import prod

def dot(row,value): return (row&value).bit_count()%2

def rank(rows):
    basis={}
    for row in rows:
        while row:
            pivot=row.bit_length()-1
            if pivot in basis: row^=basis[pivot]
            else:
                basis[pivot]=row
                break
    return len(basis)

matrices=list(product(range(4),repeat=2))
counts=Counter(rank(rows) for rows in matrices)
assert counts=={0:1,1:9,2:6}
for delta in [1,2,3]:
    collisions=sum(all(dot(row,delta)==0 for row in rows) for rows in matrices)
    assert collisions==4
singular=sum(rank(rows)<2 for rows in matrices)
assert singular==10
print('2x2 rank histogram:',sorted(counts.items()))
print('fixed nonzero difference collision, some difference collision:',4/16,singular/16)
rows=[1,1]
kernel=[delta for delta in range(4) if all(dot(row,delta)==0 for row in rows)]
assert kernel==[0,2]
print('rank-one rows 01,01: kernel',kernel)
full_rank_8=prod(1-2**(j-8) for j in range(8))
print('8x8 ideal random full rank probability:',round(full_rank_8,6))
```


## 확인 문제와 해설

1. 내적 행 256개를 모두 검사하면 비영 차이의 충돌 행은? **128개입니다.** 앞의 비트 반전 짝짓기로 정확히 반반입니다.
2. 지문이 같으면 원문도 같은가요? **보장하지 않습니다.** 충돌이 가능하며 같은 지문은 후보 필터로 쓸 수 있습니다.
3. 공개 난수는 공짜 물리 자원인가요? **아니요.** 이는 통신 복잡도 모형의 공유 가정입니다. 실제 배포의 동기화 비용은 별도입니다.

추가 읽기: [Stanford CS369E, Randomized Communication Complexity](https://theory.stanford.edu/~tim/w15/l/l3.pdf).
