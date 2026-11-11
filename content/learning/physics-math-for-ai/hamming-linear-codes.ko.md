## 유한체에서 패리티를 계산하기

GF(2)는 0과 1을 원소로 가지며 덧셈은 XOR, 곱셈은 AND와 같습니다. 따라서 1+1=0입니다. 이 산술을 사용하면 “비트 합이 짝수인가?”라는 검사를 행렬 곱으로 표현할 수 있습니다. 실수 위의 최소제곱과 달리 거리나 역행렬을 실수 연산으로 계산해서는 안 됩니다.

Hamming(7,4)는 정보 4비트에 패리티 3비트를 더합니다. H의 열은 위치 1~7의 이진수이며, 아래 행 순서는 낮은 자리부터입니다.

$$
H=\begin{pmatrix}1&0&1&0&1&0&1\\0&1&1&0&0&1&1\\0&0&0&1&1&1&1\end{pmatrix}.
$$

## 정보에서 부호어로

정보 비트를 위치 3,5,6,7에 낮은 비트부터 놓고 패리티 위치 1,2,4를 정합니다. 각 검사 행과 부호어 c의 내적이 GF(2)에서 0이어야 합니다. 유효 부호어 집합은 H의 영공간이며 차원 4입니다. 생성행렬 G를 쓰면 c=Gm, HG=0으로도 표현합니다.

정보 정수 9는 낮은 자리부터 (1,0,0,1)입니다. 이를 넣으면 위치 1부터 부호어는 0011001입니다. 위치 3이 뒤집힌 수신열의 syndrome은 H의 세 번째 열, 즉 (1,1,0)입니다. 낮은 자리부터 해석하면 정수 3이므로 그 위치를 다시 뒤집습니다.

$$
\begin{aligned}
r&=c+e,\\s&=Hr=He\pmod 2.
\end{aligned}
$$

## 왜 한 오류를 고칠 수 있는가

H의 모든 열이 서로 다른 비영 벡터입니다. 따라서 가중치 1의 유효 부호어도, 가중치 2의 유효 부호어도 없습니다. 반면 1번 열 XOR 2번 열은 3번 열이므로 위치 1,2,3에 1인 벡터는 영공간에 있습니다. 최소 거리는 정확히 3이고 보장 정정 능력은 1입니다.

$$
d_{\min}=3,\qquad t=\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor=1.
$$

## 검사 통과와 올바른 복원의 차이

같은 정보 9에서 위치 1과 2를 뒤집으면 syndrome이 다시 3입니다. 복호기는 한 오류로 가정하고 위치 3을 뒤집어 다른 유효 부호어로 이동합니다. 복원 정보는 8이며, 정정 뒤 syndrome은 0입니다. **패리티 통과는 원래 메시지라는 증명이 아닙니다.** 원래 Hamming 부호는 최대 두 오류를 검출할 수 있지만 이 단일 오류 정정기는 비영 syndrome을 모두 정정하려 하므로 두 오류를 잘못 고칠 수 있습니다.

실습의 파랑은 전송, 주황은 수신 비트입니다. 표에서 정정 결과도 비교하세요. 실제 시스템의 SECDED는 전체 패리티 등을 더해 단일 정정과 이중 검출을 구분하지만 현재 7비트 실험에는 포함하지 않습니다.

## 완전 부호: 모든 수신열에 정확히 한 중심

H의 열 1,2,4는 GF(2) 표준기저이므로 rank H=3이고 영공간 차원은 7−3=4입니다. 영공간은 2⁴=16개의 부호어입니다. 최소 거리 3이므로 각 중심과 일곱 단일 오류를 포함한 반경 1 공들은 서로 겹치지 않습니다. 그 크기를 모두 더하면 16(1+7)=128=2⁷로 전체 수신 공간과 같습니다. 이것이 Hamming(7,4)가 완전 부호라는 뜻입니다.

![Hamming(7,4)의 서로 다른 16개 부호어를 중심으로 한 16개 공이 각각 중심 1개와 단일 오류 이웃 7개를 포함하여 128개 수신열을 중복 없이 덮는 그림. 각 중심은 위치 1부터 읽는다.](/assets/learning/coding-hamming-perfect-cover-ko.svg)

Syndrome은 각 수신열을 8개 coset 중 하나로 보냅니다. 같은 syndrome의 두 벡터는 차이가 영공간에 있으므로 같은 coset이고, 각 coset의 크기는 16입니다. Syndrome 0의 대표는 0, syndrome j≠0의 대표는 위치 j의 단일 오류입니다. 따라서 syndrome 복호는 이 유일한 반경 1 중심을 고르는 최근접 복호와 같습니다.

## 네 비트 payload의 블록 오류를 공정하게 비교하기

이 복호기는 실제 오류 가중치가 0 또는 1일 때만 원래 부호어로 돌아갑니다. 2 이상인 오류 e는 복호 후 e에 0 또는 단일 오류를 더하므로 0일 수 없습니다. 결과는 다른 유효 부호어이고, 정보 추출은 부호어와 4비트 payload 사이의 일대일 대응이라 payload도 다릅니다. 독립 BSC에서는

$$
P_{e,\mathrm{Ham}}=1-(1-\varepsilon)^7-7\varepsilon(1-\varepsilon)^6.
$$

비교 대상을 4개 정보 비트로 맞춥시다. 각 비트를 세 번 보내는 방법은 12회 사용하고, 그 4개 독립 블록 중 하나라도 틀릴 확률은 1−(1−Pₑ,rep3)⁴입니다. Hamming은 7회 사용합니다. 단일 비트 오류 .028과 네 비트 블록 오류 .1496944를 곧바로 비교하면 기준이 다릅니다.

| ε | Hamming 블록 오류, 7회 | 반복 블록 오류, 12회 |
|---|---|---|
| .01 | .002031042 | .001191467 |
| .05 | .044380542 | .028686147 |
| .1 | .149694400 | .107383193 |

코드는 H의 영공간을 직접 열거해 공의 중복이 없고 128개 수신을 덮는지 확인하며, 모든 가중치의 오류에 대한 성공 조건도 검사합니다. 그림과 전수조사는 이 작은 모형의 정확한 결과입니다. 일반적인 큰 부호의 효율적인 최적 복호를 보장하지는 않습니다.

**직접 풀기:** 최근접 복호 결과의 syndrome이 0이면 실제 오류가 한 개였나요? **해설:** 아닙니다. 모든 수신열이 어떤 반경 1 공에 속하므로 복호기는 항상 유효 부호어를 만듭니다. 실제 전송 중심과 같다는 보장은 별도입니다.

```python
from fractions import Fraction as F
from math import comb

def syndrome(word):
    value=0
    for j in range(1,8):
        if (word>>(j-1))&1: value^=j
    return value

code=[word for word in range(128) if syndrome(word)==0]
assert len(code)==16
owner={}
for center in code:
    for error in [0]+[1<<j for j in range(7)]:
        received=center^error
        assert received not in owner
        owner[received]=center
assert len(owner)==128
for received in range(128):
    s=syndrome(received)
    corrected=received^(1<<(s-1) if s else 0)
    assert corrected==owner[received]
for weight in range(8):
    for error in range(128):
        if error.bit_count()!=weight: continue
        s=syndrome(error)
        corrected=error^(1<<(s-1) if s else 0)
        assert (corrected==0)==(weight<=1)
for epsilon in [F(1,100),F(1,20),F(1,10)]:
    hamming=1-(1-epsilon)**7-7*epsilon*(1-epsilon)**6
    bit_repetition=3*epsilon**2*(1-epsilon)+epsilon**3
    four_repetitions=1-(1-bit_repetition)**4
    print('epsilon, Hamming four-bit block error (7 uses), repetition four-bit error (12 uses):',
          float(epsilon),round(float(hamming),9),round(float(four_repetitions),9))
```


## 확인 문제와 해설

1. 패리티 비트의 오류도 고치나요? **예.** H의 모든 위치에 고유 열이 있어 정보와 패리티를 같은 방식으로 검사합니다.
2. 부호어 두 개의 XOR도 유효한가요? **예.** H(c₁+c₂)=0이므로 선형 부분공간에 남습니다.
3. n=7, 정보 4비트의 전송률은? **4/7입니다.** 실수 벡터 차원 대신 GF(2)에서의 정보 비트 수를 셉니다.

추가 읽기: [MIT MAS.160, Hamming Codes and Syndromes](https://ocw.mit.edu/courses/mas-160-signals-systems-and-information-for-media-technology-fall-2007/94144b6cc83df4bea4f9cef22c91baf4_1017_error.pdf).
