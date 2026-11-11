## 확률에서 실제 이진 부호로

자기정보 −log₂p는 이상적인 길이와 연결되지만, 실제 부호는 이진 문자열입니다. 부호집 0은 네 기호에 0,10,110,111을 배정하고 길이는 1,2,3,3입니다. 부호집 1은 00,01,10,11로 모두 길이 2입니다. 어느 단어도 다른 단어의 접두사가 아니므로 둘 다 순차 복호화할 수 있습니다.

$$
L=\sum_i p_i\ell_i,\qquad \sum_i2^{-\ell_i}\leq1\quad\text{for binary prefix codes}.
$$

## 여덟 관측의 비트 수 세기

관측 개수가 (4,2,1,1)이면 부호집 0의 데이터 길이는 4×1+2×2+1×3+1×3=14 bits입니다. 부호집 1은 8×2=16 bits입니다. 경험 엔트로피는 1.75 bits/기호이고 첫 부호의 평균 길이도 1.75입니다. 이 경우 길이가 −log₂ 빈도와 맞는 특별한 예제입니다.

다른 빈도에서는 같지 않을 수 있습니다. 균등 개수 (2,2,2,2)에서는 첫 부호의 데이터가 18 bits, 고정 길이 부호가 16 bits입니다. 모든 부호집이 모든 분포에 최적인 것은 아닙니다. 알려진 유한 정보원의 최적 접두 평균 길이에 대한 엔트로피 경계와 현재 선택한 고정 부호집의 길이를 구분하세요.

## 수신자에게 모형도 알려주기

어떤 부호집을 썼는지 알리는 헤더를 포함합니다. 부호집 0의 헤더는 10, 부호집 1은 0으로 이 헤더들도 접두 부호입니다. 따라서 기본 데이터의 두 부분 총 길이는 2+14=16 bits와 1+16=17 bits입니다. 여기서는 관측 수 N이 알려져 있어 종료를 알리는 별도 길이는 비교에서 제외합니다.

$$
L_{\rm total}=L(\text{codebook})+L(\text{data}\mid\text{codebook}).
$$

이것은 사전에 정한 두 유효 부호집 사이의 작은 MDL 예제입니다. 연속 모수를 학습하거나 정교한 universal coding을 구현한 것은 아닙니다. 부호집 자체를 데이터로 설계한다면 그 설명 비용도 별도로 정해야 합니다.

## 짧아 보이지만 무효인 부호

부호집 2는 0,01,10,11입니다. 0이 01의 접두사이고 Kraft 합은 1.25입니다. 실제 문자열 010을 0|10 또는 01|0으로 읽을 수 있어 모호합니다. 화면은 원시 평균 길이를 보여주되 유효한 총 길이는 —로 표시하고 MDL 후보에서 제외합니다. 접두사가 아니라는 사실만으로 모든 일반 부호의 복호화 가능성을 판정하는 것은 아니지만, 이 목록은 직접적인 모호성도 갖습니다.

파랑은 두 유효 후보의 데이터 비트 수, 주황은 헤더를 포함한 총 길이입니다. 표는 현재 선택한 기호 부호를 보여줍니다. 반복 횟수를 늘리면 데이터 비용은 비례하지만 고정 헤더 비용은 그대로입니다.

## Kraft 합으로 평균 길이의 하한 증명하기

고정한 화면 부호집과 별도로, 양의 질량 p_i를 가진 알려진 유한 정보원과 이진 접두 부호를 생각합니다. 기호가 둘 이상이고 길이 ℓ_i는 양의 정수입니다. Kraft 합 K=Σ2^(−ℓ_i)는 0<K≤1입니다. r_i=2^(−ℓ_i)/K로 정규화하면 유효한 확률분포가 됩니다. bits 단위 KL을 전개하면

$$
L-H(P)=D_{\rm KL}(P\Vert r)-\log_2K\geq0.
$$

따라서 접두 부호의 평균 길이는 엔트로피보다 작을 수 없습니다. 등호에는 KL=0과 K=1이 모두 필요하므로 p_i=2^(−ℓ_i)인 경우입니다. 특정 부호가 이 하한에 가깝다는 것과 모든 문자열의 길이가 같다는 것은 다릅니다.

## 길이를 정수로 올리고 실제 단어 만들기

Shannon 길이 ℓ_i=ceil(−log₂p_i)를 택하면 2^(−ℓ_i)≤p_i이므로 Kraft 합은 1 이하입니다. 짧은 길이부터 이진 구간을 배정하면 접두 부호를 만들 수 있습니다. 길이 ℓ의 단어를 [0,1) 안의 크기 2^(−ℓ)인 dyadic 구간으로 읽으세요. 앞서 놓은 더 큰 구간들은 현재 격자에 정렬되고, 누적 크기가 1을 넘지 않으므로 다음 구간을 놓을 자리가 있습니다. 겹치지 않는 구간은 어느 단어도 다른 단어의 접두사가 아님을 보장합니다.

각 길이가 −log₂p_i+1보다 작으므로 다음 경계를 얻습니다. 이는 최적 접두 부호가 아니라 이 구성 자체에 대한 상한입니다.

$$
H(P)\leq L_{\rm Shannon}<H(P)+1.
$$

p=(1/2,1/3,1/6)에서는 길이 (1,2,3), 단어 (0,10,110), Kraft 합 7/8, 평균 5/3입니다. 엔트로피는 약 1.45915입니다. 다른 유효 부호 (0,10,11)는 평균 3/2로 더 짧습니다. Shannon 구성이 항상 최적이라는 주장은 아닙니다.

## 블록으로 묶으면 기호당 정수화 비용이 줄어든다

iid m기호 블록의 확률은 질량의 곱이고 엔트로피는 mH(P)입니다. 같은 구성을 블록에 적용하면 기호당 평균 길이 L_m/m에 대해

$$
H(P)\leq\frac{L_m}{m}<H(P)+\frac1m.
$$

다음은 같은 p에서의 실제 구성 결과입니다. Rate는 bits/기호이고 Gap은 Rate−H(P)입니다. m=1과 2가 같듯이 실제 차이가 매 단계 엄격히 줄어들 필요는 없습니다.

| m | Rate | Gap |
|---|---|---|
| 1 | 1.66667 | 0.20752 |
| 2 | 1.66667 | 0.20752 |
| 4 | 1.58854 | 0.12939 |

m이 커지면 블록 알파벳 크기는 3^m으로 늘어납니다. 평균 비트 효율의 경계가 계산·저장 비용까지 사라지게 하지는 않습니다. 여기서는 분포와 블록 길이를 양쪽이 알고 있다고 가정합니다. 학습한 분포나 부호집을 전달하는 비용은 기존 MDL의 모형 설명 길이처럼 별도로 더해야 합니다.

## 분수 길이 계산과 복호화 검산

코드는 부동소수점 로그의 ceil 대신 분수 비교로 길이를 결정합니다. 길이를 정렬한 canonical 이진 단어를 만들고 각 단어가 접두사가 아닌지, 모든 두 블록 메시지가 정확히 복호화되는지 검사합니다. 원래 화면의 두 고정 후보와는 별도 부호 구성입니다.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod, log2

p = [Q(1, 2), Q(1, 3), Q(1, 6)]
H = -sum(float(v)*log2(float(v)) for v in p)
for m in [1, 2, 4]:
    masses = [prod(block) for block in product(p, repeat=m)]
    lengths = []
    for mass in masses:
        length = 0
        while Q(1, 1 << length) > mass:
            length += 1
        lengths.append(length)
    words = [None]*len(masses)
    code, previous = 0, 0
    for i in sorted(range(len(masses)), key=lambda i: (lengths[i], i)):
        length = lengths[i]
        code <<= length-previous
        assert code < 1 << length
        words[i] = format(code, '0'+str(length)+'b')
        code += 1
        previous = length
    assert all(not v.startswith(w) for i, w in enumerate(words)
               for j, v in enumerate(words) if i != j)
    for i, j in product(range(len(words)), repeat=2):
        remaining, decoded = words[i]+words[j], []
        while remaining:
            k = next(k for k, w in enumerate(words) if remaining.startswith(w))
            decoded.append(k)
            remaining = remaining[len(words[k]):]
        assert decoded == [i, j]
    kraft = sum(Q(1, 1 << n) for n in lengths)
    rate = sum(v*n for v, n in zip(masses, lengths))/m
    assert kraft <= 1 and H <= float(rate) < H+1/m
    print("m, alphabet size, Kraft sum, rate, gap:", m, len(masses), kraft, rate, float(rate)-H)
```

## 연구에 적용하기 전에 묻기

1. −log₂p_i가 그대로 파일 길이인가요? **아니요. 정수 단어·헤더·종료 규약을 구분합니다.**
2. 엔트로피 경계가 모든 표본의 길이를 보장하나요? **여기서는 알려진 정보원 아래 평균 길이의 경계입니다.**
3. m을 늘릴 때 무엇을 함께 기록하나요? **기호당 비용뿐 아니라 알파벳·부호표 크기와 모형 전달 비용도 기록합니다.**

배경은 [MIT 6.441, Chapter 6, Theorems 6.5–6.6](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/67d8e68cd8fda55366e3f9f0a9465c17_MIT6_441S16_chapter_6.pdf)을 참고하세요. 세 질량 블록의 실제 단어와 길이는 직접 구성했습니다.

## 확인 문제와 해설

1. 균등 개수의 더 짧은 두 부분 코드는? **총 17 bits인 고정 길이 후보입니다.**
2. invalid 후보가 짧으면 선택해도 되나요? **아니요. 유효한 복호화 규칙이 먼저입니다.**
3. 결정적 경험 빈도이면 이 고정 부호가 0 bits/기호인가요? **아니요. 네 기호 부호집을 유지한 예제입니다.**

추가 읽기: [Grünwald, A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077).
