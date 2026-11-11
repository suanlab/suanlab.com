## 오류를 허용하면 얼마나 압축할 수 있을까

무손실 압축은 원본을 정확히 복원해야 합니다. 손실 압축은 정해진 왜곡 척도와 예산 안에서 정보를 버립니다. 실습은 iid 공정 이진 정보원 X와 Hamming 비용 d(x,y)=1{x≠y}에 한정합니다. 왜곡은 평균적으로 다른 비트의 비율입니다. 다른 정보원이나 왜곡에는 같은 식을 그대로 사용할 수 없습니다.

$$
R(D)=\inf_{P_{Y|X}:E[d(X,Y)]\leq D}I(X;Y).
$$

## 이진 정보원의 정확한 경계

이 모형에서는 D≤.5에서 R(D)=1−h₂(D), D≥.5에서는 0입니다. D=.1이면 약 .531 bits/기호입니다. D=0에서는 공정 비트 하나당 1 bit를 보존해야 합니다. D=.5 이상은 언제나 0을 복원해도 평균 왜곡 .5를 달성하므로 정보를 보낼 필요가 없습니다.

$$
R(D)=\begin{cases}1-h_2(D),&0\leq D\leq1/2,\\0,&1/2\leq D\leq1.\end{cases}
$$

## 왜 이 경계인가

오류 확률 d가 D 이하일 때 이진 Fano 관계로 H(X|Y)≤h₂(d)≤h₂(D)를 얻습니다. 따라서 I(X;Y)≥1−h₂(D)입니다. 반대로 반전 확률 D인 이진 대칭 시험 채널은 같은 왜곡과 정보량을 달성합니다. 이 정보량 최적화와 긴 블록의 실제 부호율을 연결하는 것은 rate–distortion 부호화 정리이며, 그림 자체가 블록 부호를 생성하는 것은 아닙니다.

## 예산과 선택 채널을 따로 조절하기

파랑은 모든 허용 시험 채널 중 최적인 R(D), 주황은 반전 확률 ε인 대칭 시험 채널의 정보량 1−h₂(ε)입니다. ε≤.5에서는 두 곡선이 맞지만 .5를 넘는 반전은 비효율적입니다. ε=.75의 정보량은 약 .189인데 같은 왜곡 예산의 최적 경계는 0입니다.

현재 채널의 실제 왜곡 ε가 예산 D 이하인지 별도 수치로 확인하세요. D=.1,ε=.25인 채널은 정보량이 더 작아도 허용 오류를 초과하므로 유효 후보가 아닙니다. 파랑 점과 주황 점은 서로 다른 역할을 가진 입력입니다.

## AI의 표현 압축과 연결하기

잠재 표현의 정보량과 복원 비용을 함께 보는 직관을 얻을 수 있습니다. 다만 이 실습의 닫힌식이 이미지·텍스트의 의미 손실이나 임의의 신경망에 그대로 적용되지는 않습니다. 단위는 점근 bits/기호이며 한 기호 파일의 정확한 바이트 수나 모든 표본의 오류 보장이 아닙니다. 정보원, 복원 알파벳, 왜곡 척도와 평균·최악 조건을 먼저 정해야 합니다.

## 유한 블록에서도 필요한 비트의 하한이 있다

iid 공정 비트 X^n을 하나의 메시지 M으로 부호화하고 이진 블록 Y^n으로 복원합시다. 메시지는 최대 2^b가지이고, 복원기는 M 외에 원본을 받지 않습니다. 결정적 부호기를 사용하거나 독립적인 내부 난수를 쓰더라도 X^n→M→Y^n이 성립하도록 가정합니다. 평균 Hamming 왜곡은 d̄=(1/n)ΣP(X_i≠Y_i)이며 d̄≤D≤1/2라고 합시다. 이는 모든 블록의 최악 왜곡 조건과 다릅니다.

$$
I(X^n;Y^n)\leq I(X^n;M)\leq H(M)\leq b.
$$

오류 벡터 E^n=X^n XOR Y^n을 두면 Y^n을 아는 상태에서 X^n과 E^n은 일대일입니다. 조건을 지우고, 엔트로피의 부분가법성과 h₂의 오목성을 순서대로 사용합니다.

$$
H(X^n\mid Y^n)=H(E^n\mid Y^n)
\leq H(E^n)\leq\sum_{i=1}^nh_2(d_i)
\leq nh_2(\bar d)\leq nh_2(D).
$$

H(X^n)=n이므로 두 식을 결합하면 b/n≥1−h₂(D)입니다. 유한 블록에서도 성립하는 converse, 즉 불가능한 영역을 제외하는 하한입니다. 이 하한만으로 같은 n의 부호가 그 경계를 달성한다고 증명한 것은 아닙니다. 점근 achievability와 유한 블록의 정확한 달성 영역은 다른 주장입니다.

## 1 bit 메시지의 실제 복원 블록 찾기

b=1이면 복원할 블록을 두 개 선택하고, 각 원본을 더 가까운 블록으로 보낼 수 있습니다. 고정한 두 복원점에 대해서는 가장 가까운 점을 택하는 것이 각 원본의 Hamming 비용을 최소화합니다. 원본은 균등하므로 모든 2^n개 비용을 평균하면 정확한 왜곡입니다.

n=2,3,4에서 가능한 서로 다른 복원점 쌍을 전부 검사합니다. 후보 쌍 수는 각각 6,28,120개입니다. D*는 이 지정한 두 복원점·최근접 부호화 족의 최적 평균 왜곡입니다. Gap은 실제 전송률 1/n에서 R(D*)를 뺀 값입니다.

| n | D* | Gap |
|---|---|---|
| 2 | $1/4$ | 0.31128 |
| 3 | $1/4$ | 0.14461 |
| 4 | $5/16$ | 0.14604 |

모든 경우에서 경계 위에 남습니다. 예를 들어 n=3에서 (000,111)은 평균 오류 비트 수 3/4, 비트당 왜곡 1/4을 줍니다. 필요한 전송률 1/3은 R(1/4)≈0.18872보다 큽니다. ‘정보량 최적 시험 채널이 있다’는 사실과 ‘세 비트를 한 비트로 보내 그 정보량 그대로 구현한다’는 주장은 같지 않습니다.

## 평균·최악과 모형 범위 기록하기

위 표는 균등 원본 아래 평균 왜곡입니다. n=3 예제에서는 어떤 블록에 1 bit 오류가 생겨 최악 비트당 왜곡은 1/3입니다. 평균 1/4을 모든 원본의 보장으로 쓰면 안 됩니다. 메시지 길이 b도 고정되어 있으며 가변 길이 평균 비트 비용을 비교한 것이 아닙니다.

코드는 복원점 쌍과 모든 원본을 완전 열거합니다. 작은 유한 문제의 정확한 분수 왜곡을 구하지만 긴 블록의 달성 정리를 새로 증명하지는 않습니다. 기존 화면의 D와 ε는 한 기호 시험 채널용이며 아래 블록 부호를 생성하지 않습니다.

```python
from fractions import Fraction as Q
from itertools import product, combinations
from math import log2

def h(p):
    return 0 if p in [0, 1] else -p*log2(p)-(1-p)*log2(1-p)
def distance(a, b):
    return sum(x != y for x, y in zip(a, b))

for n, expected in [(2, Q(1, 4)), (3, Q(1, 4)), (4, Q(5, 16))]:
    source = list(product([0, 1], repeat=n))
    results = []
    for pair in combinations(source, 2):
        costs = [min(distance(x, c) for c in pair) for x in source]
        results.append((Q(sum(costs), n*len(source)), pair, max(costs)))
    best = min(row[0] for row in results)
    assert best == expected
    rate = Q(1, n)
    boundary = 1-h(float(best))
    assert float(rate) >= boundary
    selected = next(row for row in results if row[0] == best)
    if n == 3:
        antipodal = [min(sum(x), n-sum(x)) for x in source]
        assert Q(sum(antipodal), n*len(source)) == Q(1, 4)
        assert Q(max(antipodal), n) == Q(1, 3)
    print("n, pairs, best distortion, rate, R(D), gap:",
          n, len(results), best, rate, boundary, float(rate)-boundary)
    print("One optimal pair and its worst per-bit distortion:", selected[1], Q(selected[2], n))
```

## 연구에 적용하기 전에 묻기

1. converse를 만족하면 그 점의 부호가 존재하나요? **하한만으로 존재가 증명되지는 않습니다.**
2. 위 D*는 최악 오류 보장인가요? **아니요. 지정한 균등 원본의 평균입니다.**
3. 실제 압축 비교에 무엇이 필요한가요? **블록 길이·부호기·복원기·전송률·왜곡 기준·계산 비용을 명시합니다.**

배경은 [MIT 6.441, Chapter 23](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/880c39878ba7dc35a5fbb35758d42a69_MIT6_441S16_chapter_23.pdf)을 참고하세요. 오류 벡터 converse와 두 복원점의 유한 열거는 위 가정에서 직접 계산했습니다.

## 확인 문제와 해설

1. D=.5의 R(D)는? **0입니다.**
2. ε=1의 대칭 채널은 정보량 0인가요? **아니요. 완전 반전은 가역적이라 1 bit입니다.**
3. 왜곡 예산을 넘은 낮은 정보량 채널은 해인가요? **아니요. 제약을 만족해야 합니다.**

추가 읽기: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
