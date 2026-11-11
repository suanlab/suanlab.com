## 후처리가 원본 정보를 새로 만들 수 있을까

입력 X에서 관측 Y를 만들고, Y만 보고 Z를 만드는 과정을 생각합니다. 핵심은 Z가 X에 별도로 접근하지 않는다는 조건입니다. 이를 X→Y→Z의 Markov 조건, 또는 X와 Z가 Y를 주었을 때 조건부 독립이라고 표현합니다.

$$
I(X;Z)\leq I(X;Y).
$$

## 두 반전 채널로 계산하기

실습은 X~Bernoulli(p)와 서로 독립인 두 잡음을 사용해 Y=X⊕N₁, Z=Y⊕N₂를 만듭니다. 한 번만 반전될 때 최종 결과가 달라지므로 유효 반전 확률은 ε₁(1−ε₂)+(1−ε₁)ε₂입니다.

p=.5,ε₁=.1,ε₂=.2이면 유효 반전은 .26입니다. 첫 상호정보량은 약 .531 bits이고 최종 정보량은 1−h₂(.26)≈.173 bits입니다. 표의 여덟 결합 질량은 P(x)P(y|x)P(z|y)의 곱이며 합은 1입니다.

## 연쇄법칙으로 이유 설명하기

결합 관측 (Y,Z)와 X 사이의 상호정보량을 두 순서로 전개합니다. Markov 조건은 첫 번째 전개에서 조건부 정보량 하나를 0으로 만듭니다. 두 번째 전개의 남은 조건부 상호정보량은 비음수입니다.

$$
I(X;Y,Z)=I(X;Y)+I(X;Z\mid Y)=I(X;Y),
$$
$$
I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)\geq I(X;Z).
$$

따라서 부등식을 얻습니다. 일반 분포에 대한 논증이며 대칭 잡음 예제에만 의존하지 않습니다. 여기서는 모든 변수가 유한하므로 엔트로피와 정보량의 무한값 차이 문제도 없습니다.

## 같은 정보와 잃은 정보

ε₂=1이면 Z=1−Y로 가역적이어서 정보 손실이 0입니다. ε₂=.5이면 Z가 원본과 독립이 되어 모든 입력 정보를 잃습니다. 파랑 곡선은 두 번째 반전 확률에 따른 최종 정보이고 주황은 첫 관측 정보의 상한입니다. 비가역 처리도 주어진 분포에서 관련 정보를 보존하는 경우에는 등호를 얻을 수 있습니다.

## 신경망 표현을 해석할 때

고정된 전처리나 표현의 후처리에 이 결과를 연결할 수 있지만, Z가 X 또는 추가 라벨·데이터를 다시 받는다면 같은 Markov 가정이 아닙니다. 예를 들어 Y는 잡음인데 Z=X로 원본을 직접 복사하면 I(X;Z)>I(X;Y)가 가능하며 부등식의 반례가 아니라 조건 위반입니다. 정보량 보존과 특정 과제의 예측 편의성·계산 비용도 서로 다른 질문입니다.

## 등호는 역함수보다 약한 조건이다

유한 변수의 Markov 사슬 X→Y→Z를 유지합니다. 위 연쇄법칙에서 정보 손실은 I(X;Y|Z)입니다. 조건부 상호정보량을 사후분포 사이의 KL 평균으로 다시 쓰면 무엇을 보존해야 하는지 보입니다. 합은 p(y,z)>0인 쌍에만 적용하며 로그 밑은 2입니다.

$$
I(X;Y)-I(X;Z)=\sum_{y,z}p(y,z)
D_{\rm KL}\!\left(P_{X\mid Y=y}\Vert P_{X\mid Z=z}\right).
$$

원래 정의에는 P(X|y,z)가 들어가지만 Markov 조건 때문에 P(X|y)로 바뀝니다. 각 KL은 비음수이고, 활성 쌍에서 두 사후가 같을 때만 0입니다. 따라서 등호는 P(X|y)=P(X|z)가 거의 확실하게 성립하는 것과 동치입니다. 즉 X→Z→Y도 성립합니다. Y 전체를 되찾을 필요 없이, X에 관한 사후를 보존하면 됩니다. 이 결론은 고정한 결합분포에 대한 것이며 모든 모수·사전에서의 충분성을 자동으로 뜻하지 않습니다.

## 같은 사후를 가진 관측 상태 합치기

Y의 세 상태 a,b,c에 질량 (1/4,1/4,1/2)을 주고 P(X=1|Y)=(1/5,1/5,4/5)로 둡니다. 이로부터 P(X=1)=1/2와 유효한 결합분포가 정해집니다. X를 원천으로 읽으면 P(Y|X)는 결합 질량을 P(X)로 나누어 얻습니다. 후처리는 Y만 사용하는 결정적 함수입니다.

- 보존 병합: a,b를 z=0으로, c를 z=1로 보냅니다. 합친 두 상태의 사후가 같아 결과 사후는 (1/5,4/5)입니다.
- 손실 병합: a,c를 z=0으로, b를 z=1로 보냅니다. 첫 사후는 (1/4×1/5+1/2×4/5)/(3/4)=3/5로 바뀝니다.

![같은 사후 1/5인 a,b는 정보 손실 없이 합치지만, 사후가 다른 a,c를 합치면 3/5로 바뀌며 정보가 줄어듭니다.](/assets/learning/information-posterior-merges-ko.svg)

| 처리 | I(X;Z) | 손실 |
|---|---|---|
| 원래 Y | 0.27807 | 0 |
| a,b 병합 | 0.27807 | 0 |
| a,c 병합 | 0.09131 | 0.18677 |

정보량 단위는 bits입니다. 보존 병합에서도 a와 b를 되찾을 수 없어 Y의 원시 상태 정보는 줄었습니다. 보존되는 대상은 X에 대한 정보입니다. 다른 목표 변수로 바꾸면 같은 병합이 정보를 잃을 수 있습니다. 두 경우 모두 원래 Markov 조건은 만족합니다.

## 사후가 달라지면 버린 정보가 남는다

손실 병합에서 z=0만 관측하면 X=1의 확률은 3/5입니다. 여기에 원래 관측 a인지 c인지까지 알려주면 각각 1/5와 4/5로 다시 달라집니다. 이것이 I(X;Y|Z)>0의 직관입니다. 반면 보존 병합의 z=0 안에서는 a인지 b인지 알아도 사후가 바뀌지 않습니다.

코드는 분수 결합 질량을 구성하고, 엔트로피 차이로 구한 손실을 사후 KL 평균과 별도로 비교합니다. 고정한 세 상태의 실행 예제이며 위 XOR 조절기를 새 병합 모형으로 바꾸지는 않습니다.

```python
from fractions import Fraction as Q
from math import log2, isclose

py, posterior = [Q(1, 4), Q(1, 4), Q(1, 2)], [Q(1, 5), Q(1, 5), Q(4, 5)]
joint = [[py[y]*(posterior[y] if x else 1-posterior[y])
          for y in range(3)] for x in range(2)]
def entropy(mass):
    return -sum(float(v)*log2(float(v)) for v in mass if v)
def mi(table):
    px = [sum(row) for row in table]
    pz = [sum(row[z] for row in table) for z in range(len(table[0]))]
    return entropy(px)+entropy(pz)-entropy([v for row in table for v in row])
def kl(p, q):
    return sum(float(a)*log2(float(a/b)) for a, b in zip(p, q) if a)

base = mi(joint)
for mapping in [[0, 1, 2], [0, 0, 1], [0, 1, 0]]:
    out = [[sum(joint[x][y] for y in range(3) if mapping[y] == z)
            for z in range(max(mapping)+1)] for x in range(2)]
    pz = [sum(row[z] for row in out) for z in range(len(out[0]))]
    post = [out[1][z]/pz[z] for z in range(len(pz))]
    gap = sum(float(py[y])*kl([1-posterior[y], posterior[y]],
                               [1-post[mapping[y]], post[mapping[y]]])
              for y in range(3))
    assert isclose(base-mi(out), gap, abs_tol=1e-12)
    if mapping == [0, 0, 1]:
        assert post == [Q(1, 5), Q(4, 5)] and gap == 0
    if mapping == [0, 1, 0]:
        assert post == [Q(3, 5), Q(1, 5)] and gap > .18
    print("map, posterior, retained bits, loss:", mapping, post, mi(out), gap)
```

## 연구에 적용하기 전에 묻기

1. 비가역 처리이면 반드시 X 정보를 잃나요? **아니요. 같은 X 사후를 가진 상태는 합칠 수 있습니다.**
2. 보존할 대상을 어떻게 명시하나요? **원천·목표 X, 관측 Y, 후처리 Z와 결합분포를 기록합니다.**
3. 정보 손실 0이면 모든 과제와 계산 비용도 같은가요? **아니요. 여기서는 지정한 X의 분포적 정보만 비교합니다.**

배경은 [MIT 6.441, Chapter 3 §3.1–3.2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/486b7d83cc428acf75b74323ecafcc11_MIT6_441S16_chapter_3.pdf)를 참고하세요. 세 상태의 병합과 수치는 위 결합분포에서 직접 계산했습니다.

## 작은 합성 정보와 표시 정밀도

두 채널의 대비 1−2ε는 합성에서 곱해집니다. 계산은 이 곱을 유지해, 합성 반전확률이 0.5로 반올림되어도 작은 양의 정보를 버리지 않습니다. 예를 들어 p=.5, ε₁=.5−2⁻³⁰, ε₂=.5−2⁻³⁵이면 최종 정보는 약 8.48×10⁻³⁹ bits입니다. 표시된 반전확률 0.5만으로 정확한 독립을 판정하지 마세요. 수학적으로 양수인 정보가 숫자 표현 범위 아래이면 선택 정보량과 점을 보류하고 수치 범위 안내를 표시합니다.

## 확인 문제와 해설

1. 두 번 모두 반전되면 최종 비트는? **원래 비트입니다.**
2. 항상 뒤집는 후처리는 정보를 지우나요? **아니요. 가역적입니다.**
3. 원본을 다시 입력받는 Z에도 같은 비교가 자동으로 적용되나요? **아니요. Markov 조건을 다시 확인해야 합니다.**

추가 읽기: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
