## 경로가 있다고 항상 같은 의존은 아니다

세 노드의 chain A→B→C, fork A←B→C, collider A→B←C를 비교합니다. Chain과 fork에서는 중간 B를 조건화하면 A,C 경로가 차단됩니다. Collider는 관측하지 않으면 경로를 차단하지만 B 또는 그 후손을 조건화하면 경로가 열릴 수 있습니다. 화면은 후손을 추가하지 않은 세 노드 예제입니다.

일반 DAG에서는 조건 집합이 모든 무방향 경로를 차단하는지 봅니다. 경로의 비충돌점이 조건 집합에 있거나, 충돌점과 그 후손 어느 것도 조건 집합에 없으면 그 경로는 차단됩니다. 여러 경로 중 하나만 차단했다고 전체 독립을 결론낼 수 없습니다.

## 실제 표로 구조의 결과를 확인하기

Chain과 fork는 공정한 루트에 독립 반전 잡음 ε를 넣습니다. Collider의 A,C는 독립 공정 루트이고 B는 noisy-OR입니다. A 또는 C가 1이면 B=1일 확률 1−ε, 둘 다 0이면 ε입니다.

ε=.1에서 B를 관측하지 않은 collider의 A,C 결합은 네 상태 각각 .25입니다. B=1을 보면 가중치는 (.025,.225,.225,.225)이고 합 .7로 나눈 사후는 (1/28,9/28,9/28,9/28)입니다. 둘 중 한 원인이 꺼져 있으면 다른 원인이 켜져 있을 가능성이 높아집니다.

## 결합과 주변의 곱을 비교하기

파랑은 선택한 조건에서의 A,C 결합, 주황은 각 주변의 곱입니다. 두 곡선이 다른 것은 독립이 아니라는 증거입니다. 수치는 선택 사건 아래의 상호정보량이며, 일반적인 I(A;C|B)처럼 B 전체에 평균한 조건부 정보량과 구분합니다.

$$
p(a,c\mid B=b)\stackrel{?}=p(a\mid B=b)p(c\mid B=b).
$$

## D-separation은 충분조건

그래프가 d-separate하면 그 그래프에 Markov인 분포에서 독립을 보장합니다. 반대 방향에는 faithfulness 같은 조건이 필요합니다. ε=.5이면 간선 함수가 입력을 무시해 활성 경로가 있어도 정보량이 0입니다. ε=0인 collider에 B=0을 조건화해도 A=C=0으로 결정되어 조건부 독립인 특수 상황이 생깁니다.

따라서 그래프의 연결 여부와 모수에서 나타난 실제 의존을 같은 것으로 취급하면 안 됩니다. 인과 식별 역시 별도의 문제입니다.

## 후손만 관측해도 경로가 열릴 수 있다

화면의 세 노드에서 한 걸음 확장해 B→D를 추가합시다. A,C는 독립 공정 이진 변수, B는 앞의 ε=.1 noisy-OR이고 D는 B를 정확도 q로 복사합니다. B 자체는 보지 않고 D=1만 관측합니다. q=.8이면 부모가 둘 다 0일 때 P(D=1|A,C)=.26, 그 외에는 .74입니다. 따라서 사후의 네 질량은 (13,37,37,37)/124입니다.

이진 A,C의 독립성은 정규화한 2×2 표의 행렬식 Δ=p00p11−p01p10이 0인지로 확인할 수 있습니다. 총합이 1이면 Δ=p11−P(A=1)P(C=1), 즉 이진 지시변수의 공분산과 같습니다. 주변의 곱을 전개하면 바로 얻는 항등식입니다.

| q | P(D=1) | P(A=1\|D=1) | Δ |
|---|---|---|---|
| 1/2 | 1/2 | 1/2 | 0 |
| 4/5 | 31/50 | 37/62 | −111/1922 |
| 1 | 7/10 | 9/14 | −9/98 |

q=1/2에서는 D가 B에 대해 아무 정보도 주지 않아 실제 의존이 사라집니다. 그래프상 경로는 활성 상태지만 모수가 그 효과를 없앤 사례입니다. q=1은 B=1을 직접 관측한 결과와 같습니다. 아래 코드는 16개 (A,C,B,D) 중 D=1의 가중치를 합하고 정확분수로 검산합니다. 이 확장 예제는 화면의 세 노드 제어에 D를 추가했다고 주장하지 않습니다.

## 작은 의존을 0으로 잃지 않기

Chain·fork에서 B를 보지 않으면 끝점의 부호 상관은 ρ=(1−2ε)²이고 각 주변은 공정합니다. 따라서 상호정보량은 아래 식이며 ρ가 작으면 ρ²/(2 ln 2)가 주항입니다.

$$
I(A;C)=\frac{(1+\rho)\ln(1+\rho)+(1-\rho)\ln(1-\rho)}{2\ln 2}.
$$

ε=.5−2⁻²⁰이면 양의 정보량은 약 9.55×10⁻²⁴ bits입니다. 크기 1인 엔트로피 셋을 빼는 구현은 이를 0으로 잃었습니다. 수정한 계산은 독립 주변의 곱 qᵢ와 차이 δᵢ를 사용해 qᵢ[(1+δᵢ/qᵢ)ln(1+δᵢ/qᵢ)−δᵢ/qᵢ]를 합합니다. 괄호는 x²/2−x³/6+…로 계산하며 선형항은 전체 합에서 상쇄됩니다. Collider의 가까운 독립점도 모형의 행렬식을 직접 써서 차이를 보존합니다. 정확히 ε=.5이거나 d-separated인 경우의 0과 구분합니다. 유한 정밀도 계산이며 임의 정밀도 인증은 아닙니다.

**직접 풀기:** q=1/2에서 그래프를 지워도 되나요? **해설:** 그 모수에서만 의존이 사라진 것입니다. q가 바뀌는 같은 모형 가족의 구조적 독립을 증명한 것은 아닙니다.

```python
from fractions import Fraction as F
from itertools import product

epsilon = F(1, 10)
for accuracy in [F(1, 2), F(4, 5), F(1)]:
    joint = {(a, c): F(0) for a, c in product([0, 1], repeat=2)}
    for a, c, b in product([0, 1], repeat=3):
        prob_b1 = 1 - epsilon if a or c else epsilon
        prob_b = prob_b1 if b else 1 - prob_b1
        prob_d1 = accuracy if b else 1 - accuracy
        joint[a, c] += F(1, 4) * prob_b * prob_d1
    evidence = sum(joint.values())
    posterior = {key: w / evidence for key, w in joint.items()}
    pa = sum(v for (a, c), v in posterior.items() if a)
    pc = sum(v for (a, c), v in posterior.items() if c)
    determinant = posterior[0, 0]*posterior[1, 1] - posterior[0, 1]*posterior[1, 0]
    assert posterior[1, 1] - pa*pc == determinant
    assert (determinant == 0) == (accuracy == F(1, 2))
    print('q, evidence, P(A=1|D=1), covariance:', accuracy, evidence, pa, determinant)
```


## 확인 문제와 해설

1. 관측하지 않은 collider는 경로를 여나요? **아니요. 차단합니다.**
2. Chain에서 B를 알면? **A,C가 조건부 독립입니다.**
3. 여기서 B=1의 사후로 구한 MI는 B 전체 평균인가요? **아니요. 선택한 사건 한 개에 조건부입니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
