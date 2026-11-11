## 메시지는 바깥 부분의 요약이다

나무에서 i→j 메시지는 j 쪽을 제외한 i의 부분나무가 j의 값에 대해 주는 가중치를 요약합니다. i의 unary와 다른 이웃에서 받은 메시지를 곱한 뒤 i를 합합니다. 수신자 j의 메시지를 다시 곱하면 같은 정보가 중복될 수 있으므로 제외합니다.

$$
m_{i\to j}(x_j)\propto\sum_{x_i}\phi_i(x_i)\psi_{ij}(x_i,x_j)
\prod_{k\in N(i)\setminus j}m_{k\to i}(x_i).
$$

메시지에 양의 상수를 곱해도 마지막 주변 정규화에서 상쇄됩니다. 구현은 안정성을 위해 각 메시지를 두 상태의 합으로 정규화합니다.

## 이 실습의 시간과 공간

그래프는 A−B−C−D이고 field h는 A에만 있습니다. 초기 메시지는 두 상태 각각 .5입니다. 동기 라운드에서는 이전 라운드의 모든 메시지로 다음 메시지를 한꺼번에 계산합니다. 한 라운드에서 정보가 한 간선씩 이동하므로 A의 영향이 D까지 가려면 세 라운드가 필요합니다.

## Belief와 정확한 주변 비교

각 노드 belief는 unary와 모든 수신 메시지의 곱을 정규화한 값입니다. 파랑은 현재 belief, 주황은 16개 할당을 직접 열거한 정확한 주변입니다. J=1,h=.7에서 한 라운드 후 C,D의 파랑은 아직 .5지만 정확한 값은 각각 약 .6753,.6335입니다.

$$
b_i(x_i)\propto\phi_i(x_i)\prod_{k\in N(i)}m_{k\to i}(x_i).
$$

세 라운드 후에는 전체 나무의 정보가 전달되어 수치 오차 수준에서 일치합니다. 일반적인 나무에서는 적절한 두 방향 스케줄로 각 방향 메시지를 한 번씩 계산할 수도 있습니다. 동기 반복 횟수와 그런 스케줄의 메시지 연산 수는 같은 단위가 아닙니다.

## 정확성 조건과 다른 목적

이 결과는 나무 구조와 정확한 sum-product 연산에 의존합니다. 순환 그래프에 같은 반복을 적용하면 다른 성질을 갖습니다. 또한 노드 주변들의 최빈값을 각자 선택한다고 전체 joint MAP를 얻는 것은 아닙니다. Joint MAP에는 max-product와 적절한 역추적을 구분해 사용합니다.

## 부분나무를 합하는 귀납 증명

간선 i−j를 끊었을 때 i가 속한 성분을 Tᵢ→ⱼ라고 합시다. 정규화하지 않은 메시지는 그 성분 내부 변수의 가중치를 모두 합하되, 경계의 xⱼ만 남기는 함수입니다. i가 잎이면 합할 변수는 xᵢ뿐이라 메시지 식이 바로 성립합니다. 일반 i에서는 이웃 k≠j의 성분들이 서로 겹치지 않으므로, 이미 계산한 메시지들을 곱해 각 성분의 합을 재사용할 수 있습니다. 마지막에 xᵢ를 합하면 i→j를 얻습니다. 귀납으로 모든 메시지가 정확하고, 한 노드에 모든 방향 메시지가 모이면 전체 주변이 정확합니다.

순환에서는 성분들이 이런 방식으로 분리되지 않아 같은 정보가 여러 경로로 되돌아올 수 있습니다. 수신자의 메시지를 제외한다는 국소 규칙만으로 그 전역 중복까지 제거되지는 않습니다.

## 이 chain의 닫힌식으로 정보 도착을 읽기

이 예제는 exp(hs₀+JΣsᵢsᵢ₊₁), sᵢ∈{−1,1}이며 h는 첫 노드에만 있습니다. 끝 부호를 차례로 합할 때마다 2cosh J가 나오므로 Z=2cosh h(2cosh J)³입니다. 조건부 평균 E[sᵢ₊₁|sᵢ]=sᵢ tanh J이고 처음 평균은 tanh h이므로 반복기대값으로 다음을 얻습니다.

$$
E[s_i]=\tanh(h)\,\tanh(J)^i,
\qquad P(X_i=1)=\frac{1+\tanh(h)\tanh(J)^i}{2}.
$$

균등 메시지에서 시작하면 r회 동기 라운드 뒤 i≤r인 노드에 첫 field의 정보가 도착하고, 더 먼 노드의 부호 평균은 아직 0입니다. 이는 첫 노드에만 field가 있는 현재 chain의 특별한 구조입니다. 일반 나무에서는 모든 source와 잎에서 필요한 정보가 도착하는 거리를 확인해야 합니다. 유한 양의 인자·정확 연산·이 초기화에서는 나무 지름만큼 라운드를 실행하면 전체 정보가 전달됩니다.

| 라운드 | field 정보가 도착한 노드 | 최대 주변 오차, J=1,h=.7 |
|---|---|---|
| 0 | A | .230141 |
| 1 | A,B | .175274 |
| 2 | A,B,C | .133488 |
| 3 | A,B,C,D | 0 |

코드는 독립 열거·닫힌식·메시지 반복을 대조하며 음의 J와 h도 검사합니다. 마지막 0은 정확 실수식의 결과이며 실행 assert에는 부동소수점 허용 오차를 사용합니다.

**직접 풀기:** J<0이면 첫 field의 영향은 어떻게 변하나요? **해설:** tanh J의 부호가 음수라 거리에 따라 평균 부호가 번갈아 바뀌고, 유한 J에서는 크기가 |tanh J|의 거듭제곱으로 줄어듭니다.

```python
from math import exp, tanh, cosh
from itertools import product

edges = [(0,1),(1,2),(2,3)]
neighbors = [[j for u,v in edges for j in ([v] if u==i else [u] if v==i else [])]
             for i in range(4)]
def normalize(v):
    return [x/sum(v) for x in v]

for J, h in [(1,.7), (-1,.7), (.4,-.7)]:
    joint = [(s, exp(h*s[0] + J*sum(s[i]*s[j] for i,j in edges)))
             for s in product([-1,1], repeat=4)]
    Z = sum(w for s,w in joint)
    exact = [sum(w*s[i] for s,w in joint)/Z for i in range(4)]
    analytic = [tanh(h)*tanh(J)**i for i in range(4)]
    assert max(abs(a-b) for a,b in zip(exact,analytic)) < 1e-12
    assert abs(Z/(2*cosh(h)*(2*cosh(J))**3)-1) < 1e-12
    messages = {(i,j): [.5,.5] for u,v in edges for i,j in [(u,v),(v,u)]}
    for rounds in range(4):
        means = []
        for i in range(4):
            v = [exp(h*x if i==0 else 0) for x in [-1,1]]
            for k in neighbors[i]:
                v = [v[t]*messages[k,i][t] for t in range(2)]
            means.append(2*normalize(v)[1]-1)
        expected = [analytic[i] if i<=rounds else 0 for i in range(4)]
        assert max(abs(a-b) for a,b in zip(means,expected)) < 1e-12
        if J==1 and h==.7:
            print('round, max marginal error:', rounds,
                  round(max(abs(a-b)/2 for a,b in zip(means,exact)),6))
        updated = {}
        for i,j in messages:
            v = []
            for y in [-1,1]:
                total = 0
                for t,x in enumerate([-1,1]):
                    weight = exp(J*x*y + (h*x if i==0 else 0))
                    for k in neighbors[i]:
                        if k!=j:
                            weight *= messages[k,i][t]
                    total += weight
                v.append(total)
            updated[i,j] = normalize(v)
        messages = updated
```


## 확인 문제와 해설

1. j로 보내는데 j에서 온 메시지를 다시 쓰나요? **아니요. 수신자는 제외합니다.**
2. 한 라운드의 작은 오차가 항상 전체 정보 전달을 뜻하나요? **아니요. 멀리 있는 관측의 경로를 확인해야 합니다.**
3. J=0에서는? **변수들이 독립이어서 초기 local belief만으로도 정확합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
