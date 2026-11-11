## 같은 메시지 식이라도 구조가 다르다

나무에서 정확한 sum-product를 triangle에 반복하면 loopy belief propagation이 됩니다. 이 방법은 유용할 수 있지만 나무의 정확성 논증을 그대로 가져올 수 없습니다. 수렴, 빠른 계산, 주변의 정확성, 결합의 정확성은 서로 다른 성질입니다.

## 작은 정확한 Oracle 만들기

이 모형은 세 이진 변수라 8개 상태를 직접 열거해 정확한 주변을 구할 수 있습니다. 파랑은 반복별 최대 주변 오차입니다. 실제 큰 그래프에서는 이 정확 기준을 계산할 수 없어서 근사법을 쓰는 경우가 많으므로 같은 오차창이 항상 제공되지는 않습니다.

## 메시지 변화는 정답 오차가 아니다

주황 잔차는 연속 두 메시지 사이의 최대 변화입니다. 작아지면 반복이 고정점 근처임을 뜻하지만 그 고정점이 정확 사후라는 뜻은 아닙니다. 기본 양의 J와 A의 field에서는 수렴하는 듯 보여도 원래 정확 주변보다 확신이 과한 belief가 나타납니다.

감쇠는 mnew=(1−η)mold+η mproposal로 업데이트합니다. 양의 정규화 메시지의 볼록 결합이므로 정규화는 유지하지만, 모든 순환 모형에서 수렴이나 정확성을 보장하지 않습니다. 첫 0회는 이전 업데이트가 없어 잔차를 —로 표시합니다.

$$
r_t=\max_{i\to j,x}|m^{(t)}_{i\to j}(x)-m^{(t-1)}_{i\to j}(x)|,\qquad e_t=\max_i|b_i^{(t)}(1)-p(X_i=1)|.
$$

## 주변이 맞아도 결합이 다를 수 있다

h=0이면 반전 대칭으로 모든 노드 주변이 .5입니다. BP도 같은 주변을 줄 수 있어 파랑 오차가 수치 0에 가깝습니다. 그러나 정확한 이웃 부호 상관은 양수일 수 있고, 주변 부호 평균의 곱은 0입니다. 주변들의 곱을 정확한 joint로 대신하면 의존을 지웁니다. 이 곱은 BP의 pair belief라고 주장하는 값도 아닙니다.

## 어떤 조건을 확인해야 할까

유한 상태 전체 열거는 정확하지만 지수 비용이 들고, VE·junction tree는 중간 scope가 작을 때 효율적입니다. Tree BP는 적절한 완료 스케줄에서 정확합니다. Loopy BP는 특정 모형·수축 조건 등에서 안정성을 분석해야 합니다. Monte Carlo의 수렴과 오차는 또 다른 표집·혼합 가정에 의존합니다. 알고리즘 이름만으로 보장 수준을 결정하지 마세요.

## 순환의 닫힌식으로 pair belief까지 검사하기

화면의 h=0인 triangle을 보되, 이번에는 노드 주변뿐 아니라 BP의 pair belief도 비교합시다. J는 유한하고 t=tanh J입니다. 각 간선에서 $e^{Js_is_j}=\cosh J(1+ts_is_j)$로 전개합니다. 전체 부호를 합하면 각 노드의 차수가 홀수인 항은 부호 상쇄로 0입니다. Triangle의 간선 부분집합 중 모든 차수가 짝수인 것은 공집합과 전체 cycle뿐이므로

$$
Z=8\cosh^3(J)(1+t^3),\qquad
E[s_As_B]=\frac13\partial_J\log Z
=\frac{t+t^2}{1+t^3}.
$$

첫 미분에서 1/3을 곱한 것은 세 간선이 같은 J를 공유하고 대칭이기 때문입니다. 일반적인 서로 다른 결합에서 임의로 1/3을 붙이지 않습니다.

균등 초기 메시지는 h=0에서 BP의 고정점입니다. 수신 부호마다 Σsᵢ exp(Jsᵢsⱼ)=2cosh J로 같기 때문입니다. 그 pair belief는 exp(Jsᵢsⱼ)/(4cosh J)라 BP pair 상관은 t입니다. 따라서 J>0일 때 정확 상관과의 차이는 t²(1−t²)/(1+t³)>0입니다. 노드 주변은 모두 .5이고 메시지 잔차가 정확히 0이어도 pair는 틀릴 수 있습니다.

![영 field triangle에서 노드 주변은 모두 1/2이지만 정확한 간선 상관, BP pair 상관, 노드 평균의 곱은 서로 다를 수 있음을 보여 주는 세 비교 막대.](/assets/learning/graphical-cycle-correlations-ko.svg)

| J | 정확한 간선 상관 | BP pair 상관 | 노드 평균의 곱 |
|---|---|---|---|
| .2 | .234529 | .197375 | 0 |
| .7 | .794288 | .604368 | 0 |
| 1.3 | .978293 | .861723 | 0 |

앞의 화면은 마지막 열을 독립 가정의 예로 표시하며 BP pair 상관이라고 부르지 않습니다. 여기서는 pair belief를 별도로 유도해 세 대상을 구분했습니다. 코드는 직접 열거로 Z와 상관을 확인하고, 균등 메시지에서 만든 pair belief와 대조합니다.

## 어떤 정확도 주장을 할 수 있을까

나무에서는 간선을 끊으면 두 성분이 분리되어 부분합의 귀납이 성립합니다. Triangle에서는 나머지 두 간선이 다시 연결하므로 그 귀납을 적용할 수 없습니다. 위 1+t³는 닫힌 cycle에서 살아남는 기여이고, pair 상관 t만으로는 이를 반영하지 못합니다. 이 예제는 한 고정점의 정확도를 분석한 것이며 임의 field·초기화·여러 cycle의 수렴 정리는 아닙니다.

수렴을 주장하려면 업데이트·노름·초기화와 모형에 맞는 가정을 제시해야 합니다. 근사 정확도를 주장하려면 고정점 잔차와 별도의 오차 근거가 필요합니다. 작은 문제의 정확 oracle은 이 둘을 분리해 검증하는 연구 출발점입니다.

**직접 풀기:** J=0에서 세 상관은 어떻게 되나요? **해설:** t=0이므로 모두 0입니다. 양의 J에서의 차이는 부동소수점 잡음이 아니라 서로 다른 식이 주는 차이입니다.

```python
from itertools import product
from math import exp, tanh, cosh

for J in [.2,.7,1.3]:
    t = tanh(J)
    states = list(product([-1,1],repeat=3))
    weights = [exp(J*sum(s[i]*s[(i+1)%3] for i in range(3))) for s in states]
    Z = sum(weights)
    corr = sum(w*s[0]*s[1] for w,s in zip(weights,states))/Z
    assert abs(Z/(8*cosh(J)**3*(1+t**3))-1) < 1e-12
    assert abs(corr-(t+t*t)/(1+t**3)) < 1e-12
    # Uniform messages are a fixed point at zero field.
    pair_weights = [exp(J*a*b) for a,b in product([-1,1],repeat=2)]
    pair_corr = sum(w*a*b for w,(a,b) in zip(pair_weights,product([-1,1],repeat=2)))/sum(pair_weights)
    assert abs(pair_corr-t) < 1e-12
    for i in range(3):
        assert abs(sum(w*s[i] for w,s in zip(weights,states))/Z) < 1e-12
    assert corr > pair_corr > 0
    print('J, exact correlation, BP pair correlation, product of means:',
          J,round(corr,6),round(pair_corr,6),0)
```


## 확인 문제와 해설

1. 작은 잔차는 정확한 사후의 증명인가요? **아니요. 반복 안정성의 진단입니다.**
2. 감쇠를 쓰면 항상 정확해지나요? **아니요. 안정화와 정확성을 구분합니다.**
3. 노드 주변이 모두 맞으면 joint도 독립인가요? **아니요. 상관 구조가 남을 수 있습니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/jt/).
