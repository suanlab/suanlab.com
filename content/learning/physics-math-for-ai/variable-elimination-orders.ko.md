## 주변화의 합을 안쪽으로 밀어 넣기

A−B−C−D chain에서 A의 주변을 구하려면 다른 세 변수를 합해야 합니다. 전체 16개 할당을 열거해도 되지만, 제거할 변수를 포함한 인자만 곱하고 그 변수를 합하면 더 작은 중간 표를 재사용할 수 있습니다.

$$
\tau(C)=\sum_D\psi_{CD}(C,D)e(D),
$$

여기서 e(D)는 관측을 나타내는 0/1 인자입니다. 중간 τ는 확률표일 필요가 없고 아직 정규화되지 않은 함수입니다.

## 같은 답과 다른 중간 표

A는 공정하고 각 간선의 일치 확률은 r입니다. r=.8,D=1이면 세 번 반전의 유효 상관은 (2r−1)³=.216이고 P(A=1|D=1)=(1+.216)/2=.608입니다. 증거 확률은 .5입니다.

D,C,B 순서로 제거하면 각 곱의 scope가 두 변수라 최대 표는 4칸입니다. B부터 제거하면 ψABψBC의 scope가 A,B,C로 커져 8칸입니다. B,C,D 순서도 정확히 계산하면 같은 .608을 줍니다.

## Scope가 비용을 결정한다

제거한 변수가 없어진다고 중간 함수가 늘 작아지는 것은 아닙니다. 남은 이웃이 연결되면서 fill-in이 생길 수 있습니다. 이진 변수 k개를 가진 곱 표는 2ᵏ칸이므로 가장 큰 중간 scope가 시간·메모리에 큰 영향을 줍니다. 이 화면의 비용은 실제로 만든 곱 표의 최대 칸 수이며 일반적인 모든 연산 수의 정확한 합은 아닙니다.

## Sum과 max는 다른 질문

현재 목적은 주변확률이므로 sum-product입니다. 전체 MAP 할당에는 max-product와 역추적 등이 필요합니다. 일부 변수를 최대화하고 일부를 합하는 marginal MAP에서는 연산의 교환이 일반적으로 안 되므로 단순히 같은 순서를 자유롭게 적용할 수 없습니다.

## 소거 단계의 불변식

현재 인자들의 곱은 이미 제거한 변수들을 원래 비정규화 결합에서 합한 결과라고 유지합니다. 다음 변수 v를 포함한 인자만 묶어 그 곱을 합하면, 나머지 인자는 v와 무관하므로 분배법칙으로 불변식이 유지됩니다. 귀납하면 남은 query와 evidence의 결합 가중치가 정확합니다. 모든 비-query 변수를 제거한 뒤 합이 양수일 때만 정규화합니다. 순서에 따른 차이는 정확한 실수 연산에서는 없고, 실제 구현에는 반올림·overflow·underflow의 순서 차이가 남을 수 있습니다.

## Marginal MAP에서 max와 sum을 바꾸면 생기는 일

행을 A, 열을 B로 둔 아래 양의 결합 분포를 보세요. A의 주변 최빈값은 A=1이지만 전체 joint의 최빈 할당은 (0,0)입니다.

| A | p(A,B=0) | p(A,B=1) | p(A) |
|---|---|---|---|
| 0 | .40 | .01 | .41 |
| 1 | .30 | .29 | .59 |

$$
\max_a\sum_b p(a,b)=.59
\quad<\quad
\sum_b\max_a p(a,b)=.69.
$$

오른쪽은 B=0에서 A=0, B=1에서 A=1을 골라 더합니다. 왼쪽은 B를 모르는 상태에서 단 하나의 A를 골라야 합니다. 선택을 B에 맞춰 바꾸는 능력을 몰래 추가했기 때문에 값이 커진 것입니다. 일반 비음수 표에서도 각 p(a,b)≤max_a' p(a',b)이므로 합한 뒤 max를 취하면 왼쪽≤오른쪽입니다. 모든 B에서 공통인 최적 A가 있으면 등호가 가능하지만 항상 그렇지는 않습니다.

순수 sum끼리는 교환할 수 있어도 이 혼합 연산은 목적을 바꿀 수 있습니다. 아래 정확분수 코드는 주변 최빈값·joint MAP·두 연산의 값을 따로 확인합니다.

**직접 풀기:** Joint MAP에서 얻은 A=0만 보고 A의 주변도 가장 크다고 결론낼 수 있나요? **해설:** 아닙니다. A=1의 두 상태 질량을 모으면 .59로 더 큽니다.

```python
from fractions import Fraction as F

# Rows: A=0,1; columns: B=0,1. All four masses are positive.
p = [[F(40, 100), F(1, 100)], [F(30, 100), F(29, 100)]]
assert sum(map(sum, p)) == 1
marginal = [sum(row) for row in p]
joint_map = max(((p[a][b], a, b) for a in range(2) for b in range(2)))
sum_of_max = sum(max(p[a][b] for a in range(2)) for b in range(2))
max_of_sum = max(marginal)
assert joint_map[1:] == (0, 0)
assert marginal.index(max_of_sum) == 1
assert sum_of_max == F(69, 100) > max_of_sum == F(59, 100)
print('A marginals:', marginal)
print('joint MAP (A,B):', joint_map[1:])
print('sum_B max_A, max_A sum_B:', sum_of_max, max_of_sum)
```


## 실습과 확인 문제

증거를 없애면 A는 .5로 돌아갑니다. 관측 인자는 한 번만 포함하며 마지막에 남은 A의 두 값을 합한 증거 확률로 나눕니다. 작은 표에서 직접 계산한 값과 정확 열거를 대조해 알고리즘과 확률 의미를 함께 확인하세요.

1. 좋은 소거 순서가 확률 모형을 바꾸나요? **아니요. 계산 비용을 바꿉니다.**
2. 중간 인자가 반드시 합이 1인가요? **아니요. 마지막 정규화 전의 함수입니다.**
3. 일반 그래프에서 최적 순서를 자동으로 찾는 것은 쉬운가요? **항상 그렇지 않으며 treewidth와 계산 난도가 중요합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/inference/ve/).
