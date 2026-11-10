## 1. 가중치를 합해 변수를 제거하기

조대화는 남길 자유도를 선택하는 데서 시작합니다. 남긴 배치의 주변 가중치는 그 배치와 양립하는 모든 미시 배치의 가중치를 합한 값입니다. 그 값에 음의 로그를 취하면 유효 작용이 됩니다. 남긴 배치 전체에 대해 가중치를 더하면 원래 분배함수를 얻습니다. 유효 작용의 가산 상수는 분배함수에 영향을 주고 정규화한 확률에서는 상쇄됩니다.

## 2. 고리에서 한 칸씩 건너 스핀 제거하기

길이가 짝수이고 외부장이 0인 강자성 이징 고리에서 무차원 결합을 $K$, 가중치를 $e^{K\sum_i s_i s_{i+1}}$로 둡니다. 짝수 번째 스핀을 남깁니다. 남긴 두 스핀 $s,t$ 사이에서 제거할 스핀 $u$를 합합니다.

$$
\sum_{u=\pm1} e^{K(s+t)u} = A e^{K' st},\qquad
K'=\tfrac12\log\cosh(2K),\qquad A=2\sqrt{\cosh(2K)}.
$$

$st=1$인 두 배치의 국소 가중치는 $2\cosh(2K)$이고 $st=-1$인 배치의 가중치는 $2$입니다. 두 가중치의 비율이 $K'$를 정하고 기하평균이 $A$를 정합니다. 제거된 스핀마다 인자 $A$ 하나가 생깁니다. 유한한 고리에서도 이 항등식으로 정확한 주변분포를 구합니다.

스핀 $N$개인 고리에서 전달 행렬의 고윳값은 $2\cosh K$와 $2\sinh K$입니다. 따라서
$$
Z_N(K)=(2\cosh K)^N[1+(\tanh K)^N].
$$
한 단계 뒤에는 $A^{N/2}Z_{N/2}(K')$로 정규화를 복원합니다. 실험에서 복원한 $\log Z$는 유한 고리의 값입니다. $a\xi$는 격자 간격 $a$에서 대응하는 무한 사슬의 상관 길이입니다.

## 3. 길이와 고정점 추적하기

$\tanh K'=(\tanh K)^2$이므로 원래 두 칸 떨어진 상관을 새 격자의 한 칸 상관으로 읽을 수 있습니다. 무한 사슬에서 $0<K<\infty$일 때 격자 간격을 단위로 잰 상관 길이는 $\xi=-1/\log(\tanh K)$입니다. 한 단계에서 이 수는 절반이 되고 격자 간격은 두 배가 됩니다. 두 값을 곱한 물리적 길이는 유지됩니다.

임의의 유한한 양의 $K$에서 반복하면 $K$는 0에 접근합니다. 완전히 정렬된 $K=\infty$도 고정점입니다. $q=e^{-2K}$로 쓰면 $q'=2q/(1+q^2)$이므로 완전 정렬 근처에서 작은 $q$는 선형 수준에서 두 배가 됩니다. 길이 단위도 두 배입니다. 미분 가능한 다변수 변환의 선형화에서 고윳값의 절댓값이 1보다 크면 해당 교란이 증가하고 1보다 작으면 감소합니다. 절댓값이 1이면 고차항을 확인합니다. [Tong, Statistical Field Theory §3.1](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S3)

## 4. 선택한 결합 항만으로 표현 가능한지 점검하기

남긴 이웃 스핀 네 개와 결합한 중심 스핀을 제거하면 국소 로그 가중치는 $f(K(s_1+s_2+s_3+s_4))$이고 $f(x)=\log(2\cosh x)$입니다. 네 부호의 곱들은 모든 부호 배치에서 균등 평균을 취할 때 직교하는 기저를 이룹니다. $s_1s_2s_3s_4$의 계수는 다음과 같습니다.

$$
c_4=\frac{f(4K)-4f(2K)+3f(0)}8.
$$

$K=0.4$에서 약 $-0.027026$입니다. 원래 모든 결합이 두 스핀 사이에 있었지만 중심 스핀을 합하면 네 스핀 항이 생깁니다. 이 항을 버리면 유효 상호작용을 더 작은 결합 집합에 투영하고 주변분포를 바꾸게 됩니다. 버린 계수의 크기와 관심 관측량의 변화를 모두 확인할 수 있습니다.

## 5. 정확한 가중치와 절단한 가중치 비교하기

```python
from itertools import product
from math import cosh, exp, log, prod, sqrt, tanh

def ring(length, coupling):
    return sum(exp(coupling * sum(s[i] * s[(i+1) % length] for i in range(length)))
               for s in product((-1, 1), repeat=length))

K = 0.4
K_new = 0.5 * log(cosh(2*K))
A = 2 * sqrt(cosh(2*K))
assert abs(ring(8, K) - A**4 * ring(4, K_new)) < 1e-10
assert abs(tanh(K_new) - tanh(K)**2) < 1e-14

def log_weight(x):
    return log(2*cosh(x))

states = list(product((-1, 1), repeat=4))
c4 = sum(prod(s) * log_weight(K*sum(s)) for s in states) / 16
closed_form = (log_weight(4*K) - 4*log_weight(2*K) + 3*log_weight(0))/8
assert abs(c4 - closed_form) < 1e-14
exact = [exp(log_weight(K*sum(s))) for s in states]
truncated = [exp(log_weight(K*sum(s)) - c4*prod(s)) for s in states]
z, z_truncated = sum(exact), sum(truncated)
tv = 0.5 * sum(abs(p/z - q/z_truncated) for p, q in zip(exact, truncated))
assert tv > 0
print(K_new, c4, tv)
```

## 6. 확인할 항목

- 약한 결합에서 $\log\cosh(2K)$를 전개하고 한 단계 뒤 $K$가 얼마나 줄어드는지 예측하세요.
- 한 결합 양 끝의 스핀이 같은 경우와 다른 경우를 비교하고 $A$가 국소 가중치를 어떻게 복원하는지 확인하세요.
- 네 이웃 예제에서 네 스핀 계수를 제거하고 16개 배치 모두에서 국소 로그 가중치 차이를 확인하세요.
- 통계·머신러닝 모형의 유효 상호작용을 근사하기 전에 주변화 뒤에도 보존할 관측량을 정하세요.
