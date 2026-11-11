## 1. 최종 관측값이 앞선 입력에 어떻게 의존하는지 묻기

초기 상태를 알고 이산 모형 $x_{n+1}=a x_n+u_n$을 반복합니다. $a$는 이전 상태가 얼마나 유지되는지를 나타내고, $u_n$은 $n$번째 입력입니다. 입력 크기를 관리하면서 최종 관측 $y$를 맞추려면 다음 손실을 정의할 수 있습니다.

$$
J=\tfrac12(x_N-y)^2+\tfrac{\gamma}{2}\sum_{n=0}^{N-1}u_n^2.
$$

순방향 계산으로 상태를 저장하고 손실을 구합니다. 민감도는 매개변수나 입력을 바꿀 때 같은 손실이 얼마나 달라지는지 나타냅니다. [Giles와 Pierce, 수반법을 이용한 설계](https://doi.org/10.1023/A:1011430410075)

## 2. 최종 민감도를 뒤로 전파하기

뒤에 오는 제어 입력을 고정했을 때 최종 오차 항을 $x_n$으로 미분한 값을 $p_n$이라 합니다. 연쇄법칙으로 끝 조건과 역방향 갱신식을 얻습니다.

$$
p_N=x_N-y,\qquad p_n=a p_{n+1}.
$$

한 단계 뒤로 갈 때마다 순방향 갱신식을 입력 상태로 미분한 값이 곱해집니다. 상태가 벡터이면 상태 야코비안의 전치가 곱해집니다.

## 3. 모든 제어 입력의 미분 읽기

$u_n$을 바꾸면 $x_{n+1}$이 직접 변하고, 입력 벌점에서는 $\gamma u_n$이 더해집니다.

$$
\frac{\partial J}{\partial u_n}=p_{n+1}+\gamma u_n,
\qquad
\frac{dJ}{da}=\sum_{n=0}^{N-1}p_{n+1}x_n.
$$

$x_0=1$, $a=0.5$, $u=(0,0)$, $y=0$, $\gamma=0.1$이면 상태는 $(1,0.5,0.25)$입니다. 두 입력에 대한 미분은 $(0.125,0.25)$, $a$에 대한 미분은 $0.25$입니다.

## 4. 목적에 맞는 민감도 계산 고르기

순방향 민감도는 하나의 교란을 앞으로 전파해 여러 출력에 미치는 영향을 구할 수 있습니다. 수반 계산은 스칼라 손실 하나의 미분을 뒤로 전파해 여러 입력의 영향을 구합니다. 역방향 계산에는 순방향 상태나 이를 다시 계산하는 방법이 필요합니다. 긴 시뮬레이션에서는 중간 상태 일부를 저장하여 메모리와 재계산 시간을 조절합니다. 이산 수반법은 코드가 실제 적용한 갱신식을 미분합니다.

## 5. 방향 유한차분으로 검산하기

```python
import numpy as np

u = np.array([.15, -.1, .2])
a, target, gamma = .7, .3, .1

def objective(a, u):
    states = [1.]
    for value in u:
        states.append(a * states[-1] + value)
    loss = .5 * (states[-1] - target)**2 + .5 * gamma * (u @ u)
    return loss, np.array(states)

loss, states = objective(a, u)
adjoint = np.empty(len(u) + 1)
adjoint[-1] = states[-1] - target
for n in range(len(u)-1, -1, -1):
    adjoint[n] = a * adjoint[n+1]
du = adjoint[1:] + gamma * u
da = adjoint[1:] @ states[:-1]

eps = 1e-6
direction, direction_a = np.array([.2, .4, -.3]), .1
numeric = (objective(a+eps*direction_a, u+eps*direction)[0]
           - objective(a-eps*direction_a, u-eps*direction)[0]) / (2*eps)
assert np.isclose(numeric, du @ direction + da * direction_a, atol=1e-9)
print(du, da)
```

## 6. 최적화 전에 미분 확인하기

교란 크기를 여러 값으로 바꾸며 수반 계산과 중앙차분을 비교합니다. 간격이 크면 비선형 효과로 인한 절단 오차가 나타날 수 있고, 지나치게 작으면 반올림 오차가 커질 수 있습니다. 대표 경계 조건과 입력 조건을 포함합니다. 미분 검산은 고른 순방향 모형의 구현을 확인합니다. 모형의 예측 자체는 독립 관측으로 검증합니다.
