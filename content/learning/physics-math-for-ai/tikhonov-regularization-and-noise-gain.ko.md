## 1. 잡음이 있는 적합에 선호 조건 추가하기

관측 $y$와 순방향 행렬 $A$에 대해 0차 티호노프 정규화는 다음 값을 최소화합니다.

$$
\|Ax-y\|_2^2+\lambda\|x\|_2^2,\qquad \lambda\ge0.
$$

첫 항은 관측과의 차이를 재고, 둘째 항은 정한 좌표에서 큰 계수에 비용을 부과합니다. 매개변수의 척도가 바뀌면 벌점의 의미도 바뀌므로 단위를 명시하거나 무차원화합니다. [Hansen, 이산 역문제 4장](https://epubs.siam.org/doi/10.1137/1.9780898718836.ch4)

## 2. 특이 모드 하나의 필터 유도하기

관측 강도 $s>0$과 관측 좌표 $y$인 모드에서 $(sx-y)^2+\lambda x^2$를 최소화합니다. 미분을 0으로 두면

$$
\widehat x_\lambda=\frac{s}{s^2+\lambda}y.
$$

잡음의 증폭률은 $g_\lambda=s/(s^2+\lambda)$입니다. $y=sx_*+\eta$이면 신호에 곱해지는 계수는 $s^2/(s^2+\lambda)$이고, 잡음에는 $g_\lambda$가 곱해집니다. $s=0$이고 $\lambda>0$이면 식은 0을 고릅니다. 이 값은 0을 기준으로 둔 벌점에서 정해집니다.

## 3. 편향과 분산 함께 계산하기

잡음의 평균이 0이고 분산이 $\sigma^2$이면 편향은 $-\lambda x_* /(s^2+\lambda)$, 분산은 $g_\lambda^2\sigma^2$입니다. 평균제곱오차는 편향의 제곱과 분산의 합입니다. $s=0.05$, $x_*=1$, $\sigma=0.05$이면 정규화 전 잡음 표준편차는 1입니다. $\lambda=s^2=0.0025$에서는 신호 보존 비율과 잡음 표준편차가 모두 0.5입니다. 이 예의 평균제곱오차는 1에서 0.5로 줄어듭니다.

## 4. 여러 모드의 행렬에 적용하기

$A=U\operatorname{diag}(s_i)V^T$로 특잇값 분해합니다. $y$를 왼쪽 특이벡터에 투영하고 각 좌표에 $s_i/(s_i^2+\lambda)$를 곱한 뒤 오른쪽 특이벡터로 복원합니다. 같은 양의 $\lambda$에서 작은 특잇값 방향은 더 많이 줄어듭니다. $A$와 $x$에 물리 단위가 있다면 벌점의 척도를 명시합니다. 여기서 $\lambda$는 목적함수에 곱하는 계수 그 자체입니다.

## 5. Python으로 직접 풀이와 모드 필터 비교하기

```python
import numpy as np

strengths = np.array([1.0, .05])
A = np.diag(strengths)
truth = np.ones(2)
y = A @ truth + np.array([0.0, .05])
for penalty in [0.0, .0025, .025]:
    x = np.linalg.solve(A.T @ A + penalty * np.eye(2), A.T @ y)
    filters = strengths / (strengths**2 + penalty)
    assert np.allclose(x, filters * y)
    print(penalty, x)
assert np.isclose((strengths[1] / (strengths[1]**2 + .0025)) * .05, .5)
```

## 6. 추론 목적에 맞춰 벌점 고르기

실제 문제에서는 참 매개변수를 모르는 경우가 많습니다. 따로 남긴 관측, 독립적으로 추정한 잡음 크기, 근거가 있는 사전 분포로 벌점의 크기를 정합니다. 과학 질문에 중요한 방향이 정규화 후에도 남는지 확인합니다. 벌점 정의와 선택에 따른 민감도를 기록합니다. 잡음을 맞추면 잔차가 작아질 수 있고, 벌점이 너무 크면 실제 신호가 줄어들 수 있습니다.
