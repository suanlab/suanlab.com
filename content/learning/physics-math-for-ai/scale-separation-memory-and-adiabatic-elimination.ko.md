## 1. 완화 시간 구분하기

두 변수가 서로 다른 시간 척도에서 변할 수 있습니다. 축소 모형은 빠른 변수를 느린 변수의 함수로 대체하기도 합니다. 궤적, 초기 조건과 감쇠율을 비교해 축소가 유효한지 검사합니다. 시간 척도의 비를 모형에 명시하고 실제 관측 시간에 어떤 차이가 중요한지 확인하세요.

## 2. 근사 없이 빠른 변수 제거하기

$D>1$, $0\leq\gamma<1$인 $\dot x=-x+y$, $\dot y=D(\gamma x-y)$를 생각합니다. 행렬의 대각합은 음수이고 행렬식은 $D(1-\gamma)>0$이므로 두 모드는 감쇠합니다. 두 번째 식을 적분 인자로 풀면 다음과 같습니다.
$$
y(t)=e^{-Dt}y_0+\gamma D\int_0^t e^{-D(t-s)}x(s)\,ds .
$$
첫 번째 식에 넣으면 느린 변수의 식에 과거의 $x$와 숨은 초기 상태의 항이 남습니다. $x_0$가 같아도 $y_0$가 다르면 $\dot x(0)$가 다릅니다. 이 선형 예제에서 동역학을 투영할 때 기억항이 생기는 과정을 볼 수 있습니다. [Mori, Transport, Collective Motion, and Brownian Motion](https://doi.org/10.1143/PTP.33.423)

## 3. 기억 적분을 순간 응답으로 바꾸기

$D$가 크면 지수 커널의 폭은 $1/D$입니다. 초기 경계층을 지난 뒤 그 폭 안에서 $x$가 거의 변하지 않으면 $y\approx\gamma x$와 $\dot x\approx-(1-\gamma)x$를 사용할 수 있습니다. $y-\gamma x$를 미분하면 다음 상계를 얻습니다.
$$
|y(t)-\gamma x(t)|
\leq e^{-Dt}|y_0-\gamma x_0|
 +\frac{\gamma L}{D}(1-e^{-Dt}),\qquad L=\sup_{0\le s\le t}|\dot x(s)|.
$$
첫 항은 초기 불일치의 영향을 나타냅니다. 두 번째 항은 빠른 변수가 응답하는 동안 느린 변수가 얼마나 움직이는지 나타냅니다. $y_0=\gamma x_0$로 시작해도 $x$가 변하면 지연이 생깁니다.

## 4. 전체 계의 시간 척도 비교하기

감쇠 고윳값은 다음과 같습니다.
$$
\lambda_\pm=-\frac{1+D}{2}\pm\frac12\sqrt{(D-1)^2+4\gamma D}.
$$
느린 감쇠율을 계산할 때 $\lambda_+=-2D(1-\gamma)/(1+D+\sqrt{(D-1)^2+4\gamma D})$를 쓰면 거의 같은 두 수의 차를 피할 수 있습니다. $\lambda_+$와 축소 감쇠율 $-(1-\gamma)$를 비교한 뒤 빠른 시간 상수 몇 배가 지난 궤적을 살펴보세요. 두 감쇠율이 가까우면 어느 한 모드도 빨리 사라지지 않습니다.

## 5. 궤적에서 근사 검증하기

```python
from math import exp, sqrt

def slow_state(t, D, gamma, y0):
    gap = sqrt((D-1)**2 + 4*gamma*D)
    fast = -(1+D+gap)/2
    slow = D*(1-gamma)/fast
    weight_slow = (y0-1-fast)/gap
    return weight_slow*exp(slow*t) + (1-weight_slow)*exp(fast*t)

gamma = 0.6
times = [i/100 for i in range(501)]
def error(D, y0):
    return max(abs(slow_state(t,D,gamma,y0) - exp(-(1-gamma)*t)) for t in times)

assert slow_state(0, 4, gamma, 0) == 1
assert error(32,gamma) < error(4,gamma)
for D in [2,4,8,32]:
    print(D, error(D,gamma), error(D,0))
```

## 6. 검증할 양 고르기

- $x_0$는 같고 $y_0$가 다른 두 궤적을 초기화하세요. 초기 기울기와 늦은 시간의 거동을 비교합니다.
- $D$를 늘리고 실제 적용에서 사용하는 시점에 근사가 개선되는지 확인하세요.
- 감쇠율 오차와 함께 궤적 오차를 기록하세요. 감쇠율이 같아도 진폭은 다를 수 있습니다.
- 시간 전진 모형을 학습할 때 숨은 자유도가 아직 완화되지 않은 상황에서 최근 관측을 추가하면 예측이 개선되는지 검사하세요.
