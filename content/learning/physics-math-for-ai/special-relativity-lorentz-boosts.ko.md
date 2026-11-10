## 1. 같은 사건의 좌표 비교하기

사건은 열차 앞쪽에서 빛이 번쩍이는 순간처럼 공간과 시간의 한 점입니다. 관성계는 그 계에서 정지하고 동기화된 시계를 써서 사건에 위치와 시각을 부여합니다. 두 관성계는 같은 사건에도 서로 다른 좌표를 줄 수 있습니다. 아래 변환은 일정한 상대 속도로 움직이는 관성계의 좌표를 연결합니다. [Einstein, On the Electrodynamics of Moving Bodies, §§1–3](https://courses.physics.ucsd.edu/2016/Winter/physics2d/einstein1905.pdf)

## 2. 부스트의 부호와 단위 정하기

두 성분의 단위를 길이로 맞추기 위해 $(ct,x)$를 씁니다. $S'$는 $S$에서 볼 때 속력 $v$로 $+x$ 방향을 향하고, $t=0$에 두 원점이 일치합니다. $\beta=v/c$, $\gamma=(1-\beta^2)^{-1/2}$이면
$$
ct'=\gamma(ct-\beta x),\qquad x'=\gamma(x-\beta ct).
$$
움직이는 원점의 선 $x=\beta ct$를 넣으면 $x'=0$입니다. $-\beta$인 부스트를 적용하면 원래 좌표로 돌아옵니다. 부스트 방향에 수직인 좌표는 그대로입니다.

## 3. 빛의 속도와 불변 간격 확인하기

$x=ct$를 대입하면 $x'=ct'$이고 $x=-ct$를 대입하면 $x'=-ct'$입니다. 각 관성계에서 빛의 좌표 속력은 $c$입니다. 두 사건의 좌표 차에 대해 직접 전개하면
$$
(\Delta ct')^2-(\Delta x')^2=(\Delta ct)^2-(\Delta x)^2.
$$
교차항이 상쇄됩니다. 보존되는 양은 유클리드 내적을 보존하는 회전에서 쓰는 제곱합과 다릅니다.

## 4. 동시성, 시계와 막대 비교하기

$\beta=0.6$에서 두 사건이 $\Delta ct=0$, $\Delta x=3$광초를 만족하면 $\Delta t'=-2.25$초입니다. 공간적으로 떨어진 사건의 시각은 각 관성계가 동기화한 시계로 읽습니다. $S'$에 정지한 시계의 두 눈금 사건은 $\Delta x'=0$이고 역변환을 적용하면 $\Delta t=\gamma\Delta t'$입니다.

$S$에서 움직이는 막대의 길이를 재려면 같은 $t$에 양 끝의 위치를 기록합니다. 그 두 사건의 정지계 위치 차이는 $\Delta x'=\gamma\Delta x$입니다. 양 끝이 정지해 있으므로 이 위치 차이는 막대의 고유 길이입니다. 움직이는 막대의 측정 길이는 고유 길이를 $\gamma$로 나눈 값입니다. 이때 사용한 양 끝 사건의 시각은 막대 정지계에서 서로 다릅니다.

## 5. 급속도로 같은 방향의 부스트 합치기

급속도 $\eta$는 $\beta=\tanh\eta$로 정의합니다. 따라서 $\gamma=\cosh\eta$, $\gamma\beta=\sinh\eta$입니다. 빛의 원뿔 좌표 $u=ct+x$, $w=ct-x$를 쓰면
$$
u'=e^{-\eta}u,\qquad w'=e^\eta w.
$$
곱 $uw$가 불변입니다. 같은 축에서 두 부스트를 연달아 적용하면 급속도를 더합니다. 속도 매개변수는 $(\beta_1+\beta_2)/(1+\beta_1\beta_2)$로 합성됩니다.

## 6. 계산 검산하기

```python
from math import isclose, sqrt

def boost(ct, x, beta):
    gamma = 1/sqrt((1-beta)*(1+beta))
    return gamma*(ct-beta*x), gamma*(x-beta*ct)

beta = 0.6
assert isclose(boost(5, 3, beta)[1], 0, abs_tol=1e-14)
assert isclose(boost(0, 3, beta)[0], -2.25)
assert isclose(boost(1, 1, beta)[0], boost(1, 1, beta)[1])

for ct, x in [(2, 1), (1, 2), (2, -2)]:
    transformed = boost(ct, x, beta)
    ctp, xp = transformed
    assert isclose(ctp**2-xp**2, ct**2-x**2, abs_tol=1e-14)
    assert all(isclose(p,q,abs_tol=1e-14) for p,q in zip(boost(*transformed,-beta),(ct,x)))
print(boost(0, 3, beta))
```

## 7. 측정 절차 점검하기

- 선 $x=\beta ct$ 위에서 움직이는 원점의 사건이 모두 $x'=0$인지 확인하세요.
- 동시에 기록한 두 사건의 $\Delta t'$를 양의 $\beta$와 음의 $\beta$에서 비교하세요.
- 한 시계 위의 두 사건으로 고유 시간을 읽으세요. 막대의 길이는 측정하는 관성계에서 동시에 기록한 양 끝 사건으로 구하세요.
- 같은 자료에 부스트와 역부스트를 적용하고 표시값을 반올림하기 전에 불변량을 비교하세요.
