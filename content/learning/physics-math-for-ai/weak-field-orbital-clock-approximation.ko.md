## 1. 작은 매개변수 찾기

약한 장과 느린 운동의 근사는 $|\Phi|/c^2$와 $v^2/c^2$에 대해 전개합니다. $\Phi$는 뉴턴 퍼텐셜이고 $v$는 그 약한 장 표현에서의 좌표 속력입니다. 일차까지 시계의 진행률은
$$
\frac{d\tau}{dt}\simeq 1+\frac{\Phi}{c^2}-\frac{v^2}{2c^2}
$$
입니다. 버린 항은 작은 양들의 이차 이상입니다. 목표 정확도가 있을 때는 그 크기를 확인해야 합니다.

## 2. 정확한 원형 궤도를 기준으로 삼기

슈바르츠실트 외부에서 $G=c=1$, $u=M/r$을 씁니다. 적도면의 원형 측지선에서 방사 방향 식은 $\Omega^2=(d\phi/dt)^2=M/r^3$을 줍니다. 계량에 $dr=0$을 대입하면 다음을 얻습니다.
$$
\left(\frac{d\tau}{dt}\right)^2=1-2u-r^2\Omega^2=1-3u.
$$
여기서 $t$는 무한대에서 정규화한 슈바르츠실트 시간입니다. 원형 궤도는 $r>6M$에서 안정하고 실험의 하한인 $6M$은 경계입니다. [Carroll, The Schwarzschild Solution](https://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll7.html)

## 3. 국소 운동과 중력에 의한 시계 진행률 나누기

같은 반지름의 정지 시계는 $\sqrt{1-2u}$로 진행합니다. 그 관측자가 재는 원형 궤도 입자의 국소 속력 제곱은 $u/(1-2u)$입니다. 정지 시계의 진행률에 이 속력의 특수상대론 시간 지연 인자를 곱하면 $\sqrt{1-3u}$를 얻습니다. 약한 장에서 중력 보정의 선도항은 $-u$, 운동 보정의 선도항은 $-u/2$이므로 합하면 $1-3u/2$입니다.

## 4. 절단 오차 수치화하기

$a=3u$로 두고 차이의 분자를 유리화하면 다음과 같습니다.
$$
1-\frac a2-\sqrt{1-a}
 =\frac{a^2}{4\big(1-a/2+\sqrt{1-a}\big)}.
$$
일차식은 정확한 진행률보다 크며, 작은 $u$에서 오차는 이차로 줄어듭니다. $r=12M$에서는 각각 약 $0.875$와 $0.8660$입니다. 차이 약 $0.009$가 비교의 목표 정확도에 들어가는지 판단하세요. 진행률의 허용오차는 무차원입니다. 반지름을 고정한 동안 쌓인 시계 오차의 상계를 구하려면 진행률 오차에 경과 좌표 시간을 곱합니다.

```python
from math import isclose, sqrt

def rates(radius):
    u = 1/radius
    a = 3*u
    exact = sqrt(1-a)
    linear = 1-a/2
    error = a*a/(4*(linear+exact))
    static = sqrt(1-2*u)
    local_v_squared = u/(1-2*u)
    assert isclose(static*sqrt(1-local_v_squared),exact)
    assert isclose(linear-exact,error,rel_tol=1e-11)
    return exact, linear, error

for r in (6,12,48):
    print(r,rates(r))
assert rates(6)[2] > rates(12)[2] > rates(48)[2]
```

## 5. 근사의 적용 범위 정하기

- $r=6M$, $12M$, $48M$에서 오차를 비교하세요.
- 요구 정확도를 고정하고 그 정확도를 만족하는 반지름을 찾으세요.
- 같은 반지름의 정지 시계와 궤도 시계 진행률을 비교한 뒤 중력과 운동의 효과를 합하세요.
- 장의 세기, 속력과 궤도에 대한 가정을 명시하세요. 다른 계에 같은 전개를 적용할 때 가정을 다시 확인합니다.
