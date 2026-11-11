## 1. 좌표는 사건에 이름을 붙인다

좌표는 사건의 표지이고 관측량은 물리적 측정 절차를 지정합니다. 좌표 시간차에서 시계 읽기를 구하려면 세계선과 그 위의 계량이 필요합니다. 벡터 성분은 기저에 따라 바뀌며, 관측자의 기저와 수축한 값은 측정량을 나타낼 수 있습니다. 평탄한 시공간의 가속 좌표로 이 차이를 계산해 봅니다. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 2. 가속 좌표계 구성하기

$c=1$로 놓고 민코프스키 시공간에서 $X>|T|$인 오른쪽 쐐기 영역을 사용합니다. 변환은
$$
T=\rho\sinh\eta,\qquad X=\rho\cosh\eta,\qquad \rho>0
$$
입니다. $\rho$가 일정한 선은 쌍곡선입니다. 역변환은 $\rho=\sqrt{X^2-T^2}$와 $\eta=\operatorname{artanh}(T/X)$입니다. 변환식을 미분하면
$$
-dT^2+dX^2=-\rho^2d\eta^2+d\rho^2
$$
를 얻습니다. 곡률이 0이어도 계량 성분 $g_{\eta\eta}=-\rho^2$는 달라집니다. 이 좌표는 민코프스키 시공간의 오른쪽 쐐기 영역을 덮습니다.

## 3. 가속하는 시계와 가속도계 읽기

$\rho$가 일정한 세계선에서 $\eta_1$과 $\eta_2$ 사이의 고유 시간은 $\rho(\eta_2-\eta_1)$입니다. 사원속도의 관성계 성분은 $(\cosh\eta,\sinh\eta)$입니다. 고유 시간으로 미분해 얻은 가속도 벡터의 노름이 가속도계 값 $a=1/\rho$입니다.

예를 들어 $\rho=1$과 $\rho=3$인 시계는 같은 단위 $\eta$ 변화에서 각각 1과 3의 고유 시간을 기록합니다. 고유 가속도는 각각 1과 1/3입니다. 같은 평탄한 시공간 안에서 서로 다른 세계선을 비교한 결과입니다.

## 4. 좌표 속도와 국소 속도 비교하기

빛의 경로에서 $d\rho/d\eta=\pm\rho$입니다. 정지 관측자의 국소 반지름 길이 변화는 $d\rho$이고 국소 시간 변화는 $\rho\,d\eta$입니다. 두 값의 비로 계산한 국소 빛의 속도는 쐐기 영역 어디서나 $\pm1$입니다. 좌표 속도가 달라져도 이 측정값은 같습니다. $\rho=0$인 좌표 경계에서 곡률이 발산하지 않습니다.

## 5. 두 좌표에서 간격 검산하기

같은 가속 세계선의 $\eta=0$과 $\eta=\Delta\eta>0$인 두 사건을 생각합니다. 관성 직선으로 이으면 고유 시간은 $2\rho\sinh(\Delta\eta/2)$입니다. 가속하는 시계가 읽는 고유 시간은 $\rho\Delta\eta$입니다. 두 계산에는 서로 다른 경로가 사용됩니다. 실험에서는 좌표값과 함께 두 읽기를 표시합니다.

```python
from math import atanh, cosh, isclose, sinh, sqrt
rho, eta = 3, 0.8
T, X = rho*sinh(eta), rho*cosh(eta)
assert isclose(sqrt((X-T)*(X+T)), rho)
assert isclose(atanh(T/X), eta)
accelerated = rho*eta
inertial = sqrt((T-X+rho)*(T+X-rho))
assert isclose(inertial, 2*rho*sinh(eta/2))
assert inertial > accelerated
assert rho/rho == 1
print(accelerated,inertial,1/rho)
```

## 6. 보고된 양 점검하기

- 사용한 좌표계와 그 정의역, 측정의 기준 세계선을 기록하세요.
- $\Delta\eta$를 고정하고 $\rho$를 바꿔 고유 시간, 고유 가속도와 좌표 빛 속도를 비교하세요.
- 사건에 변환과 역변환을 적용해 좌표를 검산하세요. 물리적 읽기는 계량과 선택한 관측자로부터 계산하세요.
