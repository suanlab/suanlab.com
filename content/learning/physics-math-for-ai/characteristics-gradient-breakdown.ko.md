## 장을 복원하기 전에 Label을 따라간다

선형 수송 u_t+c u_x=0의 특성은 X'=c입니다. 경로를 따라 d/dt u(X(t),t)=0이므로 초기 위치 xi의 label이 값 u0(xi)를 X=xi+ct로 운반합니다. 평행한 특성은 label 순서와 초기 모양을 유지합니다.

Burgers 방정식 u_t+u u_x=0에서는 속도 자체가 운반되는 값입니다. 매끄러운 고전적 해가 있는 동안 각 특성은 상수 값을 가져 직선이지만 이웃 선의 기울기는 다릅니다. 빠른 값이 느린 값을 따라잡을 수 있습니다.

## Affine 초기값으로 역변환을 드러내기

실험의 u0(xi)=a-b xi에 대해 다음입니다.

$$
X(\xi,t)=\xi+t(a-b\xi)=(1-bt)\xi+at,
\qquad \partial_\xi X=1-bt.
$$

Jacobian이 사라지기 전에는 xi를 풀어 Eulerian 장을 복원합니다. 운반된 값을 미분하면 다음입니다.

$$
u_x=\frac{u_0'(\xi)}{\partial_\xi X}
=-\frac b{1-bt}.
$$

b>0이면 첫 양의 breakdown 시간은 1/b입니다. b=1, t=1/2에서 gradient는 -2이고 t=1에는 이 affine 모형의 모든 특성이 위치 a에 모입니다. 운반된 값이 유한해도 공간 gradient가 폭발할 수 있습니다. 진폭 유계성과 고전적 매끄러움의 지속은 다릅니다.

## 교차한 경로를 고전적 해로 그리지 않기

위 그림은 t=0부터 2까지의 raw 특성 직선과 현재 시간 점을 보여줍니다. 교차는 label 구성의 성질이며 하나의 매끄러운 장을 계속 정의한다는 뜻이 아닙니다. Breakdown 이후에는 고전적 Eulerian gradient를 비워 둡니다.

아래 그림도 u 대 x가 아니라 현재 위치 X 대 초기 label xi입니다. 기울기가 0이 되거나 음수가 된 뒤 pre-breakdown 공식을 무심코 이어 쓰면 원래 고전적 해의 소실을 숨깁니다. 다음 단원의 entropy 약한 해에는 추가 선택 원리가 필요합니다.

## 모든 초기값이 미래에 폭발하지는 않는다

b=0이면 Burgers 특성도 평행합니다. b<0에서는 양의 시간에 Jacobian이 커져 이 affine profile의 gradient breakdown이 없습니다. 일정 속도 수송에서는 b와 무관하게 Jacobian이 항상 1입니다. 그래프의 교차뿐 아니라 어떤 방정식인지 확인하세요.

## 가역성·기울기 폭발·역변환 민감도

유계 도함수를 갖는 C¹ 유계 Burgers 초기값에서는 X(ξ,t)=ξ+tu₀(ξ)의 Jacobian이 1+tu₀′(ξ)다. 이것이 균일하게 양수인 동안 실수 전체의 역함수가 있다. 유계 u₀ 때문에 X는 무한대로 빠져나가는 점을 무한대로 보내고, 양의 도함수 때문에 엄격히 증가한다. 각 특성을 따라

$$
u_x(X(\xi,t),t)=\frac{u_0'(\xi)}{1+tu_0'(\xi)},\qquad
\frac d{dt}u_x=-u_x^2.
$$

m=inf u₀′<0이면 균일한 gradient 상계를 처음 잃는 시간은 T*=−1/m이다. Infimum이 달성되면 그 특성에서 기울기가 폭발한다. u₀(ξ)=−tanh ξ에서는 ξ=0에서 m=−1이므로 T*=1이지만 운반 값은 계속 −1과 1 사이에 있다. 초기 기울기가 비음수이면 이런 순방향 압축 폭발은 없다.

역도함수 ∂ξ/∂x=1/(1+tu₀′)는 민감도이기도 하다. 압축 중심의 위치 오차 δx는 약 δx/(1−t)의 label 오차를 만든다. 특성 위치가 정확해 보여도 breakdown 근처 Eulerian 장의 복원이 나쁠 수 있다. 실제 affine 실습도 같은 Jacobian을 가지지만 전역 유계 초기값은 아니다. 그 명시적 공식으로 breakdown 전 역함수를 직접 얻는다.

| tanh 중심의 t | Jacobian | 역변환 민감도 |
|---|---:|---:|
| 1/2 | 1/2 | 2 |
| 9/10 | 1/10 | 10 |
| 99/100 | 1/100 | 100 |

```python
from math import tanh, cosh
for t in (.5, .9, .99):
    slope0 = -1.
    J = 1+t*slope0
    slope = slope0/J
    slope_time_derivative = -slope0**2/J**2
    assert abs(slope_time_derivative+slope*slope) < 1e-9
    for xi in (-1., 0., 1.):
        u0 = -tanh(xi)
        derivative = -1/cosh(xi)**2
        assert 1+t*derivative > 0
        position = xi+t*u0
        assert abs(position-xi) <= t
    print(t, J, 1/J, slope)
```

도함수·Riccati 항등식은 해석적으로 유도했고 코드는 T* 전 대표값을 검산한다. 이후 교차한 label 그림은 단일값 고전적 해가 아니다. 접힌 사상의 수치 역산이 아닌 entropy 선택이 물리적 연속을 정한다.

## 직접 풀기: 교차의 의미

a=1/2, b=1, Burgers에서 label 0, t=1/2의 위치는 1/4, 운반 값은 1/2, gradient는 -2입니다. t=1에는 모든 label의 위치가 1/2로 붕괴합니다. 서로 다른 운반 값이 그 위치의 한 고전적 값을 정의할 수 없는 이유를 설명하세요. 상수 속도 1/2로 바꾸면 label 순서가 유지되는지도 확인하세요.

더 읽기: [Clawpack 저자의 Burgers 설명](https://www.clawpack.org/riemann_book/html/Burgers.html).
