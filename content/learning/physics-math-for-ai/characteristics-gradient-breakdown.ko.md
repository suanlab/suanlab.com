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

## 직접 풀기: 교차의 의미

a=1/2, b=1, Burgers에서 label 0, t=1/2의 위치는 1/4, 운반 값은 1/2, gradient는 -2입니다. t=1에는 모든 label의 위치가 1/2로 붕괴합니다. 서로 다른 운반 값이 그 위치의 한 고전적 값을 정의할 수 없는 이유를 설명하세요. 상수 속도 1/2로 바꾸면 label 순서가 유지되는지도 확인하세요.

더 읽기: [Clawpack 저자의 Burgers 설명](https://www.clawpack.org/riemann_book/html/Burgers.html).
