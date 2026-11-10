## 실행한 연산 그래프를 미분한다

F(x,y)=sin(xy)+x²를 생각합시다. 입력 두 개, 곱 a=xy, b=sin(a), 또 다른 곱 c=x*x, 마지막 합 F=b+c의 여섯 node로 나타낼 수 있습니다. x는 여러 경로에 등장하므로 모든 기여를 합쳐야 합니다. 제곱을 곱셈으로 구현하면 두 인자 모두 x라는 점도 중요합니다.

실험은 값·부모 index·local 편미분을 가진 실제 primitive tape를 만듭니다. 그림에서 primal·tangent·adjoint를 전환하세요. 각 값은 실제 전달 계산에서 나오며, 별도로 둔 해석적 gradient는 검산 기준입니다.

## Forward는 방향을 앞으로 전달한다

입력 방향 v=(vx,vy)를 고르면 chain rule로 tangent를 계산합니다.

$$
\dot a=yv_x+xv_y,\quad
\dot b=\cos(a)\dot a,\quad
\dot c=2xv_x,\quad
\dot F=\dot b+\dot c=Jv.
$$

한 번의 forward sweep은 선택한 입력 방향에 대한 Jacobian action을 구합니다. 출력이 많아도 그 방향의 출력 변화를 함께 얻습니다. 전체 Jacobian이 필요하면 여러 독립 방향이 필요하므로 입력·출력 차원과 원하는 미분 곱을 구별해야 합니다.

## Reverse는 Cotangent를 뒤로 전달한다

Scalar 출력에 adjoint w를 주고 tape를 거꾸로 읽으며 부모에게 local 기여를 더합니다. x는 a를 거치는 경로와 c=x*x의 두 부모 기여를 모두 받습니다.

$$
\bar x=w\left(y\cos(xy)+2x\right),\qquad
\bar y=w x\cos(xy).
$$

공유 부모의 값을 덮어쓰면 경로 하나가 사라집니다. 그림은 읽기 쉽도록 x에서 square로 가는 선을 하나로 합치지만, 표와 실제 역전파에는 두 부모 기여를 모두 기록합니다.

Scalar 출력에서는 reverse 한 번으로 모든 입력 편미분에 w를 곱한 VJP를 얻습니다. Forward 상태를 저장하거나 다시 계산해야 하므로 메모리 문제가 생기며 checkpointing 단원에서 이어집니다. AD는 실행한 프로그램의 primitive 미분 규칙을 전달하며, 모든 식을 기호적으로 정리하거나 가까운 두 함수값을 빼는 방법은 아닙니다.

## Pairing으로 서로 검산하기

w(Jv)=(wJ)v는 다른 방향으로 전달한 두 결과의 일치 조건입니다. (x,y)=(1,0)의 gradient는 (2,1)입니다. v=(1,1), w=1이면 양변이 3입니다. v만 (0,1)로 바꾸면 JVP는 1이 되고 primal과 VJP는 그대로입니다. w=0으로 바꾸면 reverse adjoint만 모두 0이 됩니다.

아래 그림은 방향을 따라간 실제 함수와 일차 tangent를 비교합니다. 접선의 일치는 국소적인 것이며 큰 보폭에서의 오차는 곡률과 관련됩니다.

## 직접 풀기: 경로 하나를 지워 보기

Square node에서 x의 기여 하나를 빠뜨리면 (1,0)에서 x 미분은 2가 아니라 1이 됩니다. 두 부모 기여를 각각 적고 정확한 JVP와 VJP를 다시 구하세요. 매끄러운 점에서는 적당한 중앙 차분으로 방향 미분을 검산할 수 있습니다. 다만 다음 단원처럼 꺾인 점에서는 차분의 일치만으로 미분값을 증명하지 못합니다.

더 읽기: [JAX 자동 미분 문서](https://docs.jax.dev/en/latest/automatic-differentiation.html).
