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

## Reverse 계산을 미분하여 Hessian action 얻기

매끄러운 점에서는 gradient를 구하는 reverse 프로그램 자체에 방향 v를 전달할 수 있다. 이 forward-over-reverse는 전체 Hessian을 저장하지 않고 Hv를 구한다. 이 함수의 독립적인 행렬식은

$$
H=\begin{pmatrix}
2-y^2\sin(xy)&\cos(xy)-xy\sin(xy)\\
\cos(xy)-xy\sin(xy)&-x^2\sin(xy)
\end{pmatrix}.
$$

아래에서는 tape의 값과 reverse adjoint를 모두 값·tangent를 갖는 dual number로 계산한다. 따라서 역전파 중 local partial과 자식 adjoint를 모두 미분한다. 어느 항을 빠뜨려도 Hessian 기여를 잃는다. Square의 두 인자를 포함한 공유 부모에도 각각 더해야 한다.

x=y=0에서는 gradient가 0이지만 Hessian은 0이 아니다.

| v | Hv |
|---|---|
| (1,0) | (2,1) |
| (0,1) | (1,0) |
| (1,−1) | (1,1) |

```python
from math import sin, cos
class D:
    def __init__(self, value, tangent=0.): self.v, self.t = value, tangent
    def __add__(self, other): return D(self.v+other.v, self.t+other.t)
    def __mul__(self, other): return D(self.v*other.v, self.t*other.v+self.v*other.t)
def sine(x): return D(sin(x.v), cos(x.v)*x.t)
def cosine(x): return D(cos(x.v), -sin(x.v)*x.t)
def hvp(x, y, vx, vy):
    values, parents = [D(x, vx), D(y, vy)], [[], []]
    def node(value, edges):
        values.append(value); parents.append(edges); return len(values)-1
    a = node(values[0]*values[1], [(0, values[1]), (1, values[0])])
    b = node(sine(values[a]), [(a, cosine(values[a]))])
    c = node(values[0]*values[0], [(0, values[0]), (0, values[0])])
    node(values[b]+values[c], [(b, D(1)), (c, D(1))])
    bars = [D(0) for _ in values]; bars[-1] = D(1)
    for i in range(len(values)-1, -1, -1):
        for parent, partial in parents[i]: bars[parent] = bars[parent]+partial*bars[i]
    return (bars[0].t, bars[1].t)
for x, y in ((0., 0.), (1., .5)):
    for vx, vy in ((1., 0.), (0., 1.), (1., -1.)):
        actual = hvp(x, y, vx, vy)
        cross = cos(x*y)-x*y*sin(x*y)
        expected = ((2-y*y*sin(x*y))*vx+cross*vy, cross*vx-x*x*sin(x*y)*vy)
        assert max(abs(a-b) for a, b in zip(actual, expected)) < 1e-14
        print((x, y), (vx, vy), actual)
```

비영 점에서도 위 해석 행렬과 비교한다. 기존 일차 미분 UI에 새 모드를 추가한 것이 아니라 별도의 이차 tape 예제다. Hessian 대칭성에는 매끄러움이 필요하며 custom 미분이나 비매끄러운 branch 관례에는 별도 분석이 필요하다.

## 직접 풀기: 경로 하나를 지워 보기

Square node에서 x의 기여 하나를 빠뜨리면 (1,0)에서 x 미분은 2가 아니라 1이 됩니다. 두 부모 기여를 각각 적고 정확한 JVP와 VJP를 다시 구하세요. 매끄러운 점에서는 적당한 중앙 차분으로 방향 미분을 검산할 수 있습니다. 다만 다음 단원처럼 꺾인 점에서는 차분의 일치만으로 미분값을 증명하지 못합니다.

더 읽기: [JAX 자동 미분 문서](https://docs.jax.dev/en/latest/automatic-differentiation.html).
