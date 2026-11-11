## 불변량은 무엇을 유지하는가

실수 평면에 네 번의 90도 회전으로 이루어진 C4가 작용한다고 합시다. 함수 F가 모든 회전에서 같은 값을 내면 불변입니다. 점 자체가 회전하지 않는다는 뜻이 아니라 함수 출력이 같다는 뜻입니다. 원래 특징 φ는 x, x², xy 중 선택합니다. 네 회전 위치의 특징을 실제로 계산해 평균하면 새 불변 함수를 만들 수 있습니다.

$$
F(gv)=F(v)\quad(g\in C_4),\qquad P\phi(v)=\frac14\sum_{k=0}^3\phi(R^kv).
$$

## 평균이 불변인 이유

입력을 한 번 회전하면 합의 순서만 한 칸 이동하고 마지막 회전은 첫 항으로 돌아옵니다. 따라서 총합은 같습니다. 일반적인 유한군에서도 g를 gh로 바꾸는 것이 군 원소들의 순열이라 같은 논증이 됩니다. 이 평균은 실수 계수에서 수행하며 군 크기로 나눌 수 있다는 조건을 사용합니다. 유한체의 표수가 군 크기를 나누면 이 식의 분모가 가역이 아닐 수 있습니다.

$$
P\phi(hv)=\frac1{|G|}\sum_{g\in G}\phi(ghv)=P\phi(v).
$$

## 예측하고 숫자로 확인하기

v=(1,2)와 φ=x²이면 특징은 1,4,1,4이고 평균은 2.5입니다. 일반 좌표에서도 평균은 (x²+y²)/2입니다. φ=x 또는 xy를 고르면 양·음 항이 상쇄되어 0입니다. 파랑 막대는 각 회전의 특징 값, 주황 막대는 같은 평균 값입니다. 이산 회전 번호 사이를 연속 시간이나 각도 함수로 보간하지 않습니다.

$$
P(x^2)=\frac{x^2+y^2}{2},\qquad P(x)=P(xy)=0.
$$

## 정보 손실과 등변성의 차이

궤도의 평균 좌표는 항상 (0,0)이므로 원점이 아닌 입력도 사라집니다. 불변량이라는 성질만으로 모든 궤도를 구분하는 완전한 특징이 되는 것은 아닙니다. 예를 들어 같은 원 위의 서로 다른 C4 궤도는 같은 x² 평균을 가질 수 있습니다. 분류에서 방향을 지우는 것이 적절할 수 있지만 위치를 예측하는 작업에서는 출력도 함께 회전하는 등변성이 더 적합할 수 있습니다.

$$
\text{invariant: }F(gv)=F(v),\qquad\text{equivariant: }F(gv)=gF(v).
$$

## 모든 C4 궤도를 구분하는 특징을 직접 만들기

방사형 평균은 (1,2)와 (2,1)을 구분하지 못하지만 두 점은 서로의 90도 회전 결과가 아닙니다. 실수 점을 복소수 z=x+iy로 읽으면 90도 회전은 i를 곱하는 연산입니다. 따라서 네제곱은 바뀌지 않습니다. 네제곱의 실수부와 허수부를 모두 보존합니다.

$$
A=x^4-6x^2y^2+y^4,\qquad B=4xy(x^2-y^2),\qquad F_4(v)=(A,B).
$$

| 입력 | x²의 평균 | 네제곱 특징 (A,B) |
|---|---|---|
| (1,2) | 5/2 | (−7,−24) |
| (2,1) | 5/2 | (−7,24) |

이 특징은 두 예제뿐 아니라 **모든** C4 궤도를 구분합니다. 영이 아닌 z,z′의 네제곱이 같다면 (z′/z)⁴=1입니다. 이 방정식의 복소수 근은 1,i,−1,−i이므로 두 점의 차이는 90도 회전의 반복입니다. z=0인 경우도 네제곱의 일치가 z′=0을 강제합니다. 역방향은 불변성에서 바로 따릅니다. 반사는 일반적으로 B의 부호를 바꿉니다. 따라서 이들은 C4 불변량이며 D4 불변량은 아닙니다. 특히 표의 두 입력은 좌표를 맞바꾸는 정사각형 반사로 연결되므로 D4 불변 특징이라면 둘을 같게 보아야 합니다.

## 궤도 구분과 다항식환 생성은 다른 주장

s=x²+y²도 불변량입니다. A,B가 궤도를 구분하더라도 s를 A,B의 다항식으로 쓸 수는 없습니다. A,B로 만든 비상수 단항식의 차수는 양의 4의 배수지만 s는 2차 동차식이기 때문입니다. 더 직접적으로 (x,y)=(t,0)에 다항식 항등식을 제한하면 t²을 t⁴의 다항식으로 표현해야 하며 계수 비교로 불가능합니다. 함수로서의 s는 다항식이 아닌 근을 사용하면 복원할 수 있습니다.

$$
A^2+B^2=s^4,\qquad s=(A^2+B^2)^{1/4}\quad(s\ge0).
$$

따라서 궤도를 구분한다는 것과 불변 다항식환 전체를 생성한다는 것은 서로 다릅니다. 여기서는 이 특징의 전자를 증명했으며 일반 불변량 생성 알고리즘을 구현했다고 주장하지 않습니다.

## 정보를 구분해도 gradient의 크기는 나쁠 수 있다

입력 v를 실수 t배 하면 F4(v)는 t⁴배 됩니다. 원점에서는 모든 일차 미분이 0입니다. x축에서 입력 이동 ε은 특징 이동 ε⁴만 만들므로, C4로 같은 입력을 식별한 뒤에도 원점 부근에서 역방향 Lipschitz 상계를 줄 수 없습니다. 정확한 궤도 구분이 잡음이 있는 특징에서의 안정적 복원이나 유용한 gradient를 보장하지는 않습니다. s 같은 낮은 차수의 불변 특징을 함께 쓰면 이 예제의 방사형 스케일은 개선할 수 있지만 실제 예측 성능은 과제에 따라 평가해야 합니다.

$$
F_4(tv)=t^4F_4(v),\qquad DF_4(0)=0.
$$

## 특징과 유한 예제를 독립적으로 검산하기

다음 표준 라이브러리 코드는 정수와 정확한 분수 스케일을 사용합니다. 25점 정사각 격자는 C4 작용에 닫혀 있고, 특징이 같은 일곱 집합은 정확히 일곱 궤도입니다. 이 유한 열거는 앞의 일반 증명을 예시합니다. 현재 페이지의 실습 막대는 여전히 x,x²,xy 중 선택한 특징의 평균을 보여주며, 네제곱 특징은 아래에서 별도로 실행하는 확장 예제입니다.

```python
from fractions import Fraction as Q
from itertools import product

def turn(v):
    x, y = v
    return (-y, x)

def orbit(v):
    points = []
    for _ in range(4):
        points.append(v)
        v = turn(v)
    return set(points)

def feature(v):
    x, y = v
    return (x**4 - 6*x*x*y*y + y**4, 4*x*y*(x*x-y*y))

def radial(v):
    x, y = v
    return Q(x*x + y*y) / 2

v, w = (1, 2), (2, 1)
assert radial(v) == radial(w) == Q(5, 2)
assert w not in orbit(v)
assert feature(v) == (-7, -24) and feature(w) == (-7, 24)
groups = {}
for point in product(range(-2, 3), repeat=2):
    key = feature(point)
    groups.setdefault(key, set()).add(point)
    assert all(feature(q) == key for q in orbit(point))
    x, y = point
    assert sum(a*a for a in key) == (x*x+y*y)**4
assert len(groups) == 7
assert all(points == orbit(next(iter(points))) for points in groups.values())
for t in [Q(1, 2), Q(1, 8)]:
    assert feature(tuple(t*a for a in v)) == tuple(t**4*a for a in feature(v))
    print("scale:", t, "quartic feature:", feature(tuple(t*a for a in v)))
print("radial:", radial(v), radial(w), "quartic:", feature(v), feature(w))
print("grid points:", 25, "orbits:", len(groups))
```

## 추가 문제와 연구 연결

1. A만 남겨도 표의 두 입력을 구분하나요? **둘 다 −7이므로 구분하지 못하며 B의 값이 필요합니다.**
2. 불변 특징이 원래 방향을 보존하나요? **선택한 군으로 연결되는 방향은 의도적으로 같게 보고 궤도를 보존합니다.**
3. 학습 실험에서는 무엇을 따로 보고해야 하나요? **선택한 군, 특징의 구분 성질, 원점 부근의 거동, 예측 결과입니다. 유한 격자의 검사만으로 일반 구분 정리가 증명되지는 않습니다.**

네제곱 증명의 [복소수 좌표](../real-complex-numbers/)와 출력도 함께 변하는 [대수적 표현](../algebraic-representations-scope/)을 연결해 복습하세요.

## 확인 문제와 해설

1. 입력 (2,1)의 x² 평균은? **2.5로 같습니다.**
2. 평균이 0이면 입력도 0인가요? **아니요. 선형 특징이 반례입니다.**
3. 불변화 후 두 번 평균하면? **같은 결과이며 P²=P입니다.**

추가 읽기: [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/).

[Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf).
