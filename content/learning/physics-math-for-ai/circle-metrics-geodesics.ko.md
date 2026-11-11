## Metric은 접변위의 길이를 정합니다

Riemannian metric은 각 접공간에 매끄럽게 주어진 양의 정부호 내적입니다. 국소 길이와 각도를 측정하는 규칙입니다. 반지름 R의 원에서 각 θ로 위치를 미분하면 속력 R이므로 다음과 같습니다.

$$
ds^2=R^2d\theta^2,\qquad
L[\theta]=\int_0^1R|\dot\theta(t)|\,dt.
$$

이는 원 위의 경로만 허용하는 내재 기하입니다. 주변 평면을 가로지르는 현은 다른 허용 경로 문제입니다. 두 거리는 제약이 다릅니다.

## 각변위와 winding을 선택하기

끝점의 각도는 2π의 정수배만큼 모호합니다. δ를 (−π,π]의 대표값으로 택하고 같은 끝점이면 0으로 둡니다. 2πm을 더하면 winding 경로가 됩니다. 일정 속도 lift는 θ(t)=θ₀+t(δ+2πm)입니다.

$$
L_m=R|\delta+2\pi m|,\qquad
E_m=\frac12\int_0^1R^2\dot\theta^2dt
=\frac12R^2(\delta+2\pi m)^2.
$$

실험은 m=−1,0,1을 허용합니다. 최단 내재 거리는 R|δ|입니다. 대척점 δ=π에서는 m=0과 m=−1의 두 방향이 동률이므로 전역 최단 방향이 유일하지 않습니다.

## Geodesic이 항상 전역 최단은 아닙니다

상수 metric 좌표의 geodesic 방정식은 θ̈=0입니다. 따라서 일정 속도로 여러 바퀴 도는 경로도 국소적으로 이 방정식을 만족합니다. 국소 미분방정식만으로 임의로 긴 구간의 전역 최단성을 보장하지 않습니다.

R=1인 사분원의 최단 거리는 π/2≈1.571, 현은 √2≈1.414입니다. 양의 방향으로 한 바퀴 추가하면 선택한 길이는 5π/2≈7.854가 되지만 끝점과 최단 거리는 같습니다. 파랑은 원 전체, 주황은 선택 arc입니다. 여러 번 감긴 경로는 같은 선을 반복해 그리므로 그림뿐 아니라 길이와 에너지도 읽어야 합니다.

## 각 Lift에서 길이와 에너지 최소성을 증명하기

구간 위의 연속 원 경로는 시작 각도를 정하면 각도 lift를 가집니다. 절대연속 lift의 끝 각변위가 Δ=δ+2πm이면 삼각부등식과 Cauchy–Schwarz로 다음을 얻습니다.

$$
L\ge R\left|\int_0^1\dot\theta\,dt\right|=R|\Delta|,
\qquad E\ge\frac{L^2}{2}\ge\frac{R^2\Delta^2}{2}.
$$

단조 lift는 길이 하한을, 일정 속도는 에너지 하한까지 달성합니다. 정수 m 전체에서 최소화하면 내재 거리가 되고 대척점에는 최소 경로가 둘입니다. 속도가 불균일해도 단조인 매개화는 같은 길이를 갖지만 에너지는 더 큽니다. 따라서 에너지는 경로의 자취뿐 아니라 통과 방식도 측정합니다.

θ(t)=Δt와 θ(t)=Δt²를 비교하세요. 후자의 에너지는 2R²Δ²/3이며 일정 속도 최소값의 4/3배입니다. 아래 코드는 속도를 독립 수치 적분합니다. 두 매개화는 같은 사분원을 덮습니다. 실습에서 separation 2π는 정확히 같은 끝점입니다. 현을 주각변위로 계산하도록 수정해 sine 반올림 잔차 대신 0을 얻습니다.

| 매개화 | R=1, Δ=π/2의 길이 | 에너지 |
|---|---:|---:|
| 일정 속도 | π/2 | π²/8 |
| 이차 시간 | π/2 | π²/6 |

## Python으로 계산 재현하기

```python
from math import pi
n, delta = 4000, pi/2
def simpson(f):
    return (f(0)+f(1)+sum((4 if i%2 else 2)*f(i/n) for i in range(1,n)))/(3*n)
for speed in [lambda t: delta, lambda t: 2*delta*t]:
    length = simpson(lambda t: abs(speed(t)))
    energy = simpson(lambda t: speed(t)**2/2)
    print(round(length, 9), round(energy, 9))
    assert abs(length-delta) < 1e-11
```

## 직접 풀기와 풀이

R=2의 대척점에서 내재 거리 2π, 현 4, [0,1]의 일정 속도 에너지 2π²를 예상해 확인하세요. 두 최단 방향의 에너지는 같습니다. 끝점 차이를 0, winding을 1로 두면 길이 4π의 한 바퀴지만 끝점 거리는 0입니다. 거리는 선택한 경로 하나의 길이가 아니라 가능한 경로 길이의 infimum입니다.

## 기하 학습과의 연결

제약된 표현의 보간·이웃 탐색·경로 계획은 metric과 허용 경로에 의존합니다. embedding의 Euclidean 근접성이 내재 근접성과 같지 않을 수 있습니다. 호 길이 좌표 s=Rθ를 쓰면 metric 계수는 1로 바뀌지만 실제 거리는 같습니다. geodesic 기반 알고리즘을 비교할 때 metric·매개 시간 구간·전역 제약을 명시하세요.

추가 읽기: [John M. Lee, Introduction to Riemannian Manifolds](https://sites.math.washington.edu/~lee/Books/RM/).
