## 속도에 작용하는 규칙을 적분합니다

미분 1-form α=P dx+Q dy는 covector field입니다. 경로 γ(t)=(x(t),y(t))의 속도에 작용시킨 뒤 적분합니다.

$$
\int_\gamma\alpha=\int\big(P(\gamma(t))\dot x(t)+Q(\gamma(t))\dot y(t)\big)\,dt,
\qquad d\alpha=(\partial_xQ-\partial_yP)dx\wedge dy.
$$

wedge는 방향을 기록합니다. dx∧dy=−dy∧dx이고 dx∧dx=0입니다. 외미분은 좌표 변화와 호환되며 d²=0이므로 exact form α=df는 closed입니다. 그 역은 정의역의 위상에 달려 있습니다.

## Disk 위의 매끄러운 form 둘 비교하기

α=x dx+y dy=d((x²+y²)/2)는 exact입니다. 선적분의 기본정리에 따라 모든 매끄러운 폐곡선 적분은 0입니다. 중심을 옮긴 원에서도 마찬가지이므로 원점 중심의 대칭 때문에 우연히 0인 것이 아닙니다.

α=(−y dx+x dy)/2의 외미분은 dx∧dy입니다. 방향 있는 disk D의 경계를 반시계로 돌면 다음과 같습니다.

$$
\oint_{\partial D}\alpha=\int_Dd\alpha=\pi R^2.
$$

방향을 뒤집으면 −πR²입니다. 외미분 계수는 여전히 1이고 disk와 유도된 경계 방향이 바뀌어 적분 부호가 바뀝니다. 주황 짧은 선분은 같은 길이로 정규화한 field 방향입니다. 실제 계수 크기나 경로 접벡터를 그린 것이 아닙니다.

## 빠진 한 점이 결론을 바꿉니다

angular form α=(−y dx+x dy)/(x²+y²)는 원점을 뺀 평면에서만 매끄럽습니다. 그곳에서는 직접 미분해 dα=0을 얻지만, 원점을 감싸는 반시계 원의 적분은 2π입니다. 원점 중심 원을 매개화하면 R²가 상쇄되고 ∫₀²π1 dt가 남습니다.

Stokes와 모순되지 않습니다. form이 disk 내부 원점에 정의되지 않아 정리의 가정이 실패합니다. disk가 원점을 피하면 두 적분은 0입니다. 경계 자체가 원점을 통과하면 이 실험은 보통 선적분 값을 정의하지 않습니다. principal value 같은 별도 정규화를 몰래 적용하지 않습니다.

## Potential을 구성하려면 정의역 가정이 필요합니다

원점을 포함하는 별 모양 열린 영역에서 연속 미분 가능한 closed 1-form은 potential을 가집니다. 방사 선분으로 적분해 F를 정의하세요. 적분 안 미분과 closedness 항등식을 적용하면 다음과 같습니다.

$$
F(x)=\int_0^1\alpha_{tx}(x)\,dt,\qquad
\partial_jF(x)=\int_0^1\frac{d}{dt}\big[t\alpha_j(tx)\big]dt=\alpha_j(x).
$$

선분 전체가 정의역에 있어야 합니다. 원점을 뺀 평면에서는 그 선분이 제외점을 지날 수 있어 이 증명으로 전역 각도 potential을 만들 수 없습니다. 대신 angular form의 폐곡선 적분은 winding number의 2π배를 기록합니다. 적절히 자른 slit 영역에는 국소 각도가 있지만 punctured plane 전체에는 단일값 전역 각도가 없습니다.

수치 코드는 중심을 옮긴 원에서 angular form을 직접 적분하고 방향도 뒤집습니다. 특이점을 지나는 경계에는 principal value를 부여하지 않고 그 경우를 제외합니다. 분할 수를 늘려 이 예제의 구적 수렴을 확인할 수 있지만, 제안한 spanning surface 전체의 매끄러움 확인을 대신할 수는 없습니다.

| 중심 x | 반지름 | 방향 | Angular form 적분 |
|---:|---:|---|---:|
| 0 | 1 | 반시계 | 2π |
| 0 | 1 | 시계 | -2π |
| 2 | 1 | 반시계 | 0 |

## Python으로 계산 재현하기

```python
from math import pi, sin, cos
n = 20000
for center, sign in [(0,1),(0,-1),(2,1)]:
    total = 0.0
    for i in range(n):
        t = sign*2*pi*(i+.5)/n
        x, y = center+cos(t), sin(t)
        dx, dy = -sin(t), cos(t)
        total += (-y*dx+x*dy)/(x*x+y*y)*sign*2*pi/n
    expected = sign*2*pi if center < 1 else 0
    assert abs(total-expected) < 1e-10
    print(center, sign, round(total, 9))
```

## 직접 풀기와 풀이

angular form에서 R=0.5, 중심 (1,0)을 택하세요. disk가 원점을 피하므로 적분은 0이고 Stokes가 적용됩니다. 중심을 (0,0)으로 옮기면 경계 적분은 2π이고 disk Stokes가 실패합니다. 중심 (0.5,0)에서는 경계가 제외점을 지나 적분이 정의되지 않습니다. 세 상태를 모두 확인하세요.

## 과학 AI와의 연결

미분형식은 flux·보존 법칙·좌표 호환 미분 연산자를 정리합니다. 학습한 field의 curl이 0처럼 보여도 전역 potential을 주장하려면 정의역 조건이 필요합니다. 구멍·제외 특이점·경계 조건이 있으면 국소 미분이 맞아도 그 추론은 실패할 수 있습니다. 유한 방향 그림은 계산을 돕는 도구이며 전역 정리의 증명이 아닙니다.

추가 읽기: [Lorenzo Sadun, Lecture Notes on Differential Forms](https://arxiv.org/abs/1604.07862).
