## 1. 접벡터를 평행하게 유지하기

측지선은 자기 경로를 따라 접벡터가 평행이동하는 곡선입니다. 아핀 매개변수 $\lambda$에서는 다음 식을 만족합니다.
$$
\frac{d^2 x^i}{d\lambda^2}
 +\Gamma^i_{jk}\frac{dx^j}{d\lambda}\frac{dx^k}{d\lambda}=0.
$$
둘째 항이 좌표 기저의 변화를 보정합니다. 리만 계량의 정칙 측지선에서는 호 길이를 아핀 매개변수로 쓸 수 있습니다. 시간꼴 자유낙하에서는 고유 시간이 아핀 매개변수입니다. 빛꼴 측지선의 고유 시간은 0이므로 다른 아핀 매개변수가 필요합니다. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 2. 극좌표에서 직선 따라가기

평면의 계량 $ds^2=dr^2+r^2d\theta^2$에서는 다음 식을 얻습니다.
$$
\ddot r-r\dot\theta^2=0,\qquad
\ddot\theta+2\dot r\,\dot\theta/r=0.
$$
둘째 식에 따라 $r^2\dot\theta$가 일정합니다. 데카르트 좌표의 직선 $(x,y)=(1,\lambda)$는 극좌표에서 $r=\sqrt{1+\lambda^2}$와 $\theta=\arctan\lambda$입니다. 경로는 직선이지만 두 좌표는 비선형으로 바뀝니다. $\lambda=1$에서 $\ddot r=1/(2\sqrt2)$이고 $r\dot\theta^2=1/(2\sqrt2)$이므로 측지선 식에서 상쇄됩니다.

## 3. 구면 제약으로 대원 구하기

단위 벡터 $P$와 $Q$로 반지름 $R$인 구면 위 두 점을 지정합니다. 두 점이 서로 다르고 대척점이 아니면 $\vartheta=\arccos(P\cdot Q)$와 $T=(Q-\cos\vartheta\,P)/\sin\vartheta$로 둡니다. 접벡터 $T$의 길이는 1이고 $P$에 수직입니다. 다음 곡선은 단위 속력을 갖습니다.
$$
X(s)=R\big(\cos(s/R)P+\sin(s/R)T\big).
$$
가속도는 중심을 향하므로 접선 방향 가속도는 0입니다. 따라서 이 곡선은 측지선이고 $s=R\vartheta$에서 $RQ$에 닿습니다.

## 4. 측지선과 전역 최단 경로 구분하기

짧은 호의 길이는 $R\vartheta$입니다. 반대 방향의 호는 길이가 $R(2\pi-\vartheta)$이고 측지선 식을 만족합니다. 이 호는 더 짧은 경로가 생기는 지점을 지나서도 이어집니다. 대척점에서는 두 길이가 모두 $\pi R$이고 무수히 많은 대원 반원이 양 끝을 잇습니다. 끝점으로 $T$를 정할 수 없을 때는 초기 방향을 별도로 지정합니다.

## 5. 경로와 길이 검산하기

```python
from math import cos, isclose, pi, sin, sqrt

R = 2
theta = 0.9
P = (1, 0)
Q = (cos(theta), sin(theta))

def position(angle):
    return R*cos(angle), R*sin(angle)

short, long = R*theta, R*(2*pi-theta)
for angle in [theta, theta-2*pi]:
    assert all(isclose(a,R*b,abs_tol=1e-12) for a,b in zip(position(angle),Q))
    assert isclose(sqrt(sum(v*v for v in position(angle))), R)
assert short < long
assert isclose(short+long, 2*pi*R)
print(short,long)
```

## 6. 측지선 결과 점검하기

- 끝점뿐 아니라 속력과 접선 방향 가속도도 확인하세요.
- 같은 직선의 극좌표와 데카르트 좌표를 비교하세요.
- 구면의 양 끝점을 고정하고 짧은 호와 긴 호를 비교하세요. 두 호가 공유하는 성질과 서로 다른 양을 설명합니다.
- 두 점을 대척점에 가깝게 놓고 초기 방향을 하나로 정할 수 없게 되는 경우를 살펴보세요.
