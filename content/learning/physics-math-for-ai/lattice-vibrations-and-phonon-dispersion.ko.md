## 1. 반복하는 셀에서 원자를 조금 움직이기

격자는 주기적으로 반복되는 병진 위치를 뜻하고 기저는 한 셀의 원자를 지정합니다. 길이 $a$인 각 셀에서 위치 $na$와 $na+a/2$에 질량 $m_A,m_B$를 둡니다. 이웃 원자는 강성 $\kappa$인 같은 용수철로 연결합니다. 작은 변위에서 결합마다 탄성에너지가 $\kappa(\Delta u)^2/2$인 이차식이므로 정상 모드를 중첩할 수 있습니다. [MIT OCW, Lattice Waves in 1D with Diatomic Basis](https://ocw.mit.edu/courses/6-730-physics-for-solid-state-applications-spring-2003/resources/lecture_9b/)

## 2. 반복 사슬을 두 성분의 고윳값 문제로 바꾸기

파수 $k$인 파동이 각 원자의 평형 위치에서 위상 $e^{ikx}$를 갖는다고 둡니다. $q=ka$로 쓰고 변위 진폭에 질량의 제곱근을 곱합니다. 뉴턴 방정식은 대칭 고윳값 문제 $D(q)v=\omega^2v$가 됩니다.
$$
D(q)=\begin{pmatrix}
2\kappa/m_A & -2\kappa\cos(q/2)/\sqrt{m_Am_B}\\
-2\kappa\cos(q/2)/\sqrt{m_Am_B} & 2\kappa/m_B
\end{pmatrix}.
$$
비대각항은 두 이웃 결합의 위상 인자를 더한 값입니다. 질량 가중 좌표에서는 보통의 직교 고유벡터를 사용할 수 있습니다.

## 3. 음향 가지와 광학 가지 읽기

$q=0$에서 두 원자의 물리적 변위가 같으면 어떤 용수철도 늘어나지 않습니다. 음향 모드의 고윳값이 0인 것은 사슬의 평행 이동입니다. 광학 모드는 반대 방향으로 움직이며 $m_Au_A+m_Bu_B=0$이고 제곱 주파수는 $2\kappa(1/m_A+1/m_B)$입니다. 0 가까이에서 음향 모드는 $\omega\approx c_s|k|$이며 $c_s=a\sqrt{\kappa/[2(m_A+m_B)]}$입니다. 기울기가 이 모형의 장파장 음속을 줍니다.

## 4. 브릴루앙 영역 끝과 같은 질량의 극한 확인하기

$q=\pi$에서는 이웃 위상이 상쇄되고 두 제곱 주파수는 $2\kappa/m_A$와 $2\kappa/m_B$가 됩니다. 질량이 다르면 두 가지 끝에 간격이 생깁니다. 질량이 같으면 그곳에서 두 가지가 만납니다. 이때 실제 주기는 $a/2$입니다. 원자 두 개를 한 셀로 쓰면 하나의 가지가 더 작은 브릴루앙 영역으로 접힙니다.

## 5. 0이 아닌 정상 모드 양자화하기

주파수가 0이 아닌 조화 모드의 에너지 준위는 $\hbar\omega(n+\tfrac12)$입니다. 포논 하나를 더하면 $\hbar\omega$만큼 에너지가 증가합니다. 포논은 여러 원자의 집단 운동을 나타냅니다. 균일 병진 모드는 질량 중심 운동으로 따로 다룹니다. 셀 $N$개인 유한 주기 사슬에는 허용 파수 $N$개가 있으며 파수마다 모드가 두 개입니다.

```python
from math import cos, isclose, pi, sqrt
import numpy as np

def squared_modes(q, a, b):
    coupling = -2*cos(q/2)/sqrt(a*b)
    return np.linalg.eigvalsh([[2/a,coupling],[coupling,2/b]])

assert np.allclose(squared_modes(0,1,4),[0,2.5])
assert np.allclose(squared_modes(pi,1,4),[0.5,2])
assert np.allclose(squared_modes(pi,1,1),[2,2])
q = .001
assert isclose(sqrt(squared_modes(q,1,4)[0])/q,sqrt(1/10),rel_tol=1e-6)
print(squared_modes(pi,1,4))
```

## 6. 분산 관계 점검하기

- 질량비를 바꿔 영역 끝의 두 주파수를 비교하세요.
- 장파장에서 두 모드의 변위 방향을 비교하세요.
- $q=0$ 주변의 기울기로 서로 다른 질량에서의 음속을 비교하세요.
- 셀 길이를 두 배로 표시하면 실제 원자 수를 늘리지 않고도 가지 표시가 달라지는 이유를 설명하세요.
