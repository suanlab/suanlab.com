## 서로 다른 접공간을 비교할 규칙이 필요합니다

다양체의 다른 점에 놓인 벡터들은 다른 접공간에 속합니다. connection은 경로를 따라 벡터장을 미분하는 규칙이고 평행이동은 그 공변 미분이 0이라는 조건입니다. Riemannian 다양체의 Levi-Civita connection은 metric compatible이며 torsion free이고 운반 중 내적을 보존합니다.

반지름 R, 북극에서 잰 colatitude θ, longitude φ를 쓰면 다음과 같습니다.

$$
ds^2=R^2d\theta^2+R^2\sin^2\theta\,d\phi^2,\qquad
v=\cos\alpha\,e_\theta+\sin\alpha\,e_\phi.
$$

θ를 0과 π에서 떨어지게 제한해 극의 좌표 특이점을 피합니다. 그림은 각 점의 국소 정규직교 접 frame에서 읽은 성분 평면입니다. 구면 전체를 펼친 평면 chart나 위에서 본 실제 이동 경로가 아닙니다.

## 위도선을 따른 평행이동 유도

θ를 고정하면 frame의 접방향 미분은 ∇φeθ=cosθ eφ, ∇φeφ=−cosθ eθ입니다. 벡터 표현을 대입하고 공변 미분을 0으로 두면 다음이 나옵니다.

$$
\frac{d\alpha}{d\phi}=-\cos\theta,\qquad
\alpha(\phi)=\alpha_0-\phi\cos\theta.
$$

파랑은 초기 방향, 주황은 선택한 longitude만큼 운반한 성분입니다. 단위 길이는 유지됩니다. frame 자체가 회전하므로 성분 각도를 단순히 고정하면 일반적으로 평행이동과 다릅니다.

## 폐곡선이 곡률을 드러냅니다

한 바퀴 후 frame은 출발점으로 돌아옵니다. 성분 각 변화는 −2πcosθ이며 2π를 법으로 해석합니다. θ=π/3이면 −π라 벡터가 반대로 돌아옵니다. 적도에서는 cosθ=0이므로 성분 방향이 그대로입니다. 이 위도선들 중 geodesic은 적도뿐이지만 평행이동은 일반 경로에도 정의됩니다.

구면의 Gaussian 곡률은 K=1/R², 북극 cap 면적은 2πR²(1−cosθ)입니다.

$$
\int_{\mathrm{cap}}K\,dA=2\pi(1-\cos\theta)
\equiv-2\pi\cos\theta\pmod{2\pi}.
$$

부호와 frame 관례를 명시했습니다. 열린 부분 경로에도 운반 각은 있지만 폐곡선 holonomy 값은 주지 않습니다. full-cap 면적은 언제나 전체 cap이며 열린 경로가 쓸고 간 영역이 아닙니다.

## Connection 계수와 경계항

구면 metric을 Levi-Civita 공식에 대입하면 대칭 쌍을 포함해 0이 아닌 좌표 Christoffel 기호는 다음과 같습니다.

$$
\Gamma^\theta_{\phi\phi}=-\sin\theta\cos\theta,\qquad
\Gamma^\phi_{\theta\phi}=\Gamma^\phi_{\phi\theta}=\cot\theta.
$$

이 계수에서 반지름은 상쇄되지만 Gaussian 곡률에서는 상쇄되지 않습니다. 정규직교 frame으로 옮기면 앞의 운반 방정식이 나옵니다. 생성 행렬이 반대칭이라 길이 제곱의 미분이 0이고 벡터 길이가 보존됩니다. 아래에서는 두 성분 ODE를 RK4로 적분해 최종 각도 공식을 한 번 평가하는 것과 독립적으로 검산합니다.

Gauss–Bonnet에는 경계 기여도 있습니다. 양의 방향 북극 cap에서 위도 경계의 geodesic curvature 적분은 2πcosθ입니다. 이를 cap 곡률 적분에 더하면 disk의 Euler characteristic에 해당하는 2π입니다. 위도선은 일반적으로 geodesic이 아니므로 경계항을 생략하면 틀립니다. θ=π/3에서는 두 기여가 각각 π입니다.

| Colatitude | Cap 곡률 적분 | 경계 Geodesic curvature 적분 |
|---|---:|---:|
| π/3 | π | π |
| π/2 | 2π | 0 |
| 2π/3 | 3π | -π |

## Python으로 계산 재현하기

```python
from math import pi, cos, sin, hypot
for theta in [pi/3, pi/2, 2*pi/3]:
    c, z, n = cos(theta), complex(1, 0), 2000
    h = 2*pi/n
    def f(w): return -1j*c*w
    for _ in range(n):
        k1=f(z); k2=f(z+h*k1/2); k3=f(z+h*k2/2); k4=f(z+h*k3)
        z += h*(k1+2*k2+2*k3+k4)/6
    expected = complex(cos(-2*pi*c), sin(-2*pi*c))
    assert abs(z-expected) < 1e-10 and abs(hypot(z.real,z.imag)-1) < 1e-10
    print(round(2*pi*(1-c), 9), round(2*pi*c, 9), round(abs(z-expected), 12))
```

## 직접 풀기와 연구 연결

θ=π/3을 고정하고 R을 두 배로 하세요. 곡률은 1/4, cap 면적은 네 배가 되어 곱과 holonomy는 같습니다. 각각 확인한 뒤 반 바퀴만 이동하면 끝점의 성분 각 비교가 왜 frame 선택에 의존하는지 설명하세요.

mesh나 굽은 latent 공간의 접특징을 비교할 때 transport와 connection이 유용합니다. 성분 회전 그림을 읽으려면 frame·경로 방향·connection을 먼저 정해야 합니다. 한 폐곡선의 회전이 2π를 법으로 0이라고 표면이 평평하다고 결론낼 수는 없습니다.

추가 읽기: [John M. Lee, Introduction to Riemannian Manifolds](https://sites.math.washington.edu/~lee/Books/RM/).
