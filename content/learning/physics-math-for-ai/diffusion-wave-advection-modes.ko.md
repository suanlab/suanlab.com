## 같은 초기 모양도 다른 법칙으로 움직인다

Periodic 구간 [0,2pi]에서 정수 mode k, 초기 변위 u0=M+A cos(kx)를 사용합니다. 확산·파동·일정 속도 수송은 같은 초기 그래프에서 출발하지만 방정식과 필요한 데이터가 다릅니다. 파동에는 초기 속도 v0=B cos(kx)도 필요합니다.

파랑은 현재 장, 주황은 초기 변위입니다. 한 순간의 변위가 0에 가까워도 이차 시간 파동의 전체 상태가 사라진 것은 아닙니다. 속도도 상태의 일부라는 점을 먼저 확인하세요.

## 확산은 짧은 공간 규모를 더 빨리 줄인다

u_t=nu u_xx의 해는 다음입니다.

$$
u(x,t)=M+A e^{-\nu k^2t}\cos(kx).
$$

왼쪽 u는 장이고 nu는 확산계수입니다. 공간 미분을 두 번 하면 -k²가 나오므로 진폭의 감쇠율은 nu k²입니다. 확산계수 0에서는 모양이 고정되고 k를 늘리면 같은 시간에 훨씬 빠르게 매끄러워집니다.

평균은 M으로 유지됩니다. Centered 변위 에너지 E=1/2 integral (u-M)²는 pi A² exp(-2nu k²t)/2이며 dE/dt=-nu integral u_x²입니다. 경계로 빠져나가지 않아도 smoothing으로 이 에너지가 감소합니다.

## 파동은 운동·공간 에너지를 교환한다

u_tt=c²u_xx에서 omega=ck라고 두면 다음과 같습니다.

$$
u(x,t)=M+\left(A\cos\omega t+\frac B\omega\sin\omega t\right)\cos(kx),
\qquad
E_{\mathrm{wave}}=\frac12\int(u_t^2+c^2u_x^2)\,dx.
$$

전체 에너지는 pi(B²+omega² A²)/2로 일정합니다. A=c=k=1, B=0, t=pi/2에서는 변위가 거의 평평하지만 속도에 에너지가 남습니다. 변위 L2만 파동의 보존량으로 읽으면 틀립니다.

## 수송은 모양을 매끄럽게 하지 않고 옮긴다

u_t+c u_x=0의 정확한 장은 M+A cos(k(x-ct))입니다. 진폭·periodic 질량·centered L2를 유지하며 이동합니다. c로 움직이는 좌표계에서는 정지한 모양입니다. 확산의 진폭 감소, 정상파의 변위·속도 교환과 구별하세요.

수치는 x=0.37에서 시간·공간 도함수를 따로 계산해 방정식 잔차를 확인합니다. 이 실험에서는 평균 속도와 periodic 경계의 순 flux가 0이라 모든 모형의 질량이 2pi M입니다.

## 에너지 추정과 역방향 열방정식의 부적정성

매끄러운 periodic 열방정식 해에 부분적분하면 d(‖u−M‖²/2)/dt=−ν‖uₓ‖²다. 두 해의 차이에 적용하면 L² 수축과 유일성을 증명한다. 파동식 uₜₜ=c²uₓₓ에는 uₜ를 곱해 (‖uₜ‖²+c²‖uₓ‖²)/2 보존을 얻는다. Gradient가 상수 mode를 제한하지 않으므로 평균 변위의 초기값도 별도로 필요하다. 수송에서는 d‖u‖²/dt=−c∫(u²)ₓ=0이다. 각 논증에서 periodic 조건이 경계항을 없앤다.

직교성으로 단일 mode 실습을 유한 Fourier 합으로 확장하고, 정당화된 근사 극한으로 에너지 해까지 확장한다. 순방향 열방정식은 안정하지만 이전 상태 복원은 mode k를 e^(νk²T)배 한다. ν=T=1, 초기값 uₖ(x,0)=cos(kx)를 택하면 T에서 L² 노름은 √πe⁻ᵏ²→0인데 초기 노름은 √π로 일정하다. 따라서 각 유한 mode를 정확히 역산할 수 있어도 L² 역연산은 연속이지 않다.

| k | νT=1에서 역방향 증폭의 로그 |
|---|---:|
| 1 | 1 |
| 2 | 4 |
| 4 | 16 |

```python
from math import exp, pi, fsum
modes = [(1, 1.), (2, .5), (4, -.25)]
nu, T = .2, .7
initial_energy = pi/2*fsum(a*a for k, a in modes)
final_energy = pi/2*fsum(a*a*exp(-2*nu*k*k*T) for k, a in modes)
dissipation = -nu*pi*fsum(k*k*a*a*exp(-2*nu*k*k*T) for k, a in modes)
assert 0 < final_energy < initial_energy and dissipation < 0
for k in (1, 2, 4):
    terminal_norm = pi**.5*exp(-k*k)
    log_inverse_gain = k*k
    assert abs(terminal_norm*exp(log_inverse_gain)-pi**.5) < 1e-14
    print(k, terminal_norm, log_inverse_gain)
print('energy:', initial_energy, final_energy, 'derivative:', dissipation)
```

정확한 유한 mode 에너지 인자와 로그 증폭을 비교해 overflow를 피한다. 확산 생성모형의 역동역학에는 분포에 의존하는 score가 있으며, 잡음 관측에 무제한 역방향 열연산을 적용하는 것과 같지 않다. 역 PDE 학습에는 정규화나 허용 상태의 사전 정보가 필요하다. 작은 순방향 잔차만으로 역산이 안정해지지 않는다.

## 직접 풀기: 맞는 보존량 고르기

Heat에서 A=1, M=0, k=2, nu=0.2, t=0.5이면 진폭은 exp(-0.4)입니다. Wave에서 k=c=1, t=pi/2의 운동 에너지를 계산해 변위가 0 근처인데도 전체 에너지가 pi/2인 이유를 설명하세요. Transport의 M=0.5에서는 질량이 pi입니다. 각 비교가 실제로 어떤 불변량을 검사하는지 적으세요.

더 읽기: [MIT PDE 과정](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/).
