## 정적인 계획을 움직이는 밀도로 바꾸기

유한한 이차 모멘트의 확률밀도에서 동적 이차 수송은

$$
\inf_{\rho,v}\int_0^1\int_{\mathbb R}\rho_t(x)v_t(x)^2\,dx\,dt,
\qquad \partial_t\rho+\partial_x(\rho v)=0
$$

을 고정 끝점 아래 최소화한다. 여기서는 1/2를 넣지 않아 최솟값이 W₂²다. 연속방정식은 가능 조건이지 최소 작용 인증 자체는 아니다. 여러 시간 속도와 우회 경로가 같은 질량과 끝점을 보존할 수 있다.

## 정확히 계산하는 Gaussian 경로

끝점은 N(m₀,σ₀²),N(m₁,σ₁²), 표준편차는 양수다. 진행률 s(t)에 따라 m=(1−s)m₀+sm₁, σ=(1−s)σ₀+sσ₁로 두자. 고정 표준 좌표 z의 입자 x=m+σz를 미분하면

$$
v_t(x)=\dot m+\frac{\dot\sigma}{\sigma}(x-m).
$$

분산이 아닌 표준편차의 보간이 단조 Gaussian 수송 map과 대응한다. 밀도는 실수 전체에서 정규화한다. 그림의 유한 창 때문에 확률 모형을 자른 것은 아니다.

## 밀도 변화와 flux 발산 검사

Gaussian 밀도 ρ에서

$$
\partial_t\rho=\rho\left[-\frac{\dot\sigma}{\sigma}
+\frac{(x-m)\dot m}{\sigma^2}
+\frac{(x-m)^2\dot\sigma}{\sigma^3}\right].
$$

구현은 ρₓ=−(x−m)ρ/σ², vₓ=σ̇/σ로 ∂ₓ(ρv)를 따로 계산한다. 해석적으로 합은 0이지만 수치 잔차를 억지로 0으로 바꾸지 않고 표시한다.

파란 선은 밀도, 주황 선은 flux ρv다. 무차원 좌표에서도 서로 다른 양이다. 음의 flux는 왼쪽 이동을 뜻할 수 있지만 밀도는 비음수다.

## 시간 속도를 바꾸면 작용이 커진다

Gaussian 모멘트로 ∫ρv²dx=ṁ²+σ̇²이므로 전체 작용은

$$
\bigl((m_1-m_0)^2+(\sigma_1-\sigma_0)^2\bigr)
\int_0^1(s'(t))^2dt.
$$

s=t의 적분은 1이라 작용이 W₂²다. s=t²에서는 4/3이다. 같은 Gaussian 가족을 지나고 연속식도 만족하지만 일정 속도가 아니어서 더 비싸다. 끝점 일치와 동적 최적성을 구별하자.

## 모멘트와 시간 속도로 작용 하한 증명하기

모멘트를 미분하고 경계항이 사라지도록 flux를 부분적분할 정칙성을 가정하자. 양의 표준편차 σ의 밀도에서 1과 (x−m)/σ는 L2(ρ)의 정규직교 함수다. 연속식으로 ṁ=∫ρv, σ̇=∫ρv(x−m)/σ이므로 Bessel 부등식은

$$
\int\rho v^2dx\ge\dot m^2+\dot\sigma^2,\qquad
\int_0^1\!\int\rho v^2dxdt
\ge(m_1-m_0)^2+(\sigma_1-\sigma_0)^2.
$$

두 번째는 각 도함수에 시간 Cauchy–Schwarz를 적용한다. Affine Gaussian 속도는 모멘트 부등식의 등호를, 선형 m과 σ는 시간 부등식의 등호를 달성한다. Gaussian 시간 두 개를 비교하는 데서 나아가, 명시한 정칙성의 허용 경로 전체에서 Gaussian 끝점의 최소 작용을 증명한다.

표시한 Gaussian 가족에서 D²=(Δm)²+(Δσ)²라 쓰자. s(0)=0,s(1)=1이고 도함수가 제곱 적분 가능한 절대연속 clock에는

$$
A-D^2=D^2\int_0^1(s'(t)-1)^2dt\ge0.
$$

D>0이면 등호는 거의 모든 시간의 속도 1을 요구한다. 끝점이 같은 D=0은 별도 퇴화 경우다. s=t+αt(1−t)에서 |α|≤1이면 clock은 비감소이고 초과 배율은 α²/3이다.

| α | 작용 / D² |
|---|---:|
| −1 | 4/3 |
| 0 | 1 |
| 1 | 4/3 |

```python
from fractions import Fraction as F
D2 = F(17, 4)
for alpha in (F(-1), F(0), F(1)):
    speed = (1+alpha, -2*alpha)  # polynomial coefficients of s'(t).
    squared = [F(0)]*3
    for i, x in enumerate(speed):
        for j, y in enumerate(speed): squared[i+j] += x*y
    integral = sum(x/F(i+1) for i, x in enumerate(squared))
    assert integral == 1+alpha*alpha/3
    assert integral >= 1
    print(alpha, integral, "action:", D2*integral)
```

유리수 다항 적분으로 유도한 clock 공식을 세 실행 예에서 검산한다. 원래 실습에 시간 조절기를 추가한 것은 아니다. 보존식·모멘트 정칙성·최소 작용의 요구를 구별하자.

![같은 Gaussian 경로를 따르는 시간표 세 개의 작용을 비교하면 일정 시간 속도가 최소를 달성합니다.](/assets/learning/transport-clock-ko.svg)

## 직접 풀기와 풀이

m₀=−1,m₁=1,σ₀=0.5,σ₁=1이면 끝점 제곱 거리는 4.25다. 선형 시간 t=0.5에서 m=0,σ=0.75, 순간 적분 운동 에너지도 4.25다. 비선형 시간의 전체 작용은 17/3≈5.666667이다.

x=0에서 밀도 시간 미분과 flux 공간 미분을 각각 읽고 합을 확인하자. 끝점을 같게 하면 속도·flux·작용은 0이지만 밀도는 정규화된 Gaussian으로 남는다. 학습된 속도장에서도 보존식·끝점·작용 최적화를 따로 검사해야 하는 이유를 설명하자.

더 읽기: [Peyré와 Cuturi의 동적 최적수송 설명](https://optimaltransport.github.io/book/).
