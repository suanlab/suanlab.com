## 불확정성을 말하기 전에 폭부터 정의합니다

짧은 pulse와 좁은 주파수 spectrum을 동시에 만들기는 어렵습니다. 그러나 좁다는 말부터 수학적으로 정해야 합니다. 여기서는 크기 제곱을 정규화한 에너지 밀도의 평균 주위 2차 모멘트를 사용합니다. 이는 시간·각주파수 위의 수학적 분포이며 미지 신호에 대한 잡음 추정 오차가 아닙니다.

복소 Gaussian을 다음과 같이 정합니다.

$$
g(t)=\exp\!\left(-\frac{(t-\tau)^2}{4\sigma^2}\right)e^{i\Omega t},\qquad
G(\omega)=\int g(t)e^{-i\omega t}\,dt.
$$

분모의 4σ²가 중요합니다. 크기를 제곱하면 exp(−(t−τ)²/(2σ²))이므로 정규화 시간 에너지 밀도의 평균은 τ, 표준편차는 σ입니다. 진폭 분모를 2σ²로 쓰는 Gaussian과 다른 매개화입니다.

## 주파수 폭을 유도하기

시간 이동은 위상 인자, modulation은 주파수 이동을 만듭니다. 완전제곱으로 적분하면 주파수 크기는 exp(−σ²(ω−Ω)²)에 비례합니다. 다시 제곱하고 정규화하면 다음과 같습니다.

$$
p_t(t)=\frac{|g(t)|^2}{\int |g|^2},\quad
p_\omega(\omega)=\frac{|G(\omega)|^2}{\int |G|^2},\quad
\Delta t=\sigma,\quad\Delta\omega=\frac1{2\sigma}.
$$

따라서 ΔtΔω=1/2입니다. σ=1이면 폭은 1과 0.5이고, σ=0.25이면 0.25와 2입니다. pulse가 네 배 좁아지면 spectrum은 네 배 넓어지며 곱은 같습니다.

## 중심 이동과 폭 변화는 다릅니다

τ를 움직이면 파란 시간 밀도의 중심이, Ω를 움직이면 주황 각주파수 밀도의 중심이 이동합니다. 평균으로부터 계산한 분산은 바뀌지 않습니다. 겹친 두 곡선의 가로 좌표는 서로 다른 물리량이므로 교점 자체에 좌표 독립적 의미는 없습니다. 그림은 [−8,8]만 표시해 꼬리 일부가 빠질 수 있습니다. 수치의 해석적 폭은 실수 전체에서 계산한 값입니다.

## 단위에 따라 상수도 달라집니다

시간당 주기 수 f를 쓰면 f=ω/(2π)입니다. 따라서 Δf=Δω/(2π), ΔtΔf=1/(4π)입니다. 시간 단위가 초라면 각주파수는 rad/s이고 보통 주파수는 Hz입니다. 2π를 놓치면 동등한 식들이 모순되어 보입니다.

## 정칙성을 명시하고 중심화한 부등식 증명하기

0이 아닌 Schwartz 신호의 시간·주파수 평균을 이동과 modulation으로 제거한 결과를 h라 합시다. 빠른 감소로 부분적분의 경계항이 사라지므로

$$
\operatorname{Re}\int t h'(t)\overline{h(t)}\,dt
=-\frac12\|h\|_2^2,
\qquad\frac12\|h\|_2^2\le\|th\|_2\|h'\|_2.
$$

둘째 부등식은 Cauchy–Schwarz입니다. Plancherel과 도함수 변환으로 ‖h′‖₂/‖h‖₂=Δω, 다른 비율은 Δt임을 얻습니다. 따라서 각주파수 단위에서 ΔtΔω≥1/2입니다. 등호에는 양의 실수 c에 대해 h′=−cth가 필요하여 Gaussian envelope를 얻습니다. 이동·modulation·일정한 복소 배율은 등호를 유지합니다. Schwartz 밖으로 확장하려면 모멘트·약도함수·경계를 제어해야 하며 유한 에너지만으로는 부족합니다.

## Chirp는 시간 폭을 유지해도 주파수를 넓힌다

별도 함수족을 생각합니다.

$$
g_\kappa(t)=e^{-t^2/(4\sigma^2)}e^{i\kappa t^2/2},\qquad
(\Delta\omega)^2=\frac1{4\sigma^2}+\kappa^2\sigma^2,
\qquad\Delta t=\sigma.
$$

Chirp는 위상만 바꾸어 |g|²를 유지합니다. 직접 미분하면 g′=(−t/(2σ²)+iκt)g입니다. 크기 제곱을 적분하고 Plancherel을 적용하면 위 주파수 분산이 증명됩니다. 여기서 공분산은 중심화한 함수의 Re ∫t ḡ(−ig′)/‖g‖²인 대칭화된 곱셈/미분 공분산이며, 양의 공동 시간·주파수 밀도를 가정하지 않습니다. 이 값은 κσ²이므로 단순 폭의 곱이 1/2보다 커져도 Δt²Δω²−Cov²=1/4입니다.

| σ=1의 κ | 시간 표준편차 | 각주파수 표준편차 | 곱 |
|---|---|---|---|
| 0 | 1 | .5 | .5 |
| .5 | 1 | .707107 | .707107 |
| 1 | 1 | 1.118034 | 1.118034 |
| 2 | 1 | 2.061553 | 2.061553 |

![Chirp를 바꾸어도 Gaussian 시간 에너지 밀도는 같지만 각주파수 밀도는 넓어진다. Chirp가 없는 곡선만 폭의 곱 1/2를 실현한다.](/assets/learning/harmonic-chirp-widths-ko.svg)

코드는 spectrum 그림에 맞춰 폭을 추정하지 않고 시간·도함수 에너지를 독립 적분합니다. Chirp는 별도 수학 예제이며 기존 Gaussian 등호 실습의 새 제어가 아닙니다. 두 밀도 패널은 서로 다른 물리 좌표와 단위를 사용합니다.

**확인 문제:** 시간 에너지 envelope가 같으면 주파수 폭도 정해지나요? **해설:** 아닙니다. 이차 위상이 envelope를 유지하면서 도함수 에너지와 주파수 공분산을 바꿉니다.

```python
from math import exp, sqrt

def simpson(f,a,b,n=20000):
    h=(b-a)/n
    return h*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))/3

sigma=1.0
energy=lambda t:exp(-t*t/(2*sigma*sigma))
norm=simpson(energy,-12,12)
time_variance=simpson(lambda t:t*t*energy(t),-12,12)/norm
for chirp in [0,.5,1,2]:
    # |g'|^2 from g=exp(-t^2/4sigma^2) exp(i chirp*t^2/2), not a spectral-width fit.
    derivative_energy=simpson(lambda t:t*t*(1/(4*sigma**4)+chirp**2)*energy(t),-12,12)
    frequency_variance=derivative_energy/norm
    covariance=chirp*time_variance
    determinant=time_variance*frequency_variance-covariance**2
    assert abs(determinant-.25)<1e-12
    print('chirp, time sd, angular-frequency sd, product:',chirp,
          round(sqrt(time_variance),6),round(sqrt(frequency_variance),6),
          round(sqrt(time_variance*frequency_variance),6))
```

## 직접 풀기와 적용 범위

σ=0.5, τ=1, Ω=3에서 폭은 0.5와 1, 중심은 1과 3, 곱은 0.5입니다. 두 중심을 이동해도 폭이 유지되는지 확인하세요.

0이 아닌 유한 에너지 신호에 필요한 모멘트·미분 정칙성이 있으면 Fourier 불확정성 ΔtΔω≥1/2가 성립하며 Gaussian은 등호를 실현합니다. 이 실험은 등호 예제이지 모든 신호의 정리를 수치적으로 증명한 것이 아닙니다. AI의 국소 특징을 이해하는 데 유용하지만 그 자체로 양자 측정이나 통계적 추정 오차에 관한 주장은 아닙니다.

추가 읽기: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
