## 한 실현 대신 법칙을 진화시킨다

SDE는 랜덤 궤적을, Fokker–Planck 식은 주변 확률 밀도를 기술합니다. 둘은 같은 물체가 아닙니다. OU drift -theta x와 상수 diffusion sigma의 보존식은 rho_t+J_x=0이며 J는 drift 이동과 밀도 gradient의 diffusion을 합칩니다.

초기 Gaussian의 평균은 m0, 표준편차 s0는 양수입니다. 양의 sigma·s0 덕분에 초기 순간까지 매끄러운 양의 밀도입니다. Dirac 초기값의 특이점은 이 control 범위에 포함하지 않습니다.

## Gaussian Moments 유도

주변 법칙은 계속 Gaussian입니다.

$$
m(t)=m_0e^{-\theta t},\qquad
v(t)=s_0^2e^{-2\theta t}+\sigma^2t f(2\theta t),
\qquad f(z)=\frac{1-e^{-z}}z,\quad f(0)=1.
$$

Moment 식은 m'=-theta m, v'=sigma²-2theta v입니다. Theta=0이면 평균은 고정되고 분산은 s0²+sigma²t입니다. 양의 theta에서는 sigma²/(2theta)로 접근하여 confinement와 잡음의 균형을 보여줍니다.

## 밀도와 Flux를 따로 미분한다

Gaussian 밀도 rho=exp(-(x-m)²/(2v))/sqrt(2pi v)로부터 다음입니다.

$$
J=-\theta x\rho-\frac{\sigma^2}{2}\rho_x,\qquad
\rho_t=\rho\left(-\frac{v'}{2v}+\frac{(x-m)m'}v+
\frac{(x-m)^2v'}{2v^2}\right).
$$

구현은 J_x를 rho·rho_x·rho_xx에서 따로 구합니다. 단순히 -rho_t로 정의하지 않아 식의 불일치를 잔차로 찾을 수 있습니다. 국소 flux가 0이 아니어도 확률은 보존될 수 있으며 divergence와 경계를 함께 봐야 합니다.

## 그림 창과 확률 정의역은 다르다

파랑·주황은 두 밀도를 포함하는 유한 창의 현재·초기 밀도입니다. 전체 실수선의 질량은 해석적으로 1이고 그림 밖 꼬리는 재정규화하지 않습니다. 보이는 구간 적분을 전체 정규화와 동일하게 읽지 마세요.

Theta=1, sigma=sqrt(2), m0=0,s0=1이면 표준 Gaussian이 정상이고 drift·diffusion flux가 상쇄됩니다. 정상 밀도여도 표본 경로는 계속 움직입니다.

## 분포의 Lyapunov 함수: 상대 엔트로피

θ>0인 OU의 정상 법칙은 π=N(0,v*), v*=σ²/(2θ)다. Gaussian 주변 법칙의 상대 엔트로피 D=KL(N(m,v)||π)를 생각하자. r=v/v*라 두고 Gaussian 적분과 moment 식을 적용하면

$$
D=\tfrac12(m^2/v_*+r-1-\log r),\qquad
D'=-\theta\left(m^2/v_*+\frac{(r-1)^2}{r}\right)\le0.
$$

전체 질량 보존보다 강한 분포 정보를 준다. log r−1+1/r≥0이므로 소산은 2θD 이상이고 D(t)≤e^(−2θt)D(0)다. 이 부등식은 log r−1+1/r를 미분해 r=1의 최소값 0을 확인하면 된다. Gaussian의 구체적 엔트로피 추정이며 일반 밀도에는 적절한 함수 부등식과 정당화된 부분적분이 필요하다.

| m, r | 상대 엔트로피 D |
|---|---|
| 0, 1 | 0 |
| v*=1에서 1, 1 | 1/2 |
| 0, 2 | (1−log2)/2 |

```python
from math import exp, log
theta, stationary, m0, v0 = 1., 1., 1., .25
def entropy(m, v):
    r = v/stationary
    return .5*(m*m/stationary+r-1-log(r))
initial = entropy(m0, v0)
for t in (0., .5, 1.):
    m = m0*exp(-theta*t)
    v = stationary+(v0-stationary)*exp(-2*theta*t)
    r = v/stationary
    D = entropy(m, v)
    dissipation = theta*(m*m/stationary+(r-1)**2/r)
    assert D <= exp(-2*theta*t)*initial+1e-14
    assert dissipation >= 2*theta*D-1e-14
    print(t, D, dissipation)
```

코드로 감소 상계와 비음수 소산을 확인한다. 실제 밀도 계산은 SDE 계수에서 flux를 ρ[−θm+(σ²/(2v)−θ)(x−m)]로 평가하고 곱의 미분으로 공간 도함수를 구하도록 보완했다. Drift·확산 flux의 해석적 합과 같지만 큰 두 항이 상쇄할 때 작은 평균 이동 기여를 유지한다. 별도 밀도 시간 미분으로 보존을 확인하며, 부동소수점 진단을 interval 인증으로 해석하지 않는다.

## 직접 풀기: 정상과 자유 확산

정상 표준 Gaussian의 v'=0을 검산하세요. Theta=0,sigma=1,m0=1,s0=1/2이면 t=1의 평균은 1, 분산은 1.25입니다. 평균보다 큰 위치의 rho_x와 diffusion flux 부호를 구하고, 전체 확률 적분에서 어떤 경계 가정을 쓰는지 설명하세요.

더 읽기: [Lawler의 Diffusion·Heat Equation 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
