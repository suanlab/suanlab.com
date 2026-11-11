## 시계와 법칙을 함께 뒤집는다

Forward 끝 시간 T를 고정하고 reverse 경과 s를 두면 물리 시간은 t=T-s입니다. 역 과정의 시작은 실제 p_T여야 합니다. 편리하다는 이유로 무관한 표준 Gaussian을 쓰면 다른 문제를 풉니다.

Markov 단원에서 Bayes 역전 행은 끝점 주변 법칙에 의존하고 support 밖에서는 정해지지 않았습니다. Diffusion도 정칙성과 밀도 조건이 필요합니다. 이 실험은 상수 양의 diffusion과 매끄러운 양의 Gaussian에 한정합니다.

## Score 보정 OU Drift 유도

Forward drift가 -theta x, diffusion 진폭이 sigma일 때 증가하는 reverse 시계의 식은 다음입니다.

$$
\widetilde b(x,s)=\theta x+\sigma^2\partial_x\log p_{T-s}(x),
\qquad \partial_x\log p_t(x)=-\frac{x-m_t}{v_t}.
$$

파랑은 이 식, 주황은 단순 부호 반전 theta x입니다. Score는 forward 법칙에서 현재 상태의 상대 가능성을 반영합니다. 임의로 생략할 항이 아닙니다. 상태 의존 diffusion에는 추가 고려가 있어 이 상수 식을 그대로 복사하면 안 됩니다.

## 역방향 Moments를 독립 검산한다

역 주변 법칙의 평균·분산은 m(T-s),v(T-s)입니다. 시계 미분은 theta m,2theta v-sigma²이고 보정 drift의 moment도 같습니다.

$$
\frac{dm}{ds}=\theta m,\qquad
\frac{dv}{ds}=2(\theta-\sigma^2/v)v+\sigma^2
=2\theta v-\sigma^2.
$$

두 경로를 비교해 역전의 성립을 검산합니다. 분산 감소가 음의 diffusion을 필요로 하지는 않습니다. Score drift가 양의 확률적 퍼짐을 이깁니다.

## 정상 Reversibility도 단순 부호 반전이 아니다

Theta=1,sigma=sqrt(2),정상 분산 1의 score는 -x입니다. 역 drift는 x-2x=-x로 원래와 같습니다. 단순 반전 +x는 정상 법칙에서 멀어집니다. 랜덤 동역학과 결정론적 속도의 역전이 다른 이유입니다.

Theta=0의 표준 Gaussian 출발에서도 forward drift는 0이지만 분포는 퍼집니다. 역전에는 score drift가 남으며 sigma=1,물리 시간 0에서는 -x입니다.

## 같은 주변 분포가 같은 경로 법칙은 아니다

상수 diffusion의 순방향 Fokker–Planck flux는 bp−σ²pₓ/2다. 속도 b−σ²score/2의 결정론 continuity 식도 같은 flux를 가지므로 필요한 적정성 조건 아래 같은 주변 분포를 만든다. OU의 증가하는 역시간 속도는 θx+σ²score/2다. 역 SDE는 θx+σ²score와 Brownian 잡음을 함께 쓴다. 절반 인자가 중요하다. 역 SDE의 잡음만 없애면 주변 분포가 유지되지 않는다.

Gaussian의 결정론 역 흐름은 명시적으로

$$
x(s)=m_{T-s}+\sqrt{v_{T-s}/v_T}\,(x_T-m_T).
$$

미분하면 m′ᵣ+(v′ᵣ/(2v))(x−m)로 역 probability-flow 속도와 같다. 끝 Gaussian을 원하는 이전 Gaussian으로 보낸다. 잡음 있는 역 SDE는 평균·분산이 같아도 시각 사이 joint 법칙이 다르다. 어느 방법도 무관한 끝 분포에서 시작하도록 허용하지 않는다.

| θ=0, σ=v=1, m=0 | 증가하는 역시간 Drift |
|---|---|
| 잡음 있는 역 SDE | −x |
| 잡음 없는 probability-flow ODE | −x/2 |
| 단순 부호 반전 | 0 |

```python
from math import exp, sqrt
theta, sigma, m0, v0, T = .7, 1., .8, .4, 1.
def moments(t):
    decay = exp(-theta*t)
    return m0*decay, v0*decay**2+sigma**2*(1-decay**2)/(2*theta)
mT, vT = moments(T)
def flow(s, terminal):
    m, v = moments(T-s)
    return m+sqrt(v/vT)*(terminal-mT)
for s in (.25, .5, .75):
    m, v = moments(T-s)
    assert abs(flow(s,mT)-m) < 1e-14
    scale = sqrt(v/vT)
    assert abs(scale*scale*vT-v) < 1e-14
    x = flow(s, 1.2)
    velocity = theta*x-sigma**2*(x-m)/(2*v)
    difference = (flow(s+1e-5,1.2)-flow(s-1e-5,1.2))/2e-5
    assert abs(difference-velocity) < 1e-9
    print(s, x, velocity)
```

이 정확 Gaussian 사상은 보조 예제이며 별도의 live ODE solver를 추가한 것은 아니다. 코드는 수송된 moments와 독립 시간 차분의 속도를 확인한다. 일반 학습 score 모형에는 근사 오차와 수치 적분 분석도 필요하다. 이상적인 주변 분포 일치가 이산 학습 sampler의 재현을 증명하지는 않는다.

## 직접 풀기: 필요한 시작 법칙

초기 평균 1·분산 1/4의 T=1 법칙을 구해 역 시작 Gaussian으로 쓰세요. Reverse 비율 1/2의 실제 forward 시각도 구하세요. 표준 Gaussian으로 임의 대체하거나 score를 T-s 대신 s에서 평가하면 어떤 moment 식이 깨지는지 설명하세요.

더 읽기: [Song 등의 Score-based SDE 논문](https://arxiv.org/abs/2011.13456).
