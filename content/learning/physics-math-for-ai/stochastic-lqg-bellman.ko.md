## 다음 잡음을 보기 전에 정책을 선택한다

확률적 제어는 미래 잡음을 미리 보지 않고 현재 정보로 행동합니다. 실험은 완전히 관측한 X_next=aX+u+sigma Z, 독립 표준정규 Z를 씁니다. Stage 비용은 (X²+R u²)/2, terminal은 qF X_N²/2입니다. R>0, 제어 제약 없음, 현재 상태를 정확히 안다는 가정입니다.

부분 관측에는 정보 상태가 필요하고 제약·상태 의존 잡음은 해법을 바꿀 수 있습니다. 아래는 구체적인 해석 가능 사례이지 모든 제어 문제의 공식이 아닙니다.

## Bellman은 곡률과 잡음 Offset을 나눈다

V_n(x)=P_n x²/2+c_n, 마지막 P_N=qF,c_N=0을 가정합니다. 다음 가치의 조건부 기대로 convex 최소화를 풀면 다음입니다.

$$
u_n=-K_nx,\quad K_n=\frac{aP_{n+1}}{R+P_{n+1}},\qquad
P_n=1+\frac{a^2R P_{n+1}}{R+P_{n+1}},\quad
c_n=c_{n+1}+\frac{\sigma^2P_{n+1}}2.
$$

뒤에서 실제 재귀합니다. 명시한 fully observed 가산 잡음 조건에서 잡음은 offset만 높이고 gain은 바꾸지 않습니다. 잡음을 없애 비용을 줄여도 같은 상태의 최적 행동은 같을 수 있습니다.

## Forward 계산으로 비용을 다시 검산한다

Feedback u=-Kx의 평균·분산은 m_next=(a-K)m, v_next=(a-K)²v+sigma²입니다. 기대 stage 비용은 (1+R K²)(m²+v)/2입니다. 모두 더하고 terminal을 포함하여 V0와 독립적으로 비교합니다. Monte Carlo의 우연한 일치에 의존하지 않습니다.

파랑은 선택 정책의 누적 기대 비용, 주황은 최적 정책입니다. Zero 정책이 초기에는 제어 비용을 덜 쓰면서 마지막에는 더 큰 비용을 낼 수 있습니다. 전체 horizon 최적성이 모든 중간 누적 비용도 작다는 뜻은 아닙니다.

## 한 번의 잡음 있는 결정

a=R=qF=sigma=x0=1,N=1이면 gain=1/2,행동=-1/2입니다. 현재 비용은 0.625,다음 평균은 1/2·분산1이라 terminal도 0.625,합은 1.25입니다. Zero 정책은 1.5입니다. 잡음을 없애면 gain은 같고 최적 비용은 0.75입니다.

## 연속시간 HJB와의 연결

제어 diffusion dX=b(X,u)dt+sigma(X,u)dW의 매끄러운 value는 형식적으로 다음입니다.

$$
V_t+\min_u\left\{\ell(x,u)+b(x,u)V_x+\tfrac12\sigma(x,u)^2V_{xx}\right\}=0,
\qquad V(T,x)=g(x).
$$

이는 조건부 기대 원리의 무한소 형태이며 diffusion generator를 포함합니다. 고전적 검증에는 정칙성·허용 adapted 제어·적분 가능성이 필요합니다. 유한 이산 실험의 Bellman 검산이 일반 연속 문제의 가정을 없애 주지는 않습니다.

## 직접 풀기: 정보의 우위 드러내기

다음 Z를 본 뒤 u를 고르면 왜 다른 쉬운 문제인지 설명하세요. 한 stage Bellman 미분과 양의 곡률 R+P_next를 구하세요. Terminal weight=0이면 마지막 행동이 0인 이유도 확인하세요. 연속시간 HJB가 같은 조건부 최소화 원리와 연결되지만 이 유한 재귀가 일반 HJB solver는 아니라는 점도 구별하세요.

더 읽기: [Boyd의 확률적 선형이차 제어 노트](https://ee363.stanford.edu/archive/lectures/stoch_lqr.pdf).
