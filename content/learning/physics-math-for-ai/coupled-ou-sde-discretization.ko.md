## SDE에는 Drift뿐 아니라 잡음 법칙이 있다

확정 초기값 x0에서 OU 방정식 dX=-theta X dt+sigma dW를 생각합시다. 정확한 전이는 초기값의 지수 감쇠와 가중 Brownian 적분입니다. Euler는 한 step 동안 drift를 고정합니다. 강한 비교에는 주변 분산만 맞추는 것이 아니라 같은 잡음 위에 두 방법을 놓아야 합니다.

파랑·주황은 같은 Brownian 증분을 사용합니다. 그 증분만으로 복원하지 못하는 정확 적분의 부분은 두 번째 독립 Gaussian으로 생성합니다. 이 결합을 명시해야 경로 오차의 뜻이 분명합니다.

## 정확 Innovation의 공분산 유도

h, z=theta h, f(z)=(1-exp(-z))/z, f(0)=1로 둡니다. I가 exp(-theta(h-s))의 Brownian 적분이면 다음입니다.

$$
\operatorname{Var}(I)=h f(2z),\qquad
\operatorname{Cov}(\Delta W,I)=h f(z).
$$

독립 표준 Gaussian Z,G를 쓰면 다음처럼 생성합니다.

$$
\Delta W=\sqrt h Z,\qquad
I=\sqrt h\left(f(z)Z+\sqrt{f(2z)-f(z)^2}\,G\right).
$$

각 주변 표준편차에 같은 Z만 곱하면 일반적으로 다른 공분산을 만듭니다. Bridge 분산은 비음수이고 theta→0에서 사라집니다. 가까운 두 항의 상쇄를 피하도록 작은 인자 급수를 사용합니다.

## 관측 오차와 기대값을 분리한다

실제 exact·Euler 재귀와 별도로 모든 독립 Gaussian의 오차 계수를 전달합니다. 이론 MSE는 평균 오차 제곱과 잡음 계수 제곱 합입니다. 보이는 한 경로의 차이를 제곱한 값이 아닙니다.

끝 평균은 x0 exp(-theta T), 분산은 sigma² T f(2theta T)입니다. Theta=0이면 두 방법 모두 x0+sigma W여서 일치하고 sigma=0이면 결정론 ODE 비교입니다.

## Grid 변경에도 결합 명세가 필요하다

한 run 안에서는 잡음을 공유하지만 grid를 바꾸면 seed에서 생성하는 step innovation의 배치가 바뀝니다. 따라서 서로 다른 grid의 표본 차이를 세분화 증명으로 제시하지 않습니다. 이론적 결합 MSE로 오차를 비교합니다. 앞 Brownian 단원의 공유 미세 grid 구성과 다릅니다.

Strong 오차는 joint coupling, weak 오차는 test function의 기대값을 비교합니다. Euler 평균은 x0(1-theta h)^N, 분산은 sigma²h와 (1-theta h)^(2j)의 합의 곱입니다. E[X²]=평균²+분산으로 weak 이차 moment를 비교할 수 있습니다. 한 weak 통계의 일치가 올바른 strong coupling을 대신하지는 않습니다.

## 직접 풀기: 먼저 공분산 검사

Theta=sigma=x0=T=1의 평균 exp(-1), 분산 (1-exp(-2))/2를 구하세요. 한 step의 지수 kernel과 제곱 적분으로 위 공분산·분산을 유도하고 왜 완전 상관 단일 Z가 맞지 않는지 설명하세요. Theta=0에서는 bridge·이론 MSE·관측 차이가 모두 0이어도 끝 분산 sigma² T는 남는지 확인하세요.

더 읽기: [Lawler의 Diffusion 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
