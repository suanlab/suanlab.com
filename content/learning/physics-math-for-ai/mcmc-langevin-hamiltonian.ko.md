## 제안·상태·불변 Target을 구별한다

Target은 potential U(x)=x²/2의 표준정규입니다. 실험은 세 알고리즘을 실제 실행합니다. 파란 trace는 거절 뒤 반복 상태까지 포함한 chain이고 주황은 target 평균 0입니다. 유한 경험 평균이 반드시 0이라는 예측이 아닙니다.

Burn-in 이후 실제 보유 상태로 moments를 계산하고 표본 수를 적습니다. 비어 있으면 추정값이 없습니다. 유한 통계가 문제를 드러낼 수 있어도 불변성·수렴을 혼자 증명하지는 못합니다.

## ULA에는 알려진 유한 Step Bias가 있다

dX=-X dt+sqrt(2)dW의 Euler는 다음입니다.

$$
X_{n+1}=(1-h)X_n+\sqrt{2h}Z_n,\qquad
V_h=\frac1{1-h/2},\quad 0<h<2.
$$

V=(1-h)²V+2h를 풀면 얻습니다. h=0.5의 분산은 1이 아닌 4/3입니다. 안정성이 이산 bias를 없애지는 않습니다. h>=2에서는 유한 정상 분산식을 쓰지 않고 비워 두며 불안정 step을 표시합니다.

## Metropolis는 대칭 제안을 보정한다

y=x+sqrt(2h)Z를 제안하고 min(1,exp(U(x)-U(y)))로 수용합니다. 대칭성이 density ratio를 단순화하며 비대칭 제안에는 양방향 density 비가 추가로 필요합니다. 거절이면 x를 유지하고 실제 결정으로 acceptance fraction을 계산합니다.

## HMC는 Momentum과 수치 동역학을 추가한다

독립 표준정규 momentum을 refresh하고 H=(q²+p²)/2의 leapfrog를 실제 수행합니다. 같은 축척 그림의 full-step 점을 잇는 선은 이산 적분 안내이며 정확한 연속 궤적은 아닙니다.

$$
p_{1/2}=p-\epsilon q/2,\qquad
q'=q+\epsilon p_{1/2},\qquad
p'=p_{1/2}-\epsilon q'/2.
$$

가역·체적 보존 제안을 momentum 반전과 min(1,exp(-Delta H))로 보정합니다. 마지막 제안이 거절되어도 그림에는 남습니다. 큰 불안정 step의 극적인 제안 그림이 유용한 sampling을 뜻하지 않습니다.

## 직접 풀기: 세 진단 나누기

h=0.5의 ULA 분산 4/3을 재귀로 확인하세요. 60개 제안·burn-in10에서 왜 초기 상태를 포함한 보유 표본이 51개인지 설명하세요. HMC 마지막 제안 끝점·실제 마지막 chain 상태·에너지 오차도 비교하세요. 높은 수용률·좋은 유한 moments·불변 target의 증명이 왜 서로 다른 근거인지 적으세요.

더 읽기: [Betancourt의 HMC 개념 설명](https://arxiv.org/abs/1701.02434).
