## Integrand를 평가하는 위치가 합을 바꾼다

매끄러운 미적분에서는 작은 구간의 왼쪽·중간 평가 차이가 사라질 수 있습니다. Brownian 증분은 누적 제곱이 사라지지 않아 차이가 극한에 남습니다. 실험은 같은 Brownian 표본으로 두 합을 구하므로 표본 변경이 원인이 되지 않습니다.

파랑은 왼쪽 W_i, 주황은 양 끝 평균 (W_i+W_next)/2를 씁니다. 이 선형 integrand에서는 symmetric midpoint 규칙이며 임의 stochastic 방정식의 모든 midpoint 알고리즘이 같다는 뜻은 아닙니다.

## 두 유한 항등식 유도

제곱을 전개하면 W_next²-W_i²=2 W_i dW_i+(dW_i)²입니다. W0=0에서 더하면 다음입니다.

$$
L_N=\sum_i W_i\Delta W_i=\frac12(W_T^2-QV_N),\qquad
S_N=\sum_i\frac{W_i+W_{i+1}}2\Delta W_i=\frac12W_T^2.
$$

모든 유한 경로의 대수 항등식으로 QV=T를 가정하지 않습니다. 구현은 각 항을 실제로 더하고 별도로 계산한 항등식 값을 보여줍니다.

## Brownian 적분 극한으로 이동하기

적절한 세분화에서 Brownian QV가 시간으로 수렴하면 다음입니다.

$$
\int_0^T W_t\,dW_t=\frac12(W_T^2-T),\qquad
\int_0^T W_t\circ dW_t=\frac12W_T^2.
$$

동그라미는 Stratonovich 관례입니다. 이 예에서는 보통 chain rule 모양이고 Itô에는 이차 보정이 있습니다. 유한 관측 QV를 먼저 T로 바꾸면 이산 오차를 측정하지 않고 지워 버립니다.

## 일반 Chain Rule과 Drift 변환

매끄러운 F와 Itô SDE dX=b dt+sigma dW에서는 이차 도함수 보정이 있습니다.

$$
dF(X)=F'(X)b\,dt+F'(X)\sigma\,dW+\tfrac12F''(X)\sigma^2\,dt.
$$

적절한 매끄러움 아래 Stratonovich는 보통 chain rule 형태입니다. 같은 과정의 일차원 drift는 b_I=b_S+sigma sigma'/2입니다. 상수 diffusion은 drift 변환 항이 없어도 비선형 F의 Itô 보정은 남습니다. 두 보정을 혼동하지 마세요. W 제곱 예제는 뒤의 차이를 분리합니다.

## 정확한 MSE를 구할 수 있다

Left sum과 Itô 적분의 오차는 (T-QV_N)/2입니다. 균등 grid에서 평균은 0, 평균제곱은 T²/(2N)입니다. Gaussian 경로 전체에 대한 기대값이며 한 경로의 오차 제곱과 같을 필요는 없습니다. 끝점을 유지한 세분화도 관측 오차의 단조 감소를 강제하지 않습니다.

T=1,N=64의 이론 MSE는 1/128, T=2,N=16은 1/8입니다. 시간·grid 배율을 경로의 sqrt(T) 배율과 구별하세요.

## 직접 풀기: 세 보정값 비교

현재 표본의 S_N-L_N, QV_N/2, T/2를 계산하세요. 처음 둘은 반올림을 제외하면 같고 마지막은 극한 보정이라 유한 grid에서 다릅니다. 증분을 결정론적 매끄러운 경로로 바꾸어도 유한 항등식은 맞지만 QV 극한은 바뀝니다. 적분 관례뿐 아니라 과정의 법칙도 필요한 이유를 설명하세요.

더 읽기: [Lawler의 확률 적분 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
