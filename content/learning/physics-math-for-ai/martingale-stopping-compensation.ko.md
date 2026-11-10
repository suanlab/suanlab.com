## Martingale은 조건부 평균에 대한 명제다

M_n이 주어진 정보 filtration의 martingale이려면 적분 가능·adapted이어야 하고 E[M_(n+1)|F_n]=M_n입니다. 경로가 평평하거나 예측 불가능해 보인다는 뜻이 아닙니다. 한 실현이 크게 움직여도 조건부 평균 관계는 정확할 수 있습니다.

증분이 확률 p로 +1, 나머지로 -1인 walk의 drift는 d=2p-1입니다. 원래 walk는 d=0일 때 martingale이고 예측 가능한 drift를 빼면 다음입니다.

$$
M_n=S_n-dn,\qquad E[\Delta M_n\mid\mathcal F_n]=0.
$$

Poisson 과정의 N(t)-lambda t도 같은 보상 원리입니다. 여기서는 유한 확률 재귀로 정지 기대값을 정확히 계산할 수 있는 walk를 사용합니다.

## 상태뿐 아니라 시계도 멈춘다

Tau를 ±B의 첫 hit와 N의 최솟값으로 둡니다. 유계 stopping time입니다. 중간 관측 상한 n의 stopped 과정은 M_(min(tau,n))이며 보상도 실제 stopped time을 씁니다.

$$
M_{\tau\wedge n}=S_{\tau\wedge n}-d(\tau\wedge n).
$$

상태가 멈췄는데 계속 dn을 빼면 가짜 drift를 만듭니다. 위 그림은 실제 한 stopped 표본이고 아래 그림은 정확한 기대값입니다. 둘은 같은 곡선일 필요가 없습니다.

## 유계 선택적 정지를 직접 검산한다

모형은 위치와 정지시각의 joint 확률을 추적합니다. 살아 있는 상태는 p와 1-p로 갈라지고 멈춘 상태는 시계를 유지합니다. 보상값의 확률 가중합은 반올림을 제외하면 0입니다. 여기서는 유계성이 필요한 적분 조건을 확보하며 모든 무한 stopping time에 정리를 적용하지 않습니다.

p=0.75, B=2, N=4에서 시간 2에 hit할 확률은 p²+(1-p)²=0.625입니다. 나머지의 정지시각은 4라 E[tau]=2.75, E[S_tau]=0.5×2.75=1.375입니다. E[M_tau]는 0입니다.

## 직접 풀기: Seed와 정리 구분

같은 편향 예제의 seed를 바꾸면 실제 정지 상태가 달라도 확률 재귀·기대값은 같습니다. B=1이면 첫 step에 반드시 멈춥니다. 관측 시간이 8이어도 보상은 d 한 번입니다. 잘못 8d를 빼면 무엇이 달라지고 왜 martingale 정리의 반례가 아닌지 계산하세요.

더 읽기: [Lawler의 Martingale·정지시각 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
