## 충분한 표본과 정확한 실패 확률

IID Bernoulli(p)의 평균으로 p를 추정합니다. 얼마나 많은 표본이 필요한지는 오차 epsilon, 실패 확률 delta와 어떤 보장을 원하는지에 달렸습니다. 현재 실습은 실제 정수 성공 횟수의 binomial 확률을 합한 exact tail과 Hoeffding의 분포 독립 상한을 나란히 계산합니다. 파랑은 exact tail, 주황은 확률 1로 clip한 upper bound입니다.

사건은 |K/N-p|≥epsilon으로 경계를 포함합니다. 성공 횟수 K는 정수이므로 작은 N에서는 이 경계가 오차 확률을 크게 바꿀 수 있습니다. 구현은 log-domain PMF로 극단적인 p의 underflow를 피하며 floating equality는 규모에 맞춘 machine-precision 수준에서 처리합니다.

## 실제 Tail의 계산

$$
P\left(\left|\frac KN-p\right|\ge\epsilon\right)
=\sum_{k\le\lfloor N(p-\epsilon)\rfloor\ \mathrm{or}\ k\ge\lceil N(p+\epsilon)\rceil}
{N\choose k}p^k(1-p)^{N-k}.
$$

현재 N은 최대 128입니다. P=0 또는 1은 한 점 확률 질량으로 별도 처리합니다. PMF를 임의의 곡선으로 바꾸거나 각 N의 정확한 결과를 매끄럽게 강제하지 않습니다. 모델 내에서 exact라는 말도 floating 계산의 반올림 한계를 없애지는 않으므로 독립 factorial 계산과 normalization을 검산합니다.

## Hoeffding 충분 조건 유도

독립인 [0,1] 변수에 대해

$$
P(|\overline X-p|\ge\epsilon)\le2e^{-2N\epsilon^2}.
$$

이를 delta 이하로 만들면

$$
N\ge\left\lceil\frac{\log(2/\delta)}{2\epsilon^2}\right\rceil.
$$

이것은 모든 p에 대해 충분한 조건입니다. 특정 p의 필요한 최소 표본 수를 정확하게 구한 것이라고 부르면 안 됩니다. 표시된 N 범위보다 충분 N이 클 수도 있습니다. Delta를 바꾸면 목표 confidence 기준은 달라지지만 같은 N,p,epsilon의 실제 tail은 바뀌지 않습니다.

## 손계산과 직접 풀기

p=.5, epsilon=.2, N=3이면 가운데 성공 횟수 1,2의 오차는 1/6이라 사건에 포함되지 않습니다. K=0 또는 3만 포함해 확률은 2/8=.25입니다. N=4에서는 K=1,3의 오차 .25도 포함하므로 확률은 (1+4+4+1)/16=.625입니다. 표본이 늘어도 개별 finite tail이 항상 감소하는 것은 아닙니다.

직접 풀기: epsilon=.2, delta=.05의 Hoeffding 충분 N을 구하세요. 풀이: log(40)/.08≈46.11이므로 47입니다. 이 bound는 이후 모든 N≥47을 제어합니다. 반면 exact curve가 처음 .05 아래로 내려간 N을 찾았다는 사실만으로 이후 모든 N에서 계속 그렇다고 결론내릴 수는 없습니다.

알려진 p의 exact curve와 실제로 미지의 p를 추정하는 문제도 구별하세요. 모든 관측이 0이라고 p=0이 증명되지는 않습니다. 예를 들어 p=.1에서도 열 번 모두 0일 확률은 .9^10으로 약 .349입니다. 화면의 p는 세계를 정하는 oracle 설정이며 학습자가 데이터만으로 알아낸 참값이 아닙니다.

## AI 데이터 요구량의 보고

Sample complexity를 말할 때 손실, 오차 사건, confidence, IID와 분포 범위를 명시하세요. 표본 의존·선택 편향·분포 이동이 있으면 같은 bound를 그대로 적용할 수 없습니다. Estimation variance, RMSE와 deviation probability는 서로 다른 양이므로 지수도 같은 방식으로 비교하면 안 됩니다. 다음 단원은 이론 경계와 실제 learning curve의 finite-window slope를 구분합니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
