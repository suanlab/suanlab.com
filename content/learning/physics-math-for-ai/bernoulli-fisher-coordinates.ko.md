## Fisher 정보는 모형에 대한 평균이다

내부 Bernoulli 분포에서 y∈{0,1}의 확률은 pʸ(1−p)¹⁻ʸ다. Score는 선택한 매개변수에 대한 로그 확률 미분이다. 확률 좌표에서는

$$
s_p(y)=\frac{y-p}{p(1-p)},\qquad
\mathbb E_p[s_p]=0,\qquad
I_p=\mathbb E_p[s_p^2]=\frac{1}{p(1-p)}.
$$

기대값은 현재 모형 분포를 사용한다. 관측 label 하나의 score 제곱은 일반적으로 이 기대값과 다르다. p=0.2에서 label 1은 score 5와 제곱 25, label 0은 score −1.25와 제곱 1.5625다. 모형 평균은 0.2·25+0.8·1.5625=6.25다.

## 접벡터와 함께 좌표를 바꾸기

Logit θ=log[p/(1−p)]에서 dp/dθ=p(1−p)다. Score는 y−p, 정보는 p(1−p)가 된다. 숫자가 다르다고 다른 통계 기하는 아니다.

$$
I_\theta=\left(\frac{dp}{d\theta}\right)^2 I_p,
\qquad I_p\,dp^2=I_\theta\,d\theta^2.
$$

Metric 계수와 접벡터 좌표를 함께 읽어야 한다. 경계 근처의 큰 확률 좌표 계수는 좌표의 늘어남을 반영한다. 이를 곧바로 모든 좌표에서 더 어려운 학습이라고 해석하면 매개변수화를 놓친다.

## 국소 KL의 이차항

작은 확률 변화 Δ에 대해

$$
D_{KL}(\operatorname{Ber}(p)\|\operatorname{Ber}(p+\Delta))
=\frac12 I_p\Delta^2+O(\Delta^3).
$$

실습의 질의는 내부 확률로 제한한다. 파란 선은 독립 표본 n개의 정확한 KL, 주황 선은 Fisher 이차 근사다. 그림은 p에서 0.01 이내의 q를 확대하고 작은 KL 값에 맞춰 세로축을 정한다. 국소적으로 가까워지는 것이지 모든 변화량에서 같은 식은 아니다. 계산은 상쇄되는 일차항을 먼저 제거한 로그 나머지로 안정화한다.

p=0.5,Δ=0.01,n=1이면 정확한 KL≈0.000200040, 근사 0.0002다. p=0.2에서는 비대칭 때문에 삼차 보정이 더 드러난다.

## 독립성과 정칙성

같은 매개변수의 독립 관측 n개는 score가 더해지고 평균 0의 교차항이 사라져 정보가 nI다. 상관된 관측에는 결합 likelihood를 사용해야 한다. 지지집합 변화·경계 매개변수·특이 모형에서는 정칙성 가정 없이 score·Hessian 항등식을 적용하면 안 된다.

Label 조절기는 관측 score만 바꾸고 true Fisher와 KL 곡선은 그대로 둔다. 표본 수는 독립 정보와 KL을 배율 조정하며, 표에는 관측 label 한 개의 예시를 계속 보여준다.

## 직접 풀기와 풀이

p=0.2,n=10이면 확률 좌표 전체 정보는 62.5이고 표본당 logit 정보는 0.16이다. Label을 1에서 0으로 바꾸어 관측 항만 달라지는지 보자. 데이터 label로 계산한 gradient outer-product 행렬과 모형 기대값 Fisher가 둘 다 양의 준정부호여도 같지 않을 수 있는 이유를 설명하자.

더 읽기: [Martens의 Natural Gradient 분석](https://www.jmlr.org/papers/v21/17-678.html).
