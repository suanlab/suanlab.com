## 데이터 하나를 바꾼다는 실험

같은 크기의 두 dataset에서 한 label만 바꾸면 학습 결과가 얼마나 달라질까요? 이 단원은 feature가 없는 intercept predictor w를 사용합니다. Label은 [-1,1]에 있고 첫 값만 교체합니다. 나머지는 +0.5,-0.5를 번갈아 둡니다. 현재 선택한 두 dataset의 민감도와 모든 허용 dataset에 대한 균일 보장을 분리해 계산합니다.

목적함수는 mean((w-y)²)/2+lambda*w²/2입니다. 정확한 해는 w=mean(y)/(1+lambda)입니다. Lambda가 0이어도 현재 intercept 문제는 곡률 1을 가져 유일한 해가 있습니다. 일반적인 과매개변수 문제의 규제 없는 유일성과 혼동하지 마세요.

## Prediction 변화의 균일 Bound

한 label이 [-1,1] 안에서 바뀌면 평균은 최대 2/N 변합니다. 따라서

$$
|w_S-w_{S'}|\le\frac{2}{N(1+\lambda)}.
$$

두 weight는 [-1,1] 안에 있고 test label도 그 구간입니다. Half-squared loss의 weight 미분 절댓값은 최대 2입니다. Loss 자체의 범위도 [0,2]이므로

$$
\beta_{uniform}\le\min\left(2,\frac{4}{N(1+\lambda)}\right).
$$

이것은 모든 허용 replace-one dataset과 test label을 양화한 상한이며 현재 한 neighbor의 정확한 값이라고 부르지 않습니다. 더 날카로운 분석이 가능해도 현재 실습은 검산하기 쉬운 유효 상한을 사용합니다.

## 현재 Neighbor의 정확한 Supremum

두 해를 w,v라 하면 loss 차이는 test label y에 대해 affine입니다.

$$
\ell(v,y)-\ell(w,y)=(v-w)\left(\frac{v+w}{2}-y\right).
$$

따라서 [-1,1]에서 절댓값의 최대는 끝점 중 하나입니다. 파랑·주황은 두 모델의 실제 loss 함수이며 signed query change와 endpoint supremum을 readout에 따로 표시합니다. 작은 query 변화 한 번을 전체 구간 supremum이나 모든 dataset의 안정성으로 바꾸어 읽지 마세요.

## 손계산과 직접 풀기

N=2, lambda=1, 원래 데이터 [-1,.5], 교체 데이터 [1,.5]이면 w=-.125, v=.375입니다. y=-1에서 loss 차이는 .5625, y=1에서는 -.4375이므로 현재 neighbor의 정확한 최대 절댓값은 .5625입니다. 균일 상한은 4/(2·2)=1로 더 보수적입니다.

직접 풀기: N=1, lambda=0에서 -1을 +1로 교체하고 test label y=1을 쓰세요. 풀이: 원래 loss는 2, 새 loss는 0이므로 절댓값 차이 2입니다. 이 경우 bounded-loss cap이 실제로 달성됩니다. Lambda나 N이 늘면 prediction bound는 작아지지만 단일 데이터의 변화가 모든 test 성능의 개선을 보장하는 것은 아닙니다.

## 안정성과 일반화의 연결

학습기 전체가 beta-uniform stable이고 표본이 IID인 조건에서는 replace-one 논증으로 기대 일반화 gap의 크기를 beta로 제어할 수 있습니다. 이는 현재 deterministic dataset의 모든 관측 gap이 beta 이하라는 점별 주장이 아닙니다. High-probability 결과에는 추가 조건과 항이 필요합니다.

AI 연구에서는 관측한 하나의 민감도, 알고리즘 전체의 균일 성질, 기대 일반화와 confidence 보장을 나누어 보고하세요. 현재 계산은 label 교체만 다루며 feature·architecture·데이터 생성 과정을 동시에 바꾸는 perturbation 실험과도 다릅니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
