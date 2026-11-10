## 보간 여부와 Test 위험은 다른 질문

모든 training label을 맞히는 것은 미관측 입력의 위험을 결정하지 않습니다. 이 단원은 n개의 Gaussian 관측, 최대 12개 feature, 참 coefficient (1,0,...,0)를 사용합니다. 같은 design의 앞 p개 열을 쓰고 label noise도 고정하여 p만 늘립니다. 서로 다른 모델마다 새 noise를 그려 곡선을 꾸미지 않습니다.

파랑은 실제 training MSE, 주황은 알려진 독립 Gaussian test 분포에서 정확하게 평가한 조건부 MSE입니다. 여기서는 제곱 손실의 절반이 아닌 MSE를 사용합니다. 점은 선택한 p를 강조하며 전체 곡선은 모든 p=0,...,12의 실제 해입니다.

P=0은 feature가 없는 실제 0 예측 모형입니다. Training MSE는 mean(y²), test MSE는 1+sigma²입니다. 같은 seed 7과 sigma=.5에서 이 기준 risk 1.25는 p=1에서 약 .25로 내려가고, p=4에서는 약 .905로 커졌다가 p=5에서 약 .628로 줄어듭니다. 각각 실제 모형의 계산이며 모든 설정이 같은 모양을 갖는다는 주장은 아닙니다.

## 최소 Norm Least Squares와 수치 Rank

작은 design X_p의 SVD를 X_p=U Sigma Vᵀ라 하면 pseudoinverse 해는 다음과 같습니다.

$$
\widehat w_p=\sum_{\sigma_j>\tau}v_j\frac{u_j^Ty}{\sigma_j},\qquad
\tau=10^{-10}\sigma_{max}.
$$

실제 구현은 one-sided Jacobi SVD로 열을 회전합니다. Normal equation에 작은 ridge를 몰래 더하지 않습니다. Threshold 아래의 singular 방향은 버리며 numerical rank와 retained singular condition을 표시합니다. P가 n 이상이어도 rank나 cutoff에 따라 실제 보간이 달라질 수 있습니다. 보고한 condition은 버린 0 방향까지 포함한 full-rank condition이 아닙니다.

## 정확한 조건부 Test Risk 유도

독립 test 입력 z가 N(0,I₁₂)이고 label이 z₁+sigma epsilon이라고 합시다. P차원 weight를 뒤에 0으로 채운 벡터를 w라 두면 독립성과 등방 공분산으로

$$
E[(z^Tw-z_1-\sigma\epsilon)^2\mid X,y]
=\|w-e_1\|^2+\sigma^2.
$$

이는 선택된 training realization에 조건부인 population 평가입니다. 새로운 test 표본의 실측 평균이라고 부르지 않습니다. 학습 coefficient의 분산·bias·조건수에 따라 값이 커질 수 있고, 한 seed에서 곡선이 매끈하거나 반드시 두 번 내려갈 필요는 없습니다.

## 손계산과 직접 풀기

별도의 가장 작은 예로 X=[1,1], y=1을 생각하세요. 최소 norm 해는 (.5,.5)이고 training loss는 0입니다. 참 coefficient가 (1,0)이고 test 공분산이 I라면 무잡음 test MSE는 .5입니다. Training을 맞히는 참 coefficient의 test MSE 0과 다릅니다. 최소 norm 선택이 모든 진짜 모형에 최적인 것은 아닙니다.

직접 풀기: 이 예에 독립 test label noise sigma=.2를 더하면 두 위험은 얼마인가요? 풀이: 최소 norm 해의 위험은 .5+.04=.54이고 참 coefficient는 .04입니다. Noise floor는 없앨 수 없는 공통 항이며 training interpolation 자체가 test risk를 0으로 만들지 못합니다.

## Double Descent를 관측하고 검증하기

Interpolation 근처의 작은 singular 값은 noise를 크게 증폭할 수 있고, 더 많은 feature에서 최소 norm이 달라지며 위험이 줄기도 합니다. 하지만 feature 분포·target·noise·regularization에 따라 모양은 바뀝니다. 현재 원고는 실제 계산한 유한 곡선만 제시합니다. 여러 seed·크기·rank와 조건수, 해법 tolerance를 함께 기록해야 일반성을 논할 수 있습니다.

독립 검산에서는 네 Moore–Penrose 항등식과 알려진 해, normal residual, nested column·noise 보존을 확인합니다. 한 그림이 기존 이론 전체를 반박한다거나 모든 과매개변수 모델의 일반화를 보장한다고 해석하지 마세요.

참고: [Belkin 등 원 논문](https://arxiv.org/abs/1812.11118), [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). 곡선과 예제는 독립 구현입니다.
