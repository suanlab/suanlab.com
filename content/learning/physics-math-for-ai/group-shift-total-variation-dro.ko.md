## 학습과 Target의 Group 비중이 바뀌면

두 group의 label이 각각 0과 1인 세계를 정합니다. 학습에서 group 1 확률은 alpha, target에서는 beta입니다. Conditional label 규칙은 유지하고 group 비중만 바뀝니다. 가설 0은 항상 0, 가설 1은 항상 1, 가설 2는 독립적인 fair coin으로 예측합니다. 세 번째의 group별 loss .5는 실제 동전 결과 하나가 아니라 무작위 예측에 대한 기대 0-1 loss입니다.

실제 training 표본의 group count로 ERM을 구합니다. 파랑은 선택한 가설의 target group 비중별 위험, 주황은 강건하게 선택한 가설의 위험이며 점은 현재 beta입니다. 알려진 세계의 oracle risk와 표본에서 추정할 수 있는 위험을 구분해야 합니다.

## Importance Weight와 Support

학습 group 확률 P_g와 target 확률 Q_g에 대해 적절한 조건에서

$$
R_Q(h)=E_P\left[\frac{Q_G}{P_G}\ell(h,G)\right].
$$

Q_g>0인 곳에 P_g>0이어야 하고 현재처럼 group 안의 label 조건부 규칙도 유지돼야 합니다. P_g=0인데 Q_g>0이면 비율은 정의되지 않으며 그 영역의 손실은 추가 모형 가정 없이 관측 표본만으로 식별되지 않습니다. 실험은 importance estimate를 미정으로 표시합니다. 세계를 완전히 정해 계산한 oracle risk는 여전히 보이지만 이것이 표본의 식별 문제를 해결하는 것은 아닙니다.

Support가 있어도 큰 weight는 높은 추정 분산을 만들 수 있습니다. 한 유한 표본의 weighted 평균이 target risk와 정확히 같을 필요는 없습니다. Unbiased라는 말은 반복 training 표본의 기대값에 대한 것입니다.

## Total Variation 모호성 집합

두 점 분포의 TV 거리는 group 1 확률 차이의 절댓값입니다. 선언한 population 중심 alpha와 radius rho로

$$
q\in[\max(0,\alpha-\rho),\min(1,\alpha+\rho)],\qquad
R_q(h)=(1-q)\ell_{h0}+q\ell_{h1}.
$$

선형 함수의 최댓값은 interval 끝점에 있으므로 실제 두 끝점을 비교하고 세 가설 중 worst risk가 가장 작은 것을 고릅니다. 이 집합은 알려진 중심의 모형 설정이지 자동으로 데이터에서 얻은 confidence set이 아닙니다. Target beta가 집합 밖이면 그 target에 대한 보호라고 주장할 수 없습니다.

## 손계산과 직접 풀기

Alpha=.9에서 항상 1인 가설의 학습 population risk는 .1입니다. Beta=.1로 바뀌면 target risk는 .9가 됩니다. Rho=.8이면 interval은 [.1,1]이고 worst risk는 항상 0인 가설 1, 항상 1인 가설 .9, fair coin .5입니다. 따라서 현재 finite class의 robust 선택은 fair coin입니다.

직접 풀기: alpha=.9, beta=.1의 두 importance weight를 구하세요. 풀이: group 0은 .9/.1=9, group 1은 .1/.9=1/9입니다. 드문 group 0이 큰 weight를 받습니다. Alpha=1, beta=0이면 group 0의 분모가 0이므로 같은 계산을 강행하거나 임의의 큰 값으로 대체하면 안 됩니다.

## AI 강건성 평가에서의 해석

어떤 shift를 허용하는지, 조건부 label을 유지하는지, radius를 어떻게 정했는지 보고하세요. Worst-case 위험은 선언한 ambiguity set과 class에 대한 결과입니다. 알려지지 않은 모든 현실 변화를 견딘다는 뜻이 아닙니다. Training 평균, target oracle, importance 추정, finite-sample 불확실성과 minimax 목적을 따로 기록하는 것이 강건성 주장의 기본입니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/), [MIT 학습 자료](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/).
