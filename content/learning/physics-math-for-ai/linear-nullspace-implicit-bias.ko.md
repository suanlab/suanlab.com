## 학습 오류 0인 모델은 하나가 아니다

한 학습식 a w₁+b w₂=y는 두 가중치를 유일하게 정하지 못합니다. 해들은 가중치 평면의 직선 위에 있습니다. 모든 해가 학습 target을 맞혀도 그 직선에 수직인 test 방향에서는 예측이 달라질 수 있습니다. Implicit bias는 명시적인 norm penalty 없이 최적화 방법과 초기화가 이 여러 해 중 어떤 것을 선택하는지 묻습니다.

위 그림은 같은 축척의 w₁,w₂ 평면입니다. 파랑은 실제 GD 경로와 끝점, 주황 점선은 학습 제약 직선, 빈 원은 최소 Euclidean norm 해입니다. 아래 그림은 실제 training loss이며 선 길이가 일반화 성능을 뜻하지 않습니다.

## Gradient의 방향과 Nullspace

x=(a,b), 손실 L(w)=(x·w-y)²/2이면

$$
w_{k+1}=w_k-\eta(x\cdot w_k-y)x.
$$

갱신은 항상 x와 평행합니다. x에 직교한 v=(-b,a)/||x||에 대해 v·w는 정확한 산술에서 바뀌지 않습니다. 따라서 초기 nullspace 성분은 학습식이 관측하지 못한 채 남습니다. 수치적으로 발산하는 궤적에서는 cancellation과 반올림 때문에 관측한 불변량이 깨질 수 있어 readout의 실제 값과 수학적 항등식을 구분해야 합니다.

## Residual과 수렴 조건

$$
r_{k+1}=(1-\eta\|x\|^2)r_k,\qquad
0<\eta\|x\|^2<2.
$$

이 조건이면 residual은 0으로 수렴하고 초기화에 따른 극한은

$$
w_\infty=w_0+\frac{y-x\cdot w_0}{\|x\|^2}x.
$$

0 초기화에서만 nullspace 성분이 0이므로 최소 Euclidean norm 해 yx/||x||²를 선택합니다. 다른 metric의 gradient, momentum, feature parameterization이나 비선형 네트워크에는 같은 선택 원리를 그대로 적용할 수 없습니다. Eta=0은 학습하지 않으며 조건 경계의 multiplier -1은 일반적으로 감쇠하지 않습니다.

## 손계산과 직접 풀기

a=b=y=1, eta=.25이면 residual multiplier는 .5입니다. 0에서 한 step은 (.25,.25), 두 step은 (.375,.375)이고 극한은 (.5,.5)입니다. 같은 식을 초기값 (1,-1)에서 시작하면 같은 row-space 변화를 더해 (1.5,-.5)로 갑니다. 두 극한 모두 합이 1입니다.

직접 풀기: test feature z=(1,-1)에서 두 극한의 예측을 비교하세요. 풀이: 최소 norm 해는 0을 예측하지만 다른 초기화의 해는 2를 예측합니다. Training loss가 같아도 미관측 방향은 같지 않습니다. 현재 null 방향을 반대로 정하면 projection 부호가 바뀔 뿐 정보가 남는다는 결론은 같습니다.

좌표를 바꾸면 Euclidean norm의 의미도 바뀝니다. 같은 식 w1+w2=1에서 v1=2w1, v2=w2로 쓰면 제약은 .5v1+v2=1입니다. V 좌표의 최소 norm 해 (.4,.8)은 원래 좌표에서 (.2,.8)로, 앞의 (.5,.5)와 다릅니다. 같은 함수 class라도 parameterization과 최적화 metric을 기록해야 하는 이유입니다.

## AI 연구의 Implicit Regularization

좋은 일반화가 관측됐다는 이유만으로 모든 optimizer가 최소 norm을 찾는다고 말할 수 없습니다. 데이터 행렬의 rank, 초기화, step 안정성, metric과 어떤 norm을 말하는지 정해야 합니다. 이 선형 예제에서는 선택 해와 불변량을 직접 증명할 수 있습니다. Neural network의 implicit bias는 별도의 가정과 분석이 필요한 연구 주제입니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/), [MIT 학습 자료](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/).
