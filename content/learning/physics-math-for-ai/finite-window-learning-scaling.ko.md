## 이론적 기대와 유한 관측의 기울기

표본 평균으로 참 평균 0을 추정합니다. 각 관측은 독립 Rademacher 모형의 ±sigma이며 추정기에 고정 bias b를 더할 수 있습니다. 실제 반복마다 가장 긴 표본을 생성한 뒤 N=2,4,...,128의 prefix를 사용합니다. 같은 반복 안의 서로 다른 N은 연결돼 있으므로 각 점을 독립 관측이라고 가정하지 않습니다.

파랑은 실제 추정기들의 제곱 오차 평균, 주황은 모형에서 유도한 기대입니다. Prediction noise floor를 켜면 각 추정기에 조건부인 정확한 test noise variance를 더합니다. 독립 test 표본을 실제 관측한 것처럼 만든 숫자가 아닙니다.

## 기대 Risk의 유도

표본 평균은 중심 0, 분산 sigma²/N이고 bias는 고정입니다.

$$
E[(\overline X+b)^2]=b^2+\frac{\sigma^2}N.
$$

새로운 독립 label의 잡음까지 예측하면 총 위험은 b²+sigma²/N+sigma²입니다. 이 조건부 risk는 학습된 각 추정기에서 정확히 평가하고 실제 반복 평균을 냅니다. 모형의 기대와 반복 평균이 같도록 강제로 맞추지 않습니다.

같은 현상도 metric에 따라 지수가 다릅니다. Mean estimator의 MSE가 1/N이면 RMSE는 1/sqrt(N)입니다. 앞 단원의 confidence 반경이나 deviation probability와 이 MSE의 지수를 직접 비교하면 정의가 섞입니다.

## 바닥과 관측 구간이 만드는 효과

R(N)=C+A/N이라면 연속적인 log-log 접선 기울기는 다음과 같습니다.

$$
\frac{d\log R}{d\log N}=-\frac{A/N}{C+A/N}.
$$

C=0이면 -1이지만 바닥 C>0이 지배하면 0으로 평평해집니다. 따라서 같은 추정기라도 bias·irreducible noise와 N 구간에 따라 보이는 slope가 달라집니다. 실제 유한 반복의 slope는 표본 변동 때문에 이 이론 범위를 벗어날 수도 있습니다.

## 실제 Fit과 미정 값 처리

현재는 양수인 repetition-average risk 점에만 ordinary least squares를 적용합니다.

$$
\widehat s=\frac{\sum_i(\log N_i-\overline{\log N})(\log \widehat R_i-\overline{\log\widehat R})}{\sum_i(\log N_i-\overline{\log N})^2}.
$$

양수 점이 두 개 미만이면 slope는 미정입니다. Risk 0을 임의의 작은 양수로 바꿔 그럴듯한 fit을 만들지 않습니다. Prefix 간의 의존성 때문에 단순 독립 잔차 standard error를 자동 보고하지도 않습니다. 여러 반복·seed·구간과 추정 절차를 함께 검토해야 합니다.

## 손계산과 직접 풀기

Sigma=1,b=0,noise floor 없음이면 N=128의 기대 risk는 1/128=.0078125입니다. Prediction noise floor를 더하면 1.0078125입니다. 후자의 큰 N 기울기가 거의 0인 것은 추정기가 갑자기 학습을 멈췄기 때문이 아니라 총 위험의 바닥이 지배하기 때문입니다.

직접 풀기: sigma=0,b=0의 모든 관측 위험과 log slope는 무엇인가요? 풀이: 모든 위험은 0이고 log는 정의되지 않아 slope도 미정입니다. Sigma=0,b=1이면 모든 위험은 1이므로 두 개 이상의 N에서 slope는 0입니다. 두 상황을 같은 평평한 성공 곡선이라고 부르면 정의를 놓칩니다.

## 새로운 AI 연구의 Scaling 주장

유한 구간의 좋은 power-law fit은 보편 neural scaling 정리가 아닙니다. 과제·모형·data distribution·평가 metric·compute·바닥·관측 구간을 기록하고 대안 설명을 비교하세요. 이 모형의 정확한 1/N 기대는 가정에서 유도됐고 파랑 관측은 실제 반복에서 나왔다는 차이를 유지하는 것이 연구 해석의 핵심입니다.

참고: [MIT 학습 자료](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/), [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
