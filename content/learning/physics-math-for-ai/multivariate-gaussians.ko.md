## 스케일과 상관을 분리하기

평균이 0인 두 변수 Gaussian의 공분산은 두 표준편차와 상관계수로 정할 수 있습니다. 표준편차는 양수이고 상관은 −1부터 1까지여야 양의 준정부호 조건을 만족합니다. 기준에서는 두 표준편차가 1, 상관이 0.5라 공분산의 비대각 성분도 0.5입니다. 대각 성분은 표준편차가 아니라 분산입니다.

$$
\Sigma=\begin{pmatrix}\sigma_x^2&\rho\sigma_x\sigma_y\\\rho\sigma_x\sigma_y&\sigma_y^2\end{pmatrix},\qquad\det\Sigma=\sigma_x^2\sigma_y^2(1-\rho^2).
$$

## 타원은 표본 구름이 아니다

독립 표준정규 벡터 U를 L로 변환하면 공분산이 LLᵀ입니다. 그림은 무작위 표본이 아니라 단위원의 경계를 L로 옮긴 곡선입니다. 두 축의 단위 길이를 같게 그려 상관에 따른 기울기를 보존합니다. 비특이인 경우 Mahalanobis 반지름 1 경계이지만 내부 확률은 이차원에서 약 39.3%입니다. 일차원 ±1 표준편차의 약 68%와 혼동하지 마세요.

$$
L=\begin{pmatrix}\sigma_x&0\\\rho\sigma_y&\sigma_y\sqrt{1-\rho^2}\end{pmatrix},\quad Z=LU,\quad P(\|U\|\le1)=1-e^{-1/2}.
$$

## 조건부 평균과 남은 불확실성

공동 Gaussian에서 X=x를 알면 Y의 조건부 평균은 x의 선형함수이고 분산은 x와 무관합니다. 기준 x=1에서는 평균 0.5, 분산 0.75입니다. 상관이 0이면 조건부 평균은 0이고 분산은 원래 분산이라 두 변수는 독립입니다. Gaussian이 아닌 일반 분포에서는 상관 0만으로 독립을 결론낼 수 없습니다.

$$
E[Y\mid X=x]=\rho\frac{\sigma_y}{\sigma_x}x,\qquad\operatorname{Var}(Y\mid X=x)=\sigma_y^2(1-\rho^2).
$$

## 특이 분포와 조건부의 의미

상관이 ±1이면 그림이 직선으로 붕괴합니다. 유효한 Gaussian 확률법칙이지만 평면 면적에 대한 밀도는 없습니다. 실습은 행렬 역수나 2차원 밀도를 억지로 만들지 않습니다. Y는 X의 선형함수라 조건부 분산은 0입니다. 연속 X의 한 점 확률은 0이므로 위 조건부는 초등 사건 비율이 아니라 Gaussian 모형의 정칙 조건부분포 버전으로 이해합니다. 그래프 기반 모델이나 Gaussian 예측에서 지지 차원과 밀도의 기준 측도를 구분해야 합니다.

## 확인 문제와 해설

1. σx=1,σy=2,ρ=1,x=1이면? **조건부 평균 2, 분산 0입니다.**
2. 행렬식 0은 확률법칙이 없다는 뜻인가요? **아니요. 평면 밀도 공식이 적용되지 않는 것입니다.**
3. 그려진 점들은 표본인가요? **아니요. 단위원 경계의 변환입니다.**

추가 읽기: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
