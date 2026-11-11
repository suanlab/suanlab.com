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

## 독립 잡음을 분리하여 조건부 공식을 유도하기

양의 표준편차와 |ρ|<1에서 독립 표준정규 U1,U2를 사용하여 X=σx U1, Y=ρσy U1+σy√(1−ρ²)U2로 씁니다. β=ρσy/σx와 η=Y−βX를 정의하면 η는 U2에만 의존하므로 X와 독립입니다. X를 알아도 Y=βX+η의 첫 항만 정해지고 잡음의 분포는 같습니다. 따라서 조건부 평균과 분산은 공식을 대입하는 대신 이 구성에서 증명됩니다. 남은 분산은 스칼라 Schur complement입니다.

$$
\beta=\frac{\Sigma_{YX}}{\Sigma_{XX}},\quad
\eta=Y-\beta X,\quad
\operatorname{Var}(\eta)=\Sigma_{YY}-\frac{\Sigma_{YX}^2}{\Sigma_{XX}}.
$$

σx=2, σy=3, ρ=1/2이면 공분산의 두 행은 (4,3),(3,9)입니다. 따라서 β=3/4, 잔차 분산은 27/4입니다. X=2일 때 조건부 평균은 3/2이고 조건부 분산은 여전히 27/4입니다. 관측한 한 점의 밀도를 사건 확률처럼 다뤄 얻는 결과가 아니라 조건부 Gaussian 법칙의 연속인 버전을 지정한 것입니다. |ρ|=1에서는 η가 항상 0이 되어 앞의 특이 분포 설명과 일치합니다.

## 조건부 평균과 최적 예측을 연결하기

η는 X와 독립이고 평균 0이므로 X의 제곱 적분 가능한 함수와 곱한 교차항도 0입니다. 모든 제곱 적분 가능한 예측기 h(X)에 대해 제곱 오차를 전개하면 다음과 같습니다.

$$
E[(Y-h(X))^2]=\operatorname{Var}(\eta)+E[(\beta X-h(X))^2].
$$

따라서 βX는 선형 예측기뿐 아니라 이 조건을 만족하는 모든 예측기 중 기대 제곱 오차를 최소화합니다. h(X)=aX로 제한하면 Var(η)+σx²(a−β)²입니다. 잔차 잡음의 독립성을 얻는 데 공동 Gaussian 구조를 사용했습니다. 일반 분포에서도 두 변수의 이차 모멘트가 유한하고 Var(X)>0이면 공분산으로 최적 중심 선형 예측기를 구할 수 있지만 조건부 평균과 조건부 분산까지 정해지는 것은 아닙니다.

## 같은 공분산으로 조건부 법칙을 바꾸는 반례

X가 −1,0,1을 같은 확률로 갖고 Y=X²−2/3이라고 합시다. 두 평균은 0이고 대칭성 때문에 공분산도 0입니다. 반면 Var(X)=2/3, Var(Y)=2/9입니다.

| X | Y | 결합 점 확률 |
|---|---|---|
| −1 | 1/3 | 1/3 |
| 0 | −2/3 | 1/3 |
| 1 | 1/3 | 1/3 |

동일한 대각 공분산을 갖는 중심 공동 Gaussian 쌍이라면 독립이고 조건부 평균 0, 조건부 분산 2/9입니다. 그러나 표의 실제 법칙에서는 X가 Y를 완전히 결정합니다. 조건부 평균은 X²−2/3, 조건부 분산은 0입니다. 최적 선형 예측의 오차는 2/9이지만 주어진 비선형 예측의 오차는 0입니다. 따라서 공분산을 맞췄다고 일반 자료에 Gaussian 조건부 법칙을 그대로 적용할 수는 없습니다.

## Schur complement와 반례를 정확하게 검산하기

다음 코드는 첫 공분산 행렬에 (X,Y)↦(X,Y−βX)를 적용하고 둘째 예제의 모멘트와 손실을 분수로 계산합니다. 코드의 대각 공분산은 대수적 검산이며 Gaussian 모형의 독립성은 앞의 독립 잡음 구성으로 정당화했습니다. 유한 반례는 이 추가 가정의 필요성을 보여줍니다. 현재 타원 조절기는 기존 중심 Gaussian 모형을 계속 표시하며 아래는 별도의 유도와 검산입니다.

```python
from fractions import Fraction as Q

Sigma = ((Q(4), Q(3)), (Q(3), Q(9)))
beta = Sigma[1][0] / Sigma[0][0]
residual = Sigma[1][1] - Sigma[1][0]**2 / Sigma[0][0]
A = ((Q(1), Q(0)), (-beta, Q(1)))
transformed = tuple(tuple(sum(A[i][k]*Sigma[k][l]*A[j][l]
                              for k in range(2) for l in range(2))
                          for j in range(2)) for i in range(2))
assert beta == Q(3, 4) and residual == Q(27, 4)
assert transformed == ((Q(4), Q(0)), (Q(0), residual))
assert beta*2 == Q(3, 2)
print("Gaussian slope, residual variance, mean at x=2:", beta, residual, beta*2)

points = [(Q(x), Q(x*x)-Q(2, 3)) for x in [-1, 0, 1]]
expect = lambda f: sum(f(x, y) for x, y in points) / 3
ex, ey = expect(lambda x, y: x), expect(lambda x, y: y)
vx = expect(lambda x, y: (x-ex)**2)
vy = expect(lambda x, y: (y-ey)**2)
cov = expect(lambda x, y: (x-ex)*(y-ey))
assert (ex, ey, vx, vy, cov) == (0, 0, Q(2, 3), Q(2, 9), 0)
linear_error = expect(lambda x, y: (y-0*x)**2)
nonlinear_error = expect(lambda x, y: (y-(x*x-Q(2, 3)))**2)
assert linear_error == Q(2, 9) and nonlinear_error == 0
for point in points:
    print("non-Gaussian x, y, mass:", *point, Q(1, 3))
print("same covariance, different conditional errors:", linear_error, nonlinear_error)
```

## 추가 문제와 연구 연결

1. 조건부 분산 0은 Y 자체의 분산도 0이라는 뜻인가요? **X의 결정적 함수로 변할 수 있으므로 아닙니다.**
2. 절대 손실에서도 선형 예측기가 최적인가요? **제곱 오차의 증명은 적용되지 않으며 조건부 중앙값이 관련 최적해입니다.**
3. 관측 공분산 블록이 특이면 어떻게 되나요? **여기서 사용한 나눗셈을 쓸 수 없으므로 지지집합과 일반화 역행렬 조건을 별도로 다뤄야 합니다.**

독립 정규 구성은 [Pishro-Nik, §5.3.2](https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php)를 참고하세요. 위 행렬·유한 반례·손실 비교는 직접 계산했습니다. 공분산으로 독립을 추론하기 전에 [결합 표 단원](../joint-marginal-conditional/)과 비교하세요.

## 확인 문제와 해설

1. σx=1,σy=2,ρ=1,x=1이면? **조건부 평균 2, 분산 0입니다.**
2. 행렬식 0은 확률법칙이 없다는 뜻인가요? **아니요. 평면 밀도 공식이 적용되지 않는 것입니다.**
3. 그려진 점들은 표본인가요? **아니요. 단위원 경계의 변환입니다.**

추가 읽기: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
