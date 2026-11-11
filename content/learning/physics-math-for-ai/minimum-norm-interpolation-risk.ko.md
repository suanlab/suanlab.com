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

## 조건부 곡선과 Design 평균을 구별하기

정확한 pseudoinverse에서 P=X⁺X, y=X*theta+noise라 하자. 학습 잡음은 독립·중심이며 공분산 sigma²I이다. Bias는 design nullspace, X⁺noise는 row space에 있어 잡음을 평균내기 전에도 교차항이 0이다. Identity test 공분산이면

$$
E_{noise}[R(\widehat w)\mid X]=\|(I-P)\theta\|^2+\sigma^2\|X^+\|_F^2+\sigma^2.
$$

따라서 작은 retained singular 값은 기대에서 sigma²/s_j²를 더한다. 특정 잡음 실현이 그 방향과 약하게 정렬될 수는 있다. 마지막 항은 독립 test noise다. 이 조건부 잡음 평균과 특정 design·실제 학습 잡음 둘 다에 조건부인 화면의 곡선은 다르다.

독립 예제로 표준 Gaussian n-by-p design, 정확한 비절단 least squares, norm 1의 well-specified theta, sigma>0을 가정하자. Inverse-Wishart moment로 학습 잡음 항은 p<n−1에서 sigma²*p/(n−p−1), p>n+1에서 sigma²*n/(p−n−1)이다. 회전 대칭에 의해 과매개변수 bias는 1−n/p이다. 이 moment는 명시한 분모가 양수여야 한다. P=n−1,n,n+1에서는 잡음 위험의 기대가 무한이지만 개별 full-rank 실현은 거의 확실히 유한하다. 유한 그림 하나가 무한 기대를 반박하지 못한다.

Moment 식은 짧은 기하적 증명도 갖는다. N≥p인 Gaussian design에서 Schur complement로 (XᵀX)^(−1)의 대각 원소는 다른 열을 사영해 없앤 한 열의 길이 제곱의 역수이다. 남은 공간의 차원은 n−p+1이고 독립 Gaussian 좌표를 갖는다. 자유도 nu인 chi-square V의 gamma 밀도를 적분하면 nu>2에서 E[1/V]=1/(nu−2), nu≤2에서는 발산한다. 대각 p개를 합하면 저매개변수 항이다. Xᵀ에 같은 논증을 쓰면 과매개변수 항을 얻는다. 회전 불변성과 trace(P)=n으로 E[P]=(n/p)I를 얻어 bias도 증명한다. 이는 정확한 inverse 대수의 분포 평균이며 수치 해법 tolerance가 아니다.

![Gaussian design의 기대 test 위험에서 잡음 기대가 발산하는 차원을 표시하고 개별 학습 design과 구별하는 그림.](/assets/learning/theory-interpolation-ko.svg)

| n=6, sigma=1/2, 독립 예제 p | 기대 전체 Test MSE |
|---|---:|
| 4 | 5/4 |
| 6 | 무한 |
| 8 | 2 |

```python
from itertools import product
from fractions import Fraction as F
sigma=F(1,2);values=[]
for signs in product((-1,1),repeat=2):
    w=(1+sigma*signs[0]/2,1+sigma*signs[1],F(0))
    values.append(sum((x-1)**2 for x in w)+sigma*sigma)
expected=sum(values)/4
assert expected==1+sigma*sigma*(F(1,4)+1)+sigma*sigma==F(25,16)
print('fixed diagonal design:',expected)
n=6
for p in (4,6,8):
    total=(sigma*sigma*(1+F(p,n-p-1)) if p<n-1 else sigma*sigma+1-F(n,p)+sigma*sigma*F(n,p-n-1) if p>n+1 else None)
    print('Gaussian design expectation:',p,total)
```

유리수 대각 예제로 조건부 잡음 분해를 별도 검산한다. Gaussian 기대 그림은 수치 singular cutoff 없이 p=16까지 확장한다. 화면의 nested design은 최대 12 feature와 cutoff 1e-10을 쓰므로 다른 질문에 답한다. [Ridgeless regression 분석](https://arxiv.org/abs/1903.08560)은 더 넓은 연구 배경이다. 이 유한 계산을 그 논문의 전체 실험이나 점근 결과 재현이라고 부르지 않는다.

## 손계산과 직접 풀기

별도의 가장 작은 예로 X=[1,1], y=1을 생각하세요. 최소 norm 해는 (.5,.5)이고 training loss는 0입니다. 참 coefficient가 (1,0)이고 test 공분산이 I라면 무잡음 test MSE는 .5입니다. Training을 맞히는 참 coefficient의 test MSE 0과 다릅니다. 최소 norm 선택이 모든 진짜 모형에 최적인 것은 아닙니다.

직접 풀기: 이 예에 독립 test label noise sigma=.2를 더하면 두 위험은 얼마인가요? 풀이: 최소 norm 해의 위험은 .5+.04=.54이고 참 coefficient는 .04입니다. Noise floor는 없앨 수 없는 공통 항이며 training interpolation 자체가 test risk를 0으로 만들지 못합니다.

## Double Descent를 관측하고 검증하기

Interpolation 근처의 작은 singular 값은 noise를 크게 증폭할 수 있고, 더 많은 feature에서 최소 norm이 달라지며 위험이 줄기도 합니다. 하지만 feature 분포·target·noise·regularization에 따라 모양은 바뀝니다. 현재 원고는 실제 계산한 유한 곡선만 제시합니다. 여러 seed·크기·rank와 조건수, 해법 tolerance를 함께 기록해야 일반성을 논할 수 있습니다.

독립 검산에서는 네 Moore–Penrose 항등식과 알려진 해, normal residual, nested column·noise 보존을 확인합니다. 한 그림이 기존 이론 전체를 반박한다거나 모든 과매개변수 모델의 일반화를 보장한다고 해석하지 마세요.

참고: [Belkin 등 원 논문](https://arxiv.org/abs/1812.11118), [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). 곡선과 예제는 독립 구현입니다.
