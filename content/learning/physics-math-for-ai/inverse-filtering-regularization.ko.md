## 전방 식을 안다고 입력을 복원할 수 있지는 않습니다

역문제는 관측에서 원래 입력을 찾습니다. 이 실험은 길이 8의 주기 convolution, h=[1−r,r], 교대 관측 잡음을 사용합니다. 참조 진실은 저주파 cosine과 Nyquist 성분을 섞습니다.

$$
x[n]=\cos(2\pi n/8)+\tfrac12(-1)^n,\qquad y=h*_8x+\eta(-1)^n.
$$

주파수에서는 Y=HX+E입니다. 모든 H bin이 0이 아닐 때만 직접 역에서 H로 나눌 수 있습니다. 작은 비영 값은 잡음을 증폭하고, 정확한 영점은 해당 성분의 식별 가능성을 없앱니다. 두 문제를 구별해야 합니다.

## 정확한 영점에서는 역이 없습니다

r=0.5이면 H[4]=1−2r=0입니다. 참조 진실의 교대 성분 크기를 어떻게 바꿔도 무잡음 관측은 같습니다. 직접 역이 정의되지 않으므로 화면은 복원 곡선이나 복원 MSE를 만들어 표시하지 않습니다. reciprocal gain의 무한대는 역의 실패를 나타내며 유한한 복원값이 아닙니다.

정규화 0의 최소 노름 pseudoinverse는 사라진 계수를 0으로 택합니다. 보이는 cosine은 복구하지만 참조 교대 진폭 0.5는 관측에서 추론할 수 없습니다. cosine을 정확히 복구해도 복원 MSE는 0.5²=0.25입니다. 최소 노름 선택은 추가 구조이지 진실을 유일하게 되찾았다는 증거가 아닙니다.

## 안정화는 적합도와 크기를 절충합니다

양수 λ에서는 Tikhonov 추정값을 씁니다.

$$
\widehat X[k]=\frac{\overline{H[k]}Y[k]}{|H[k]|^2+\lambda}.
$$

현재 DFT 관례에서 이는 Σₙ|(h*_8z)[n]−y[n]|²+λΣₙ|z[n]|²를 최소화합니다. Parseval의 공통 배율은 상쇄됩니다. λ는 불안정한 gain을 줄이지만 식별 가능한 성분도 축소합니다. r=0.5, 잡음 0.05, λ=0.1이면 MSE는 약 0.2555입니다. 잃어버린 성분이 여전히 오차의 큰 부분입니다. 파랑은 참조 진실, 주황은 복원이며 표에는 blur와 잡음이 섞인 관측도 따로 나옵니다.

## 영점에 가까운 경우도 위험합니다

r=0.49, λ=0, 직접 역을 선택하세요. Nyquist gain은 0.02이므로 관측 잡음 0.05가 입력 오차 진폭 2.5로 증폭됩니다. MSE는 2.5²=6.25입니다. 관측 잔차가 아주 작아도 복원이 나쁠 수 있습니다. 잡음까지 정확히 맞추는 일과 원인을 복구하는 일은 다릅니다.

## 최소해와 공통 gain 경계 유도하기

지정한 주기 경계에서 DFT는 이차 목적함수를 (1/N)Σ_k(|H_k Z_k−Y_k|²+λ|Z_k|²)로 바꿉니다. 각 복소 스칼라를 최소화하면 앞의 켤레 필터를 얻습니다. λ>0에서는 Hessian이 양정이어서 최소해가 유일합니다. λ=0,H_k=0이면 그 좌표를 관측하지 못하며 최소 노름 관례가 Z_k=0을 선택합니다.

a=|H_k|,λ>0에서 (a−√λ)²≥0이므로

$$
\frac{a}{a^2+\lambda}\le\frac1{2\sqrt\lambda}.
$$

양의 λ가 영점 근처를 포함한 모든 복원 gain을 제한합니다. 잃어버린 참 입력 성분을 복구한 것은 아닙니다.

## 두 모드 예제에서 편향과 잡음을 정확히 분리하기

d=1−2r,q=|H(π/4)|²라 둡니다. 참조의 cosine과 교대 벡터는 직교하며 평균 제곱 노름은 각각 1/2와 1입니다. λ>0에서 cosine 오차 진폭은 −λ/(q+λ), 교대 오차 진폭은 (dη−λ/2)/(d²+λ)이므로

$$
\operatorname{MSE}=\frac12\left(\frac{\lambda}{q+\lambda}\right)^2+
\left(\frac{d\eta-\lambda/2}{d^2+\lambda}\right)^2.
$$

d≠0,λ=0의 직접 역 오차는 η²/d²이며 정확한 모형의 관측 잔차는 0입니다. d=0,λ=0이면 pseudoinverse 관례를 써서 MSE=1/4, 잔차 MSE=η²입니다. 위 분수를 0/0으로 계산하지 않습니다.

| λ, r=.49, η=.05 | 복원 MSE | 관측 잔차 MSE |
|---|---|---|
| 0 | 6.25000000 | 0 |
| .01 | .14799603 | .00338563 |
| .1 | .24368894 | .00826477 |
| 1 | .39432516 | .12781750 |

코드는 80자리 Decimal로 8차원 시간 normal equation을 독립적으로 풀고 얻은 MSE를 모드 공식과 비교합니다. 처음에는 정규화로 복원이 좋아지며 잔차는 커지고, 너무 강한 shrinkage에서는 복원 오차가 다시 커집니다.

실습 오차 수치는 반올림된 FFT 복원값의 차가 아니라 해석적인 모형 진폭으로 계산합니다. r=0,η=10⁻¹⁰⁰의 직접 MSE는 10⁻²⁰⁰입니다. 약 10⁻³²의 FFT 반올림 잔차를 이 모형 오차로 부르지 않습니다. binary64 범위 아래의 양의 오차는 —·수치 범위 안내로 표시하고, 정확한 무잡음 직접 역의 0은 유지합니다. FFT 곡선과 표는 유한 정밀도 복원이므로 그 값들의 뺄셈 오차가 해석적 수치와 bit 단위로 같을 필요는 없습니다.

**확인 문제:** 관측 잔차 0이면 참조 입력이 맞다고 인증하나요? **해설:** 아닙니다. λ=0의 잡음 역은 관측을 정확히 맞추면서 교대 잡음을 증폭합니다.

```python
from decimal import Decimal as D, localcontext

def solve(A,b):
    a=[list(row)+[v] for row,v in zip(A,b)]
    for j in range(len(a)):
        pivot=max(range(j,len(a)),key=lambda i:abs(a[i][j]))
        a[j],a[pivot]=a[pivot],a[j]
        value=a[j][j]
        a[j]=[v/value for v in a[j]]
        for i in range(len(a)):
            if i!=j:
                scale=a[i][j]
                a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    return [row[-1] for row in a]

with localcontext() as ctx:
    ctx.prec=80
    r,noise=D('.49'),D('.05')
    q=D(2).sqrt()/2
    truth=[1,q,0,-q,-1,-q,0,q]
    truth=[D(v)+D('.5')*(-1)**n for n,v in enumerate(truth)]
    H=[[((1-r) if i==j else D(0))+(r if i==(j+1)%8 else D(0)) for j in range(8)] for i in range(8)]
    observed=[sum(a*b for a,b in zip(row,truth))+noise*(-1)**i for i,row in enumerate(H)]
    for lam in [D(0),D('.01'),D('.1'),D(1)]:
        A=[[sum(H[k][i]*H[k][j] for k in range(8))+(lam if i==j else 0) for j in range(8)] for i in range(8)]
        rhs=[sum(H[k][i]*observed[k] for k in range(8)) for i in range(8)]
        estimate=solve(A,rhs)
        mse=sum((a-b)**2 for a,b in zip(estimate,truth))/8
        residual=sum((sum(a*b for a,b in zip(row,estimate))-y)**2 for row,y in zip(H,observed))/8
        h4=1-2*r
        h1_squared=(1-r)**2+r*r+2*r*(1-r)*q
        analytic=D('.5')*(lam/(h1_squared+lam))**2+((h4*noise-D('.5')*lam)/(h4*h4+lam))**2
        assert abs(mse-analytic)<D('1e-65')
        print(str(lam),round(mse,8),round(residual,8))
```

## 직접 풀기와 연구 연결

r=0, 잡음 0에서 직접 역은 항등이며 MSE는 0입니다. 이어 r=0.5로 돌려 직접 역·pseudoinverse·양수 λ 추정값을 비교하세요. 어느 경우가 정의되지 않고, 어느 경우가 가능한 입력 중 하나를 선택하며, 어느 경우가 shrinkage를 도입하는지 설명해 보세요.

영상 복원·센서 재구성·학습 역문제에는 경계·잡음·prior를 명시해야 합니다. 정규화는 해를 선택하거나 안정화할 수 있지만 관측의 식별 가능성을 새로 만들지는 않습니다. 참조 진실이 있을 때 복원 오차와 관측 잔차를 분리해 보고하세요.

추가 읽기: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
