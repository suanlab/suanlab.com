## 세 오차를 어떤 목적함수에서 정의하는가

근사·추정·최적화 오차라는 이름만 나열하면 모든 항이 자동으로 양수라고 생각하기 쉽습니다. 실제로는 어떤 oracle과 어떤 목적함수를 비교하는지 먼저 정해야 합니다. 이 단원은 x가 -1,0,1에 균일하고 참 함수가 f(x)=x+c x²인 regression 세계를 사용합니다. Label은 f(x)에 독립적인 ±sigma 잡음을 더합니다. 손실은 제곱 오차의 절반입니다.

가설은 w 하나가 곱해진 상수 feature 또는 x feature입니다. 실제 표본의 이차 손실에서 GD를 실행하고, 같은 표본의 정확한 ERM과 population class oracle을 별도로 구합니다. 파랑은 population risk, 주황은 empirical risk이며 두 곡선은 서로 다른 평균입니다.

## 세 기준점을 연결한 정확한 항등식

Bayes risk는 sigma²/2입니다. Class oracle w_H는 class 안에서 population risk를 최소화하고, w_S는 sample risk를 최소화합니다. 현재 GD 값 w_k에 대해

$$
R(w_k)-R_{Bayes}=[R(w_H)-R_{Bayes}]
+[R(w_S)-R(w_H)]+[R(w_k)-R(w_S)].
$$

첫째는 근사, 둘째는 현재 표본이 만든 추정 효과입니다. 마지막은 population에서 평가한 최적화 효과로 음수일 수 있습니다. 반면 같은 empirical 목적함수의 간극 Rhat(w_k)-Rhat(w_S)는 비음수입니다. 이를 마지막 항과 바꾸면 위 등식은 일반적으로 성립하지 않습니다.

## 실제 이차 목적함수와 GD

Feature를 phi라 두면 표본 곡률 A=mean(phi²), 선형 항 B=mean(phi y)입니다.

$$
w_{k+1}=w_k-\eta(Aw_k-B),\qquad
w_S=B/A\quad(A>0).
$$

모든 표본의 feature가 0이면 A=B=0으로 empirical 목적함수가 평평합니다. 현재 규칙은 w_S=0을 선택하고 GD도 0에 남습니다. Population에서 feature가 정보를 갖는다는 사실과 현재 표본이 그 정보를 포함한다는 사실은 다릅니다. 이 경우를 작은 ridge로 몰래 바꾸지 않습니다.

## Early Stopping의 Spectral Filter와 Bias 비용

별도의 intercept 문제에서 Y=mu+sigma*sign, sign은 독립 공정 부호라고 하자. Half-squared empirical loss를 w0=0에서 0<eta<1로 최적화하면 recurrence를 풀어 w_k=g_k*Ybar, g_k=1−(1−eta)^k를 얻는다. Population oracle은 mu이고 중심 잡음의 독립성으로 Bayes 위험 위의 기대 초과는

$$
E[R(w_k)-R_{Bayes}]=\frac12(1-g_k)^2\mu^2+\frac12g_k^2\frac{\sigma^2}{N}.
$$

첫째는 shrinkage bias, 둘째는 추정 분산이다. G를 자유롭게 고를 수 있다고 미분하면 분모가 양수일 때 g*=mu²/(mu²+sigma²/N)이다. 정수 iteration이 이 값을 정확히 실현할 필요는 없다. 여기서 mu·잡음 분산은 oracle 값이므로 데이터만으로 정한 중단 규칙이 아니다.

Mu=1/4, sigma=1, N=4, eta=1/5에서는 한 step이 기대 초과를 줄이지만 ERM까지 가면 더 커진다. 유리한 seed 하나를 고른 결과가 아니라 열여섯 잡음열 전체에 대한 기대다.

| Iteration | Bayes 위의 기대 초과 |
|---|---:|
| 0 | 1/32 |
| 1 | 1/40 |
| 극한 | 1/8 |

```python
from itertools import product
from fractions import Fraction as F
N=4;mu=F(1,4);eta=F(1,5)
for k in (0,1,8):
    gain=1-(1-eta)**k
    exact=sum(((gain*(mu+F(sum(signs),N))-mu)**2/2 for signs in product((-1,1),repeat=N)),F(0))/2**N
    formula=((1-gain)**2*mu**2+gain**2/N)/2
    assert exact==formula
    print(k,exact,float(exact))
assert ((1-(1-eta))**2/N+(1-eta)**2*mu**2)/2==F(1,40)
assert F(1,2*N)==F(1,8)
```

현재 실습의 population 효과는 중심화된 항등식으로 계산한다. 근사 항은 상수 feature에서 1/3+c²/9, x feature에서 c²/3이고 추정 항은 A_pop*(w_S−w_H)²/2이다. Population 최적화 효과는 두 제곱 차이를 인수분해해 signed 값을 유지한다. 실제 표본·ERM 동률 규칙·GD recurrence는 보존한다. 두 표시 위험이 같은 잡음 바닥으로 반올림돼도 양의 3.3×10^(−17) 근사 오차를 보존할 수 있다. 직접 뺀 값은 `expandedEffects`에 남긴다. 이 해석적 진단은 반올림한 표시 문자열을 뺀 실측값이 아니다.

## 손계산과 직접 풀기

c=0, sigma=1, N=2, seed 9이면 실제 표본은 (-1,-2),(1,2)입니다. Linear feature의 ERM은 w=2입니다. Population risk는 (w-1)²/3+1/2이므로 class oracle은 1이고 ERM의 추정 효과는 1/3입니다. 0에서 eta=1/2로 한 번 GD하면 w=1에 도착해 population 최적화 효과가 -1/3입니다. Empirical 간극은 여전히 1/2입니다.

직접 풀기: c=1에서 constant class의 approximation risk를 구하세요. 풀이: 참 값은 [0,0,2], 평균은 2/3입니다. 최적 상수의 평균 제곱 오차 절반은 4/9입니다. Linear feature의 oracle은 w=1이고 근사 오차는 1/3입니다. 반복 횟수만 늘려 class의 표현 제한을 없앨 수 없습니다.

## Early Stopping을 해석하기

현재 예제는 early stopping이 더 나은 population 위험을 줄 수 있음을 보여 주지만 항상 그렇다는 정리는 아닙니다. 데이터로 최적화한 목적함수, 평가 분포, class와 초기화·학습률을 함께 보고하세요. 실제 연구에서 세 효과를 각각 식별하려면 알려진 oracle이나 독립 추정·실험 설계가 필요합니다. 이 유한 세계는 그 정의를 검증하기 위한 투명한 기준입니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
