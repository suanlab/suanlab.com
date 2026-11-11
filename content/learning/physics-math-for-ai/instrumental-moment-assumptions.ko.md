## 처치를 바꾸는 외부 변동

숨은 U가 처치 X와 결과 Y에 모두 영향을 주면 Y를 X에 회귀한 계수에는 선택 효과가 섞일 수 있다. 도구변수 Z는 별도의 처치 변동을 제공하지만 결과와의 관계에도 추가 가정이 필요하다. 처치와 상관이 있다는 관련성만으로 유효한 도구가 되지는 않는다. 여기서는 명시한 선형 모집단 모형을 계산한다. 상수 효과 식을 임의의 비선형·이질적 효과 모형으로 일반화하지 않는다.

## 구조식과 가정 명시

H,V,E_X,E_Y는 평균 0, 분산 1인 독립 innovation이다.

$$
Z=H,\quad U=\rho H+\sqrt{1-\rho^2}V,
$$

$$
X=\gamma Z+\alpha U+\sigma E_X,\qquad
Y=\beta X+\delta U+\kappa Z+E_Y.
$$

구조적 처치 효과는 beta다. 관련성은 Cov(Z,X)가 0이 아님을 요구한다. kappa는 Z에서 Y로 가는 직접 경로의 배제 위반이고 rho는 U를 통한 도구와 결과 교란항의 연관이다. rho=0이면 생성 규칙상 Z와 U가 독립이다. kappa=rho=0이고 E_Y도 독립이면 도구 모멘트에서 교란 항이 사라진다.

## 비율을 직접 유도하기

단위 분산과 innovation 독립성을 이용하면

$$
D=\operatorname{Cov}(Z,X)=\gamma+\alpha\rho,
\quad\operatorname{Cov}(Z,Y)=\beta D+\delta\rho+\kappa.
$$

D가 0이 아닐 때

$$
\beta_{IV}=\beta+\frac{\delta\rho+\kappa}{D}.
$$

OLS는 Cov(X,Y)/Var(X)를 사용한다. 처치 분산은 gamma²+alpha²+2 gamma alpha rho+sigma²이고, beta 외에 더해지는 분자는 delta(gamma rho+alpha)+kappa D다. 관측 표본이 커져도 이 가정 위반 항이 자동으로 사라지는 것은 아니다.

## 이질적 반응의 이진 도구가 식별하는 Local 효과

현재 선형 상수 효과 모형은 moment 가정이 맞으면 beta를 식별한다. 반응이 이질적이고 Z가 이진 무작위 도구이면 Wald 비율은 다른 estimand를 식별할 수 있다. Z가 모든 잠재 처치·결과와 독립이고, 배제 Y=Y(X), 일관성, 관련성, 단조성 X1≥X0을 가정하자. Always-taker와 never-taker는 Z가 바뀌어도 처치가 같고 complier만 바뀐다.

배정 arm 둘을 빼고 변하지 않은 반응을 상쇄하면

$$
E[X\mid Z=1]-E[X\mid Z=0]=P(C),\quad
E[Y\mid Z=1]-E[Y\mid Z=0]=P(C)E[Y_1-Y_0\mid C].
$$

따라서 비율은 complier LATE이며 전체 ATE와 자동으로 같지 않다. Defier가 있으면 반대 부호로 기여해 추가 제약 없이는 이 해석이 깨진다. 단조성은 개체 수준 제한이지 평균 first stage가 양수라는 조건만은 아니다.

| 독립 모집단의 Principal stratum | 질량 | 평균 결과 효과 |
|---|---:|---:|
| Never-taker | 3/10 | 1 |
| Complier | 1/2 | 2/5 |
| Always-taker | 1/5 | −1 |

```python
from fractions import Fraction as F
# x0,x1,y0,y1,mass; the complier stratum has two outcome response types.
atoms=[(0,0,0,1,F(3,10)),(0,1,0,1,F(1,5)),(0,1,0,0,F(3,10)),(1,1,1,0,F(1,5))]
assert sum(row[-1] for row in atoms)==1
means=[]
for z in (0,1):
    ex=ey=F(0)
    for x0,x1,y0,y1,mass in atoms:
        x=x0 if z==0 else x1;ex+=mass*x;ey+=mass*(y0 if x==0 else y1)
    means.append((ex,ey))
first=means[1][0]-means[0][0];reduced=means[1][1]-means[0][1]
ate=sum(mass*(y1-y0) for x0,x1,y0,y1,mass in atoms)
assert first==F(1,2) and reduced==F(1,5)
assert reduced/first==F(2,5) and ate==F(3,10)
print(first,reduced,reduced/first,ate)
```

First-stage 차이는 1/2, reduced form은 1/5이므로 Wald는 2/5지만 ATE는 3/10이다. 네 response atom으로 직접 구성한 결과이며 현재 연속 선형 잡음 조절기와는 별개다.

IV의 분자·분모를 저장된 binary64 계수의 정확한 dyadic 다항식으로 평가한 뒤 비율을 ties-to-even으로 binary64에 반올림한다. 작은 곱과 큰 항 상쇄 뒤의 잔여 효과를 모두 보존한다. 직접 나눈 rounded moment는 `expandedIV`, 보통 산술의 beta+(delta*rho+kappa)/D는 `factoredIV`에 남긴다. 수학적으로 0인 분모나 표현 불가능한 최종 비율은 계속 미정이다. 표의 반올림 moment가 0이어도 저장 입력의 정확한 분모는 비영일 수 있다. 작은 다항식의 정확한 중간 산술이지 입력 자료 전체의 임의 정밀도나 추론 인증은 아니다. 이 수치 보완이 약한 도구의 표본 추론을 해결하거나 배제·독립 가정을 증명하는 것은 아니다.

## 손계산 예제

gamma=alpha=beta=delta=sigma=1, kappa=rho=0이면 Var(X)=3이다. IV는 1, OLS는 1+1/3이다. 같은 교란 모집단을 보지만 유효한 도구 모멘트만 U의 기여를 제거한다.

gamma=0.1, kappa=0.1로 바꾸고 rho=0을 유지하면 IV=1+0.1/0.1=2다. 처치 관련성이 있어도 작은 배제 위반이 큰 오차를 만든다. 반대로 gamma=alpha=delta=beta=1, kappa=0, rho=0.5이면 D=1.5이고 IV=4/3이다. 배제만 충족해도 도구와 교란항이 연관되어 있으면 충분하지 않다.

## 확인 문제와 풀이

gamma=rho=kappa=0, alpha=sigma=1이면 IV는 무엇인가? 구조식의 처치 효과도 사라지는가?

**풀이.** D=0이므로 도구가 처치 변동을 제공하지 않아 IV 비율은 미정이다. 선언한 결과식의 beta는 여전히 처치 효과를 정의한다. 이 모멘트로 식별하지 못한 것이지 효과가 0이 된 것은 아니다. Var(X)>0이므로 OLS는 정의될 수 있다.

## 민감도 그림 읽기

가로축은 다른 계수를 유지하고 kappa를 바꾼다. 파랑은 실제 모집단 IV, 주황은 참 beta다. 그림은 [-5,5] 범위만 보여주며 벗어난 파란 점은 생략 개수로 알린다. 비율을 5로 잘라 바꾸는 계산이 아니다. 분모가 0이거나 기계 수로 표현할 수 없는 비율은 —로 표시한다. 표의 모멘트와 분모도 함께 확인하자.

## 연구와 연결하기

모집단 비율이 beta를 식별해도 추정 모멘트의 유한 표본 비율이 불편하다는 뜻은 아니다. 약한 도구는 표본 변동과 추론에도 문제를 만든다. 실제 연구에서는 도구 배정, 배제 경로, 모집단, 효과 이질성, 불확실성을 설명해야 한다. 예측 모형의 도구 관련성만으로 이 근거를 대신할 수 없다.

## 원전

Hernán·Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook), 16장. 공분산 모형과 그림, 민감도 예제는 직접 구성했다.
