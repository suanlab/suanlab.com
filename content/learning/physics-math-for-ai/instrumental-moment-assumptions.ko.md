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
