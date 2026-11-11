## 실제로 관측한 사례에 관한 질문

모집단 개입은 전체 모집단에서 배정식을 교체했을 때의 결과를 묻는다. 반사실은 특정 사실을 관측한 사례가 다른 행동 아래에서 어떻게 되었을지 물을 수 있다. 그 증거는 어떤 외생 상태가 가능한지 바꾼다. 따라서 새로운 사람을 무작위로 뽑아 개입을 반복하는 계산과 같을 필요가 없다.

앞 단원의 이진 SCM을 사용한다. 독립 외생 비트 U=(Z,E_X,E_Y)에서 X=Z XOR E_X를 생성한다. 결과식은 X XOR E_Y, Z XOR E_Y, X XOR Z XOR E_Y 중 하나다. 이 함수들은 같은 기저 상태가 대안 행동에 어떻게 반응할지를 지정한다. 관측 주변분포만 아는 것보다 더 많은 구조적 정보를 제공하는 가정이다.

## Abduction: 증거를 설명하는 상태 추론

사실 증거를 e={X=x_f,Y=y_f}라고 하자. 여덟 외생 상태에서 자연 배정식을 실행하고 증거와 비교한다. 양립하는 상태의 사전 질량을 남겨 정규화한다.

$$
P(u\mid e)=\frac{P(u)\,1\{X(u)=x_f,Y(u)=y_f\}}{P(e)}.
$$

이는 관측한 사례의 외생 상태에 대한 불확실성이다. 사전분포에서 새 상태를 추출하는 과정이 아니다. 표에는 사전 질량, 사후 질량, 사실 X·Y를 모두 표시한다. 사전 확률 0인 행은 남아 있지만 증거에 기여하지 않는다. P(e)=0이면 이 모형에서 사후분포를 정의할 수 없다. 이때 임의의 분포를 만들지 않고 —로 표시한다.

## Action과 Prediction: 같은 U 공유

행동 a를 적용할 때 X 식만 X_a=a로 바꾸고, 추론한 U와 결과 함수는 유지한다. 각 상태에서 Y_a(u)를 다시 계산한 뒤 평균한다.

$$
E[Y_a\mid e]=\sum_u Y_a(u)P(u\mid e).
$$

쌍둥이 그림의 사실 세계와 행동 세계는 하나의 잠재 U를 공유한다. 사실 X·Y의 사각형은 증거이고 X_a는 지정한 행동이다. E_Y에 관한 정보를 얻은 뒤 새 E_Y를 뽑으면 다른 질문이 된다. 같은 상태를 유지하므로 일관성도 성립한다. 증거와 양립하는 모든 상태에서 사실 처치를 다시 적용하면 사실 결과가 나온다.

## 무작위 배정도 모든 Cross-world 결합을 식별하지는 못한다

무작위 처치가 (Y0,Y1)과 독립이면 trial의 두 arm에서 p0=P(Y0=1), p1=P(Y1=1)을 식별한다. 그러나 한 개체의 두 결과를 동시에 보지는 못한다. T=P(Y0=1,Y1=1)라 쓰고 네 반응 유형 질량의 비음수성을 요구하면 날카로운 Fréchet 구간은

$$
\max(0,p_0+p_1-1)\le t\le\min(p_0,p_1),\quad
P(Y_1=1,Y_0=0)=p_1-t.
$$

P1>0이면 무작위 배정으로 P(Y0=0|X=1,Y=1)=(p1−t)/p1도 얻는다. 이 necessity 확률은 명시한 무작위 체제에서 실제 처치받고 성공한 집단에 관한 것이다. 관측 처치였다면 조건에서 X를 빼기 전에 추가 가정이 필요하다.

P0=3/10,p1=3/5에서 ATE는 모든 결합에서 3/10이지만 necessity는 1/2에서 1까지 가능하다. 코드는 세 결합 법칙을 직접 구성하고 무작위 factual (X,Y) 표가 같음을 검사한다.

| Cross-world t | Benefit 확률 | 처치 성공 집단의 Necessity |
|---|---:|---:|
| 0 | 3/5 | 1 |
| 3/20 | 9/20 | 3/4 |
| 3/10 | 3/10 | 1/2 |

![같은 무작위 관측과 평균 처치 효과를 유지하면서 가능한 cross-world 결합에 따른 benefit·harm 확률을 비교한 그림.](/assets/learning/causal-counterfactual-coupling-ko.svg)

```python
from fractions import Fraction as F
p0,p1=F(3,10),F(3,5);baseline=None
for t in (F(0),F(3,20),F(3,10)):
    response={(0,0):1-p0-p1+t,(0,1):p1-t,(1,0):p0-t,(1,1):t}
    assert min(response.values())>=0 and sum(response.values())==1
    factual={(x,y):sum(mass/2 for (y0,y1),mass in response.items() if (y0 if x==0 else y1)==y) for x in (0,1) for y in (0,1)}
    if baseline is None:baseline=factual
    assert factual==baseline
    benefit=response[0,1];harm=response[1,0]
    assert benefit-harm==p1-p0
    print(t,benefit,benefit/p1)
```

단조성 Y1≥Y0를 추가하면 harm을 제거해 이 예에서는 t=p0로 고정한다. 주변 위험의 순서만으로 단조성이 입증되지는 않는다. 화면의 XOR SCM은 같은 잡음을 공유해 특정 결합 하나를 제공한다. 그림은 그와 달리 trial 주변분포를 유지하면서 결합을 바꾼다.

매우 희귀한 factual 증거에서는 `logEvidence`를 유지하고 호환 log mass로 posterior를 정규화한다. Raw `evidence`가 0으로 underflow해도 사후 Support가 양수일 수 있어 ‘posterior defined’는 1로 남는다. 실제로 불가능한 증거는 유한 log mass가 없어 여전히 사후분포가 없다. 원래 atom 확률도 비교용으로 유지하며 임의 floor나 균등 상태로 바꾸지 않는다.

## 끝까지 계산하는 예제

Chain 결과식 Y=X XOR E_Y를 고르고 pi=0.5, 배정 잡음 0.1, 결과 잡음 0.2로 두자. X_f=1,Y_f=1을 관측하면 E_Y=0을 알 수 있다. 증거 확률은 0.5×0.8=0.4다.

양의 사후 질량을 가진 상태는 둘이다. (Z,E_X,E_Y)=(0,1,0)의 사후 질량은 0.1, (1,0,0)은 0.9다. X_a=0으로 바꾸면 두 상태 모두 Y_a=0이다.

$$
E[Y_0\mid X=1,Y=1]=0,
\qquad E[Y\mid do(X=0)]=0.2.
$$

뒤의 모집단 개입은 전체의 결과 잡음 비율 20%를 유지한다. 앞의 반사실은 증거가 결과 잡음 비트를 알려준 집단에 관한 값이다. 두 평균의 차이는 계산 오류나 모순이 아니다.

## 기제를 바꾸면 답도 바뀐다

Fork 결과식 Y=Z XOR E_Y에서는 같은 외생 상태의 X를 바꾸어도 Y가 변하지 않는다. 사실 Y=1을 관측했다면 양립하는 모든 상태에서 어느 행동을 적용해도 Y_a=1이다. 모집단 개입 평균은 여전히 0.5일 수 있다. 실제 잠재 상태가 무엇인지 단정하지 않으면서도 동일 상태의 반응과 모집단 구성을 구분하는 예다.

## 확인 문제와 풀이

Y=X XOR Z XOR E_Y를 고르고 두 잡음 확률은 0, pi=0.5로 두자. X_f=1,Y_f=0을 관측했다. 행동 a=0의 결과는 무엇인가?

**풀이.** 자연 상태에서 X=Z이므로 증거가 Z=1을 알려준다. Z=1,E_Y=0을 유지하고 X_a=0으로 바꾸면 Y_a=1이다. a=1을 다시 적용하면 Y_a=0으로 사실 결과와 일치한다. 증거 자체가 확률 0인 사건이었다면 이 SCM의 조건부 계산으로 어느 답도 만들 수 없다.

## 적합한 예측기만으로 부족한 정보

지도학습기가 P(Y|X)를 정확히 추정하더라도 잠재 결과들을 연결하는 결합 방식까지 정해지는 것은 아니다. 반사실 답에는 구조 함수, 잡음 분포, 두 세계에서 같은 잡음을 유지한다는 가정이 필요하다. AI 설명에서 입력 하나를 바꾸어 예측기를 다시 실행했다고 실제 사람이나 시스템의 인과 반사실이 되는 것은 아니다. 증거, 행동, 유지할 외생 가정, 결과 기제를 명시하자. 다음 단원에서는 같은 관측분포를 설명하는 여러 모형으로 비식별을 확인한다.

## 원전

- Judea Pearl, [Sufficient Causes: On Oxygen, Matches, and Fires](https://ftp.cs.ucla.edu/pub/stat_ser/r484.pdf), 2019: abduction–action–prediction의 구체적 적용.
- Alexander Balke·Judea Pearl, [Counterfactuals and Policy Analysis in Structural Models](https://arxiv.org/abs/1302.4929): 구조모형의 반사실 질의 평가.

이진 예제, 사후 질량, 쌍둥이 그림은 직접 구성했다.
