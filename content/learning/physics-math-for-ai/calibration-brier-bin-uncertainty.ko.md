## 정확도·확률·불확실성의 서로 다른 뜻

예측 p=.9는 모든 label을 맞힌다는 뜻이 아닙니다. 같은 조건에서 약 90%의 label이 1이라는 확률과 실제 한 label의 성공은 다릅니다. 이 단원은 같은 비중의 세 stratum, 알려진 참 Bernoulli 확률 q, 예측 [.15,.5,.85]를 사용합니다. Temperature는 logit을 나누어 확률을 바꾸지만 양수이므로 현재 hard prediction의 .5 경계는 유지됩니다.

파랑은 예측 p 위치에서의 참 q이고 주황은 calibration 대각선입니다. 실제 seed 표본의 label 빈도는 표에 따로 표시합니다. 알려진 q는 교육용 oracle이며 실제 데이터의 참 확률을 자동으로 관측할 수 있다는 뜻은 아닙니다.

## Brier Score의 정확한 분해

Bernoulli Y에 대해

$$
E[(p-Y)^2]=(p-q)^2+q(1-q).
$$

첫 항은 확률 예측 오차이고 둘째는 label의 고유 분산입니다. 이 단원은 stratum별 조건부 분해를 평균냅니다. 다른 binning이나 marginal variance를 쓰는 전체 Murphy 분해의 항을 이름만 바꾼 것이 아닙니다. P=q여도 q가 0 또는 1이 아니면 Brier score는 0이 아닙니다.

## 실제 Bin 빈도의 불확실성

Stratum마다 N개의 실제 label을 생성합니다. 그 평균 qhat에 대해

$$
E[\widehat q]=q,\qquad\operatorname{Var}(\widehat q)=\frac{q(1-q)}N.
$$

예를 들어 q=.5, N=20이면 분산 .0125, 표준편차 약 .1118입니다. 참 calibration이 정확해도 관측 빈도가 p와 다를 수 있습니다. 이 분산은 모형의 이론적 반복 표본 변동이고 개별 label의 확실성이나 자동적인 parameter credible interval이 아닙니다.

## Bin 합치기가 오류를 숨기는 예

현재 ECE는 bin별 |mean prediction-mean true/observed probability|의 가중 평균입니다. Merge 모드는 진단 bin만 하나로 합치며 실제 예측이나 Brier score는 바꾸지 않습니다. 참 q를 [.85,.5,.15]로 뒤집으면 각 끝 stratum은 .7만큼 틀리지만 전체 평균 p와 q는 모두 .5입니다.

$$
\mathrm{ECE}_{separate}=\frac{.7+0+.7}{3}\approx .4667,\qquad
\mathrm{ECE}_{merged}=0.
$$

Perfect aggregate 점 하나가 stratum 내부의 calibration을 증명하지 못합니다. 평균을 낸 뒤 절댓값을 취하는 것과 절댓값을 평균내는 것은 다릅니다. 이 상황의 Brier는 약 .495로 여전히 나쁩니다.

## Recalibration은 직교 사영이고 Bin 병합은 진단이다

보고한 score를 P, C(P)=E[Y|P]라 하자. Y−P=(Y−C)+(C−P)를 제곱하고 교차항을 P에 조건부로 평균내면 E[Y−C|P]=0이므로

$$
E[(Y-P)^2]=E[(Y-C(P))^2]+E[(C(P)-P)^2].
$$

Score를 C로 바꾸면 기대 Brier가 정확히 두 번째 항만큼 개선된다. Tower property로 E[Y|C]=C도 성립한다. 진짜 조건부 확률을 알아야 하는 oracle 결과이며 같은 label에서 C를 추정하고 평가하면 과적합할 수 있다.

상수 E[Y]도 calibrated이므로 calibration만으로 resolution·유용한 판별력을 증명할 수 없다. 뒤집은 세 stratum 예제에서 실제 recalibration은 score를 (.85,.5,.15)로 바꾼다. 반면 진단 bin 병합은 원래 (.15,.5,.85) 예측을 그대로 둔다.

| 뒤집은 예제의 절차 | 진단 ECE | 기대 Brier |
|---|---:|---:|
| 원래 Score와 별도 Bin | 7/15 | 99/200 |
| 같은 Score와 합친 진단 | 0 | 99/200 |
| Score를 참 C(P)로 교체 | 0 | 101/600 |

![실제로 다른 세 score 정책의 기대 Brier를 고유 label 분산과 확률 예측 오차로 구분한 비교 그림.](/assets/learning/theory-calibration-ko.svg)

```python
from fractions import Fraction as F
p=[F(3,20),F(1,2),F(17,20)];q=list(reversed(p))
def brier(scores):return sum((b*(1-a)**2+(1-b)*a*a for a,b in zip(scores,q)),F(0))/3
intrinsic=sum((b*(1-b) for b in q),F(0))/3
projection=sum(((a-b)**2 for a,b in zip(p,q)),F(0))/3
assert brier(p)==intrinsic+projection==F(99,200)
assert brier(q)==intrinsic==F(101,600)
assert sum(p)==sum(q) and sum(abs(a-b) for a,b in zip(p,q))/3==F(7,15)
for scores in (p,[F(1,2)]*3,q):print(scores,brier(scores))
```

그림은 상수 1/2 predictor를 포함한 실제 score 정책을 비교한다. 가운데 막대는 진단 bin만 합친 결과가 아니다. 검증 label과 독립적으로 고정한 B개 bin, 각각 독립 Bernoulli 결과 N개에서는 sqrt(log(2B/delta)/(2N))의 union Hoeffding 반경으로 모든 bin 평균을 동시에 제어한다. Bin count가 무작위이면 조건부 또는 count를 고려한 경계가 필요하며 adaptive temperature·bin 선택은 별도 평가가 필요하다. 기존 고정 stratum 분산 자체가 동시 confidence interval은 아니다.

## 손계산과 직접 풀기

q=p=[.15,.5,.85]이면 확률 오차 성분은 0이고 평균 intrinsic variance는 (.1275+.25+.1275)/3≈.16833입니다. Hard accuracy는 (.85+.5+.85)/3≈.73333입니다. Perfect calibration과 완벽한 정확도는 같은 조건이 아닙니다.

직접 풀기: temperature를 .5로 줄이면 hard prediction과 Brier는 어떻게 되나요? 풀이: 예측은 더 극단적으로 변하지만 .5의 양쪽 방향은 같아 hard accuracy는 그대로입니다. 원래 q와 같던 확률에서 벗어나 (p-q)² 항이 양수가 되어 기대 Brier가 증가합니다. 실제 finite sample의 empirical score는 표본 변동도 포함합니다.

## AI 확률 평가의 보고 기준

정확도, proper score, bin 정의, 표본 수와 uncertainty를 함께 제시하세요. Bin이나 temperature를 같은 평가 데이터에서 반복 선택하면 선택 절차도 별도로 평가해야 합니다. 알려진 q를 사용한 현재 분해를 미지의 실제 데이터에 그대로 oracle처럼 적용하지 마세요.

참고: [MIT 통계적 학습 자료](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/), [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
