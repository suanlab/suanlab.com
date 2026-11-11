## 분포를 고르는 학습과 확률 보장의 가정

여섯 threshold 각각에 확률을 주는 posterior q를 생각합니다. Gibbs classifier는 예측할 때 이 분포에서 가설을 무작위로 골라 사용합니다. 그 위험은 E_q R(h)입니다. 가장 무거운 가설 하나를 쓰거나 majority vote를 하는 것과 같은 분류기가 아닙니다. Posterior라는 이름만으로 Bayesian 모형의 참 사후확률이 되는 것도 아닙니다.

이 단원은 IID 표본, [0,1]의 0-1 loss, 표본과 독립적으로 미리 고정한 양의 prior pi와 사전에 정한 delta를 가정합니다. Prior tilt를 데이터를 본 뒤 조정하면 숫자는 계산되더라도 아래 confidence 보장을 그대로 해석할 수 없습니다. 파랑은 실제 Gibbs posterior 질량, 주황은 prior 질량입니다.

## 실제 Posterior와 KL

표본 위험을 계산한 뒤 다음 분포를 만듭니다.

$$
q(h)=\frac{\pi(h)e^{-\beta\widehat R(h)}}{\sum_j\pi(j)e^{-\beta\widehat R(j)}},\qquad
\mathrm{KL}(q\Vert\pi)=\sum_hq(h)\log\frac{q(h)}{\pi(h)}.
$$

Beta=0이면 q=pi이고 KL=0입니다. 큰 beta는 낮은 empirical 위험에 질량을 모으지만 알려진 population 위험을 직접 최소화하는 식은 아닙니다. KL은 참 모델에 대한 거리라고 자동 해석하지 않고, 선언한 prior에서 얼마나 벗어났는지를 nats로 측정합니다.

## Bound를 세 단계로 유도하기

고정 h에 대해 Hoeffding의 moment 부등식은 E_S exp(lambda(R-Rhat))≤exp(lambda²/(8N))입니다. 이를 prior로 평균내고 Markov를 적용합니다. 미리 정한 lambda 집합 {1,2,4,8,16,32,64,128}에 union bound를 쓰면 확률 1-delta 이상으로 모든 그리드 값에서 동시에

$$
\log E_\pi e^{\lambda(R-\widehat R)}\le\frac{\lambda^2}{8N}+\log\frac8\delta.
$$

여기에 유한 change-of-measure 부등식 E_q f≤KL(q||pi)+log E_pi exp(f)를 적용하면 모든 q에 대해 동시에

$$
R(q)\le\widehat R(q)+\frac{\mathrm{KL}(q\Vert\pi)+\log(8/\delta)}\lambda+\frac\lambda{8N}.
$$

이 때문에 posterior를 데이터로 골라도 되며 미리 정한 그리드 안의 lambda 선택도 보정됩니다. 연속적인 모든 lambda를 공짜로 사후 최적화하거나 prior까지 자유롭게 데이터로 고르는 결론은 아닙니다.

## Gibbs 최적화와 Prior Family 선택 비용

Beta>0, 양의 고정 prior에서 log(q_beta/pi)=−beta*r−log Z를 KL(q||q_beta)에 넣고 정리하면

$$
E_q r+\frac{\mathrm{KL}(q\Vert\pi)}\beta=-\frac{\log Z}\beta+\frac{\mathrm{KL}(q\Vert q_\beta)}\beta.
$$

마지막 KL의 비음수성으로 Gibbs 분포가 분포 위의 regularized empirical 목적함수를 최소화함을 증명한다. 이것이 참 Bayesian 사후나 population 최소자를 식별한다는 뜻은 아니다. 코드는 최적점의 등식뿐 아니라 세 후보 분포에서 전체 항등식을 검사한다.

미리 선언한 유한 prior family도 비용을 지불하면 데이터를 본 뒤 고를 수 있다. Prior j에 고정 양의 질량 w_j를 주고 합을 1로 둔다. 각 prior/grid 쌍에 실패 확률 delta*w_j/8을 배정하면 기존 moment·union 논증의 log(8/delta)가 log(8/(delta*w_j))로 바뀐다. 한 사건에서 모든 posterior와 family/grid 선택을 제어한다. 관측 label을 보고 임의로 만들어 낸 prior까지 허용하는 결과는 아니다.

| 고정 Prior-family 비중 | 추가 분자 비용 |
|---|---:|
| 1/2 | log 2 |
| 1/4 | log 4 |
| 1/8 | log 8 |

```python
from math import exp,log
prior=[.5,.3,.2];risk=[0,.25,.75];beta=2
Z=sum(p*exp(-beta*r) for p,r in zip(prior,risk))
gibbs=[p*exp(-beta*r)/Z for p,r in zip(prior,risk)]
def kl(q,p):return sum(a*log(a/b) for a,b in zip(q,p))
for q in (prior,[1/3]*3,gibbs):
    objective=sum(a*r for a,r in zip(q,risk))+kl(q,prior)/beta
    certificate=-log(Z)/beta+kl(q,gibbs)/beta
    assert abs(objective-certificate)<1e-14
    assert objective>=-log(Z)/beta-1e-14
    print(objective,kl(q,gibbs)/beta)
```

현재 조절기는 이 family 확장이 아니라 선언한 prior 하나를 쓴다. 작은 tilt의 KL은 정규화 Gibbs log-ratio u와 convex generator exp(u)*u−expm1(u)로 계산한다. 급수의 시작은 u²/2+u³/3으로, 직접 합의 음수를 0으로 clip하는 대신 양의 이차 효과를 보존한다. 해석적 Gibbs KL은 표시 확률의 정밀도보다 작은 tilt도 구별한다. 기존 q*log(q/pi) 합은 `expandedKL`에 남긴다. 둘 다 binary64 계산이며 interval 인증이나 임의 정밀도 결과는 아니다.

## 손계산과 직접 풀기

N=32, delta=.05, lambda=32, beta=0이면 penalty는 log(160)/32+1/8≈0.2836입니다. 이 값을 실제 Gibbs empirical 위험에 더합니다. 1을 넘으면 bounded risk의 자명한 상한 1로 clip할 수 있지만 원래 상한도 readout에 남겨 vacuity를 숨기지 않습니다.

직접 풀기: uniform prior, noise=.1에서 beta=0의 population 위험을 구하세요. 풀이: 여섯 threshold 위험 [.42,.26,.10,.26,.42,.58]을 평균내어 .34입니다. 이 값은 empirical 평균과 같을 필요가 없습니다. 신뢰확률은 반복 표본 추출의 사건에 대한 것이며 개별 label을 맞힐 posterior 확률 .95라는 뜻이 아닙니다.

## AI 일반화 분석에서의 사용

Prior의 데이터 독립성, loss 범위, 표본 수, confidence와 hyperparameter 선택 비용을 기록하세요. 전체 dataset을 보고 만든 prior에 같은 식을 그대로 쓰면 보장이 달라집니다. 또한 Gibbs 위험 bound를 vote 위험 bound로 단순히 이름만 바꿔 보고하면 안 됩니다. 현재 실험은 유한 class에서 모든 항을 직접 계산해 그 가정을 학습하는 기준입니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). 위 grid 보정과 예제는 직접 유도했습니다.
