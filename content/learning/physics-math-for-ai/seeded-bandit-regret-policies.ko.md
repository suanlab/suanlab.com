## 운 좋은 보상 합계보다 의사결정을 비교한다

앞 단원은 Bayesian 사전으로 평균하고 짧은 계획 문제를 풀었습니다. 여기서는 실제 Bernoulli 평균 mu0,mu1을 실험에서 고정합니다. 정책에는 그 평균을 주지 않고 선택한 보상만 관측하게 합니다. 이를 통해 탐색 규칙을 비교하고 낮은 평균의 arm을 고른 손실과 우연히 변동하는 보상을 구분합니다.

모든 정책은 처음에 arm0과 arm1을 한 번씩 뽑습니다. 이후 규칙은 각자의 관측 이력을 사용합니다. Thompson 변형에도 이 공통 초기화를 적용합니다. Score 동률은 arm0을 택합니다. 같은 실험을 공유해도 다른 행동이 다른 보상을 드러내므로 이력까지 같다는 뜻은 아닙니다.

## 미래를 알려주지 않고 잠재 보상을 공유한다

Arm마다 pull 횟수로 색인한 seed 기반 Bernoulli 보상열을 만듭니다. Arm i를 n번째 뽑으면 전체 round와 관계없이 그 열의 n번째 보상을 받습니다. 세 정책은 이 보상열을 공유하고 탐색 난수는 별도 stream을 씁니다. 다음 보상을 읽기 전에 행동을 먼저 선택합니다.

이 설계는 짧은 horizon과 긴 horizon의 앞부분을 보존하며 알고리즘이 난수를 더 소비했다는 이유로 환경 보상까지 바뀌지 않게 합니다. 그렇다고 정책 비교가 수학적 기대값이 되는 것은 아닙니다. 모형은 정상적인 IID 보상을 가정하고, seed 구현은 그 모형의 재현 가능한 수치 실험입니다.

## 고정 epsilon-greedy

초기화 이후 확률 epsilon으로 균등한 무작위 arm을, 나머지에는 경험 평균이 큰 arm을 택합니다. 무작위 분기도 활용과 같은 arm을 고를 수 있으므로 탐색 사건과 나쁜 arm 선택은 다릅니다. Epsilon=0은 초기 pull 이후 순수한 경험 평균 greedy이며, 잘못된 초기 데이터 때문에 갇힐 수 있습니다.

실습은 감소하는 일정이 아닌 고정 epsilon을 사용합니다. 두 평균의 차이가 Delta일 때 무작위 분기만으로 이후 매 round에 epsilon*Delta/2의 기대 regret가 생깁니다. 초기의 열등 arm 한 번까지 포함하면

$$
\mathbb E[R_T]\geq \Delta+\frac{\epsilon\Delta}{2}(T-2),\qquad T\geq2
$$

입니다. 활용이 결국 좋은 arm을 알아내더라도 고정 양의 탐색률은 점근적으로 선형 regret를 만들 수 있습니다. 신중하게 감소시키는 epsilon 일정의 로그 보장을 이 고정 구현에 그대로 붙이지 않습니다.

## UCB1: 불확실한 추정을 낙관적으로 평가한다

실행 round t 직전까지의 총 pull 수는 m=t-1입니다. 각 arm의 행동 전 count n_i와 경험 평균으로

$$
U_i(t)=\widehat\mu_i+\sqrt{\frac{2\ln(t-1)}{n_i}}
$$

를 계산해 큰 index를 택합니다. 로그는 자연로그이며 초기화 뒤 모든 n_i는 양수입니다. Index가 1을 넘을 수도 있는데, 이는 낙관적인 score이지 Bernoulli 확률이 아니기 때문입니다. 원래 UCB1 도식의 완료된 pull 시계에 맞춰 t-1을 사용합니다. 다른 시간 규약을 쓰려면 손계산과 섞지 않고 명시해야 합니다.

유계 IID 보상의 고정 표본 수 n에서는 Hoeffding 부등식으로 경험 평균이 실제 평균을 놓칠 확률을 제한합니다. Bonus sqrt(2*ln(m)/n)을 대입하면 해당 한쪽 꼬리 확률은 m^-4 이하입니다. 그러나 표본 수를 적응적으로 정하므로 완전한 논증에서는 가능한 count와 시간도 함께 처리해야 합니다. Gap 기반 분석은 열등 arm의 전형적인 pull 수를 ln(T)/Delta², 기대 regret를 ln(T)/Delta 차수로 제한하며 추가 항이 있습니다. 이는 가정 아래의 기대 보장이며 짧은 모든 seed에서 UCB가 이겨야 한다는 요구가 아닙니다.

## Thompson Sampling: 그럴듯한 환경을 표본 추출한다

각 arm에 균등 Beta(1,1) 사전을 사용합니다. 관측 성공 S회, 실패 F회이면 Beta(S+1,F+1)에서 후보 평균을 뽑고 큰 표본의 arm을 선택합니다. 현재 사후 평균이 큰 쪽만 고르는 규칙과 다릅니다. 또한 남은 예산 전체의 Bellman 최적값을 직접 구하는 유한 Bayesian 최적 정책과도 다릅니다.

여기서는 정수 매개변수의 성질을 사용합니다. 독립 균등 난수 a+b-1개 중 a번째로 작은 값은 Beta(a,b) 분포입니다. 실제로

$$
\Pr(U_{(a)}\leq x)=\sum_{j=a}^{a+b-1}\binom{a+b-1}{j}x^j(1-x)^{a+b-1-j}
$$

입니다. 적어도 a개의 균등 표본이 x 이하인 사건을 세었으며 Beta CDF를 얻습니다. 작은 실습에서 투명하게 구현할 수 있는 방법입니다. 테스트는 순서통계량의 손계산·모멘트·CDF 빈도를 확인하며 표본 추출 검사로 regret 정리를 증명했다고 주장하지 않습니다.

## 서로 다른 두 성능 수치를 함께 자르지 않는다

mu_star=max(mu0,mu1)로 놓으면 선택한 행동 경로의 누적 pseudo-regret는

$$
R_T=\sum_{t=1}^T(\mu_{star}-\mu_{A_t})
$$

입니다. 음수가 아니며 행동이 이전 관측에 의존하므로 이 양도 확률적입니다. 실제 평균 benchmark에 대한 shortfall은

$$
S_T=T\mu_{star}-\sum_{t=1}^T Y_t
$$

입니다. 관측 보상이 기대 benchmark를 넘으면 음수가 될 수 있습니다. 별도로 표본 추출한 전지적 보상 경로와 비교한 양은 아닙니다. 명시한 비예견적 IID 모형에서는 E[S_T]=E[R_T]이지만 한 실행에서 두 값이 같을 필요는 없습니다. 음수 shortfall을 0으로 자르면 통계량을 바꾸고 평균도 왜곡합니다.

위 막대는 세 정책의 마지막 pseudo-regret를 비교하며 파랑 테두리는 선택 정책입니다. 아래 파랑 계단은 그 정책의 누적 pseudo-regret, 주황은 UCB1입니다. 표시할 때 번호 붙은 round 사이의 값을 유지하며 결정과 증가는 정수 round에서 일어납니다. UCB1을 선택하면 두 경로가 겹칩니다. 표에서 행동 전 score, 선택, 보상, count와 두 성능 양을 확인하세요. 한 seed로 기대 성능의 순위를 증명하지 않습니다.

## 연구 확장: 학습 regret와 보상 잡음을 분리하기

실현 shortfall을 $S_T=T\mu_*-\sum_tY_t$, pseudo-regret을 $R_T=\sum_t(\mu_*-\mu_{A_t})$라 하자. 차이는 $S_T-R_T=\sum_t(\mu_{A_t}-Y_t)$다. 매번 행동을 먼저 정하고 독립 Bernoulli 보상을 받으면 조건부 증분 평균이 0인 martingale이 된다. 조건부 증분 범위 길이는 1이므로 Hoeffding-Azuma에서

$$P\left(|S_T-R_T|\ge\sqrt{(T/2)\log(2/\delta)}\right)\le\delta$$

다. 이는 무작위 pseudo-regret 주위의 보상 잡음 상한이며 pseudo-regret 자체의 상한은 아니다. 세 정책과 모든 $t\le T_{max}$를 동시에 보려면 문턱을 $\sqrt{(t/2)\log(6T_{max}/\delta)}$로 바꾼다. 보상 스트림을 공유하는 정책들의 독립성은 union bound에 필요 없다.

| $T$, $\delta=0.05$ | 고정 시점 문턱 |
|---|---|
| 2 | $\sqrt{\log40}$ |
| 8 | $2\sqrt{\log40}$ |
| 32 | $4\sqrt{\log40}$ |

```python
from fractions import Fraction as F
from itertools import product
from math import sqrt,log
mu=[F(1,4),F(3,4)]; delta=.05
for T in [2,4,8]:
    expected=F(0); tail=F(0); total=F(0)
    threshold=sqrt(T/2*log(2/delta))
    for rewards in product([0,1],repeat=T):
        arm=0; mass=F(1); gap=F(0)
        for y in rewards:
            mass*=mu[arm] if y else 1-mu[arm]
            gap+=mu[arm]-y
            if not y:arm=1-arm
        total+=mass; expected+=mass*gap
        if abs(gap)>=threshold:tail+=mass
    assert total==1 and expected==0 and float(tail)<=delta
    print(T,expected,float(tail))
```

정확 열거는 성공하면 유지하는 별도의 적응 정책을 확인한다. 유한한 결정론적 seed 집합은 이상적 IID 확률공간과 같지 않으며 한 seed에서 더 좋은 경로가 기대 정책 우월성을 증명하지 않는다.

## 손계산 문제: 결정론적 UCB 선택

mu0=1,mu1=0,T=4로 놓습니다. 처음에는 arm0에서 1, arm1에서 0을 받습니다. Round 3 직전 m=2이고 두 count가 1이므로 score는 1+sqrt(2*ln(2))와 sqrt(2*ln(2))입니다. 따라서 arm0을 택합니다. Round 4 직전 count는 2와 1이며 score는 1+sqrt(ln(3))와 sqrt(2*ln(3))으로 다시 arm0을 택합니다.

**해설.** 행동은 0,1,0,0이며 총보상은 3, pseudo-regret는 1, 실제 benchmark shortfall도 1입니다. 결정론적 예제로 난수 결과와 score 규약을 분리하고, score의 count가 행동 전 값인지 확인할 수 있습니다.

두 번째 문제는 mu0=mu1=0.5,T=2,seed=1입니다. 초기 두 보상이 모두 1이어서 모든 행동 gap이 0이고 pseudo-regret도 0입니다. 반면 shortfall은 2*0.5-2=-1입니다. 세 번째 안내 예제는 이 음수를 그대로 유지합니다.

## 적용 범위와 AI 연구

Seed를 바꾸면 짧은 실행의 순위가 뒤집힐 수 있습니다. 평균, horizon, 정상성 가정을 바꾸면 문제 자체가 바뀔 수도 있습니다. 현재 모형에는 지연 보상, 문맥 변수, 변하는 평균과 적대적 피드백이 없습니다. 그런 경우에는 다른 모형과 보장이 필요합니다.

AI의 bandit 평가에서는 정책이 어떤 정보를 받는지, 보상 난수를 어떻게 공유하는지, 탐색이 고정인지 적응적인지, 어떤 기대로 regret를 정의하는지 밝혀야 합니다. 경험적 주장을 하려면 여러 seed와 불확실성 평가가 유용합니다. 이 실습은 개별 행동과 정확한 성능 회계를 살펴보는 기준이며 앞의 사전 평균 계획과 목적을 혼동하지 않고 보완합니다.

## 원전과 읽을거리

Auer·Cesa-Bianchi·Fischer의 [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)은 원래 UCB1 도식과 유계 보상의 기대 regret 설정을 제공합니다. Agrawal·Goyal의 [Analysis of Thompson Sampling](https://proceedings.mlr.press/v23/agrawal12.html)은 Bernoulli 사후 표본 추출의 배경입니다. 보상열 공유 실험, 고정 epsilon 계산, 예제와 그림은 이 교재를 위해 직접 구성했습니다.
