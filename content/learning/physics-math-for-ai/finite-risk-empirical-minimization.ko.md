## 무엇에 대한 평균인가

학습 오류는 이미 본 데이터의 평균이고, population risk는 앞으로 만날 결합분포에 대한 기대입니다. 둘은 같은 손실을 쓰더라도 다른 양입니다. 이 단원은 x=0,1,2,3,4가 같은 확률로 나오고, x≥2이면 보통 label 1, 아니면 보통 0인 세계를 정합니다. 확률 epsilon으로 반대 label이 나옵니다. 모형의 진짜 분포를 알기 때문에 미관측 위험을 Monte Carlo 추정 없이 정확히 계산할 수 있습니다.

가설은 여섯 threshold h_t(x)=1{x≥t}, t=0,...,5입니다. t=0과 5는 각각 항상 1, 항상 0입니다. 파랑은 실제 seed 표본의 empirical risk, 주황은 population risk입니다. 점은 별도로 선택한 query 가설이며 ERM과 반드시 같지는 않습니다.

## Risk와 ERM 정의

$$
R(h)=E_{(X,Y)\sim P}[1\{h(X)\ne Y\}],\qquad
\widehat R_S(h)=\frac1N\sum_{i=1}^N1\{h(X_i)\ne Y_i\}.
$$

ERM은 선언한 class에서 경험 위험을 최소화합니다. 이 실습은 동률이면 가장 작은 threshold를 선택합니다. Class를 바꾸거나 tie 규칙을 바꾸면 같은 데이터에서 다른 모델이 선택될 수 있습니다. 데이터와 관계없이 정한 하나의 가설에 대한 집중 부등식을 데이터가 고른 ERM에 그대로 적용하는 것도 정당하지 않습니다. 유한 class의 union bound나 별도 복잡도 분석이 필요합니다.

## 이 세계의 Population Oracle

노이즈가 절반보다 작으므로 각 x에서 더 흔한 label을 예측하는 threshold 2가 oracle입니다. 현재 모형에서는 다음을 직접 유도할 수 있습니다.

$$
R(h_t)=\epsilon+(1-2\epsilon)\frac{|t-2|}{5},\qquad R(h_2)=\epsilon.
$$

Oracle도 label의 고유 잡음 아래로 내려갈 수 없습니다. 그러나 sample ERM의 학습 오류는 epsilon보다 작거나 클 수 있습니다. E[min Rhat]≤min E[Rhat]는 기대값에 대한 관계입니다. 개별 표본에서 generalization gap의 부호를 강제로 양수로 만들면 실제 데이터를 왜곡합니다. 화면은 signed gap을 그대로 표시합니다.

## 균일 제어에서 Oracle 부등식으로

고정 class 전체에서 |R(h)−Rhat(h)|≤e가 동시에 성립하는 사건을 가정하자. h*는 population 최소자이고 학습기는 empirical 최적값보다 최대 a만큼 큰 approximate ERM을 반환한다. 두 population 위험 사이에 empirical 위험 두 개를 끼우면

$$
R(\widehat h)-R(h^*)\le [R(\widehat h)-\widehat R(\widehat h)]+a+[\widehat R(h^*)-R(h^*)]\le2e+a.
$$

선택된 가설에도 적용되는 이유는 사건이 class 전체에 동시에 성립하기 때문이다. 고정 M개 가설, [0,1] 손실, IID 표본에서는 양쪽 Hoeffding과 union bound로 e=sqrt(log(2M/delta)/(2N))를 얻는다. 손실 범위보다 클 수도 있는 상한이며 실제 관측 gap 자체가 아니다.

Realizable 상황에서는 더 빠른 별도 논증이 가능하다. 위험 0인 가설이 class에 있고 정확하게 consistent한 가설을 찾는다고 하자. 위험이 e보다 큰 고정 가설이 N개 label에서 한 번도 틀리지 않을 확률은 (1−e)^N≤exp(−Ne) 이하이다. 나쁜 가설들을 union bound하면 log(M/delta)/N 규모의 보장을 얻는다. 잡음이나 consistency 실패가 있으면 이 단계를 쓸 수 없다. 작은 training loss만으로 가정을 증명할 수는 없다.

아래 독립 예제는 같은 비중의 두 입력, Y=X와 세 증가 threshold를 쓴다. 가장 작은 threshold를 고르는 ERM은 모든 관측이 1일 때만 틀린다. 실패 확률 2^(−N)을 모든 표본의 열거로 확인한다.

| 독립 예제의 N | ERM의 양의 초과 위험 확률 |
|---|---:|
| 1 | 1/2 |
| 2 | 1/4 |
| 4 | 1/16 |

```python
from itertools import product
from fractions import Fraction as F
for N in (1,2,4):
    failure=F(0)
    risk=[F(1,2),F(0),F(1,2)]
    for sample in product((0,1),repeat=N):
        empirical=[F(sum(int(x>=t)!=x for x in sample),N) for t in range(3)]
        chosen=min(range(3),key=empirical.__getitem__)
        uniform=max(abs(a-b) for a,b in zip(risk,empirical))
        assert risk[chosen]<=2*uniform
        if risk[chosen]>0:failure+=F(1,2**N)
    assert failure==F(1,2**N)
    print(N,failure)
```

화면의 여섯 threshold 실습은 다른 다섯 입력 세계와 실제 seed 실현을 사용한다. 알려진 population 곡선은 교육용 oracle이다. Class·중단 규칙·test 질의를 데이터를 보고 고르면 별도 동시 제어나 독립 평가가 필요하다. 결정적 생성기는 IID 모형의 재현 가능한 예시를 주지만 유한 seed 집합 자체가 정리의 확률 공간은 아니다.

## 손계산과 직접 풀기

무잡음에서 population risk는 t=0부터 [0.4,0.2,0,0.2,0.4,0.6]입니다. N=1, seed 7의 실제 표본은 (x=0,y=0)입니다. Threshold 1부터 5는 모두 학습 오류 0이고 tie 규칙은 1을 고릅니다. 그 모델의 population risk는 0.2입니다. 학습 오류 0이 일반화 오류 0을 뜻하지 않는 가장 작은 예입니다.

직접 풀기: noise=0.1에서 t=0과 t=2의 위험을 구하세요. 풀이: t=0은 두 음성 영역을 잘못 예측하므로 0.1+0.8·2/5=0.42, t=2는 0.1입니다. 한 표본의 label 비율을 이 정확한 기대값과 같은 것으로 해석하지 마세요.

## 실제 AI 평가로 연결하기

일반 데이터에서는 진짜 P를 모르므로 별도 test set, 평가 절차와 불확실성이 필요합니다. 현재의 정확한 population 곡선은 알려진 유한 세계의 교육용 oracle입니다. Test 결과로 class나 hyperparameter를 반복 선택하면 다시 선택 편향이 생깁니다. 입력 분포, 고정 class, 실제 표본, 선택 규칙과 평가 대상을 함께 기록하는 습관이 다음의 일반화 이론을 이해하는 출발점입니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/), [MIT 통계적 학습 자료](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/). 원고와 유한 예제는 직접 작성했습니다.
