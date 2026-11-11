## 평균 방향을 보존하면서 잡음을 줄이기

Control variate는 상관된 양을 빼고 그 알려진 기대값을 다시 더한다. 원하는 평균을 유지하면서 분산을 줄이는 원리다. 유한합 최적화에서는 같은 성분의 기울기를 가까운 두 점에서 계산하면 강한 상관을 얻을 수 있다.

실습은 fᵢ(x)=hᵢ(x−aᵢ)²/2, h=(1,2,3,4), a=(−1,1,2,0)를 사용한다. 전체 기울기는 G(x)=2.5x−1.75이므로 최적점은 0.7이다. 성분별 최적점이 다르므로 전체 최적점에서 개별 기울기가 모두 0일 필요는 없다.

## SVRG 추정량의 유도

Anchor x̃에서 전체 기울기를 한 번 계산한 뒤 균등 성분 하나를 고른다.

$$
v_i(x,\widetilde x)=g_i(x)-g_i(\widetilde x)+G(\widetilde x),
\qquad\mathbb E_i[v_i]=G(x).
$$

두 성분 기울기는 같은 인덱스를 사용한다. 관계없는 성분을 빼도 추가 독립 가정 아래 평균이 맞을 수 있지만 공분산이 달라져 의도한 상쇄가 사라진다.

이 이차 모형에서는 target에 의존하는 항이 정확히 상쇄된다.

$$
v_i=h_i(x-\widetilde x)+2.5\widetilde x-1.75,
\qquad\operatorname{Var}(v_i)=1.25(x-\widetilde x)^2.
$$

1.25는 곡률 h의 모집단 분산이다. 이 식은 현재 문제족에서 유도한 것이며 모든 SVRG 문제의 정확한 분산 공식은 아니다.

## 세 가지 anchor 예제

x=x̃=0.7이면 모든 SVRG 추정량이 0이다. SGD 기울기는 (1.7,−0.6,−3.9,2.8)로 평균 0, 분산 6.575다. 성분들이 같은 최소점을 공유하지 않기 때문에 최적점에도 잡음이 남는다.

x=0.7을 유지하고 anchor를 −1로 옮기면 분산은 1.25·1.7²=3.6125다. 여전히 비편향이지만 같은 anchor보다 시끄럽다. 임의로 먼 anchor에서는 보통 SGD보다 분산이 커질 수도 있다. 이름이 variance reduction이라고 모든 점 쌍에서 분산 감소를 보장하는 것은 아니다.

x=x̃=2로 바꾸면 SVRG 분산은 0인데 모든 추정량이 3.25다. 정확하지만 비영인 기울기다. 따라서 분산 0은 최적성 인증과 다르다.

## 계산 비용과 바깥 반복

그림은 한 현재 점·anchor 쌍에서 가능한 네 성분 추정량을 비교한다. 완전한 SVRG 학습 실행이 아니다. G(x̃)를 계산하려면 여기서는 네 성분 평가가 든다. 내부 보폭도 anchor 정보를 저장하지 않으면 보통 현재 점과 anchor에서 두 기울기를 평가한다. 여러 내부 보폭에 걸쳐 전체 기울기 비용을 분산시키고 anchor 갱신 주기를 명시해야 한다.

수렴률에는 성분 손실의 smoothness, 적절한 볼록성 또는 비볼록 정상성 목표, 학습률 등의 가정이 더 필요하다. 한 보폭의 낮은 분산만으로 전체 계산량과 최종 정확도가 정해지지 않는다.

## Control Variate에서 Epoch 수축 상계로

각 성분이 볼록·L-smooth이고 평균 F가 Euclidean 공간에서 μ-강볼록이라고 가정합니다. x*를 F의 최소점이라 하세요. 매끄러운 볼록 gradient 부등식을 성분 평균하면 E‖gᵢ(x)−gᵢ(x*)‖²≤2L(F(x)−F*)입니다. ∇F(x*)=0이므로 선형항이 지워집니다. SVRG 추정량을 현재 차이와 중심화한 anchor 차이로 나누면 다음을 얻습니다.

$$
\mathbb E\|v_i\|^2\le4L\big(F(x)-F^*+F(\widetilde x)-F^*\big).
$$

이차 모멘트의 상계이지 최적점에서 모든 성분 기울기가 0이라는 주장이 아닙니다. 한 갱신의 거리 제곱을 전개하고 볼록성을 적용해 내부 m보폭을 더한 뒤 그 평균을 새 anchor로 선택합니다. η<1/(2L)에서 기대 목적 수축의 배율은 다음과 같습니다.

$$
\rho=\frac1{\mu\eta(1-2L\eta)m}+\frac{2L\eta}{1-2L\eta}.
$$

이 논증으로 수축을 보장하려면 ρ<1이어야 합니다. 실습의 이차함수족에서 L=4,μ=5/2,η=1/32,m=64이면 ρ=3/5입니다. 오래된 anchor 하나만으로 알고리즘이 정해지지 않으며 갱신·평균 규칙이 중요합니다.

| 이 상수에서 내부 보폭 m | 상계 ρ |
|---:|---:|
| 16 | 7/5 |
| 32 | 13/15 |
| 64 | 3/5 |

Python 확장은 실제 표본 epoch와 성분 평가 수를 제공합니다. 한 경로가 매 epoch 기대 상계를 만족해야 하는 것은 아닙니다. 기존 제어는 한 추정량 비교를 유지합니다. 분산은 이제 중심화한 해석 항등식으로 계산하여 x=0,anchor=10⁻²⁰에서 표시 추정량이 같게 반올림돼도 1.25×10⁻⁴⁰을 보존합니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as Q
from random import Random
L,mu,eta=Q(4),Q(5,2),Q(1,32); m=64
for length in [16,32,64]:
    rho=1/(mu*eta*(1-2*L*eta)*length)+2*L*eta/(1-2*L*eta)
    print('m:',length,'rho:',rho)
h=[1,2,3,4]; a=[-1,1,2,0]; calls=0; rng=Random(7)
def gradient(i,x):
    global calls
    calls+=1
    return h[i]*(x-a[i])
anchor=Q(-1)
for epoch in range(4):
    before=calls; full=sum(gradient(i,anchor) for i in range(4))/4
    x=anchor; iterates=[]
    for _ in range(m):
        iterates.append(x); i=rng.randrange(4)
        x-=eta*(gradient(i,x)-gradient(i,anchor)+full)
    anchor=sum(iterates)/m
    assert calls-before==4+2*m
    print('epoch:',epoch+1,'gap:',float(Q(5,4)*(anchor-Q(7,10))**2),'calls:',calls-before)
```

## 직접 풀기와 풀이

x=0.7,x̃=−1, 선택 성분 0이면 SGD 기울기는 1.7, SVRG 추정량은 1.7−g₁(−1)+G(−1)=−2.55다. η=0.1에서 다음 점은 각각 0.53,0.955다. 최적점에서 기대 기울기가 0이어도 개별 보폭은 양쪽으로 멀어질 수 있다. 네 추정량을 직접 평균하여 G(x)와 같은지 검산하자.

더 읽기: [Johnson과 Zhang의 분산 감소 논문](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
