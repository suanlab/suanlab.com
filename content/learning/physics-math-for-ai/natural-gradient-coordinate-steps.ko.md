## 미분을 gradient로 바꾸는 내적

목적의 미분은 접벡터 방향의 변화를 알려주는 선형 함수다. Gradient는 선택한 내적에서 이 미분을 표현하는 벡터다. Natural gradient는 특정 좌표의 유클리드 내적 대신 통계 모형의 Fisher metric을 사용한다.

실습은 내부 확률 p와 target r에서 L(p)=−r log p−(1−r)log(1−p)를 최소화한다. 최적점은 p=r이고 목적 초과는 KL(Ber(r)||Ber(p))다. KL의 어느 인자가 target인지 명시하자.

## 좌표 불변인 접방향 유도

확률 좌표 미분은 (p−r)/[p(1−p)]다. 음의 미분에 Fisher 역을 곱하면

$$
\dot p=-I_p^{-1}\frac{dL}{dp}=r-p.
$$

Logit에서는 미분 p−r, 정보 p(1−p)이므로 θ̇=(r−p)/[p(1−p)]다. dp/dθ=p(1−p)로 접벡터를 되옮기면 같은 ṗ=r−p를 얻는다. 이는 접벡터와 연속 흐름에 대한 주장이다.

## 유한 Euler 보폭은 좌표에 의존한다

확률 Euler는 p⁺=p+η(r−p)다. Logit Euler는

$$
p^+=\operatorname{sigmoid}\left(\operatorname{logit}(p)+
\eta\frac{r-p}{p(1-p)}\right).
$$

두 식은 η의 일차까지만 일치한다. 비선형 좌표 변환과 유한 덧셈 보폭은 순서를 바꾸어도 같은 연산이 아니다. 정확한 natural flow는 p(t)=r+(p₀−r)e⁻ᵗ이며 같은 흐름을 일관되게 표현하면 좌표 불변이다.

파란 선은 보폭 매개변수에 따른 정확한 흐름이다. 주황 선은 선택한 좌표에서 한 번의 Euler 갱신을 보폭별로 비교한다. 여러 번 학습한 반복 궤적이 아니다.

## 좌표 비교 손계산

p=0.2,r=0.8,η=0.1에서 확률 Euler는 0.26, logit Euler는 약 0.266727, 정확한 흐름은 약 0.257098이다. 옮겨 읽은 접속도는 여전히 0.6으로 같다. 유한 결과의 차이가 접벡터 불변성의 반례는 아니다.

같은 예의 일반 확률 GD는 0.575다. p=0.1,r=0.9,η=0.5이면 약 4.544444로 정의역을 벗어난다. Natural 확률 Euler는 표시한 범위에서 p와 r의 볼록결합이라 0.5로 남는다. 일반 logit GD는 내부를 유지하지만 다른 내적의 방향을 따른다.

## 이 흐름과 확률 보폭의 손실 보장

Cross entropy는 확률 인자에 대해 볼록하다. 정확한 흐름은 p(t)=e⁻ᵗp₀+(1−e⁻ᵗ)r이라는 볼록결합이므로 목적 오차를 명시적으로 제어한다. 0≤η≤1의 확률 Euler도 혼합 가중치를 η로 바꾸어 같은 논리를 쓸 수 있다.

$$
L(p(t))-L(r)\le e^{-t}[L(p_0)-L(r)],\qquad
L(p^+)-L(r)\le(1-\eta)[L(p)-L(r)].
$$

이 증명은 지정한 Bernoulli 손실·내부 끝점·확률 갱신을 사용한다. 모든 natural-gradient 이산화나 모든 보폭의 logit Euler에 대한 정리는 아니다. 연속 흐름을 직접 미분해도 dL/dt=−(p−r)²/[p(1−p)]를 얻는다.

## 같은 identity damping이 접벡터 불변성을 깨는 이유

두 스칼라 좌표에서 Fisher I를 I+d로 바꾸자. a=dp/dθ=p(1−p)라고 쓰면 확률 좌표의 damping d는 같은 metric을 나타내기 위해 logit에서 a²d로 변환해야 한다. 두 좌표에 같은 숫자 d를 더하면 되옮긴 속도가 달라진다.

p=1/5,r=4/5의 정확한 유리수 계산은 다음과 같다.

| d | 확률 속도 | 되옮긴 logit 속도 |
|---|---:|---:|
| 0 | 3/5 | 3/5 |
| 1/10 | 75/127 | 24/65 |
| 1 | 15/29 | 12/145 |

```python
from fractions import Fraction as F
p, target = F(1, 5), F(4, 5)
a = p*(1-p)
gp, gt = (p-target)/a, p-target
for damping in (F(0), F(1, 10), F(1)):
    vp = -gp/(1/a+damping)
    mapped = a*(-gt/(a+damping))
    covariant = a*(-gt/(a+a*a*damping))
    assert covariant == vp
    if damping: assert mapped != vp
    print(damping, vp, mapped)
```

마지막 검산은 변환된 damping을 사용하여 접벡터 일치를 복구한다. 큰 모형의 근사 Fisher 해결기를 구현한 예제가 아니라 metric과 좌표의 계산이다. 불변성을 일반화하기 전에 damping과 유한 보폭을 함께 명시하자.

## 직접 풀기와 적용 한계

첫 예에서 η를 절반으로 하여 두 natural Euler 확률 차이를 비교하자. 이 경우 선도 오차는 이차로 줄어든다. 목적 방향 미분 −(p−r)²/[p(1−p)]를 유도하고 비양수인지 확인한다.

실습은 정확하고 비특이한 스칼라 Fisher를 사용한다. 큰 신경망에서는 구조적 근사나 반복 풀이가 필요하고 좌표별 identity damping은 불변성을 바꿀 수 있다. 근사 Fisher와 고정 학습률에 정확한 흐름의 모든 성질이 자동으로 이전되지는 않는다. Metric·좌표·해결기·이산화를 명시해야 한다.

더 읽기: [Martens의 Natural Gradient 분석](https://www.jmlr.org/papers/v21/17-678.html).
