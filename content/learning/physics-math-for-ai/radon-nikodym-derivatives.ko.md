## 두 측도를 같은 기준으로 비교하기

Q가 P에 절대연속이라는 것은 모든 P-영 사건이 Q에서도 영 사건이라는 뜻입니다. 기호는 Q≪P이며 방향이 중요합니다. 두 확률측도에서 이 조건이 성립하면 비음수 가측함수 w를 통해 Q의 적분을 P의 가중 적분으로 쓸 수 있습니다. 그 함수가 Radon–Nikodym 도함수 dQ/dP입니다. 미분의 이름이 붙지만 기본 의미는 측도 사이의 국소적인 질량 비율입니다.

$$
Q\ll P\iff P(A)=0\Rightarrow Q(A)=0,\qquad Q(A)=\int_A w\,dP.
$$

## 유한 공간에서는 질량의 비율

기준 P=(1/2,1/2,0,0), Q=(1/4,3/4,0,0)입니다. 첫 두 점의 비율은 1/2과 3/2입니다. 뒤 두 점에는 0/0을 계산하지 않고 표에서 —로 표시합니다. 전체 도함수의 버전을 만들 때는 그 점들에서 0을 선택합니다. P-거의 모든 곳의 유일성만 있으므로 다른 비음수 값을 선택해도 같은 Q를 만듭니다. 선택한 버전의 P 적분은 1입니다.

$$
w_i=\frac{Q_i}{P_i}\quad(P_i>0),\qquad\int w\,dP=Q(\Omega)=1.
$$

## 기대값의 측도 변경

φ(ω)=ω로 두면 기준 Q 기대값은 0.75입니다. P 아래에서 wφ의 기대를 계산해도 0.75입니다. 모든 점이 양의 P 질량을 갖는 모드는 두 계산 모두 1.5를 줍니다. 이 공식은 비음수 함수나 필요한 적분가능성을 만족하는 부호 함수에 적용합니다. 일반 σ-유한 측도에도 적절한 Radon–Nikodym 정리가 있지만, 실습은 유한 확률측도를 사용합니다.

$$
E_Q[\varphi]=E_P\!\left[\frac{dQ}{dP}\varphi\right].
$$

## 지지집합이 없으면 비율로 복구할 수 없다

실패 모드는 Q가 점 2에 질량 1/4을 주지만 P는 0을 줍니다. P에 대한 어떤 가중 적분도 그 점에 새 질량을 만들 수 없으므로 전체 RN 도함수가 없습니다. 표는 P가 양수인 점의 후보 비율만 보여주고 도함수 적분·변경 기대값은 정의하지 않습니다. 중요도 샘플링에서는 이 지지 조건을 먼저 확인해야 합니다. 조건이 성립하더라도 일반 모형의 추정량 분산이 유한하거나 작다는 보장은 별도로 필요합니다.

$$
P(\{2\})=0<Q(\{2\})\implies Q\not\ll P.
$$

## 절대연속은 존재 조건이며 분산 상계가 아니다

Q≪P이고 w=dQ/dP이며 h가 Q에 대해 적분가능하다고 합시다. P에서 독립 표본 N개를 뽑아 통상적인 중요도 가중 평균을 만들면 기대값은 I=E_Q[h]입니다. 그러나 유한 분산 공식에는 절대연속 외에 이차 모멘트 조건이 필요합니다.

$$
\widehat I_N=\frac1N\sum_{i=1}^Nw(\omega_i)h(\omega_i),\qquad
\operatorname{Var}(\widehat I_N)=\frac{E_P[w^2h^2]-I^2}{N}.
$$

조건은 E_P[w²h²]<∞입니다. Q 아래 h의 적분가능성은 P 아래 wh의 일차 절댓값 모멘트를 보장하지만 이차 모멘트를 보장하지는 않습니다. 여기서는 명시한 가중값의 평균을 다루며 자기 정규화 비율 추정량은 별개입니다.

연속 예제로 (0,1)에서 P는 균등분포, Q의 밀도는 q(x)=1/(2√x)로 둡니다. 이 밀도의 적분은 1이고 Q는 P에 절대연속입니다. 유계 함수 h=1에서도 가중값의 평균은 1이지만 이차 모멘트는 무한입니다.

$$
E_P[w]=\int_0^1\frac{dx}{2\sqrt x}=1,\qquad
E_P[w^2]=\frac14\int_0^1\frac{dx}{x}=\infty.
$$

유한 개 표본의 가중치는 거의 확실히 모두 유한하므로 관측한 표본분산이 유한하다고 빠진 모집단 모멘트를 증명할 수 없습니다. 이 평균에는 통상적인 유한 분산 CLT를 적용할 수 없습니다. h=1의 자기 정규화 추정량이라면 항상 1입니다. 반례는 위에서 정의한 비정규화 가중 평균에 관한 것입니다.

## clipping은 꼬리와 함께 기대값도 바꾼다

K≥1/2에서 가중치를 min(w,K)로 자릅니다. 전환점은 x=1/(4K²)입니다. 상수 구간과 원래 가중치 구간을 나누어 적분하면 다음과 같습니다.

$$
E_P[\min(w,K)]=1-\frac1{4K},\qquad
E_P[\min(w,K)^2]=\frac14+\frac14\log(4K^2).
$$

잘린 추정량은 유한 분산을 갖지만 이 정규화 적분에 대한 기대값은 1보다 작습니다. 꼬리 제어와 불편성은 서로 다른 주장입니다. 이 정확한 편향 계산이 모든 clipping이나 자기 정규화의 편향이 같다는 뜻은 아닙니다.

## 가정을 명시하여 최적 제안분포 유도하기

공통 기준 측도에 대한 목표·제안 밀도를 q,p로 둡니다. 0<C=E_Q|h|<∞, |h|>0이 Q-거의 확실히 성립한다고 가정하고 Q≪P를 요구합니다. Cauchy–Schwarz로 다음을 얻습니다.

$$
C^2=\left(\int\frac{|qh|}{\sqrt p}\sqrt p\right)^2
\le\int\frac{q^2h^2}{p},\qquad
p_* = \frac{q|h|}{C}.
$$

적분항은 p>0에서 계산하며 q=0인 곳의 기여는 0입니다. 제안한 p*는 적분 1이고 비영 h 가정 아래 Q의 지지를 유지하며 등호를 달성합니다. 따라서 최소 단일 표본 분산은 C²−I²입니다. 영이 아닌 고정 부호 h이면 가중값이 상수가 되어 이 값이 0입니다. C를 알거나 p*에서 정확히 표집하는 것 자체가 어려울 수 있으므로 자동으로 실행할 수 있는 알고리즘이 아니라 이론적 최적해입니다. h가 양의 Q 질량에서 0이면 위 지지 논증을 수정해야 합니다.

Q=(1/4,3/4), h=(1,3)이면 I=C=5/2입니다. 표의 행은 균등 제안분포, P=Q, 최적 제안분포 순서입니다.

| $P(0)$ | $P(1)$ | $\operatorname{Var}(wh)$ |
|---|---|---|
| $1/2$ | $1/2$ | 4 |
| $1/4$ | $3/4$ | $3/4$ |
| $1/10$ | $9/10$ | 0 |

## 유한 제안분포와 연속 유도식을 평가하기

다음 코드는 유한 모형의 가중 모멘트를 정확한 분수로 계산합니다. 연속 예제는 유도한 제한 적분·clipping 적분을 평가하며 무작위 표집으로 무한 극한의 거동을 추론하지 않습니다. 기존 네 점 RN 조절기는 존재와 변경 기대값을 계산하며 아래 분산 예제는 별도의 연습입니다.

```python
from fractions import Fraction as Q
from math import log

q, h = (Q(1, 4), Q(3, 4)), (Q(1), Q(3))
I = sum(a*b for a, b in zip(q, h))
C = sum(a*abs(b) for a, b in zip(q, h))
optimal = tuple(a*abs(b)/C for a, b in zip(q, h))
assert optimal == (Q(1, 10), Q(9, 10)) and I == Q(5, 2)
for p, expected_variance in [((Q(1, 2), Q(1, 2)), Q(4)),
                             (q, Q(3, 4)), (optimal, Q(0))]:
    values = tuple(qi*hi/pi for qi, hi, pi in zip(q, h, p))
    mean = sum(pi*value for pi, value in zip(p, values))
    second = sum(pi*value**2 for pi, value in zip(p, values))
    assert mean == I and second >= C*C
    assert second-I*I == expected_variance
    print("proposal, weighted values, variance:", p, values, second-I*I)
for m in [1, 2, 4, 8]:
    # Uniform proposal, w(x)=1/(2 sqrt(x)), restricted to x >= 4**(-m).
    retained_mean = 1-Q(1, 2**m)
    restricted_second = m*log(2)/2
    print("lower cutoff:", Q(1, 4**m), "mean:", retained_mean,
          "second moment:", restricted_second)
for K in [1, 2, 4, 16]:
    clipped_mean = 1-Q(1, 4*K)
    clipped_second = .25+.25*log(4*K*K)
    assert clipped_mean < 1 and clipped_second-float(clipped_mean)**2 > 0
    print("clipping:", K, "mean:", clipped_mean,
          "finite variance:", clipped_second-float(clipped_mean)**2)
```

## 추가 문제와 연구 연결

1. h=1인데도 가중 평균이 왜 변동하나요? **함수는 상수이지만 무작위 가중치를 통해 적분을 추정하기 때문입니다.**
2. 표집한 모든 점에서 가중치가 유한하면 모집단 분산도 유한한가요? **위 원점 부근 적분이 발산하므로 아닙니다.**
3. 중요도 샘플링 보고서에서 무엇을 구분해야 하나요? **지지, 일차·이차 모멘트, clipping 편향, 추정량 정의와 표집 비용입니다.**

측도 변경의 존재는 [Durrett, 정리 A.4.8·§1.6](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf)을 참고하세요. 위 분산·clipping·제안분포 비교는 주어진 법칙과 Cauchy–Schwarz에서 직접 유도했습니다. 밀도 q는 [제곱 변수변환](../random-variable-transformations/)에도 등장합니다.

## 확인 문제와 해설

1. P-영점의 도함수 값은? **버전에 따라 달라질 수 있습니다.**
2. 일부 점에서 비율을 계산할 수 있으면 전체 도함수가 있나요? **아니요. 절대연속 조건이 필요합니다.**
3. 연속 밀도 p,q를 같은 기준으로 쓸 때 비율은? **p>0인 곳에서 q/p이며 지지 조건을 확인해야 합니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
