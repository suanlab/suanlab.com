## 두 집단의 차이가 곧 처치 효과는 아니다

시스템 설정을 바꾼 뒤 실패율을 비교한다고 하자. 성능이 좋은 설정을 주로 어려운 과제에 배포했다면, 모든 난이도 집단에서 도움이 되더라도 관측 실패율은 더 높을 수 있다. 처치 집단과 비처치 집단이 서로 다른 모집단을 섞고 있기 때문이다. 교란은 이 비교를 설명하는 구조적 개념이지, 입력 변수끼리 상관이 있다는 말과 같지 않다.

실습은 두 세계를 분리한다. 세계 A에는 X와 Y의 공통 원인 Z, 그리고 X→Y 효과가 있다. 세계 B에서는 X와 Y가 독립이고 두 변수가 함께 표본 선택 S를 결정한다. 두 세계의 수치를 섞으면 서로 다른 생성 기제를 혼동하게 된다. 그래프와 확률표의 A·B 표시를 함께 읽자.

## 세계 A: 효과의 부호가 뒤집히는 계산

Z는 확률 pi로 1이다. 두 Z 층에서 X=1일 확률을 e_0, e_1이라 하자. 결과 기제는 다음과 같다.

$$
q(z,x)=P(Y=1\mid Z=z,X=x)=b_z+\tau x,
\qquad b_0=0.2,\quad b_1=0.7.
$$

tau의 범위는 [-0.2,0.2]여서 모든 결과 확률이 유효하다. Z가 주어졌을 때 처치 잡음과 결과 잡음이 독립이라는 가정도 포함한다. 따라서 어느 층에서든 X를 바꾸면 결과 평균의 변화는 정확히 tau다.

pi=0.5, e_0=0.1, e_1=0.9, tau=-0.1로 두자. X=1 관측 중 90%는 Z=1이고 X=0 관측 중에는 10%만 Z=1이다.

$$
E[Y\mid X=1]=0.9\times0.6+0.1\times0.1=0.55,
$$

$$
E[Y\mid X=0]=0.1\times0.7+0.9\times0.2=0.25.
$$

단순 관측 차이는 +0.30이다. 개입은 원래 P(Z)를 유지하므로 X=1 평균은 0.35, X=0 평균은 0.45다. 인과효과는 -0.10이다. Y가 실패라면 도움이 되는 처치가 단순 관측에서는 해롭게 보인다. 오류는 층별 효과의 계산보다 두 비교 집단의 구성에서 발생했다.

## 표준화와 비어 있는 셀

일관성, Z 조건부 교환가능성, 목표 모집단 층에서의 처치 positivity가 성립하면 다음 backdoor 조정을 사용한다.

$$
E[Y\mid do(X=x)]=\sum_z E[Y\mid X=x,Z=z]P(Z=z).
$$

실습의 조정값은 열거한 관측 셀에서 조건부 평균을 구한 뒤 원래 P(Z)로 평균한다. 비어 있는 셀을 oracle 결과식으로 몰래 채우지 않는다. 두 층이 모두 존재하는데 e_0=0, e_1=1이면 각 층에 반대 처치가 없다. 이때 관측 조정값은 —로 표시한다.

선언한 SCM의 효과는 여전히 계산할 수 있다. 구조식이 보지 못한 결과를 가정으로 제공하기 때문이다. 그러나 관측자료가 그 값을 식별했다고 말할 수는 없다. 반대로 목표 모집단 확률이 0인 층은 조정 합에 기여하지 않으므로 그 층의 처치 셀이 없어도 된다.

## 세계 B: 선택만으로 생기는 연관

이번에는 X와 Y를 서로 독립인 공정한 비트로 둔다. 둘 사이 인과 화살표는 없다. (X,Y)=(0,0)일 때 선택 확률은 1-rho, 나머지 세 상태에서는 1이다. S=1을 조건으로 선택한 결합확률은 다음 값에 비례한다.

$$
(1-\rho,\ 1,\ 1,\ 1).
$$

rho=1이면 선택 표본에서 00만 빠진다. X=0인 선택 관측에서는 Y=1이 확실하지만, X=1에서는 Y=1일 확률이 0.5다. 선택된 차이는 -0.5이고 모집단 인과효과는 0이다. 공통 결과를 조건으로 삼으면 원래 독립이던 원인 사이에 연관이 생길 수 있다. 이것이 X→S←Y라는 collider 경로의 핵심이다. 모든 선택이 항상 편향을 만든다는 뜻은 아니다.

## Double Robustness의 곱 잔차와 교란 가정

고정 행동 x에서 q_x(z)=E[Y|X=x,Z=z], e_x(z)=P(X=x|Z=z)라 하자. 목표 층에서 g가 양수인 후보 함수 m,g로 augmented 모집단 functional을 정의한다.

$$
\Psi_x(m,g)=E\left[m(Z)+\frac{1\{X=x\}}{g(Z)}(Y-m(Z))\right],
\quad\Psi_x-\mu_x=E\left[\left(1-\frac{e_x(Z)}{g(Z)}\right)(m(Z)-q_x(Z))\right].
$$

Z에 조건부로 평균하면 residual의 기대가 e_x*(q_x−m)이어서 항등식이 증명된다. Nuisance 하나가 정확하면 잔차가 0이다. 두 모델이 training data에 잘 맞는다는 이유로 잔차가 없어지는 것은 아니다. G≥eta>0이면 Cauchy–Schwarz로 절댓값을 ||g−e_x||₂*||m−q_x||₂/eta 이하로 제어한다. 이 곱은 rate 조건의 동기이지만 점근 추론에는 표본 가정·분산 제어·적절한 sample splitting 또는 cross-fitting도 필요하다.

유리수 예제는 세계 A의 기본 joint law를 유지한다. Outcome 평균만 정확하거나 propensity만 정확해도 ATE −1/10을 복원하지만 두 함수가 모두 틀리면 편향된 +3/10을 얻는다.

| 후보 Nuisance 함수 | Augmented ATE |
|---|---:|
| 정확한 Outcome·틀린 Propensity | −1/10 |
| 틀린 Outcome·정확한 Propensity | −1/10 |
| 둘 다 틀림 | 3/10 |

![Outcome 오차와 세 propensity 선택에 따른 augmented 평균의 모집단 편향으로 곱 잔차를 설명한 그림.](/assets/learning/causal-double-robust-ko.svg)

그림은 참 propensity 1/5·outcome 평균 7/10인 한 층을 분리하며 아래 코드는 세계 A의 두 층 전체를 쓴다.

```python
from fractions import Fraction as F
P=[F(1,2)]*2;e=[F(1,10),F(9,10)];tau=F(-1,10)
q=[[F(1,5),F(1,5)+tau],[F(7,10),F(7,10)+tau]]
wrong=[[F(2,5)]*2 for _ in range(2)]
def augmented(m,g,x):
    true_e=[e[z] if x else 1-e[z] for z in range(2)]
    fitted_e=[g[z] if x else 1-g[z] for z in range(2)]
    value=sum(P[z]*(m[z][x]+true_e[z]/fitted_e[z]*(q[z][x]-m[z][x])) for z in range(2))
    truth=sum(P[z]*q[z][x] for z in range(2))
    remainder=sum(P[z]*(1-true_e[z]/fitted_e[z])*(m[z][x]-q[z][x]) for z in range(2))
    assert value-truth==remainder
    return value
for m,g in ((q,[F(1,2)]*2),(wrong,e),(wrong,[F(1,2)]*2)):
    print(augmented(m,g,1)-augmented(m,g,0))
```

고정 함수의 모집단 항등식이지 같은 기록에서 학습한 nuisance의 유한 표본 불편성 주장은 아니다. Backdoor 식별 가정도 필요하다. 미측정 공통 원인·Support 부재·collider 조건화는 이 대수로 고칠 수 없다. 조건부 셀 비율은 scaled log mass를 써 raw 곱이 underflow하는 희귀 양의 셀을 유지한다. 실제로 없는 처치 층은 여전히 미정이며 세계 B의 선택 과정도 별도로 유지한다.

## 확인 문제와 해설

세계 A의 나머지 기본값을 유지하고 e_0=e_1=0.5로 처치를 무작위 배정하자. 세계 B에서는 rho=1을 유지한다. 어떤 불일치가 사라지는가?

**해설.** A의 두 관측 처치 집단은 모두 원래의 50:50 층 구성을 갖는다. 관측 차이는 -0.1이 되어 개입 효과와 일치한다. B에서는 여전히 00을 선택하지 않으므로 선택 차이 -0.5가 남는다. 한 세계의 배정을 바꾸는 것으로 별도 세계의 선택 기제를 고칠 수는 없다. 예제를 실행하고 확률표의 세계 표시를 확인하자.

## AI 평가에 적용하는 순서

난이도, 정책 배정, 결과, benchmark 포함 여부를 만드는 과정을 먼저 그리자. 처치와 결과에 모두 관련된 변수라도 공통 원인, 결과, collider 중 무엇인지는 다를 수 있다. 조정 변수를 무조건 늘리는 것이 해결책은 아니다. 지금 예제는 정확한 모집단 계산으로 구조적 문제를 분리한다. 실제 연구의 표본 불확실성, 측정 오류, 모형 오지정은 추가로 검토해야 한다.

## 원전과 더 읽기

- Judea Pearl, [A Causal Calculus for Statistical Research](https://proceedings.mlr.press/r0/pearl95a.html): 개입을 관측식으로 바꾸는 그래프 조건.
- Miguel Hernán·James Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook): 교란·선택 편향과 식별 조건.

두 수치 모집단과 각각의 확률표는 직접 구성한 교육용 예제다.
