## 직접 효과라는 말의 여러 뜻

처치 X는 결과 Y에 직접 영향을 줄 수도 있고 매개변수 M을 통해 영향을 줄 수도 있다. M을 특정 값에 고정하는 질문과 다른 처치 아래 자연스럽게 생겼을 값에 두는 질문은 다르다. 특히 X와 M의 상호작용이 있으면 두 직접 효과가 같을 필요가 없다. 회귀계수를 효과라고 해석하기 전에 참조 세계를 정해야 한다.

실습은 독립이고 평균 0인 매개 잡음 E_M=±0.5, 결과 잡음 E_Y=±0.25를 사용한다. 네 잡음 쌍의 확률은 각각 1/4이다. 두 세계의 결합 방식을 구조식으로 제공한 예이며 임의의 관측자료가 효과를 모두 식별한다는 주장은 아니다.

## 네 잠재 결과 만들기

처치 x와 고정 매개값 m에 대해

$$
M_x=ax+E_M,\qquad Y(x,m)=cx+bm+dxm+E_Y.
$$

각 잡음 상태에서 Y(0,M0), Y(1,M0), Y(0,M1), Y(1,M1)을 계산한다. 모든 세계에서 같은 잡음을 유지한다. 총 효과는 Y(1,M1)과 Y(0,M0)의 평균 차이다. 표의 각 행에서 어느 처치와 어느 자연 매개값을 쓰는지 확인하자.

## 자연 효과의 올바른 짝

순수 자연 직접 효과 PNDE는 M0를 유지하고 처치만 바꾼다. 총 자연 간접 효과 TNIE는 처치를 1에 두고 M0를 M1로 바꾼다.

$$
PNDE=E[Y(1,M_0)-Y(0,M_0)]=c,
$$

$$
TNIE=E[Y(1,M_1)-Y(1,M_0)]=a(b+d).
$$

다른 순서로는 처치 0에서 매개값을 바꾸어 PNIE=ab를 얻고, M1을 유지하며 처치를 바꾸어 TNDE=c+ad를 얻는다.

$$
TE=c+ab+ad=PNDE+TNIE=PNIE+TNDE.
$$

이 항등식은 각 잡음 상태에서 중간 항이 상쇄된 뒤 평균해도 유지된다. 상호작용이 있으면 두 간접 효과는 다르다. 참조 처치를 정하지 않고 언제나 간접 효과가 ab라고 할 수 없다.

## 통제 직접 효과

매개값 m을 고정하면 CDE(m)=c+dm이다. 파란 곡선은 고정할 값을 바꾸고 주황선은 PNDE=c를 보여준다. PNDE는 자연 M0의 평균이 0인 현재 모형의 결과다. 상호작용이 있으면 특정 고정값에서의 효과는 자연 직접 효과와 크기뿐 아니라 부호도 다를 수 있다.

그림은 서로 다른 두 추정 대상을 비교한다. CDE(m)에 임의의 간접 효과 하나를 더하면 언제나 TE가 된다는 뜻이 아니다. 올바르게 짝지은 자연 효과의 분해는 별도 수치와 상태 표에서 확인한다.

## 무작위 처치만으로 Mediation Formula가 성립하지는 않는다

측정한 baseline C에서 익숙한 nested-counterfactual functional은

$$
E[Y_{x,M_{x'}}]=\sum_{c,m}E[Y\mid X=x,M=m,C=c]P(M=m\mid X=x',C=c)P(C=c).
$$

C와 M_x'에 조건부로 평균하고 Y_xm과 M_x'가 C 조건에서 독립이라는 cross-world 관계로 기대를 분리한다. 일관성·처치 및 매개 교환가능성·Support를 써 위 관측 항들로 바꾼다. 적절한 그래프의 독립 구조 잡음이 그 관계를 제공할 수 있다. X의 무작위 배정만으로 매개–결과 교환가능성이 생기거나 처치가 만든 매개–결과 교란이 없어지지는 않는다.

명시적 반례로 빠진 정보를 보자. U와 무작위 X는 독립 공정 비트이고 M0=U,Y(0,m)=0이라 하자. 세계 A는 M1=1−U,Y(1,m)=1{m=U}, 세계 B는 M1=U,Y(1,m)=1{m≠U}를 쓴다. 두 세계 모두 모든 (X,M) 셀의 Support가 양수이고 양쪽 처치에서 factual M은 균등, Y는 0이다. 그러나 E[Y(1,M0)]는 A에서 1, B에서 0이다.

| 공통 관측 법칙 또는 완성의 양 | 값 |
|---|---:|
| 관측 Mediation Functional | 0 |
| 세계 A의 Nested 평균 | 1 |
| 세계 B의 Nested 평균 | 0 |

```python
from fractions import Fraction as F
from itertools import product
observed=[];nested=[]
for world in (0,1):
    law={(x,m,y):F(0) for x,m,y in product((0,1),repeat=3)}
    target=F(0)
    for u in (0,1):
        m0=u;m1=1-u if world==0 else u
        y1=lambda m:int(m==u) if world==0 else int(m!=u)
        target+=F(1,2)*y1(m0)
        for x in (0,1):
            m=m0 if x==0 else m1;y=0 if x==0 else y1(m)
            law[x,m,y]+=F(1,4)
    observed.append(law);nested.append(target)
assert observed[0]==observed[1] and nested==[1,0]
assert all(sum(observed[0][x,m,y] for y in (0,1))>0 for x,m in product((0,1),repeat=2))
assert sum(mass*y for (x,m,y),mass in observed[0].items())==0
print('same observed law, different nested means:',nested)
```

반례는 매개와 결과의 공유 의존성을 가지며 실제 독립 잡음 상호작용 모형과 다르다. 필요한 식별 가정을 강조하는 것이지 그 가정이 맞을 때의 mediation formula를 반박하는 예는 아니다. 매개 분포를 개입적으로 바꾸는 효과는 다른 질문이므로 자체 estimand와 조정 조건이 필요하다.

현재 모형에서는 중심 잡음을 해석적으로 적분해 c,a(b+d),ab,c+ad,c+dm을 직접 계산한다. 반올림한 atom 결과 차이의 평균은 `expanded`에 남긴다. 표시 상태별 차이가 0으로 반올림돼도 작은 효과를 유지할 수 있다. 분해 residual은 강제로 0으로 바꾸지 않고 별도의 산술 검사로 유지한다.

## 확인 문제와 풀이

a=1,b=0.5,c=0.2,d=0.5에서 두 분해와 CDE(-1)을 구하라.

**풀이.** TE=0.2+0.5+0.5=1.2다. 첫 분해는 PNDE=0.2와 TNIE=1.0, 둘째는 PNIE=0.5와 TNDE=0.7이다. CDE(-1)=0.2-0.5=-0.3이므로 PNDE가 양수인데도 음수일 수 있다. 음수 매개값을 고정하는 일과 자연 기준값을 평균하는 일이 다르기 때문이다.

d=0으로 상호작용을 없애면 두 간접 효과는 모두 0.5, 두 직접 효과는 모두 0.2가 된다. 어느 m에서도 CDE=0.2다. 익숙한 선형 경로 곱 공식은 이 특수한 경우에 돌아온다.

## 정의와 관측 식별 구분

구조식이 있으면 중첩 반사실을 계산할 수 있다. 실제 자료에서 추정하려면 처치–결과, 처치–매개, 매개–결과 교란과 처치가 만든 교란, 일관성, Support, 필요한 cross-world 관계를 따로 검토해야 한다. 처치를 무작위화했다고 매개–결과의 모든 문제가 해결되는 것은 아니다. 측정 시점과 매개변수를 설정한다는 조작의 의미도 필요하다.

AI 내부 표현을 매개변수로 부르려면 어떤 개입으로 그 값을 정하고 나머지 계산 기제를 무엇으로 유지하는지 명시하자. 단순한 ablation을 곧 자연 효과라고 부를 수는 없다.

## 원전

Pearl, [Direct and Indirect Effects](https://ftp.cs.ucla.edu/pub/stat_ser/r273-jsm05.pdf), R-273의 ASA/JSM 판본. 유한 잡음과 상호작용 모형은 직접 구성했다.
