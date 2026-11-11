## 표현은 군 구조를 선형화합니다

표현 ρ는 군 원소를 가역 선형 연산자로 보내고 곱을 보존합니다. 차수 4의 순환군 C4 생성자가 네 표본을 (Px)[n]=x[(n−1) mod 4]로 한 칸 옮기게 합니다. P⁴=I이므로 R⁴ 위의 표현입니다.

불변 부분공간은 모든 군 원소가 그 안으로 보내는 공간입니다. 불가약 표현은 비영이며 적절한 비영 불변 부분공간이 없는 표현입니다. 계수 체를 명시해야 합니다. 실수 불가약성과 복소화 후 불가약성은 다를 수 있습니다.

## 정규직교 분해 구하기

다음 네 기저를 사용합니다.

$$
u_0=\tfrac12(1,1,1,1),\quad
u_2=\tfrac12(1,-1,1,-1),\quad
u_c=\tfrac1{\sqrt2}(1,0,-1,0),\quad
u_s=\tfrac1{\sqrt2}(0,1,0,-1).
$$

첫째는 trivial 표현, 둘째는 한 칸 이동마다 부호가 바뀌는 alternating 표현을 만듭니다. 마지막 둘은 평면입니다. 계수는 a_c=(x₀−x₂)/√2, a_s=(x₁−x₃)/√2이고 한 칸 이동은 이 쌍을 π/2 회전시킵니다.

$$
(a_c,a_s)\longmapsto(-a_s,a_c),\qquad
\mathbb R^4=V_0\oplus V_2\oplus V_{cs}.
$$

그림의 파랑·주황 벡터는 이 평면 계수의 이동 전후이며 시간 표본 자체가 아닙니다. 표는 세 표현 block을 모두 사용해 이동한 표본을 복원하고 직접 cyclic shift와 비교합니다.

## 실수에서 평면이 불가약인 이유

2차원 평면의 적절한 비영 실수 부분공간은 직선입니다. 불변 직선이 있으면 quarter-turn J의 실수 고유벡터가 있어야 하지만 특성다항식 λ²+1에는 실수 근이 없습니다. 따라서 실수에서 이 평면은 불가약입니다.

복소수에서는 J의 고유값 ±i와 1차원 고유직선이 있어 복소화한 평면이 두 character로 갈라집니다. 유한 가환군 C4의 복소 표현이 1차원 character들로 분해된다는 사실과 실수 2차원 block은 모순되지 않습니다.

## 에너지와 손계산 예제

정규직교성으로 Σxₙ²=a₀²+a₂²+a_c²+a_s²입니다. x=(1,0,0,0)이면 scalar 계수는 각각 0.5, 평면은 (1/√2,0)입니다. 에너지는 1/4,1/4,1/2로 합이 1입니다. 한 칸 이동하면 평면은 (0,1/√2), alternating 계수는 −0.5이고 전체 복원값은 (0,1,0,0)입니다.

## 불변 보완공간과 사영을 구성하기

유한 실수 표현에서는 임의의 양의 정부호 내적을 군 전체에 평균해 직교 표현으로 만들 수 있습니다. 평균도 양의 정부호이고 합의 index를 바꾸면 불변성이 증명됩니다. 불변 부분공간의 직교 여공간도 이 평균 metric에서는 불변입니다. 차원에 대한 귀납법으로 완전 가약성을 얻습니다. 군의 차수로 나누는 단계가 핵심이므로 표수가 군 차수를 나누는 체에서는 같은 증명이 실패합니다.

C4 regular 예제는 원래 Euclidean metric부터 불변입니다. 다음 사영으로 상수·교대·나머지 평면 block을 분리합니다.

$$
E_0=\frac{I+P+P^2+P^3}{4},\quad
E_2=\frac{I-P+P^2-P^3}{4},\quad
E_{cs}=I-E_0-E_2.
$$

각 사영은 멱등이며 서로 곱하면 0이고 합은 항등입니다. trace는 rank인 1,1,2입니다. sine/cosine 기저부터 고르지 않아도 분해를 구성할 수 있습니다. 평면에서 교환하는 연산자는 aI+bJ이므로 실수 대수로 보면 복소수의 복사본입니다. 따라서 실수 불가약성만으로 모든 교환 연산자가 실수 scalar라고 결론내릴 수 없습니다. 대수적으로 닫힌 체의 익숙한 Schur 보조정리에는 다른 가정이 있습니다.

| 사영 | Rank | 한 칸 이동이 상에서 하는 작용 |
|---|---:|---|
| E₀ | 1 | +1 |
| E₂ | 1 | -1 |
| Ecs | 2 | Quarter-turn |

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
n=4
I=[[F(i==j) for j in range(n)] for i in range(n)]
P=[[F(i==(j+1)%n) for j in range(n)] for i in range(n)]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
powers=[I,P]
for _ in range(2): powers.append(mm(powers[-1],P))
E0=[[sum(M[i][j] for M in powers)/4 for j in range(n)] for i in range(n)]
E2=[[sum((-1)**k*M[i][j] for k,M in enumerate(powers))/4 for j in range(n)] for i in range(n)]
Ecs=[[I[i][j]-E0[i][j]-E2[i][j] for j in range(n)] for i in range(n)]
for a,A in enumerate([E0,E2,Ecs]):
    assert mm(A,A)==A
    for b,B in enumerate([E0,E2,Ecs]):
        if a!=b: assert all(x==0 for row in mm(A,B) for x in row)
    print('rank:', sum(A[i][i] for i in range(n)))
```

## 직접 풀기와 연구 연결

상수 신호 (1,1,1,1)은 a₀=2만 남아 에너지가 4입니다. 교대 신호 (1,−1,1,−1)은 a₂=2만 남고 한 칸 후 부호가 바뀝니다. 직접 확인하세요.

불가약 block은 허용 feature 종류와 등변 선형 결합을 정리합니다. 군·작용·체를 항상 명시해야 합니다. 한 군에서 불가약인 표현이 부분군에 제한하면 가약이 될 수도 있으며 뒤의 tensor 두 배 각도 예제로 확인합니다.

추가 읽기: [Etingof et al., Introduction to Representation Theory](https://math.mit.edu/~etingof/books.html).
