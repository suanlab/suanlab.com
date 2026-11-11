## 함수 전체를 입력으로 받는 목적

보통 목적함수는 벡터를 수로 보낸다. 범함수는 곡선·장·확률밀도 같은 함수 자체를 입력으로 받을 수 있다. 변화를 보려면 전체 함수를 yε=y+εh로 움직인다. 방향 h도 함수이고, ε는 그 크기를 정하는 스칼라다.

실습은 무차원 t∈[0,1], 양 끝값 0인 다음 에너지를 사용한다.

$$
J[y]=\frac12\int_0^1\bigl((y')^2+\alpha y^2\bigr)\,dt,
\qquad y_\epsilon=y+\epsilon h,\quad\alpha\ge0.
$$

Sine 기저는 끝점 조건을 정확히 보존한다. 파란 선은 원래 함수, 주황 선은 섭동한 함수다. 시간에 따른 최적화 궤적과는 다르다.

## 일차와 이차 변분

적분하기 전에 제곱을 전개하면

$$
\delta J[y;h]=\left.\frac{d}{d\epsilon}J[y+\epsilon h]\right|_{\epsilon=0}
=\int_0^1(y'h'+\alpha yh)\,dt,
\qquad
\delta^2J[y;h,h]=\int_0^1((h')^2+\alpha h^2)\,dt.
$$

일차 변분은 h에 선형이다. 방향별 Gâteaux 미분과 달리 Fréchet 미분은 정한 노름에서 모든 작은 섭동에 대한 근사 오차를 균일하게 제어한다. 이 이차 에너지는 영 경계 Sobolev 공간 H₀¹에서 유계 쌍선형 형식으로 그 성질을 확인할 수 있다. 일반 범함수에도 같은 말을 하려면 함수 공간과 노름부터 정해야 한다.

## 두 mode의 정확한 계산

y=a sin πt+b sin 2πt, h=c sin πt+d sin 2πt로 두자. Sine과 도함수 cosine의 직교성으로 A=π²+α, B=4π²+α를 쓰면

$$
J[y]=(Aa^2+Bb^2)/4,\quad
\delta J=(Aac+Bbd)/2,\quad
\delta^2J=(Ac^2+Bd^2)/2.
$$

Sine 제곱 적분이 1/2라는 질량 계수가 중요하다. 정규화 없이 두 계수를 정규직교 좌표처럼 생각하면 계수 미분의 배율을 틀리게 된다.

## 유한 변화와 미분의 구별

여기서는 J[y+εh]−J[y]=εδJ+ε²δ²J/2가 정확하다. ε≠0인 차분 몫에는 δJ뿐 아니라 εδ²J/2도 남는다. 코드는 거의 같은 에너지를 빼는 대신 전개식으로 계산한다. 섭동 후 에너지는 바뀐 계수에서 직접 계산하여 곡선 전체가 상쇄될 때 정확히 0으로 남긴다. ε=0에서 몫 자체는 미정의지만 극한인 일차 변분은 존재한다.

한 방향의 일차 변분이 0이어도 범함수가 상수라는 뜻은 아니다. 직교 방향에서 이차 에너지 비용이 남을 수 있다. 정지성은 모든 허용 방향에서 일차 변분이 0이라는 조건이다.

## 함수 공간의 노름이 정리를 바꾸는 이유

Fréchet 미분 가능성을 직접 증명하자. H₀¹(0,1)에 도함수 노름 ‖v‖V=‖v′‖₂를 주자. v(t)=∫₀ᵗv′(s)ds와 Cauchy–Schwarz로 ‖v‖₂≤‖v′‖₂를 얻는다. B(y,h)=∫(y′h′+αyh)라고 쓰면 |B(y,h)|≤(1+α)‖y‖V‖h‖V다. 제곱을 전개하면 모든 섭동에 공통인 나머지 상계를 얻는다.

$$
J[y+h]-J[y]-B(y,h)=\tfrac12 B(h,h),\qquad
\frac{|J[y+h]-J[y]-B(y,h)|}{\|h\|_V}
\le\tfrac12(1+\alpha)\|h\|_V\longrightarrow0.
$$

이 상계는 크기가 줄면서 방향도 바뀌는 h를 포함한다. 몇 가지 방향의 도함수를 계산하는 것만으로는 보장되지 않는 균일성이 바로 이 부분이다.

이제 함수 집합은 그대로 두고 섭동을 L2 노름으로만 재자. hₙ(t)=sin(nπt)/n은 진폭과 L2 노름이 0으로 가지만 기울기 에너지는 사라지지 않는다.

$$
\|h_n\|_2^2=\frac{1}{2n^2},\qquad
\|h_n'\|_2^2=\frac{\pi^2}{2},\qquad
J[h_n]=\frac{\pi^2}{4}+\frac{\alpha}{4n^2}.
$$

따라서 H₀¹에 L2 노름만 준 위상에서는 J가 0에서 연속조차 아니므로 Fréchet 미분도 없다. 눈에 보이는 진폭은 작아도 빠른 진동이 큰 도함수 에너지를 유지한다. 학습한 장의 잔차 그림이 작게 보인다는 이유만으로 에너지 노름의 정확도를 보장할 수 없는 이유와도 연결된다.

α=0에서 비교하면 다음과 같다.

| n | $\|h_n\|_2^2$ | $\|h_n'\|_2^2/\pi^2$ | $J[h_n]/\pi^2$ |
|---|---:|---:|---:|
| 1 | 1/2 | 1/2 | 1/4 |
| 10 | 1/200 | 1/2 | 1/4 |
| 100 | 1/20000 | 1/2 | 1/4 |

## 정확한 나머지와 고주파 반례 실행하기

다음 표준 라이브러리 코드는 α=0에서 에너지를 π²의 유리수 배수로 표현한다. ε=10⁻²⁰에서도 반올림된 binary64 에너지를 빼지 않고 나머지를 검산한다. 마지막 세 행은 별도의 고주파 함수족이다. 위 실습의 조절기는 원래 두 mode를 그대로 사용한다.

```python
from fractions import Fraction as F

# alpha=0; energies and H1 seminorms are divided by pi**2.
a, h, eps = (F(1), F(1, 3)), (F(1, 2), F(-1, 4)), F(1, 10**20)
energy = lambda v: sum(F(k*k, 4)*x*x for k, x in enumerate(v, 1))
first = sum(F(k*k, 2)*x*z for k, (x, z) in enumerate(zip(a, h), 1))
second = sum(F(k*k, 2)*z*z for k, z in enumerate(h, 1))
changed = tuple(x+eps*z for x, z in zip(a, h))
remainder = energy(changed)-energy(a)-eps*first
assert remainder == eps*eps*second/2
assert remainder > 0
print("exact quadratic remainder / pi^2:", remainder)
for n in (1, 10, 100):
    l2_squared, h1_squared_over_pi2 = F(1, 2*n*n), F(1, 2)
    energy_over_pi2 = F(1, 4)
    print(n, l2_squared, h1_squared_over_pi2, energy_over_pi2)
```

심화 학습에서는 충분히 정칙인 개별 y에서 L2 기울기를 표현하는 일과, L2 위상에서 에너지 전체가 미분 가능한지를 구별하자. 경계항이 사라지고 −y″+αy가 L2에 속하면 부분적분으로 그런 표현을 얻을 수 있다. 그 표현이 고주파 반례를 없애지는 않는다.

## 직접 풀기와 풀이

a=c=1,b=d=α=0,ε=0.1이면 일차·이차 변분 모두 π²/2다. 유한 변화는 0.0525π²≈0.518154다. c=0,d=1로 바꾸면 일차항은 0이지만 변화는 0.01π²≈0.098696이다. 직교성으로 이유를 설명하고, ε=0에서 변화 0과 도함수 값을 구별하자. 이 방식은 경로·PDE 해·확률분포를 학습 변수로 다룰 때의 출발점이다.

더 읽기: [MIT의 변분법 강의](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
