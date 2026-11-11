## 정상성보다 가능성이 먼저입니다

R 전체의 미분 가능한 볼록 최적화에서는 gradient 0이 최소를 인증합니다. 제약이 있으면 허용 방향이 달라집니다. 무제약 gradient가 0이어도 가능집합 밖의 점은 제약 최적해가 아닙니다.

f(x)=(x−a)²/2를 비어 있지 않은 닫힌 C=[l,u]에서 최소화합니다. l≤0≤u라 항상 가능점이 있습니다. 강볼록성 때문에 구간이 {0}으로 줄어도 해는 유일합니다.

## 올바른 일차 조건 유도

미분 가능한 볼록 f와 볼록 C에서 x*의 최적성은 가능성과 다음 조건에 동치입니다.

$$
f'(x^*)(y-x^*)\ge0\quad\text{for every }y\in C.
$$

필요성은 가능한 y 쪽으로 조금 움직인 뒤 방향 미분을 취해 얻고, 충분성은 f(y)≥f(x*)+f'(x*)(y−x*)에서 얻습니다. 비볼록 stationary point에는 이 결론을 적용할 수 없습니다.

1차원에서는 g=f'(x)≥0이면 y=l에서, g<0이면 y=u에서 곱이 최소입니다. 그 값이 음수이면 실제 가능한 하강 방향이 있습니다. 불가능 후보에는 이 인증값을 주지 않습니다.

## Normal Cone과 경계의 비영 Gradient

바깥 normal cone은 N_C(x)={n:n(y−x)≤0 for all y∈C}이며 최적성은 다음과 같습니다.

$$
0\in f'(x)+N_C(x).
$$

비퇴화 구간 내부는 {0}, 하한은 (−∞,0], 상한은 [0,∞)이고 singleton에서는 R 전체입니다. C 밖에서는 공집합입니다. 따라서 필요한 normal −f'(x)가 경계의 비영 gradient를 상쇄할 수 있습니다.

## 잘린 이차 함수 손계산

해는 x*=clip(a,l,u)입니다. a=2,C=[−1,1]이면 최적점은 1이고 gradient는 −1, 필요한 normal +1은 상한의 cone 안입니다. 가능한 변위 y−1은 비양수라 gradient −1과의 곱은 비음입니다.

파랑은 무제약 목적함수, 주황은 가능한 부분입니다. 후보와 최적점은 별도로 표시합니다. 후보를 x=2로 두면 gradient는 0이어도 구간 밖이므로 가능한 목적 초과값 인증은 없습니다.

## 사영 경사의 고정점과 감소 증명

비어 있지 않은 닫힌 볼록 C에서 q=P_C(z)의 필요충분조건은 모든 y∈C에 대해 ⟨z−q,y−q⟩≤0입니다. z=x−η∇f(x)를 대입하면 다음 동치를 얻습니다.

$$
x=P_C(x-\eta\nabla f(x))
\quad\Longleftrightarrow\quad
\langle\nabla f(x),y-x\rangle\ge0\ (y\in C),\qquad\eta>0.
$$

f가 미분 가능한 볼록함수이면 전역 최적성을 인증합니다. 관련 볼록 정의역에서 gradient가 L-Lipschitz라고 하고 d=x_next−x로 둡니다. 사영 조건은 ∇f(x)·d≤−‖d‖²/η입니다. Descent lemma와 합치면 f(x_next)≤f(x)−(1/η−L/2)‖d‖²이고 η≤1/L에서는 최소 ‖d‖²/(2η)만큼 감소합니다. 가능성과 단계 크기 가정이 모두 명제의 일부입니다.

정확분수 코드는 삼각형의 세 변을 열거해 2차원 이차함수를 사영합니다. 구간 실습의 별도 확장입니다. 경계에서는 비영 gradient가 normal과 균형을 이룰 수 있어 gradient만 보는 종료 조건이 실제 최적점을 거부할 수 있습니다.

| 조건 | 보장하는 것 |
|---|---|
| 가능 고정점, 볼록·미분 가능한 f | 전역 최적해 |
| 가능 사영 단계, η≤1/L | 목적 감소 |
| 반올림 높이를 뺀 작은 차이 | 그 자체로 인증 아님 |

구간 수치 표는 이제 최적점에서의 변위로 목적 초과값을 계산합니다. a=1,C=[−1,0],x=−10⁻²⁰이면 두 목적 높이는 모두 1/2로 반올림되지만 실제 초과값은 약 10⁻²⁰입니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
c=[F(4,5),F(3,5)]; eta=F(1,2)
def norm2(v): return sum(x*x for x in v)
def clip(x): return max(F(0),min(F(1),x))
def project(z):
    t=clip((z[0]-z[1]+1)/2)
    candidates=[[F(0),clip(z[1])],[clip(z[0]),F(0)],[t,1-t]]
    if min(z)>=0 and sum(z)<=1: candidates.append(z)
    return min(candidates,key=lambda q:norm2([q[i]-z[i] for i in range(2)]))
def value(x): return norm2([x[i]-c[i] for i in range(2)])/2
x=[F(0),F(0)]
for k in range(8):
    nxt=project([x[i]-eta*(x[i]-c[i]) for i in range(2)])
    step=[nxt[i]-x[i] for i in range(2)]
    assert value(nxt) <= value(x)-norm2(step)/(2*eta)
    print(k,tuple(x),float(value(x)))
    x=nxt
```

## 직접 풀기와 풀이

a=0.5,C=[−1,1],후보 0이면 gradient −0.5, y=1에서 최소 곱 −0.5라 하강 방향을 찾습니다. 해는 0.5입니다. 두 경계를 0,a=2로 바꾸면 가능한 변위가 모두 0이라 gradient −2인 유일 가능점도 최적입니다.

## 연구와의 연결

사영 방법·제약 매개화·기하 최적화에서는 무제약 gradient만이 아니라 가능한 방향을 봅니다. 집합·미분 가능성·볼록 가정을 명시하고 잔차를 전역 인증으로 해석하세요. 강볼록성은 유일성·거리 상계를 추가하지만 일반 볼록성만으로 그 결론까지 얻지는 못합니다.

추가 읽기: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
