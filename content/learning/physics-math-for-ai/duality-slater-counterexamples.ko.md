## 하한과 값의 등호 정리는 다릅니다

가능한 부등식 승수의 Lagrangian infimum은 모든 primal 가능값의 하한입니다. 이를 최대화하면 d*≤p*라는 약 쌍대성입니다. 강 쌍대성은 최적값의 등호이며 유한 승수에서 supremum을 달성하는지는 별도입니다.

Slater는 유한 최적값의 볼록 문제에 대한 충분 qualification입니다. 적절한 relative interior의 가능점에서 비선형 부등식을 엄격히 만족하고 affine 등식 조건을 갖춥니다. 충분조건이 실패했다는 사실만으로 값의 등호 여부는 정해지지 않습니다.

## Strict 가능성과 달성 승수

첫 모드는 min (x−2)²/2, x≤1입니다. x=0은 strict 가능하고 primal 최적값은 x=1에서 0.5입니다.

$$
d(\lambda)=\lambda-\lambda^2/2,\quad\lambda\ge0,
\qquad\lambda^*=1,\quad d^*=0.5.
$$

dual 곡선은 primal 기준 높이에 닿습니다. 이 정상 예제에서는 witness 크기를 사용하지 않습니다.

## 값은 같지만 유한 최적 승수가 없는 경우

min x, x²≤0은 가능점 0뿐이라 p*=0이며 strict 가능점이 없습니다. λ>0에서 x+λx²를 최소화하면 다음과 같습니다.

$$
x_\lambda=-1/(2\lambda),\qquad d(\lambda)=-1/(4\lambda).
$$

λ=0의 infimum은 x=−M으로 −∞입니다. 양수 λ에서는 λ→∞에서만 d→0이라 d*=p*=0이지만 유한 최적 승수는 없습니다. 최적점의 KKT 정상성은 1+2λ·0=0이어야 하므로 불가능합니다. 볼록성만으로 승수 존재를 얻지는 못했습니다.

## 실제 양의 간극

세 번째는 Boyd·Vandenberghe 연습문제 5.21을 명시해 사용합니다.

$$
\min_{x\in\mathbb R,\ y>0} e^{-x}
\quad\text{subject to}\quad x^2/y\le0.
$$

perspective quadratic x²/y는 y>0에서 볼록·비음입니다. 가능성은 x=0을 강제해 primal 값 1을 달성합니다. 모든 유한 λ≥0에서 L은 비음이고 x=M,y=M⁴이면 e⁻ᴹ+λ/M²→0이므로 d(λ)=0, 최적값 간극은 1입니다.

이 sequence는 함수 정의역 안이지만 primal 불가능점입니다. 더 좋은 가능 해가 아니라 무제약 L infimum의 witness입니다. 여기서는 dual supremum은 달성하지만 내부 L infimum은 달성하지 않습니다. 세 달성 주장을 구별하세요.

## Perturbation 기하로 두 실패 설명하기

![두 볼록 perturbation 값 함수로 연속이지만 유한 지지 기울기가 없는 경우와 끝점의 위쪽 점프가 양의 쌍대 간극을 만드는 경우를 비교합니다.](/assets/learning/convex-perturbation-ko.svg)

Affine 등식을 제거한 뒤 부등식 경계를 0에서 u로 완화한 infimum을 p(u)로 정의합니다. 가능 witness를 섞으면 p의 볼록성이 나옵니다. 이 증명에서는 공통 볼록 정의역의 relative interior에 있는 유한 점이 모든 부등식을 엄격히 만족하고 p(0)이 유한한 Slater 조건을 가정합니다. 그러면 0은 유효 perturbation 정의역의 내부이고 그곳에 지지 subgradient s가 있습니다. 완화에 대한 단조성으로 s≤0이며 λ=−s≥0으로 두면 목적 정의역의 모든 x에 대해 다음과 같습니다.

$$
p(u)\ge p(0)-\lambda^Tu,\qquad
f(x)+\lambda^Tg(x)\ge p(0).
$$

두 번째 식은 u=g(x)를 선택해 얻습니다. Lagrangian infimum을 취하고 약 쌍대성과 합치면 값의 등호와 달성 승수를 얻습니다. 이 지지 논증이 qualification의 이유이며 수치 dual 그림이 대신할 수는 없습니다.

min x, x²≤u의 값은 u≥0에서 −√u입니다. 원점에서 연속이지만 유한 지지 기울기가 없어 승수 미달성을 설명합니다. Perspective 예제에서는 p(0)=1, 모든 u>0에서 p(u)=0, u<0에서 +∞입니다. 끝점의 위쪽 점프 때문에 epigraph가 닫히지 않고, 하반연속 hull의 원점 값 0이 이 예제의 dual 하한과 일치합니다.

| Perturbation 모형 | 원점 값 | 오른쪽 극한 | 유한 지지 승수 |
|---|---:|---:|---|
| x²≤u, min x | 0 | 0 | 없음 |
| x²/y≤u, min exp(−x) | 1 | 0 | Primal 값을 인증하는 지지 없음 |

두 번째에서 u>0이면 y=M²/u로 어떤 x=M도 가능하고 목적을 0으로 보냅니다. 이는 교란된 문제의 가능점이며 u=0에서는 가능하지 않습니다. 원래 문제에 더 좋은 가능 해가 있다는 주장과 구별해야 합니다.

## Python으로 계산 재현하기

```python
from math import exp,sqrt
for u in [1,.1,.01,.0001]:
    print('u:',u,'first value:',-sqrt(u))
    for M in [2,4,8,16]:
        y=M*M/u
        assert abs(M*M/y-u) < 1e-14
    print('second feasible objective at M=16:',exp(-16))
# The analytic sequence M -> infinity proves the infimum, not this finite list.
```

## 직접 풀기와 해석

세 번째 모드에서 M을 4에서 16으로 늘리면 witness L은 0으로 내려가고 primal 값은 1입니다. 이것이 약 쌍대성 반례가 아닌 이유를 설명하세요. 값의 간극은 0이지만 승수 달성이 없는 두 번째와 비교하세요.

AI 최적화 인증에는 primal·dual 정의역, 볼록성, qualification을 명시해야 합니다. 유한 그림만으로 무한 영역의 infimum을 확정할 수 없으며 명시한 sequence와 하한 논증이 그 근거입니다.

추가 읽기: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
