## 무한 함수 탐색이 유한 계수 탐색이 되는 조건

Representer 정리는 경험 손실이 유한한 함수 평가값만 보고 정규화가 RKHS 노름의 적절한 증가 함수일 때 적용됩니다. 최소해가 존재한다면 관측점 kernel 절편들의 span에서 해를 선택할 수 있습니다. 벌점이 노름에 대해 엄격히 증가하면 최소해의 불필요한 직교 성분은 사라집니다. 존재·유일성·계수 유일성까지 가정 없이 자동 보장하는 정리는 아닙니다.

$$
f=f_\parallel+f_\perp,\qquad f_\perp\perp\operatorname{span}\{k_{x_i}\}.
$$

재생 성질에서 f⊥(xᵢ)=0이므로 관측 손실은 그대로입니다. 반면 제곱 노름은 두 직교 성분의 제곱 노름 합입니다. 손실에 도움이 없는 직교 방향을 제거하면 벌점이 줄어듭니다.

## 관측에 보이지 않는 방향을 만들기

실습은 특징 φ(x)=(1,x,x²), kernel 1+xz+x²z²를 사용합니다. 계수 공간은 유클리드 R³이며 관측은 (−1,1),(1,1)입니다. 단위 벡터 v=(1,0,−1)/√2는 두 관측 특징 모두와 직교합니다.

$$
v\cdot\varphi(x)=\frac{1-x^2}{\sqrt2},\qquad v\cdot\varphi(\pm1)=0.
$$

따라서 이 방향을 c만큼 더해도 훈련값은 바뀌지 않지만 중앙의 함수값은 c/√2만큼 바뀝니다. 관측값과 함수 전체가 같은 대상이 아님을 보여 줍니다.

## 정확한 최소해와 벌점 증가

목적함수는 평균 제곱 잔차 + λ‖w‖²이며 λ>0입니다. 대칭성과 정상 조건으로 최소해는 w=(t,0,t), t=1/(2+λ)입니다. 단위 직교 성분을 추가하면 노름 제곱은 2t²+c²입니다.

$$
J(w+cv)-J(w)=\lambda c^2.
$$

λ=1이면 t=1/3, 훈련 MSE는 1/9, 최소 목적함수는 1/3입니다. c=1을 추가하면 관측값은 같지만 목적함수는 4/3입니다. λ=.1이면 같은 직교 성분의 추가 비용은 .1입니다. 파랑은 최소해, 주황은 수정한 함수, 점은 실제 관측입니다.

## 일반화할 때 주의할 점

정리가 데이터 밖 예측의 정확도를 보장하지는 않습니다. 손실이 미분값·적분 관측을 사용하면 관측 범함수의 연속성과 대표를 다시 확인해야 합니다. Gram이 singular이면 동일한 함수를 다른 계수 α로 표현할 수도 있어 함수 유일성과 계수 유일성도 다릅니다.

이 실험은 유한 RKHS에서 정리의 직교 논증을 그대로 계산합니다. 실제 무한 차원 kernel 회귀에서도 같은 논증이 유한 관측 span으로 문제를 줄이지만, 수치 조건과 통계적 일반화는 별도 과제입니다.

## Representer 축소·존재·유일성을 분리하기

L:Rⁿ→[0,∞)가 하반연속이고 달성 가능한 평가 벡터 중 적어도 한 곳에서 유한하다고 합시다. λ>0에서 L(f(x₁),…,f(x_n))+λ‖f‖²를 최소화할 때 직교 사영으로 유한 차원인 닫힌 kernel 절편 span에 제한할 수 있습니다. L≥0이고 제곱 벌점이 coercive이므로 그 안의 최소화 열은 유계입니다. 수렴 부분열과 하반연속성으로 최소값이 달성됩니다. L이 볼록이면 엄격 볼록인 Hilbert 제곱 노름 덕분에 최소 함수도 유일합니다.

이는 representer 명제에 존재 논증을 더한 것입니다. 예를 들어 정규화 없는 상수족 t>0에서 관측값 0의 제곱 손실은 infimum 0이지만 최소해가 없습니다. 닫힘과 coercivity를 임의로 지울 수 없습니다. 벌점이 비감소이면 최소해가 있을 때 span 최소해를 고를 수 있고, 엄격 증가이면 모든 최소해의 직교 성분이 0입니다.

## 함수가 유일해도 계수는 여러 개일 수 있다

상수 kernel k=1과 관측 두 개를 쓰고 두 label을 모두 1로 둡니다. 합 손실과 λ=1/2에서 함수는 상수 t이며 목적함수는 2(t−1)²+λt²입니다. 미분하면 t=4/5, 목적함수 2/5입니다. 그러나 f=α₁k(·,x₁)+α₂k(·,x₂)는 α₁+α₂에만 의존합니다.

| 계수 (α₁,α₂) | 함수 t | 목적함수 | (K+λI)α=y의 해 |
|---|---|---|---|
| (4/5,0) | 4/5 | 2/5 | 아니요 |
| (0,4/5) | 4/5 | 2/5 | 아니요 |
| (2/5,2/5) | 4/5 | 2/5 | 예 |

K가 singular일 때 계수 정상 조건은

$$
K\big((K+\lambda I)\alpha-y\big)=0.
$$

가역 선형계 (K+λI)α=y를 풀면 유효한 표준 계수 하나를 선택합니다. 충분조건이지만 처음 두 행처럼 같은 최소 함수를 나타내는 모든 계수의 필요조건은 아닙니다. 코드는 세 경우를 정확히 확인합니다. 이 상수 kernel 예제는 기존 세 특징·평균 손실 실습과 별도입니다.

**확인 문제:** λI를 더했는데 함수 표현은 왜 일대일이 아니죠? **해설:** 함수로 가는 사상 α↦∑α_i k_i에는 원래 Gram 영공간이 남습니다. 계수를 정하는 선형계의 선택 관례는 다른 사상입니다.

```python
from fractions import Fraction as F

lam=F(1,2)
t=F(2)/(2+lam)
coefficients=[(t,F(0)),(F(0),t),(t/2,t/2)]
for alpha in coefficients:
    fitted=sum(alpha)
    objective=2*(fitted-1)**2+lam*fitted**2
    assert fitted==F(4,5) and objective==F(2,5)
    # K alpha depends only on the sum; K is the all-ones matrix.
    stationarity=[fitted-1+lam*a for a in alpha]
    assert sum(stationarity)==0  # K(K+lambda I)alpha=K y.
    canonical=all(v==0 for v in stationarity)
    print(tuple(str(x) for x in alpha),str(fitted),str(objective),canonical)
# For the separate unregularized class g_t=t, t>0, with zero labels,
# inf_t g_t(0)^2=0 is approached but not attained: the class is not closed.
```

## 확인 문제와 해설

1. c를 2배로 하면 추가 비용은? **4배입니다.** 제곱 노름 벌점입니다.
2. 직교 성분이 훈련에서 0이면 테스트에서도 항상 0인가요? **아니요.** 중앙에서의 변화가 반례입니다.
3. λ=0에서도 직교 성분을 반드시 없애나요? **아니요.** 손실이 같고 벌점이 없으면 여러 최소해가 가능합니다. 현재 조작 범위는 λ>0입니다.

추가 읽기: [Schölkopf, Herbrich and Smola, A Generalized Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf).
