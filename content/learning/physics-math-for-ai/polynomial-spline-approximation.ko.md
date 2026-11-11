## 표현할 수 있는 것과 선택한 근사 방식

[0,1]의 f(x)=x²를 근사해 봅시다. 이 함수는 2차 다항식 자체이므로 최선의 2차 다항식 오차는 0입니다. 하지만 특정한 근사 절차가 항상 최선의 계수를 고르는 것은 아닙니다. 이번 실험은 Bernstein 연산자와 균등 매듭 선형 spline이라는 두 구체적인 절차를 비교합니다.

Bernstein의 n은 기저의 명목 차수입니다. 이 예제에서는 결과가 이차식으로 단순화되어도 n+1개 표본 가중치를 사용하는 연산자의 차수 n이라는 이름을 유지합니다.

## Bernstein 가중치를 확률로 이해하기

Bₙf는 표본 f(k/n)에 이항분포 가중치를 붙입니다. K~Binomial(n,x)라면 Bₙf(x)=E[f(K/n)]입니다. 제곱 함수에는 E[Z²]=(EZ)²+Var(Z)를 적용합니다.

$$
\begin{aligned}
B_nf(x)&=\sum_{k=0}^n f(k/n){n\choose k}x^k(1-x)^{n-k},\\
B_n(x^2)&=x^2+\frac{x(1-x)}n.
\end{aligned}
$$

오차는 항상 위쪽이고 중앙에서 1/(4n)입니다. n=4이면 1/16, n=8이면 1/32입니다. 이는 Bernstein 절차의 이 대상에 대한 오차이며 다항식 공간의 최선 오차가 아닙니다.

## 국소 선형 spline의 다른 속도

매듭 xⱼ=j/n 사이에서는 두 끝의 함수값을 직선으로 잇습니다. 이차 함수는 볼록이므로 직선이 위에 놓이고 오차는 다음과 같습니다.

$$
S_nf(x)-x^2=(x-x_j)(x_{j+1}-x),\quad x\in[x_j,x_{j+1}].
$$

각 구간 중앙의 최대는 h²/4=1/(4n²)입니다. 네 구간이면 1/64, 여덟이면 1/256입니다. 파랑은 참조 함수, 주황은 선택한 절차의 결과이며 점은 현재 질의입니다. 매듭에서는 spline 오차가 0이어도 구간 사이의 최대 오차는 남습니다.

## 일반 함수에서 필요한 조건

연속 함수에 대한 근사 존재와 오차 속도는 다른 주장입니다. 예를 들어 선형 보간의 h²‖f''‖∞/8 경계에는 bounded인 2차 도함수가 필요합니다. 뾰족점이나 불연속이 있으면 같은 속도를 그대로 쓰지 않습니다. Bernstein의 일반 수렴 역시 연속성과 구간 조건을 사용합니다.

AI의 표현력 비교에서는 가설 공간, 계수 선택 알고리즘, 데이터, 평가 노름을 분리해야 합니다. 같은 n+1개 숫자를 쓰더라도 전역 기저와 국소 기저의 계산 구조·정칙성 가정·근사 오차가 다릅니다.

## 구간의 모든 연속 함수에 대해 수렴을 증명하기

f가 [0,1]에서 연속이고 M=‖f‖∞라 합시다. 연속성 계수 ω_f(δ)는 |u−v|≤δ인 점 쌍에서 |f(u)−f(v)|의 supremum입니다. 균등연속성 때문에 δ가 줄면 이 값도 0으로 갑니다. E|f(K/n)−f(x)|를 |K/n−x|≤δ인 사건과 나머지로 나눕니다. 가까운 사건에는 ω_f(δ), 먼 사건에는 2M과 Chebyshev 경계를 씁니다. Var(K/n)=x(1−x)/n≤1/(4n)이므로

$$
\|B_nf-f\|_\infty\le\omega_f(\delta)+\frac{M}{2n\delta^2}.
$$

먼저 δ를 골라 첫 항을 줄이고 이후 n을 키워 둘째 항을 줄이면 균등 수렴입니다. 따라서 이 구간에서 다항식의 조밀성도 얻습니다. 모든 연속 함수에 공통인 속도까지 준 것은 아닙니다. L-Lipschitz 함수라면 Cauchy–Schwarz로 L/(2√n)이라는 더 좋은 경계를 얻습니다. x²의 정확 오차 1/(4n)은 이보다 빠릅니다. 일반 경계가 특정한 매끄러운 대상에 날카로울 필요는 없습니다.

## 국소 spline 경계를 유도하고 같은 양끼리 비교하기

f∈C²[a,b]이고 I_hf가 선형 보간이면 내부 x의 나머지는 어떤 ξ∈(a,b)에 대해 f″(ξ)(x−a)(x−b)/2입니다. f−I_hf에서 (t−a)(t−b)의 적절한 배수를 빼 세 영점을 만든 후 Rolle 정리를 두 번 쓰면 얻습니다. 따라서

$$
\|f-I_hf\|_{L^\infty[a,b]}\le\frac{(b-a)^2}{8}\|f''\|_{L^\infty[a,b]}.
$$

표는 x²의 두 정확 오차와 L=2인 Lipschitz 경계 1/√n을 비교합니다. 마지막 열은 정리의 상계이며 실습에 구현한 세 번째 근사 방식이 아닙니다. 분수 코드는 Bernstein 가중치를 독립적으로 합하며 격자 오차 곡선으로 최대 오차를 추정하지 않습니다.

| n | Bernstein 정확 오차 | 선형 spline 정확 오차 | Lipschitz 상계 |
|---|---|---|---|
| 2 | .125 | .0625 | .707107 |
| 4 | .0625 | .015625 | .5 |
| 8 | .03125 | .00390625 | .353553 |
| 16 | .015625 | .0009765625 | .25 |

**확인 문제:** 연속성만으로 spline의 O(h²)를 주장할 수 있나요? **해설:** 아닙니다. 연속성 계수로 균등 수렴을 보일 수 있지만 위 2차 속도에는 명시한 2차 도함수 제어가 필요합니다.

실습의 질의 오차는 두 근사값을 빼는 대신 위의 양의 오차 식으로 계산합니다. n=16,x=1−2⁻⁵³에서는 두 방식 모두 약 6.94×10⁻¹⁸의 양의 오차를 갖지만 보간값에서 x²를 빼면 0으로 사라질 수 있습니다. 표현 범위 아래의 양의 오차는 —·안내로 표시하고 그 질의점은 보류합니다. 끝점과 해당 spline 매듭의 정확한 0은 유지합니다. 아래 코드는 binary 입력값의 정확분수로 이 차이를 검산합니다.

```python
from fractions import Fraction as F
from math import comb, sqrt

for n in [2,4,8,16]:
    # Direct Bernstein sum at the midpoint, not the closed error formula.
    value=sum((F(k,n)**2*comb(n,k)*F(1,2)**n for k in range(n+1)),F(0))
    bernstein_error=value-F(1,4)
    spline_midpoint=(F(0)**2+F(1,n)**2)/2-F(1,2*n)**2
    assert bernstein_error==F(1,4*n)
    assert spline_midpoint==F(1,4*n*n)
    # f(x)=x^2 has Lipschitz constant 2 on [0,1].
    lipschitz_bound=1/sqrt(n)
    print(n,float(bernstein_error),float(spline_midpoint),round(lipschitz_bound,6))

x=F.from_float(1-2**-53)
errors=[x*(1-x)/16,(x-F(15,16))*(1-x)]
assert all(error>0 for error in errors)
print('Exact query errors near one:',*[float(error) for error in errors])
```

## 확인 문제와 해설

1. n=1이면 두 방식은? **둘 다 x입니다.** 끝점 (0,0),(1,1)을 연결합니다.
2. Bernstein n=4가 오차를 가지면 x²를 4차 공간에서 표현할 수 없나요? **아니요.** 특정 절차가 최선 해를 고르지 않은 것입니다.
3. spline 매듭만 평가하면 무엇을 놓치나요? **매듭 사이 오차입니다.** 이 예에서는 모든 매듭에서 정확해도 중앙 오차가 있습니다.

추가 읽기: [MIT 18.102, Functional Approximation Background](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
