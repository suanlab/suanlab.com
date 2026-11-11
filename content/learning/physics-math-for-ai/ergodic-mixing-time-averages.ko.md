## 세 명제를 하나로 줄이지 않는다

정상 법칙은 전이에 대해 불변입니다. Mixing은 주변 분포가 불변 법칙으로 수렴하는 성질입니다. Ergodic 시간 평균은 궤적을 따라 평균하는 성질입니다. 관련은 있어도 한 단어로 합치면 중요한 경계 사례를 놓칩니다.

두 상태 chain에서 파랑은 실제 state1 주변 확률, 주황은 유일할 때의 정상 확률입니다. 표본 평균·그 정확한 기대값·유한 창 분산은 별도로 표시합니다.

## 두 번째 고유값과 주변 감쇠

a+b>0이면 pi1=a/(a+b)이고 다음입니다.

$$
p_n-\pi_1=(1-a-b)^n(p_0-\pi_1).
$$

두 번째 고유값 절댓값이 1보다 작으면 이 모형의 분포가 수렴합니다. Reducible chain도 유일한 attracting closed class를 가질 수 있어 기약성과 이 수렴 flag는 같지 않습니다. a=b=0이면 모든 초기 법칙이 정상이라 유일성이 없습니다.

## 주기적 반례도 시간 평균은 좋을 수 있다

a=b=1에서는 결정론적으로 교대하여 고정 초기 상태의 주변 분포는 mixing하지 않습니다. 그러나 짝수 길이 창의 평균은 정확히 1/2입니다. 기약·주기 chain도 시간 평균 법칙을 가질 수 있으므로 ergodicity 용어의 관례를 명시하세요.

a=b=0에서는 한 경로가 처음 선택한 상태에 영원히 남습니다. 시간 평균은 0 또는 1이지 p0 자체가 아닙니다. 실현 간 분산은 창 길이와 무관하게 p0(1-p0)입니다.

## 상쇄 없이 유한 창 분산 계산

조건부 기대가 기울기 lambda=1-a-b의 affine 식입니다. Centered 전이는 martingale innovation을 주며 조건부 분산은 현재 상태로 결정됩니다. 평균을 초기 centered 상태와 직교 innovation으로 전개하면 다음처럼 비음수 합입니다.

$$
\operatorname{Var}(\overline X)=w_0^2\operatorname{Var}(X_m)
+\sum_j w_j^2 E[\operatorname{Var}(X_{j+1}\mid X_j)].
$$

가중치는 유한 등비 합을 표본 수로 나눈 것입니다. 주기 경계에서 가까운 이차 moment를 빼는 상쇄를 피합니다. Burn-in이 모든 관측을 없애면 평균·분산을 0으로 만들지 않고 비워 둡니다.

## 상관이 평균의 유효 정보를 바꾼다

0<π₁<1의 정상 두 상태 법칙에서 centered indicator f(X)=X−π₁는 고유값 λ=1−a−b의 전이 고유함수다. 반복 조건화로 Cov(X₀,Xₖ)=π₁(1−π₁)λᵏ다. M개 평균의 모든 공분산 쌍을 더하면

$$
\operatorname{Var}(\overline X_M)=\frac{\pi_1(1-\pi_1)}{M^2}
\left[M+2\sum_{k=1}^{M-1}(M-k)\lambda^k\right].
$$

|λ|<1에서 M을 곱하고 수렴 등비 급수를 더하면 극한 분산 인자는 τ=(1+λ)/(1−λ)다. 양의 상관은 불확실성을 키우고 음의 상관은 iid 기준보다 줄일 수 있다. M/τ는 분산으로 환산한 유효 표본 수이며 독립 관측의 개수가 아니고 M보다 클 수 있다. 주기 끝점 λ=−1은 별도의 유한 창 분석이 필요하다.

| λ | 극한 분산 인자 τ |
|---|---:|
| −1/2 | 1/3 |
| 0 | 1 |
| 1/2 | 3 |

```python
from fractions import Fraction as F
for lam in (F(-1,2), F(0), F(1,2)):
    tau = (1+lam)/(1-lam)
    for M in (10, 40):
        variance = F(1,4*M*M)*(M+2*sum((M-k)*lam**k for k in range(1,M)))
        # Independent stationary innovation weights.
        geometric = lambda n: sum(lam**j for j in range(n))
        innovation = (1-lam*lam)/4
        other = F(1,4)*(geometric(M)/M)**2
        other += sum(innovation*(geometric(M-j)/M)**2 for j in range(1,M))
        assert variance==other and variance>=0
        print(lam, M, '4 M Var:', 4*M*variance, 'limit:', tau)
```

유리수 공분산 계산은 실제 비음수 innovation 공식을 독립 보완한다. 실제 mixing flag도 a+b>0이면서 결정론 교대 a=b=1이 아닌 구조적 조건으로 수정했다. 작은 양의 전이에서 표시 고유값이 1로 반올림되어도 수학적으로는 mixing할 수 있고 시간 규모가 표시 trace보다 훨씬 길다. Flag가 40-step의 수치 평형을 보장하는 것은 아니다.

## 직접 풀기: 두 수렴 실패 비교

a=b=1,p0=1,index0..19의 평균·분산은 1/2와 0이지만 마지막 주변 분포는 pi와 다릅니다. a=b=0,p0=0.3에서는 모든 비어 있지 않은 창의 평균 분산이 0.21입니다. 얼어 있는 한 궤적을 더 길게 보아도 초기 class 불확실성이 사라지지 않는 이유를 설명하세요.

더 읽기: [MIT 대학원 확률과정 노트](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/).
