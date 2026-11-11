## 유한 점 집합의 거리를 함께 보존하기

Johnson–Lindenstrauss의 핵심은 고정한 유한 점 집합의 모든 쌍 거리를 작은 상대 오차로 보존하는 선형 사상입니다. 보통 필요한 출력 차원은 점 개수의 로그와 허용 오차의 역제곱에 의존합니다. 여기서는 노름이 아니라 거리의 제곱에 (1±ε) 허용 범위를 적용합니다.

실습은 R¹²⁸의 네 점 0,e₁,e₂,모든 좌표가 1/√128인 점을 사용합니다. 전체 점은 투영 행렬을 정하기 전에 고정되어 있습니다. 여섯 쌍의 원래 거리 제곱과 투영 뒤 비율을 표에 기록합니다.

## 왜 1/√m로 나눌까

이상적 행렬 A의 각 원소가 독립 N(0,1/m)이면 고정 벡터 v에 대해 ||Av||²/||v||²은 χ²ₘ/m을 따릅니다. 따라서 기대 제곱 노름이 보존됩니다. 한 쌍의 평균 보존은 모든 쌍이 동시에 가까운 것과 다르므로 꼬리 경계가 필요합니다.

## 한 쌍에서 여섯 쌍으로

0<ε<1에서 Gaussian 제곱 노름의 꼬리는 2exp[−m(ε²/4−ε³/6)]로 상계할 수 있습니다. 여섯 쌍에 union bound를 적용하면 다음 값이며, 확률 표시에서는 1과의 최솟값을 씁니다.

$$
P(\text{some pair fails})\leq12\exp[-m(\varepsilon^2/4-\varepsilon^3/6)].
$$

실패확률을 .05 이하로 만드는 충분한 m은 ln(12/.05)을 괄호의 계수로 나눈 값의 올림입니다. 충분조건이므로 실제로 필요한 최소 차원은 아닙니다. 이 네 점은 작은 부분공간에 있으므로 일반 경계가 보수적일 수 있습니다.

## Seed의 실현과 확률 보장 구분

화면은 seed를 고정한 유사난수와 Box–Muller 변환으로 재현 가능한 행렬을 만듭니다. 이는 이상적인 독립 Gaussian 추출의 수학적 보장과 구분해야 합니다. seed를 바꾸면 실제 오차는 바뀌지만 같은 m,ε의 이론 상계는 같습니다. 한 seed가 실패하거나 성공해도 정리를 증명하거나 반박한 것은 아닙니다.

파랑 점은 쌍별 비율이고 주황 수평선은 1과 1±ε입니다. 연결선은 쌍 번호를 따라 읽도록 돕는 것이며 확률밀도 곡선이 아닙니다. 제곱 거리 오차와 원래 거리 오차의 ε를 같은 값으로 혼용하지 마세요.

## 적용 범위

이 보장은 미리 고정한 유한 점 집합에 대한 것입니다. 행렬을 본 뒤 악의적으로 점을 고르거나 전체 연속 공간의 모든 벡터를 저차원으로 등거리 보존한다는 주장은 아닙니다. 학습·검색 성능도 거리 보존만으로 자동 보장되지는 않습니다.

## χ² 꼬리의 상수를 직접 유도하기

이상적인 독립 Gaussian 투영에서 고정한 비영 벡터의 제곱 노름 비율을 Z/m, Z~χ²_m으로 씁니다. Gaussian 제곱의 지수 적분을 제곱 완성하면 E exp(tG²)=(1−2t)^(−1/2), t<1/2입니다. 독립 성분의 곱으로

$$
E e^{tZ}=(1-2t)^{-m/2},\qquad t<1/2.
$$

상단 사건 Z≥m(1+ε)에는 t>0의 Markov 부등식을 적용하고 t=ε/[2(1+ε)]에서 최소화합니다. 하단 사건 Z≤m(1−ε)에는 t<0를 사용하며 최적값은 −ε/[2(1−ε)]입니다. 0<ε<1에서 다음 두 비율을 얻습니다.

$$
a_+(\varepsilon)=\frac{\varepsilon-\ln(1+\varepsilon)}2,\qquad
 a_-(\varepsilon)=\frac{-\varepsilon-\ln(1-\varepsilon)}2,
$$

$$
P\left(\left|\frac Zm-1\right|\geq\varepsilon\right)
\leq e^{-m a_+}+e^{-m a_-}.
$$

두 a는 양수입니다. ln(1+ε)≤ε−ε²/2+ε³/3을 쓰면 a_+≥ε²/4−ε³/6이고 a_-도 이보다 작지 않습니다. 따라서 위의 단순한 양측 경계를 회복합니다. 자연로그를 쓰며 허용 오차는 제곱 거리에 대한 것입니다.

## 모든 쌍의 의존성은 union bound를 막지 않는다

투영 전에 고정한 N개 점에서 서로 다른 점의 쌍 수는 L≤N(N−1)/2입니다. 같은 행렬을 쓰므로 쌍별 오차가 독립일 필요는 없습니다. 각 쌍의 상계를 더하면 전체 실패확률은 L(e^(−ma_+)+e^(−ma_-)) 이하입니다.

네 점·여섯 쌍·목표 실패확률 .05에서 다음 정수 차원을 비교합니다. Two-tail은 이 두 지수의 합으로 된 상계를 .05 이하로 만드는 최소 정수이고, Coarse는 기존 단순 상계의 충분 차원입니다. 어느 쪽도 실제로 필요한 최소 투영 차원이라는 뜻은 아닙니다.

| ε | Two-tail | Coarse |
|---|---|---|
| 0.25 | 367 | 421 |
| 0.50 | 102 | 132 |
| 0.80 | 46 | 74 |

코드는 Gaussian MGF를 직접 수치 적분해 확인하고 정수 차원을 탐색합니다. 적분 검사는 근사 검산이며 유도한 확률 부등식을 대신하지 않습니다. 화면은 기존 단순 상계와 고정 seed를 계속 보여주고, 이 표는 별도 이론 비교입니다.

## 행렬을 본 뒤 고른 벡터에는 같은 보장을 쓸 수 없다

m<d인 선형 사상은 영 공간에 비영 벡터를 가집니다. 그 벡터를 행렬을 본 뒤 선택하면 투영된 노름은 0이므로 ε<1의 상대 오차 보존을 만족하지 않습니다. 유한 집합이 미리 고정되었다는 조건과 전체 R^d를 보존한다는 주장을 구분해야 합니다. 데이터에 맞추어 사상을 학습하거나 평가점을 선택한다면 추가적인 논증이 필요합니다.

```python
from math import exp, log1p, log, ceil, sqrt, pi, isclose

for eps in [.25, .5, .8]:
    upper_rate = (eps-log1p(eps))/2
    lower_rate = (-eps-log1p(-eps))/2
    coefficient = eps*eps/4-eps**3/6
    assert min(upper_rate, lower_rate) >= coefficient
    def union(m):
        return 6*(exp(-m*upper_rate)+exp(-m*lower_rate))
    m = 1
    while union(m) > .05:
        m += 1
    assert union(m-1) > .05
    coarse = ceil(log(12/.05)/coefficient)
    print("epsilon, two-tail sufficient m, coarse m:", eps, m, coarse)
    for t in [eps/(2*(1+eps)), -eps/(2*(1-eps))]:
        def f(x):
            return exp(-(.5-t)*x*x)/sqrt(2*pi)
        steps, lo, hi = 4000, -12, 12
        h = (hi-lo)/steps
        integral = f(lo)+f(hi)
        integral += sum((4 if j % 2 else 2)*f(lo+j*h) for j in range(1, steps))
        integral *= h/3
        assert isclose(integral, 1/sqrt(1-2*t), rel_tol=1e-10)
print("MGF integration checks passed; these do not measure seed-specific failure probabilities.")
```

## 연구에 적용하기 전에 묻기

1. 쌍별 오차가 의존하면 union bound가 무효인가요? **아니요. 각 사건의 확률 상한을 더할 수 있습니다.**
2. 표의 최소 정수는 무엇에 대한 최소인가요? **지정한 확률 상계를 만족하는 정수에 대한 것입니다.**
3. 재현 기록에는 무엇이 필요한가요? **고정 점 집합·분포·차원·제곱 오차·실패확률·seed와 이론의 구분을 적습니다.**

배경은 [Stanford CS 368, Lecture 4](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf)를 참고하세요. 여기서는 제곱 거리 허용 오차에 맞추어 최적 Chernoff 두 꼬리와 정수 표를 직접 계산했습니다.

## 확인 문제와 해설

1. 행렬 배율은? **표준 Gaussian 원소를 √m으로 나눕니다.**
2. 현재 실습의 점 쌍은 몇 개인가요? **6개입니다.**
3. 실패 상계 1은 반드시 실패한다는 뜻인가요? **아니요. 이 설정에서 경계가 자명하다는 뜻입니다.**

추가 읽기: [Stanford CS 368, Lecture 4 — Johnson–Lindenstrauss](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf).
