## 유수는 극의 차수가 아니다

고립 특이점 a 주변 Laurent 전개에서 유수는 (z−a)⁻¹의 계수입니다.

$$
f(z)=\sum_{n=-\infty}^{\infty}c_n(z-a)^n,\qquad
\operatorname{Res}(f,a)=c_{-1}.
$$

1/(z−a)는 유수 1인 단순극입니다. 1/(z−a)²는 2차 극이지만 유수는 0입니다. 극의 차수를 그대로 적분에 곱하면 틀립니다. 실습의 차수 m 컨트롤로 이 차이를 확인할 수 있습니다.

## 감김 수와 유수 정리

경로가 a를 지나지 않을 때 감김 수는

$$
\operatorname{Ind}(\gamma,a)=\frac1{2\pi i}\oint_\gamma\frac{dz}{z-a}.
$$

원점을 중심으로 한 반시계 원은 내부 점에 대해 1, 외부 점에 대해 0입니다. 경로가 한 번 이상 돌거나 방향이 바뀌면 그에 맞게 정수값이 달라집니다. 그림에서 단순히 가까워 보이는지로 판정하는 값은 아닙니다.

정칙성·고립 특이점·경로 조건을 갖춘 유수 정리는

$$
\oint_\gamma f(z)\,dz=2\pi i\sum_a\operatorname{Ind}(\gamma,a)\operatorname{Res}(f,a)
$$

입니다. 실습의 단순 원에서는 |a|<R인지 정확히 비교해 감김 수를 결정합니다.

## 내부·외부·경로 위 세 경우

a=0,R=1,m=1이면 유수 1, 감김 1이어서 2πi입니다. a=2로 옮기면 감김 0이어서 적분 0입니다. a=1이면 극이 경로 위에 있어 보통 경로 적분을 정의하지 않습니다.

유한 중점 격자가 극을 직접 찍지 않아 합이 유한하게 나오더라도 적분을 구한 것이 아닙니다. 실습은 먼저 경로 위의 극을 해석적으로 검사하여 수치 구적을 거부합니다. Cauchy 주값·작은 우회 경로 등은 별도의 규약이며 여기서 몰래 적용하지 않습니다.

## 고차 극을 직접 이해하기

m≥2에서 f=(z−a)⁻ᵐ은 특이점을 뺀 영역에서 원시함수 (z−a)¹⁻ᵐ/(1−m)을 가집니다. 따라서 극을 지나지 않는 닫힌 경로 적분은 0입니다. 유수 0인 계산과 일치합니다. 원시함수 자체가 특이점에서 정의되지 않는다는 사실은 닫힌 경로 적분 0과 모순되지 않습니다.

## 수치 구적의 조건

극이 경로에 가까워지면 함수가 급격히 변하여 고정 패널 구적의 오차가 커질 수 있습니다. 실습은 해석값, 수치값, 그 차이, 극과 원 사이의 거리를 분리합니다. 작은 수치 오차 하나를 정리의 증명으로 읽지 말고, 큰 오차가 정리의 실패인지 수치 해상도 문제인지 확인하세요.

빈 원은 입력 평면에서 제외된 극의 위치이며 그 점의 함수값을 그린 것이 아닙니다. 원의 방향은 반시계로 고정합니다.

## 극 근처 중점 구적 오차를 계산하기

R>0,a≥0,ρ=a/R≠1인 단순극을 생각하세요. 반시계 적분을 I/(2πi)로 정규화하면 ρ<1에서 1,ρ>1에서 0입니다. 실습의 중점 노드 w_k=exp(2πi(k+1/2)/N)를 쓰면 정규화한 유한 구적은

$$
Q_N(\rho)=\frac1N\sum_{k=0}^{N-1}\frac{w_k}{w_k-\rho}
=\frac1{1+\rho^N}.
$$

입니다. N은 양의 정수입니다. ρ<1에서 w/(w−ρ)=Σⱼ≥₀ρʲw⁻ʲ로 전개하세요. w⁻ʲ의 평균은 j가 N의 배수일 때만 남고 그 값은 $(-1)^{j/N}$입니다. 따라서 1−ρᴺ+ρ²ᴺ−⋯를 얻습니다. ρ>1에서는 −Σⱼ≥₁(w/ρ)ʲ를 사용하면 같은 유리식을 얻습니다. 각각의 정의역에서 수렴하는 급수로 유도한 것입니다.

정규화한 절대 오차는 내부에서 ρᴺ/(1+ρᴺ), 외부에서 1/(1+ρᴺ)입니다. 이는 적분 오차를 2π로 나눈 값이며 실습에 표시되는 적분 자체의 절대 오차와 구분해야 합니다.

| N | 정확한 목표가 1인 Q_N(0.99) | 내부 정규화 오차 | 정확한 목표가 0인 Q_N(1.01) |
|---|---|---|---|
| 32 | 0.579717 | 0.420283 | 0.421063 |
| 128 | 0.783545 | 0.216455 | 0.218634 |
| 512 | 0.994210 | 0.005790 | 0.006093 |

ρ=1에서도 중점 노드가 1을 직접 찍지 않으므로 유한 합은 연속적으로 1/2이 됩니다. 하지만 그 위치의 보통 경로 적분은 정의되지 않습니다. 실제 실습은 구적 전에 이 극을 거부합니다. 유한 합으로 정의역 검사를 대신할 수는 없습니다.

## 극의 거리와 함께 해상도 선택하기

고정된 ρ>0,ρ≠1에서 정규화 오차 목표 ε∈(0,1/2)을 만족시키려면

$$
N\ge\frac{\log((1-\varepsilon)/\varepsilon)}{|\log\rho|}.
$$

이면 됩니다. ρ=0에서는 중점 구적이 정확합니다. ρ가 1에 가까우면 |log ρ|가 작아져 더 많은 노드가 필요합니다. 극도 함께 경로에 접근한다면 균등 수렴을 말할 수 없습니다. ρ_N=exp(−c/N),c>0이면 모든 N에서 오차가 1/(1+exp c)로 남습니다.

대각 행렬 A=diag(0.99,1.01)의 각 성분에 유수를 적용하면 단위 원의 resolvent 사영은

$$
P=\frac1{2\pi i}\oint_{|z|=1}(zI-A)^{-1}\,dz
=\operatorname{diag}(1,0).
$$

입니다. 중점 구적에서는 P_N=diag(Q_N(0.99),Q_N(1.01))을 얻습니다. 성분은 스펙트럼 선택에 가까워지지만 유한 N에서는 P_N²=P_N을 만족하지 않을 수 있습니다. 이 대각 예제는 경로 적분 기반 스펙트럼 필터를 보여주며 일반적인 비정규 행렬에는 조건수 분석이 더 필요합니다.

## Python으로 구적과 필터 재현하기

다음 코드는 유리식과 별도로 중점 합을 계산합니다. 정확한 분수로 표와 사영의 멱등성 오차를 구하고 수치 합으로 경로 매개화를 검산합니다. 극이 경로에 접근하는 경우와 오차 예산 식도 확인합니다. 경계의 원시 유한 합은 유효한 적분과 따로 표시합니다.

```python
from cmath import exp
from fractions import Fraction as Q
from math import pi, ceil, log

def midpoint(rho, n):
    total = 0j
    for k in range(n):
        w = exp(2j*pi*(k+0.5)/n)
        total += w/(w-rho)
    return total/n

for n in [32, 128, 512]:
    inside = 1/(1+Q(99, 100)**n)
    outside = 1/(1+Q(101, 100)**n)
    for rho, expected in [(Q(99, 100), inside), (Q(101, 100), outside)]:
        assert abs(midpoint(float(rho), n) - float(expected)) < 2e-11
    assert 0 < inside < 1 and 0 < outside < 1
    defect = max(abs(inside*inside-inside), abs(outside*outside-outside))
    assert defect > 0
    print(n, float(inside), float(1-inside), float(outside), float(defect))

for n in [32, 128, 512]:
    assert abs(midpoint(exp(-1/n).real, n) - 1/(1+exp(-1).real)) < 2e-11

# A finite raw sum at the pole is NOT an ordinary contour integral.
assert abs(midpoint(1.0, 512) - 0.5) < 2e-11

epsilon = Q(1, 10000)
for rho in [Q(99, 100), Q(101, 100)]:
    n = ceil(log(float((1-epsilon)/epsilon))/abs(log(float(rho))))
    estimate = 1/(1+rho**n)
    error = 1-estimate if rho < 1 else estimate
    assert error <= epsilon
    print('tolerance', rho, n, float(error))
```

출처 연결: [Trefethen과 Weideman의 The Exponentially Convergent Trapezoidal Rule](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2014_149.pdf) §3과 §18에서 주기적 aliasing과 resolvent 구적을 확인했습니다. 이동한 노드의 식·표·대각 필터는 직접 유도했으며 원전의 다른 응용과 실험 전체를 재현한 것은 아닙니다.

## 확인 문제와 답

1. m=3인 순수 극의 유수는? **0**입니다.
2. 외부 단순극의 적분은? **0**입니다.
3. 극이 경로 위일 때 “오차 0”을 표시해도 되나요? **아니요. 기준 적분 자체가 정의되지 않습니다.**

물리·신호 계산에서 응답 극은 안정성·공진 분석과 연결되지만, 동역학 안정성 결론에는 변환 규약과 시간 영역의 가정이 별도로 필요합니다.

추가 읽기: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). 공식 강좌 범위를 확인해 보충 자료로 연결합니다. 본문의 작은 계산과 예제는 직접 구성했습니다.
