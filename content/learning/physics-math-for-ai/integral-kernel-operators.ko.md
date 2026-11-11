## 행렬 곱의 연속 버전

적분 연산자는 입력 함수 f(t)를 kernel K(x,t)로 가중해 적분하고 출력 함수 Tf(x)를 만듭니다. 유한 행렬의 열 번호가 연속 변수 t로 바뀐 것처럼 볼 수 있지만 적분 측도와 함수 공간을 반드시 지정해야 합니다.

$$
Tf(x)=\int_0^1K(x,t)f(t)\,dt.
$$

L2의 두 변수에서 제곱 적분 가능한 kernel은 Hilbert–Schmidt 연산자를 정의하고 bounded·compact입니다. 이는 충분조건이며 모든 bounded 적분 사상이 이 조건을 만족한다는 뜻은 아닙니다.

## 정확히 풀리는 rank-one kernel

실습은 g(x)=1+a x와 K(x,t)=g(x)g(t)를 사용합니다. 그러면 Tf=g⟨g,f⟩라 모든 출력이 g의 배수입니다. g는 0 함수가 아니므로 rank는 정확히 1입니다. T는 자기수반이고 ⟨Tf,f⟩=|⟨g,f⟩|²≥0입니다.

$$
Tg=\|g\|_2^2g,\qquad \lambda=\|g\|_2^2=1+a+a^2/3.
$$

g에 직교하는 함수는 모두 kernel에 있습니다. 유일한 비영 고유값 λ가 연산자 노름과 같고 Hilbert–Schmidt 노름도 λ입니다. 두 변수 적분 ∫∫K²가 (∫g²)²임을 직접 확인할 수 있습니다.

## 중점 근사와 정확한 사상을 분리하기

입력이 f=1+b t이면 계수 q=∫gf는 1+(a+b)/2+ab/3입니다. tᵢ=(i+.5)/N의 중점 규칙 qₙ을 사용하면 오차는 정확히 −ab/(12N²)입니다.

$$
T_Nf(x)=g(x)\frac1N\sum_i g(t_i)f(t_i).
$$

a=b=1, N=2에서 q=7/3, qₙ=37/16이고 차이 절댓값은 1/48입니다. N=8이면 1/768입니다. 파랑과 주황은 각각 정확한 Tf와 근사 Tₙf입니다. a=0 또는 b=0이면 integrand가 일차 이하라 중점 적분이 정확합니다.

## 수치 행렬에도 가중치를 넣어야 한다

중점 Nyström 행렬의 원소는 K(tᵢ,tⱼ)/N입니다. 1/N을 빼면 고유값의 크기가 잘못됩니다. 이 대칭 행렬의 가능한 비영 고유값은 ∑g(tᵢ)²/N이며 정확한 λ와 차이는 −a²/(12N²)입니다. 모든 표본의 g가 0이면 행렬은 rank 0입니다. 예를 들어 a=−2, N=1의 중점은 g(.5)=0이라 원래 연산자가 rank 1이어도 수치 행렬은 영행렬입니다. 해상도를 바꾸는 것은 이산 근사를 바꾸는 것이지 원래 연산자 자체를 바꾸는 일이 아닙니다.

AI의 kernel 방법·Gaussian process·neural operator에 이 관점이 연결됩니다. 일반 attention 가중치는 현재 kernel처럼 대칭·양의 준정부호라는 보장이 없으므로 같은 스펙트럼 결론을 그대로 적용하지 않습니다.

## Hilbert–Schmidt 조건이 bounded·compact를 주는 이유

실수 L²[0,1]에서 Cauchy–Schwarz를 먼저 t에, 다음 x 적분에 적용하면 ‖Tf‖₂≤‖K‖L²([0,1]²)‖f‖₂입니다. 두 변수 L² kernel은 직사각형 step 함수의 유한 합으로 근사할 수 있습니다. 각 항 a(x)b(t)의 적분 연산자는 유한 rank이고, kernel L² 오차가 연산자 노름 오차를 제어하므로 그 노름 극한인 T가 compact입니다. 모든 bounded 연산자를 이렇게 표현한다는 역명제는 아닙니다.

## 점 구적식과 L² 전체의 연산자를 구분하기

화면의 f=1+b t는 매끄러워 점 값이 정해져 있습니다. 따라서 표시한 중점 구적식은 이 함수 가족에서 유효합니다. 하지만 f(t_i)는 일반 L² 동치류에서 정의되지 않고, 연속 함수로 제한해도 L²에 대한 bounded 관측이 아닙니다. 그 구적식을 무조건 L² 전체에서 bounded인 T_N으로 간주하면 앞 쌍대 단원의 반례와 충돌합니다.

대신 P_N을 각 셀 평균을 쓰는 L² 직교 투영이라 두고 g_N=P_Ng, K_N(x,t)=g_N(x)g_N(t)를 사용하면 T_N^cell f=g_N⟨g_N,f⟩는 L² 전체에서 bounded입니다. 선형 g에서는 셀 평균이 중점 값과 같습니다. 정규직교 셀 기저에서 이 연산자의 행렬은 g(t_i)g(t_j)/N이라 Nyström 행렬과 일치합니다. 입력 벡터를 셀 평균으로 해석하고 출력도 step 함수로 재구성한 경우의 일치이며, 화면의 연속 출력 g(x)q_N과 모든 x에서 같은 함수라는 뜻은 아닙니다.

$$
\|T-T_N^{\mathrm{cell}}\|
\le(\|g\|_2+\|g_N\|_2)\|g-g_N\|_2,
\qquad \|g-g_N\|_2^2=\frac{a^2}{12N^2}.
$$

g_N과 e=g−g_N은 직교합니다. p=‖g_N‖, d=‖e‖이면 그 두 방향에서 T−T_N^cell의 행렬은 [[0,pd],[pd,d²]]입니다. 따라서 노름은 (d²+sqrt(d⁴+4p²d²))/2입니다. p=0인 rank-one 퇴화 경우에도 같은 값이 나옵니다.

| N, a=−2,b=2 | 셀 연산자 노름 오차 | 위 상계 | 매끄러운 입력의 계수 구적 오차 |
|---|---|---|---|
| 1 | .333333 | .333333 | .333333 |
| 2 | .191898 | .311004 | .083333 |
| 4 | .091773 | .164020 | .020833 |
| 8 | .044026 | .083007 | .005208 |

코드는 분수 셀 모멘트와 2×2 특성식을 대조합니다. 이 예제에서 연산자 오차는 O(1/N), 특정 매끄러운 입력의 구적 계수 오차는 O(1/N²)로 다른 양입니다. 더 작은 구적 오차 하나가 전체 연산자의 같은 수렴률을 증명하지 않습니다.

작은 a,b에서는 q_N−q의 직접 뺄셈도 반올림 잡음을 남깁니다. a=b=10⁻⁸,N=16의 실제 구적 오차는 약 3.26×10⁻²⁰인데 기존 뺄셈은 약 2.22×10⁻¹⁶이었습니다. 이제 계수 구적 오차는 정확한 bias −ab/(12N²)의 절댓값으로 계산합니다. 표의 두 계수가 같은 표시값으로 반올림돼도 이론상 bias는 양수일 수 있습니다. 표현 범위 아래에서는 —·안내, a=0 또는 b=0인 실제 0은 그대로 표시합니다.

**직접 풀기:** 중점 행렬과 같은 행렬을 얻었으면 원래 함수 공간에서의 연산자도 같나요? **해설:** 아닙니다. 입력 계수의 의미와 출력 재구성까지 정해야 합니다. 점 샘플과 셀 평균은 일반 L²에서 다른 관측입니다.

```python
from fractions import Fraction as F
from math import sqrt

# g=1-2x; g_N is its cell average, equal to its midpoint value.
mu=F(1,3)
for N in [1,2,4,8]:
    midpoints=[F(2*i+1,2*N) for i in range(N)]
    values=[1-2*t for t in midpoints]
    projected_norm_squared=sum(v*v/N for v in values)
    error_squared=F(1,3*N*N)
    assert projected_norm_squared+error_squared==mu
    # In the orthogonal directions g_N and g-g_N, T-T_N has this 2x2 form.
    cross=sqrt(float(projected_norm_squared*error_squared))
    d=float(error_squared)
    exact_operator_error=(d+sqrt(d*d+4*cross*cross))/2
    upper=(sqrt(float(mu))+sqrt(float(projected_norm_squared)))*sqrt(d)
    assert exact_operator_error<=upper+1e-12
    assert abs(cross*cross+d*exact_operator_error-exact_operator_error**2)<1e-12
    # For the separate smooth input f=1+2t, midpoint coefficient bias is +1/(3N^2).
    exact_coefficient=F(-1,3)
    midpoint=sum((1-2*t)*(1+2*t)/N for t in midpoints)
    assert midpoint-exact_coefficient==error_squared
    print('N, bounded cell-operator norm error, its upper bound, smooth-input quadrature bias:',
          N,round(exact_operator_error,6),round(upper,6),round(float(error_squared),6))
```


## 확인 문제와 해설

1. a=0의 연산자 노름은? **1입니다.** 상수 kernel의 적분 연산자입니다.
2. a=−1의 비영 고유값은? **1/3입니다.** g=1−x의 제곱 적분입니다.
3. 수치 행렬 rank가 1이면 함수 공간도 유한 차원인가요? **아니요.** 정의역 L2는 무한 차원이지만 이 연산자의 상만 1차원입니다.

추가 읽기: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
