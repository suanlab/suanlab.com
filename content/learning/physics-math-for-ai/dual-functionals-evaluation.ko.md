## 함수를 숫자로 읽는 연속 측정

선형 범함수는 벡터나 함수를 숫자로 보냅니다. 연속 쌍대 X*는 그중 bounded인 것들의 공간이며 노름은 단위 입력에서 출력 절댓값의 supremum입니다. 모든 대수적 선형 사상을 무조건 연속 쌍대에 넣지는 않습니다.

$$
\|\ell\|_{X^*}=\sup_{\|f\|_X\le1}|\ell(f)|.
$$

## Hilbert 공간의 Riesz 대표

실수 Hilbert 공간에서 모든 연속 선형 범함수는 유일한 g에 대해 ℓ(f)=⟨f,g⟩로 표현됩니다. L2[0,1]에서는 ℓg(f)=∫gf입니다. Cauchy–Schwarz가 |ℓg(f)|≤‖g‖₂‖f‖₂를 주고, g≠0이면 f=g/‖g‖₂를 선택해 상한을 달성하므로 쌍대 노름은 ‖g‖₂입니다.

$$
\ell_g(f)=\int_0^1gf,\qquad \|\ell_g\|=\|g\|_2.
$$

g=1+a t, f=1+b t이면 내적은 1+(a+b)/2+ab/3입니다. a=b=1에서는 값과 Cauchy–Schwarz 상한이 모두 7/3이며 쌍대 노름은 √(7/3)입니다. 값과 범함수의 노름은 서로 다른 양입니다.

## 점 평가가 실패하는 두 이유

일반 L2는 영집합 차이를 무시한 동치류이므로 f(0)는 잘 정의되지 않습니다. 연속 대표로 제한해 이 문제를 피하더라도, L2 노름에 대한 점 평가의 bounded 성질은 여전히 실패합니다.

$$
v_m(t)=\max(1-mt,0),\quad v_m(0)=1,\quad \|v_m\|_2=\frac1{\sqrt{3m}}.
$$

실습의 주황 삼각형은 폭이 좁아져도 높이가 1입니다. m=4에서 L2 노름은 약 .289, m=16에서는 약 .144입니다. 점 값/노름 비는 √(3m)로 끝없이 커져 하나의 상수 C로 |f(0)|≤C‖f‖₂를 보장할 수 없습니다.

## RKHS와 센서 모델로 연결하기

어떤 함수 공간에서 점 평가를 연속으로 만들려면 다른 노름이나 추가 정칙성이 필요합니다. RKHS는 점 평가가 bounded인 Hilbert 공간이며 대표 kₓ를 사용해 f(x)=⟨f,kₓ⟩로 씁니다. 일반 L2가 자동으로 RKHS인 것은 아닙니다. Sobolev 공간의 점 평가 역시 차원과 미분 차수에 따른 embedding 조건을 확인해야 합니다.

AI의 관측을 점 샘플로 표현할 때 입력 공간의 노름과 관측 안정성을 함께 정해야 합니다. 반대로 유한 폭 센서의 가중 적분은 그 가중치가 L2라면 현재 Riesz 예제처럼 연속 관측이 됩니다.

## Riesz 대표는 어디에서 오는가

ℓ≠0인 실수 Hilbert 공간의 연속 선형 범함수를 잡습니다. Ker ℓ은 닫혀 있으므로 다음 단원에서 증명할 직교 투영을 사용할 수 있습니다. ℓ(u)≠0인 u를 택하고 v=u−Pkerℓ u라 두면 v≠0, v⊥ker ℓ, ℓ(v)=ℓ(u)입니다. 임의 f에서 f−[ℓ(f)/ℓ(v)]v는 kernel에 있으므로

$$
\langle f,v\rangle=\frac{\ell(f)}{\ell(v)}\|v\|^2,
\qquad g=\frac{\ell(v)}{\|v\|^2}v,
\qquad \ell(f)=\langle f,g\rangle.
$$

대표 두 개의 차이가 모든 f와 내적 0이면 f를 그 차이로 택해 노름이 0임을 얻으므로 유일합니다. ℓ=0의 대표는 g=0입니다. 복소 공간에서는 내적의 선형인 인자에 맞춰 켤레를 넣어야 하며 여기서는 실수 경우를 사용합니다.

## 센서 폭을 줄이면 측정 노름이 커진다

폭 h의 평균 센서를 ℓ_h(f)=h⁻¹∫₀ʰf로 두면 대표는 g_h=h⁻¹1_[0,h]이고 노름은 h⁻¹/²입니다. 각 고정 h>0에서는 L² 연속이지만 h→0의 공통 상한은 없습니다. Spike v_m와 h=1/m를 함께 택하면 점 값은 1, 평균 센서값은 정확히 1/2입니다. 평균값이 모든 연속 f에서 점 값으로 수렴한다는 명제는 f도 m마다 바뀌는 이 예제에 균등하게 적용되지 않습니다.

| m | spike L² 노름 | 평균 센서 쌍대 노름 | spike H¹ 노름 |
|---|---|---|---|
| 1 | .577350 | 1 | 1.154701 |
| 4 | .288675 | 2 | 2.020726 |
| 16 | .144338 | 4 | 4.002603 |

미분 제곱 적분은 m이므로 spike의 H¹ 노름은 sqrt(m+1/(3m))로 커집니다. 1차원 H¹의 절대연속 대표에서는 기본정리와 Cauchy–Schwarz로 모든 x에 대해

$$
|u(x)|\le\|u\|_{L^2(0,1)}+\|u'\|_{L^2(0,1)}
\le\sqrt2\,\|u\|_{H^1(0,1)}.
$$

첫 부등식은 |u(x)|≤|u(y)|+∫|u'|를 y에 대해 평균해 얻습니다. 따라서 더 강한 노름 아래의 bounded 점 평가는 L² spike 반례와 모순되지 않습니다. 이 H¹ 대표의 근거는 약도함수 단원에서 연결합니다. 일반 고차원에서 같은 상수를 그대로 쓰지는 않습니다.

**직접 풀기:** 각 ℓ_h가 연속이면 그 극한 점 평가도 L² 연속인가요? **해설:** 아닙니다. 노름 h⁻¹/²이 발산하므로 연속성 상수가 균등하지 않습니다.

```python
from fractions import Fraction as F
from math import sqrt

for m in [1,4,16]:
    h=F(1,m)
    spike_integral=h/2
    sensor_value=spike_integral/h
    signal_l2_squared=F(1,3*m)
    sensor_dual_squared=F(1,h)
    derivative_l2_squared=F(m)
    assert sensor_value==F(1,2)
    assert sensor_value**2<=signal_l2_squared*sensor_dual_squared
    assert 1<=2*(signal_l2_squared+derivative_l2_squared)
    print('m, spike L2 norm, averaging dual norm, spike H1 norm:',
          m,*(round(sqrt(float(v)),6) for v in
              [signal_l2_squared,sensor_dual_squared,
               signal_l2_squared+derivative_l2_squared]))
```


## 확인 문제와 해설

1. g=1인 범함수의 노름은? **1입니다.** 길이 1인 구간에서 ‖1‖₂=1입니다.
2. g=f=1+t의 내적이 노름과 같은가요? **내적은 노름의 제곱입니다.** 7/3과 √(7/3)을 구분하세요.
3. m=16의 결과만으로 비유계성을 증명했나요? **아니요.** 유한 그림은 직관이고 m→∞의 정확한 비율이 증명입니다.

추가 읽기: [MIT 18.102, Functional Analysis](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/).
