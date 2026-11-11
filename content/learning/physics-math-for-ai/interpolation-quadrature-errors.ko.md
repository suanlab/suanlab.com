## 같은 표본으로도 다른 질문을 할 수 있다

[0,1]의 함수 표본을 주었을 때 보간은 node 사이의 값을, 수치 적분은 전체 적분을 근사합니다. 같은 데이터를 써도 결과의 의미는 다릅니다. 실험은 exp(x) 또는 x^4를 n=2m개의 균등 panel로 나누어 n+1번 평가합니다.

파랑은 실제 함수, 주황은 표본을 직선으로 연결한 보간입니다. Node에서는 일치하지만 사이에서는 일반적으로 다릅니다. 주황 polyline의 적분은 composite trapezoid와 정확히 같습니다. Simpson은 같은 표본에 다른 가중치를 주며 주황 그림 자체의 적분은 아닙니다.

## 보간 오차를 국소적으로 유도하기

이웃 node xi와 xi+h 사이에서 선형 보간은 양 끝값의 convex combination입니다. 두 번째 도함수의 나머지로 다음 상한을 얻습니다.

$$
f(x)-I_hf(x)=\frac{f''(\xi)}2(x-x_i)(x-x_i-h),\qquad
\|f-I_hf\|_\infty\le M_2h^2/8.
$$

나머지 식의 xi 모양 기호 ξ는 어떤 중간점이며, M2는 구간 전체의 |f''| 상한입니다. 볼록한 두 예제에서는 chord가 실제 곡선 위에 있습니다. Node에서 질의하면 실제 오차가 0이어도 전역 상한은 양수일 수 있습니다.

## 같은 표본, 다른 적분 가중치

h=1/n, fj=f(jh)라고 쓰면 다음과 같습니다.

$$
T=h\left(\frac{f_0+f_n}2+\sum_{j=1}^{n-1}f_j\right),\qquad
S=\frac h3\left(f_0+f_n+4\sum_{j\text{ odd}}f_j+2\sum_{\substack{j\text{ even}\\0<j<n}}f_j\right).
$$

단위 구간에서 trapezoid 절대 오차는 M2 h²/12, Simpson은 M4 h^4/180 이하입니다. 뒤의 결과는 네 번째 도함수의 유계성과 짝수 panel 수를 가정합니다. 고차 정확도는 아무 함수·아무 grid에도 자동 적용되는 성질이 아닙니다.

## 세 표본으로 손계산하기

f=x^4, n=2의 표본은 (0,1/16,1)입니다. Simpson 값은 5/24, 정확한 적분은 1/5이므로 오차는 1/120입니다. 네 번째 도함수가 상수 24여서 상한과도 정확히 같습니다. x=1/4에서 선형 보간은 1/32이지만 실제 값은 1/256입니다.

exp(x)의 정확한 적분은 e-1이고 M2=M4=e입니다. 같은 세 표본에서 Simpson 오차는 약 0.000579, trapezoid는 약 0.03565입니다. 한쪽에 더 많은 표본을 주지 않고 비용 조건을 맞춘 비교입니다.

## 외삽을 유도하고 추정량과 인증 구별하기

충분한 매끄러움 아래 S(h)=I+ch⁴+O(h⁶)라 하자. h와 h/2의 식에서 c를 소거하면 R=(16S(h/2)−S(h))/15=I+O(h⁶)다. 세밀한 격자의 선도 오차는 |S(h/2)−S(h)|/15로 추정한다. 고차 나머지를 따로 제어하지 않으면 엄밀한 상계는 아니다. 비매끄럽거나 부족하게 표본화한 함수에서는 부정확한 두 값도 일치할 수 있다.

x⁴의 거듭제곱 합으로 composite 오차를 정확히 구하면

$$
T_n-\tfrac15=\frac1{3n^2}-\frac1{30n^4},\qquad
S_n-\tfrac15=\frac2{15n^4},\qquad
\frac{16S_{2n}-S_n}{15}=\tfrac15.
$$

이 다항식에는 고차 항이 없어 유리수 외삽이 정확하다. 하지만 세밀한 격자의 추가 표본을 사용하므로 원래 표본 수에서 비용 없이 좋아진 것은 아니다.

| n | Simpson 오차 | 외삽 오차 |
|---|---:|---:|
| 2 | 1/120 | 0 |
| 4 | 1/1920 | 0 |
| 8 | 1/30720 | 0 |

```python
from fractions import Fraction as F
def simpson(n):
    values = [F(i, n)**4 for i in range(n+1)]
    return (values[0]+values[-1]+sum((4 if i % 2 else 2)*values[i]
            for i in range(1, n)))/(3*n)
for n in (2, 4, 8):
    coarse, fine = simpson(n), simpson(2*n)
    extrapolated = (16*fine-coarse)/15
    assert coarse-F(1, 5) == F(2, 15*n**4)
    assert extrapolated == F(1, 5)
    print(n, coarse-F(1, 5), extrapolated-F(1, 5))
```

두 격자와 가중치를 독립적으로 만든다. 일반 매끄러운 함수에서는 선도항을 없애는 것이지 적분을 정확히 만드는 것은 아니다. 꺾인 함수에서는 사차 전개 가정부터 실패할 수 있다. 오차 추정·도함수 기반 상계·실제 표본 비용을 구별하자.

## 직접 풀기: 상한·오차·비용

x^4에서 Simpson panel 수를 두 배로 늘리면 오차는 16분의 1이 되고 표본 수는 n+1에서 2n+1로 늘어납니다. 보간 node의 오차 0과 양의 전역 상한이 왜 모순이 아닌지 설명하세요. 꺾인 함수에 같은 h^4 상한을 그대로 적용할 수 없는 이유도 도함수 가정과 연결해 적으세요.

더 읽기: [SciPy Simpson 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html).
