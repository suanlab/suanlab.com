## 비선형 추정량의 변동

기초 추정량이 점근 정규이면 미분 가능한 변환의 일차 Taylor 전개로 분포를 근사할 수 있습니다. 실습은 Xi가 N(μ,1)인 iid 표본이라 표본평균 법칙 N(μ,1/n)이 유한 n에서도 정확한 경우를 사용합니다. 그림은 표준정규 잡음을 무작위 추출하지 않고 ε 값에 따른 변환을 계산합니다. 파랑은 실제 변환, 주황은 μ 근처의 선형화입니다.

$$
\sqrt n(\widehat\theta_n-\theta)\Rightarrow N(0,V)\implies\sqrt n(g(\widehat\theta_n)-g(\theta))\Rightarrow N(0,[g^\prime(\theta)]^2V).
$$

## 제곱의 정확한 모멘트와 비교

g(x)=x²이면 g′(μ)=2μ이고 일차 분산 근사는 4μ²/n입니다. Gaussian 모멘트로 계산한 정확 분산에는 2/n² 항이 더 있습니다. 또한 기대값은 μ²가 아니라 μ²+1/n이라 유한 표본 편향도 있습니다. 기준 μ=1,n=10에서는 정확 분산 0.42, 일차 근사 0.4입니다. 큰 n에서의 근사를 유한 n의 등식으로 바꾸지 마세요.

$$
E[\bar X^2]=\mu^2+\frac1n,\qquad\operatorname{Var}(\bar X^2)=\frac{4\mu^2}n+\frac2{n^2}.
$$

## 도함수 0은 실제 분산 0이 아니다

μ=0에서는 일차 분산이 0이지만 정확 분산은 2/n²입니다. √n 척도에서는 퇴화한 극한이 맞지만, n 척도로 보면 n·표본평균²가 χ² 자유도 1을 따릅니다. 이 예제에서는 그 법칙이 유한 n에서도 정확합니다. 주황 선이 평평해지는 모습과 파랑 곡선의 변동을 비교하면 왜 고차 전개와 다른 척도가 필요한지 볼 수 있습니다.

$$
\mu=0:\qquad n\bar X^2\sim\chi_1^2,\qquad\operatorname{Var}(\bar X^2)=2/n^2.
$$

## 지수 변환과 적용 범위

g(x)=exp(x)에서는 도함수가 0이 아니지만 곡률 때문에 편향과 근사 오차가 생깁니다. 정확 기대값은 exp(μ+1/(2n)), 분산은 (exp(1/n)−1)exp(2μ+1/n)입니다. 일차 근사는 exp(2μ)/n입니다. 그림의 먼 ε에서는 선형화가 크게 벗어나거나 제곱 변환의 선형화가 음수가 될 수 있습니다. Delta Method는 국소 점근 주장이고 유한 표본의 전체 꼬리에 대한 균일 정확도를 보장하지 않습니다.

$$
\operatorname{Var}(e^{\bar X})=(e^{1/n}-1)e^{2\mu+1/n},\qquad V_{\mathrm{delta}}=e^{2\mu}/n.
$$

일반적인 분포수렴만으로 모멘트 수렴까지 보장되지는 않습니다. 여기서는 지정한 Gaussian 법칙에서 직접 계산한 유한 모멘트와 일차 근사를 비교합니다.

## 고차 스칼라 delta 극한을 유도하기

√n(θ̂_n−θ)가 실수 확률변수 Z로 분포수렴한다고 합시다. 양의 정수 r에 대해 g:R→R는 가측이고 θ 근방에서 C^r이며 1차부터 r−1차까지 도함수는 0, g^(r)(θ)는 비영이라고 가정합니다. Taylor 나머지로 다음을 얻습니다.

$$
g(\theta+h)-g(\theta)=\frac{g^{(r)}(\theta)}{r!}h^r+o(|h|^r),\qquad
n^{r/2}\{g(\widehat\theta_n)-g(\theta)\}\Rightarrow\frac{g^{(r)}(\theta)}{r!}Z^r.
$$

왜 나머지가 사라질까요? 분포수렴은 √n(θ̂_n−θ)가 확률적으로 유계임을 주며 따라서 θ̂_n→θ로 확률수렴합니다. 나머지의 계수는 확률적으로 0으로 가고 [√n(θ̂_n−θ)]^r은 확률적으로 유계이므로 곱은 0으로 갑니다. 주항에는 연속 사상 정리를 적용합니다. 이로써 표시한 분포 극한을 증명하며 분포수렴에서 모멘트 수렴을 추론한 것은 아닙니다.

r=2, Z∼N(0,V), V>0이면 극한은 비퇴화 정규분포가 아니라 g″(θ)Vχ1²/2입니다. g″의 부호에 따라 카이제곱 변수의 음수 배가 될 수도 있습니다. 현재 원점의 제곱 예제는 g″=2,V=1입니다.

## 정상점으로 줄어드는 근방에서 비교하기

n번째 행의 자료가 iid N(μ_n,1)이고 고정 실수 c에 대해 μ_n=c/√n이라고 합시다. 표본평균은 표준정규 Z를 사용하여 분포상 정확히 (c+Z)/√n으로 쓸 수 있습니다. 따라서 다음과 같습니다.

$$
n\bar X_n^2\overset d=(Z+c)^2,\qquad
n(\bar X_n^2-\mu_n^2)\overset d=Z^2+2cZ.
$$

첫 변수는 자유도 1, 비중심 모수 c²의 비중심 카이제곱입니다. 평균은 1+c², 분산은 2+4c²입니다. 이 줄어드는 근방에서도 제곱 추정량의 원래 척도 편향은 1/n입니다.

표는 실제 분산과 일차 근사에 각각 n²을 곱한 값입니다.

| c | 정확 | 일차 |
|---|---|---|
| 0 | 2 | 0 |
| $1/2$ | 3 | 1 |
| 2 | 18 | 16 |

정확 분산은 (2+4c²)/n²이고 선형화 분산은 4c²/n²입니다. 두 비율은 n이 커져도 그대로입니다. c=1/2에서는 일차 값이 계속 정확한 값의 1/3뿐입니다. 따라서 μ=0 근방 전체에서 이 상대 분산 오차가 n 증가만으로 사라지지는 않습니다. 고정한 비영 μ의 점근 명제를 이런 균일 보장으로 바꾸면 안 됩니다.

## 정규 오차막대를 가정하는 대신 모멘트 계산하기

(Z+c)²와 그 제곱을 전개하고 E[Z]=E[Z³]=0, E[Z²]=1, E[Z⁴]=3을 사용합니다. 제곱 평균을 빼면 2+4c²입니다. 이 유한 Gaussian 모멘트가 정확한 표를 정당화하며 고차 분포 정리만으로 분산 계산까지 따라오는 것은 아닙니다.

다음 코드는 Gaussian 다항식 모멘트를 분수로 평가하고 고정 기초 값에서 거듭제곱 척도 항등식을 확인합니다. 무작위 표집은 하지 않습니다. 기존 조절기는 평균과 n을 직접 지정하며 μ_n=c/√n으로 연결한 수열은 별도로 실행하는 예제입니다.

```python
from fractions import Fraction as Q
from math import prod

def moment(k):
    return Q(0) if k % 2 else Q(prod(range(1, k, 2)))

def expected(poly):
    return sum(a*moment(k) for k, a in enumerate(poly))

def square(poly):
    out = [Q(0)]*(2*len(poly)-1)
    for i, a in enumerate(poly):
        for j, b in enumerate(poly):
            out[i+j] += a*b
    return out

for c in [Q(0), Q(1, 2), Q(2)]:
    transformed = [c*c, 2*c, Q(1)]  # (Z+c)^2, Z standard normal.
    mean = expected(transformed)
    variance = expected(square(transformed))-mean**2
    linear_variance = 4*c*c
    assert mean == 1+c*c and variance == 2+4*c*c
    for n in [10, 100]:
        exact = variance/(n*n)
        first_order = linear_variance/(n*n)
        assert exact-first_order == Q(2, n*n)
        print("c, n, exact variance, first order:", c, n, exact, first_order)
    print("scaled variance ratio:", linear_variance/variance)
for r in [2, 3, 4]:
    for scale in [2, 4, 8]:
        for z in [Q(-1), Q(1, 2), Q(2)]:
            estimate = z/scale  # n=scale^2, theta=0, g(x)=x^r.
            assert scale**r * estimate**r == z**r
print("Higher-order power scaling checked algebraically on fixed base values.")
```

## 추가 문제와 연구 연결

1. 일차 미분 0이면 변환한 추정량이 결정적이라고 해도 되나요? **고차항이 요동을 정할 수 있으므로 아닙니다.**
2. 고차 극한이면 유효한 정규 신뢰구간도 자동으로 얻나요? **극한이 Gaussian의 다항식일 수 있어 아닙니다.**
3. 정상점 근처 점근 주장에는 무엇이 필요한가요? **모수 수열, 척도, 극한 법칙과 모멘트·균일성 가정입니다.**

Taylor 기반 분포·모멘트 근사는 [MIT 18.655, Lecture 16, 정리 5.3.3과 분산 논의](https://ocw.mit.edu/courses/18-655-mathematical-statistics-spring-2016/2ca793c6ea6c4a6fece39c0da3440ba2_MIT18_655S16_LecNote16.pdf)를 참고하세요. 위 스칼라 r차·국소 Gaussian 계산은 명시적으로 유도했습니다.

## 확인 문제와 해설

1. μ=0,n=10에서 제곱의 실제 분산은? **0.02입니다.**
2. 제곱 추정량은 μ²에 비편향인가요? **아니요. 편향이 1/n입니다.**
3. 정규성 가정이 바뀌어도 정확 모멘트 식은 그대로인가요? **아니요. 점근 정리와 정확 분포를 구분해야 합니다.**

추가 읽기: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
