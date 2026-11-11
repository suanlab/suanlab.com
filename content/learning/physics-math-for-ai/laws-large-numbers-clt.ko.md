## 표본과 표본평균의 분포

Xi가 서로 독립이고 같은 Bernoulli(p) 분포를 따른다고 가정합니다. 합 S는 이항분포이고 평균은 S/n입니다. 실습은 무작위로 뽑은 표본 경로가 아니라 가능한 모든 k=0,…,n의 확률을 계산합니다. 파랑은 이항식의 부동소수점 계산값입니다. 표본 수가 늘면 개별 시행의 분산은 그대로여도 평균의 분산은 줄어듭니다.

$$
P(S=k)={n\choose k}p^k(1-p)^{n-k},\quad E[\bar X]=p,\quad\operatorname{Var}(\bar X)=\frac{p(1-p)}n.
$$

## 대수의 법칙은 고정 오차에 대한 주장

약한 대수의 법칙은 고정된 양의 ε에 대해 평균이 p에서 ε 이상 벗어날 확률이 0으로 간다는 주장입니다. 모든 유한 표본평균이 p와 같다는 뜻이 아닙니다. Chebyshev 부등식은 이 예제에서 분산/(ε²)로 꼬리확률을 제어합니다. 확률 상한은 1을 넘을 필요가 없어 1로 제한합니다. 표에는 실제 이항 꼬리합과 이 상한이 함께 나옵니다.

$$
P(|\bar X-p|\ge\varepsilon)\le\frac{p(1-p)}{n\varepsilon^2}\longrightarrow0.
$$

## CLT는 요동의 척도를 바꾼다

중심극한정리는 평균을 중심화하고 자체 표준편차로 나눈 변수가 정규분포에 분포수렴한다는 내용입니다. 현재는 0<p<1일 때 적용할 수 있습니다. 표본평균 자체가 고정된 분산의 정규분포로 변한다는 말이 아닙니다. 독립·동일 분포와 유한 비영 분산 가정이 중요하며 의존 자료에는 별도의 정리가 필요합니다.

$$
\frac{S-np}{\sqrt{np(1-p)}}\;\Rightarrow\;N(0,1).
$$

## 질량을 밀도 높이와 비교하지 않기

주황은 정규 밀도의 높이가 아니라 각 정수 k 주위의 폭 1 구간에 정규근사가 주는 질량입니다. 연속성 보정으로 이항 질량과 단위를 맞춥니다. 표시 범위 밖 정규 꼬리는 다시 정규화하지 않아 질량합이 1보다 작을 수 있습니다. 정규 CDF는 절대 오차 약 10의 −6승 이내인 수치 근사를 사용합니다. p=0 또는 1에서는 이항법칙이 한 점에 집중하고 표준화와 정규근사는 정의하지 않습니다.

$$
P(S=k)\approx\Phi\!\left(\frac{k+1/2-np}{\sqrt{np(1-p)}}\right)-\Phi\!\left(\frac{k-1/2-np}{\sqrt{np(1-p)}}\right).
$$

## Gaussian 극한이 나오는 항을 확인하기

p를 (0,1)에 고정하고 σ²=p(1−p)로 둡니다. 실수 확률변수 W의 특성함수는 E[exp(itW)]입니다. 독립성은 합의 특성함수를 곱으로 바꿉니다. 중심화한 Bernoulli 변수는 유계이므로 지수함수의 Taylor 전개에서 삼차 나머지를 제어할 수 있습니다. 고정된 실수 t마다 다음을 얻습니다.

$$
E\exp\!\left(\frac{it(X-p)}{\sigma\sqrt n}\right)
=1-\frac{t^2}{2n}+O(n^{-3/2}).
$$

E[X−p]=0이므로 일차항이 사라지고 E[(X−p)²]=σ²가 이차항의 계수를 줍니다. 이를 n제곱하거나 1 부근에서 로그를 취하면 다음과 같습니다.

$$
\varphi_{Z_n}(t)=\left[1-\frac{t^2}{2n}+O(n^{-3/2})\right]^n
\longrightarrow e^{-t^2/2},\qquad Z_n=\frac{S_n-np}{\sigma\sqrt n}.
$$

극한은 표준정규 특성함수입니다. Lévy 연속성 정리를 사용하면 이 점별 특성함수 극한이 분포수렴을 줍니다. 여기서는 그 정리를 사용하며 측도론적 증명은 별도 단계입니다. 몇 값의 그림으로 그 정리가 증명되는 것은 아닙니다. 나머지 상수는 p와 t에 의존할 수 있습니다. 이 논증은 p를 고정하며, p가 n과 함께 바뀔 때의 균일한 정규근사 오차 상계를 제공하지 않습니다.

## 정규분포로 가지 않는 희귀 사건 수열

다른 실험으로 각 행에 독립 Bernoulli(1/n) 시행 n개가 있다고 합시다. 한 행 안에서는 동일 분포이지만 행 사이에서는 시행의 분포가 바뀝니다. 횟수 S_n의 평균은 1, 분산은 1−1/n입니다. 다음과 같이 표준화합니다.

$$
p_n=1/n,\qquad Z_n=\frac{S_n-1}{\sqrt{1-1/n}}\quad(n\ge2).
$$

모든 n≥2에서 Z_n≤−1/2는 정확히 S_n=0 사건입니다. 횟수가 1이면 Z_n=0이고 횟수가 0이면 −1보다 작습니다. 따라서 다음과 같습니다.

$$
P(Z_n\le-1/2)=(1-1/n)^n\longrightarrow e^{-1}\ne\Phi(-1/2).
$$

| n | −1/2에서 실제 CDF | 같은 위치의 표준정규 CDF |
|---|---|---|
| 5 | 0.327680 | 0.308538 |
| 20 | 0.358486 | 0.308538 |
| 100 | 0.366032 | 0.308538 |

극한 e⁻¹은 약 0.367879입니다. 정규 CDF는 −1/2에서 연속이므로 이 불일치가 표준정규로의 분포수렴을 배제합니다. 분산이 1이라고 Gaussian 모양이 되는 것은 아닙니다. 여기서는 예상 희귀 사건 수가 커지는 대신 1에 머뭅니다. 앞에서 증명한 고정 분포 CLT의 가정과 다른 행별 실험이므로 모순이 아닙니다.

## 평균의 집중은 여전히 별도 문제다

행 평균의 기대값은 1/n, 분산은 (n−1)/n³입니다. Chebyshev 상계로 행 평균과 그 기대값의 차이가 확률적으로 0에 수렴하며, 평균 자체도 0에 수렴함을 알 수 있습니다. 표준화 횟수가 정규분포로 가지 않는 현상과 양립합니다. 오류율이 작아지는 평가에서는 평균이 집중하더라도 정규분포로 만든 횟수 오차막대가 부정확할 수 있습니다. 표본 수뿐 아니라 예상 사건 수와 표집 가정을 확인하세요.

## 정확한 분수 확률과 특성함수 비교하기

다음 표준 라이브러리 코드는 희귀 사건의 영 횟수 질량을 정확한 분수로 계산하고, 고정 p의 특성함수를 정규 극한과 수치 비교합니다. 앞부분은 정확한 유한 계산이고 뒷부분은 증명의 예시이며 증명이나 수렴률 인증을 대신하지 않습니다. 현재 조절기는 n≤30에서 지정한 이항법칙 하나를 계산하며, 아래의 큰 n과 변하는 p는 별도로 실행하는 확장 예제입니다.

```python
from fractions import Fraction as Q
from math import exp, erf, sqrt, floor
from cmath import exp as cexp

normal_cdf = (1 + erf(-.5/sqrt(2))) / 2
for n in [5, 20, 100, 1000]:
    p = Q(1, n)
    zero_mass = (1-p)**n
    cutoff = 1 - .5*sqrt(1-1/n)
    assert floor(cutoff) == 0  # S integer: Z <= -1/2 iff S == 0.
    variance_mean = p*(1-p)/n
    assert variance_mean == Q(n-1, n**3)
    print(n, "CDF at -1/2:", float(zero_mass),
          "normal:", normal_cdf, "mean variance:", float(variance_mean))
assert abs(float((1-Q(1, 1000))**1000)-exp(-1)) < .0002
assert abs(exp(-1)-normal_cdf) > .05

p, t = .25, .7
sigma = sqrt(p*(1-p))
errors = []
for n in [10, 100, 1000]:
    step = t/(sigma*sqrt(n))
    one = (1-p)*cexp(-1j*step*p) + p*cexp(1j*step*(1-p))
    characteristic = one**n
    error = abs(characteristic-exp(-t*t/2))
    errors.append(error)
    print("fixed-p characteristic:", n, characteristic, "error:", error)
assert errors[-1] < errors[0]
```

## 추가 문제와 연구 연결

1. 대수의 법칙이면 같은 가정에서 CLT도 따라오나요? **집중과 표준화된 분포 극한은 서로 다른 주장입니다.**
2. 왜 점 질량과 밀도 높이 대신 CDF를 비교하나요? **단위가 같은 확률이며 극한 CDF의 연속점에서 분포수렴을 판정할 수 있기 때문입니다.**
3. 위 Taylor 나머지가 균일한 CDF 오차율을 증명하나요? **추가 논증이 필요하므로 아닙니다.**

연속성 정리와 특성함수 방법은 [Sheffield, MIT 18.175, Lecture 15](https://math.mit.edu/~sheffield/175/Lecture15.pdf), 고정 i.i.d. CLT 명제는 [Pishro-Nik, §7.1.2](https://www.probabilitycourse.com/chapter7/7_1_2_central_limit_theorem.php)를 참고하세요. 희귀 사건 반례는 명시적인 영 횟수 확률에서 직접 유도했습니다.

## 확인 문제와 해설

1. p=0.5,n=10의 평균 분산은? **0.025입니다.**
2. 표본 수가 세 배면 이 분산은? **삼분의 일입니다.**
3. p가 0에 가까우면 작은 n의 정규근사가 항상 좋나요? **아니요. 근사 오차를 직접 확인해야 합니다.**

추가 읽기: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
