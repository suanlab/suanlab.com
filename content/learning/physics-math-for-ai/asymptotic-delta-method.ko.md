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

## 확인 문제와 해설

1. μ=0,n=10에서 제곱의 실제 분산은? **0.02입니다.**
2. 제곱 추정량은 μ²에 비편향인가요? **아니요. 편향이 1/n입니다.**
3. 정규성 가정이 바뀌어도 정확 모멘트 식은 그대로인가요? **아니요. 점근 정리와 정확 분포를 구분해야 합니다.**

추가 읽기: [MIT 18.650, Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-notes/).

[Wasserman, All of Statistics — author’s page](https://stat.cmu.edu/~larry/all-of-statistics/index.html).
