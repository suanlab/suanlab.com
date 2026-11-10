## 가로 구간 대신 값의 층 보기

단순함수는 유한 개 값을 갖는 가측함수입니다. 서로소 가측집합 위에서 일정한 값들을 갖도록 쓰면 적분은 값과 집합 측도의 곱을 더한 것입니다. Riemann 적분의 직사각형 직관과 연결되지만 집합이 꼭 구간일 필요는 없습니다. 실습은 [0,1]의 Lebesgue 측도와 f(x)=x²를 사용하며 함수값을 아래쪽 이산 층으로 반올림합니다.

$$
s=\sum_{k=1}^K a_k\mathbf1_{A_k},\qquad\int s\,d\mu=\sum_{k=1}^K a_k\mu(A_k)\quad(a_k\ge0).
$$

## 실제 층의 크기를 계산하기

m=2이면 간격은 1/4입니다. s_m=2의 −m제곱 곱하기 floor(2의 m제곱 f)로 정합니다. k번째 층의 x 구간은 √(k/2^m)부터 √((k+1)/2^m)까지입니다. 폭이 일정한 x 격자가 아니라 값의 층으로 나눈 것이므로 구간 폭이 서로 다릅니다. x=0.6이면 f=0.36이고 근사는 0.25입니다.

$$
s_m(x)=2^{-m}\lfloor2^m x^2\rfloor,\quad\mu(A_k)=\sqrt{\frac{k+1}{2^m}}-\sqrt{\frac{k}{2^m}}.
$$

## 비음수 Lebesgue 적분의 정의

일반 비음수 가측함수의 적분은 그 함수 아래에 있는 비음수 단순함수 적분들의 상한으로 정의합니다. 적분값은 무한일 수도 있습니다. 현재 예제는 s_m이 단조 증가하여 x²에 접근합니다. 단조수렴정리로 적분도 1/3에 접근합니다. 유한 슬라이더 단계만으로 정리를 증명하는 것이 아니라, 명시적인 근사식과 오차 상계로 그 과정을 검산합니다.

$$
\int f\,d\mu=\sup_{0\le s\le f,\ s\text{ simple}}\int s\,d\mu,\qquad0\le\int_0^1x^2dx-\int_0^1s_mdx\le2^{-m}.
$$

## 끝점과 부호 함수의 주의점

x=1에서는 모든 s_m의 값이 1이지만, m=0의 함수는 그 한 점을 제외하면 0입니다. 따라서 적분은 0입니다. 그림의 채운 원과 빈 원은 이런 끝점 규약을 구분하며, 표의 층에 추가되는 끝점 한 점의 기여는 0입니다. 부호 함수는 양의 부분과 음의 부분을 나누어 적분합니다. 둘 다 무한이면 ∞−∞이므로 정의할 수 없습니다. 유한 기대값을 원하면 절댓값 적분가능성을 확인해야 합니다.

$$
f=f^+-f^-,\quad |f|=f^++f^-,\qquad\int|f|\,d\mu<\infty\implies\int f\,d\mu\in\mathbb R.
$$

## 확인 문제와 해설

1. m=0,x=1의 값과 적분은? **값은 1, 적분은 0입니다.**
2. 층을 세밀하게 하면 아래 근사가 줄어드나요? **아니요. 단조 증가합니다.**
3. 적분은 표본점 개수의 평균인가요? **아니요. 선택한 측도에 대한 가중입니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
