## 함수의 극한과 적분의 극한

함수열이 점별로 수렴해도 적분의 극한이 일치하는 것은 아닙니다. 어떤 수렴이고 어떤 측도로 적분하는지 먼저 정해야 합니다. 이 단원은 [0,1]에서 Lebesgue 적분을 사용합니다. 첫 예제 f_n=x의 n제곱은 x<1에서 0, x=1에서 1로 수렴합니다. 극한 함수가 거의 모든 곳에서 0이므로 극한의 적분은 0입니다.

$$
\int_0^1x^n\,dx=\frac1{n+1}\longrightarrow0.
$$

## 우세수렴이 요구하는 공통 지배

우세수렴정리는 가측 f_n이 거의 모든 곳에서 f로 수렴하고 모든 n에 대해 |f_n|≤g인 하나의 적분가능 g가 있으면 L1 차이가 0으로 가며 적분도 수렴한다고 말합니다. x^n에는 g=1을 쓸 수 있습니다. 항마다 따로 적분가능하다는 사실이나 n에 따라 달라지는 경계만으로는 충분하지 않습니다. 슬라이더가 보여주는 유한 항보다 전체 수열의 가정이 중요합니다.

$$
f_n\to f\text{ a.e.},\quad |f_n|\le g\in L^1\implies\int|f_n-f|\to0.
$$

## 질량이 집중하는 반례

두 번째 함수는 열린 구간 (0,1/n)에서 높이 n이고 나머지에서 0입니다. 각 고정 x>0은 결국 지지집합 밖으로 나가고 x=0에서도 항상 0이므로 점별 극한은 0입니다. 하지만 폭 1/n 곱하기 높이 n은 1입니다. 따라서 적분 극한은 1, 극한의 적분은 0입니다. 이 결과 자체가 전체 수열을 지배하는 적분가능 함수가 존재할 수 없음을 보여줍니다.

$$
f_n=n\mathbf1_{(0,1/n)},\qquad\lim_n\int f_n=1\ne0=\int\lim_n f_n.
$$

## 단조수렴과 Fatou의 역할

비음수 함수가 단조 증가하면 단조수렴정리는 무한값까지 허용해 적분의 극한을 교환합니다. 세 번째 예제는 x>0에서 min(n,1/√x), x=0에서 0입니다. 적분은 2−1/n이고 극한 적분은 2입니다. 여기서는 지배 함수 1/√x도 적분가능하므로 우세수렴도 적용됩니다. 그래프는 극한 함수가 높이 2n을 넘는 원점 근처를 생략하며 그 구간이 0이라는 뜻이 아닙니다. 일반 비음수 수열에는 Fatou의 부등식이 유용합니다.

$$
\int\liminf_n f_n\le\liminf_n\int f_n,\qquad 0\le f_n\uparrow f\implies\int f_n\uparrow\int f.
$$

## 확인 문제와 해설

1. x^n의 점별 극한은 모든 점에서 0인가요? **아니요. 1에서는 1입니다.**
2. 집중 반례가 Fatou를 위반하나요? **아니요. 0≤1입니다.**
3. 세 번째 예제 n=4의 적분은? **1.75입니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
