## 같은 공간의 거리와 분포의 거리

확률수렴은 같은 확률공간 위에서 X_n과 X의 차이가 고정 오차를 넘을 확률이 0으로 가는 것입니다. 약수렴 또는 분포수렴은 두 변수의 법칙을 비교합니다. 실수에서는 모든 유계 연속 시험함수 h의 기대값이 수렴하는 조건으로 표현할 수 있습니다. 법칙만 주어졌다면 두 변수를 같은 결과에서 빼는 연산은 아직 정의되지 않을 수 있습니다.

$$
X_n\to X\text{ in probability}\iff P(|X_n-X|>\varepsilon)\to0\ \forall\varepsilon>0,\quad X_n\Rightarrow X\iff E[h(X_n)]\to E[h(X)]\ \forall h\in C_b.
$$

## 희소 사건과 L1 수렴의 차이

Uniform[0,1]에서 X_n=1_(0,1/n)이면 편차 확률과 L1 차이는 ε<1에서 모두 1/n입니다. 거의 확실히도 0으로 수렴합니다. 높이를 n으로 바꾼 모드 2는 여전히 확률수렴하지만 L1 차이는 1입니다. 따라서 L1 수렴은 확률수렴을 함의하지만 역은 일반적으로 성립하지 않습니다. 수렴 표식은 이 명시한 전체 수열에 대한 본문 논증을 반영하며 유한 그래프에서 자동 추론한 결과가 아닙니다.

$$
P(|X_n-X|>\varepsilon)\le\frac{E|X_n-X|}{\varepsilon}.
$$

## 동일한 법칙이어도 가까운 변수는 아니다

X가 −1과 1을 같은 확률로 갖고 X_n=(−1)^n X라고 합시다. 모든 n의 분포는 X와 같아 약수렴합니다. 하지만 같은 X와 비교하면 홀수 항의 차이는 항상 2이고 짝수 항은 0입니다. ε=0.5에서 편차 확률은 1과 0으로 번갈아 나타나므로 X로 확률수렴하지 않습니다. 비상수 극한에서 결합을 무시한 법칙 비교와 경로의 근접성을 혼동하지 마세요.

$$
\mathcal L(X_n)=\mathcal L(X),\qquad P(|X_{2k+1}-X|>1/2)=1.
$$

## CDF는 극한의 연속점에서 비교한다

결정론적 X_n=1/n은 0으로 수렴합니다. 하지만 F_n(0)=0인 반면 극한 원점 질량의 F(0)=1입니다. 약수렴의 CDF 조건은 극한 CDF의 연속점에서만 요구되므로 모순이 아닙니다. 이 예제의 CDF 최대 차이는 항상 1이어서 그 거리의 수렴과 약수렴은 다릅니다. 상수로의 분포수렴은 확률수렴도 함의하는 특별한 경우입니다. 일반 약수렴에서는 유계가 아닌 손실 기대값의 수렴을 추가 가정 없이 결론낼 수 없습니다.

$$
F_n(t)\to F(t)\quad\text{at every continuity point }t\text{ of }F.
$$

## 확인 문제와 해설

1. 교대 부호 모드 n=3의 L1 차이는? **2입니다.**
2. 집중 높이 n의 예제는 왜 L1 수렴하지 않나요? **L1 차이가 1로 유지됩니다.**
3. δ1/n의 0에서 CDF 불일치는 약수렴 실패인가요? **아니요. 그 점은 극한의 불연속점입니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
