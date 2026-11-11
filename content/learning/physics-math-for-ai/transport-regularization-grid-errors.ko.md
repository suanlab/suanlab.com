## 세 오차는 각각 검사해야 한다

Entropy 수송 계산이 원하는 연속 문제와 다른 이유에는 정규화·미완료 최적화·주변 분포 이산화가 있다. 격자만 늘린다고 나머지 둘이 해결되지는 않는다. 실습은 연속 정답을 독립적으로 아는 가족으로 이를 나눈다.

μ는 [0,1] uniform, ν는 s만큼의 평행 이동이다. W₂=|s|다. 각각 N=2ᵏ개의 midpoint 원자로 바꾸어도 대응점을 연결하면 모든 N에서 W₂=|s|여서 실제 pair 거리 이산화 오차는 0이다.

## 양의 상한과 실제 0 오차

각 주변 분포는 cell 적분으로

$$
W_2(\mu,\mu_N)=\frac{1}{\sqrt{12}N},\qquad
|W_2(\mu,\nu)-W_2(\mu_N,\nu_N)|
\le\frac{2}{\sqrt{12}N}.
$$

상한은 양수여도 이 평행 이동 가족의 실제 pair 오차는 0이다. 상계와 등식은 다르다. 또한 여기 값은 거리 단위이고 아래 정규화 인증은 제곱 비용 단위다.

## 원문제 상한 전에 가능성 복구

유한 Sinkhorn 행렬은 주변 질량이 틀릴 수 있다. 실습은 질량 단위 2⁻²⁴로 가능 계획을 만든다. 각 행·열 정수 용량은 2²⁴/N이다. 남은 용량 안에서 각 cell 질량의 내림값을 남긴 뒤 잔여 정수 용량을 배분한다.

결과는 비음수인 단위 질량의 정수배이고 행·열 합이 이진 산술에서 정확히 1/N이다. 이는 가능성 보정이지 또 다른 Sinkhorn 수렴 정리가 아니다. L1 보정량에는 주변 복구와 양자화가 함께 들어간다.

## Entropy 상한과 최적화 인증

가능 보정 계획과 현재 potential의 정규화 Fenchel gap을 G라 하자. 균등 N점 주변의 결합 entropy는 log N과 2log N 사이이므로 원래 permutation 계획과 비교하면

$$
0\le \langle C,\gamma_{feas}\rangle-OT
\le\epsilon\log N+G.
$$

εlog N은 정확한 정규화 해의 수송 비용 편향을 상계하고 G는 정규화 최적화 오차를 따로 제어한다. 어느 것도 위의 주변 이산화 상한과 같은 값이 아니다. 최악 entropy 상한이 N에 따라 커진다고 실제 편향 증가가 증명되지는 않는다.

파란 선은 가능 보정 후 초과 비용, 주황 선은 인증이다. 초과는 s²와의 상쇄를 피하려고 Σγᵢⱼ(xᵢ−xⱼ)²로 계산한다. 정규화 목적·쌍대와 안정적인 generalized-KL gap도 함께 표시한다.

## 제곱 비용 인증을 거리 오차 예산으로 바꾸기

균등 주변에서 H(P)=H(X)+H(Y|X)≥log N이고 H(P)≤H(X)+H(Y)=2log N이다. 가능한 permutation 최적해 P₀는 H(P₀)=log N이다. 가능하게 보정한 P의 정규화 gap이 G이면 Jε(P)≤Jε(P₀)+G다. Entropy 관례의 상수를 소거하면

$$
\langle C,P\rangle-OT_N
\le\epsilon[H(P)-H(P_0)]+G
\le\epsilon\log N+G=:B.
$$

각 주변 근사의 검증된 W₂ 오차 합을 δ라 하자. 삼각부등식으로 |W₂(μ,ν)−W₂(μ_N,ν_N)|≤δ다. 가능 계획의 제곱 비용 c≥OT_N과 |√c−√OT_N|≤√B를 결합하면

$$
|\sqrt{c}-W_2(\mu,\nu)|\le\delta+\sqrt B.
$$

비용 단위에서 거리 단위로 명시적으로 바꾸었다. δ를 B에 직접 더하면 단위가 섞인다. Midpoint 가족에서는 δ=1/(√3N)이다. 거리 허용 오차 τ를 위해 δ≤τ/2, εlog N≤τ²/8, G≤τ²/8을 택하면 충분하다. 마지막은 정지 조건이지 특정 반복 수가 이를 만족한다는 주장이 아니다.

| τ | N | ε 예산 |
|---|---:|---|
| 1/2 | 4 | τ²/(8log N) |
| 1/4 | 8 | τ²/(8log N) |
| 1/8 | 16 | τ²/(8log N) |

```python
from math import log, sqrt
for tau, N in ((.5, 4), (.25, 8), (.125, 16)):
    delta = 1/(sqrt(3)*N)
    epsilon_budget = tau*tau/(8*log(N))
    gap_target = tau*tau/8
    bound = delta+sqrt(epsilon_budget*log(N)+gap_target)
    assert delta <= tau/2 and bound <= tau
    assert epsilon_budget < .05
    print(tau, N, "epsilon budget:", epsilon_budget,
          "gap target:", gap_target, "distance bound:", bound)
```

계산한 ε 예산은 실습 최솟값 ε=0.05보다 작다. 프로그램은 별도의 이론적 정확도 예산 예제이며 조절기를 바꾸지 않고 이 허용 오차를 달성했다고 주장하지 않는다. 실제 평행 이동 쌍의 격자 오차는 여전히 0이라 주변 상계가 보수적일 수 있다. 일반 비균등 주변에는 별도 entropy·가능성 분석이 필요하다.

## 직접 풀기와 풀이

N=8,s=0.5,ε=0.1,20회에서 주변 양자화 거리는 약 0.036084, 실제 pair 오차 0, 초과 비용 약 0.038481이다. 정규화 상한은 약 0.207944, gap은 약 5.71×10⁻⁸이다.

N을 두 배로 하면 주변 양자화 거리는 절반이지만 최적화 잔차와 entropy 효과는 따로 봐야 한다. 반복 0에서도 보정 후 가능성은 정확하지만 정규화 인증은 클 수 있다. 연속 수송 근사를 주장하기 전에 이 정확도들을 함께 보고하자.

더 읽기: [Peyré와 Cuturi의 최적수송 교재](https://optimaltransport.github.io/book/).
