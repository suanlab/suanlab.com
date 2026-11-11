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

## 확률수렴이 약수렴을 함의함을 증명하기

|h|≤M인 유계 연속 시험함수 h를 택합니다. 작은 η>0에 대해 P(|X|>R)<η인 R을 고릅니다. h는 compact 구간 [−R−1,R+1]에서 균등 연속이므로, 그 안의 두 값 차이가 δ 이하이면 함수값 차이가 η 이하가 되도록 0<δ<1을 고릅니다. {|X|≤R, |X_n−X|≤δ}에서는 이 경계가 적용되고 나머지에서는 2M으로 제어합니다.

$$
|E[h(X_n)]-E[h(X)]|
\le\eta+2M\bigl(P(|X|>R)+P(|X_n-X|>\delta)\bigr).
$$

R,δ를 고정한 뒤 확률수렴을 사용하면 마지막 확률 항이 0으로 갑니다. 이어 η를 0으로 보내면 유계 연속 시험함수에 의한 약수렴 정의를 증명합니다. 모든 연속함수가 실수 전체에서 균등 연속이라고 잘못 가정할 필요가 없습니다.

## 상수 극한에서는 역방향도 성립함을 증명하기

X_n이 상수 c로 분포수렴한다고 합시다. ε>0에 대해 a=c−ε/2, b=c+ε/2를 택하면 둘 다 극한 CDF의 연속점입니다. 극한 CDF 값이 각각 0과 1이므로 다음과 같습니다.

$$
P(|X_n-c|>\varepsilon)\le F_n(a)+1-F_n(b)\longrightarrow0.
$$

따라서 상수로는 확률수렴도 합니다. 지정한 결합에서 일반 비상수 극한까지 확장되는 성질은 아니며 앞의 교대 부호 반례가 그 이유를 보여줍니다. 법칙의 수열만으로 특정 목표 확률변수와의 결합이 정해지지는 않습니다.

## 약수렴·전변동·수송 거리 비교하기

높이 n인 희귀 사건의 법칙은 μ_n=(1−1/n)δ0+(1/n)δn이고 목표는 δ0입니다. 모든 유계 연속 h에 대해 적분 차이는 (h(n)−h(0))/n이며 크기가 2||h||∞/n 이하이므로 약수렴을 직접 증명할 수 있습니다.

여기서 전변동은 가측집합 A에 대한 |μ_n(A)−δ0(A)|의 상한으로 정의합니다. 이산 공간에서는 질량 절댓값 차이 합의 절반과 같습니다. {n}에서 달성되므로 값은 1/n입니다. 첫 Wasserstein 거리는 두 법칙의 모든 결합에서 E|U−V|의 하한입니다. δ0와의 모든 결합에서 V는 거의 확실히 0이므로 이 거리는 반드시 E|U|=1입니다.

| n | TV | $W_1$ | 평균 |
|---|---|---|---|
| 2 | $1/2$ | 1 | 1 |
| 4 | $1/4$ | 1 | 1 |
| 16 | $1/16$ | 1 | 1 |

이 예제는 약수렴하고 전변동 거리도 0으로 가지만 평균과 W1은 목표의 값 0으로 가지 않습니다. 멀리 있는 희귀 결과를 어떤 방식으로 제어하는지에 따라 거리가 답하는 질문이 다릅니다. 수송 항등식은 점 질량 목표에 대해 유도했으며 일반 최적 수송 solver를 구현한 것은 아닙니다.

## 유계 시험함수와 비유계 손실을 구분하기

유계 연속 시험함수 h(x)=x/(1+|x|)의 기대값은 1/(n+1)→0입니다. 반면 비유계 손실 |x|의 기대는 1입니다. 손실을 고정한 K로 자르면 min(n,K)/n→0이지만, n>K이면 꼬리 기여가 1입니다. 따라서 이 모음은 균등 적분가능하지 않습니다. 각 고정 절단값에서 극한을 구한 뒤 균등 꼬리 논증 없이 절단을 제거하면 안 됩니다. L1 충분조건은 [우세수렴과 균등 적분가능성](../convergence-dominated-integration/)에서 확인하세요.

## 유한 법칙을 정확하게 검산하기

다음 코드는 전변동, CDF 거리, 점 목표의 수송 비용, 유계 시험함수, 절단·꼬리 손실을 분수로 검사합니다. 전체 수열의 성질은 위 공식으로 증명합니다. 현재 조절기는 기존 네 수열을 비교하며 수송·절단 계산은 별도의 확장입니다.

```python
from fractions import Fraction as Q

for n in [2, 4, 16, 64]:
    law = {0: 1-Q(1, n), n: Q(1, n)}
    target = {0: Q(1)}
    support = sorted(set(law) | set(target))
    mean = sum(Q(x)*law.get(x, 0) for x in support)
    tv = sum(abs(law.get(x, 0)-target.get(x, 0)) for x in support)/2
    # Every coupling with delta_0 has second coordinate zero.
    w1 = sum(abs(Q(x))*law.get(x, 0) for x in support)
    cdf_gap = max(abs(sum(m for x, m in law.items() if x <= t)
                      -sum(m for x, m in target.items() if x <= t))
                  for t in support)
    assert tv == cdf_gap == Q(1, n) and mean == w1 == 1
    bounded_test = sum(Q(x, 1+abs(x))*m for x, m in law.items())
    assert bounded_test == Q(1, n+1)
    for K in [1, 4, 16]:
        truncated = sum(min(Q(x), Q(K))*m for x, m in law.items())
        tail = sum(Q(x)*m for x, m in law.items() if x > K)
        assert truncated == Q(min(n, K), n)
        assert tail == int(n > K)
    print("n, TV, W1, mean, bounded test:", n, tv, w1, mean, bounded_test)
```

## 추가 문제와 연구 연결

1. 전변동이 아주 작으면 모든 비유계 손실의 오차도 작은가요? **표의 기대 절댓값 손실이 1로 남으므로 아닙니다.**
2. CDF 최대 거리와 W1이 다를 수 있나요? **여기서는 각각 1/n과 1입니다.**
3. 분포 이동에 대한 주장에 무엇을 명시해야 하나요? **거리 또는 시험함수 모음, 과제 손실과의 관계, 필요한 꼬리 모멘트입니다.**

약수렴과 시험함수는 [Durrett, §3.2](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf)를 참고하세요. 위 두 함의와 유한 희귀 사건의 거리 계산은 명시적으로 유도했습니다.

## 확인 문제와 해설

1. 교대 부호 모드 n=3의 L1 차이는? **2입니다.**
2. 집중 높이 n의 예제는 왜 L1 수렴하지 않나요? **L1 차이가 1로 유지됩니다.**
3. δ1/n의 0에서 CDF 불일치는 약수렴 실패인가요? **아니요. 그 점은 극한의 불연속점입니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
