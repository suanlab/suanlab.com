## 함수의 극한과 적분의 극한

함수열이 점별로 수렴해도 적분의 극한이 일치하는 것은 아닙니다. 어떤 수렴이고 어떤 측도로 적분하는지 먼저 정해야 합니다. 이 단원은 [0,1]에서 Lebesgue 적분을 사용합니다. 첫 예제 f_n=x의 n제곱은 x<1에서 0, x=1에서 1로 수렴합니다. 극한 함수가 거의 모든 곳에서 0이므로 극한의 적분은 0입니다.

$$
\int_0^1x^n\,dx=\frac1{n+1}\longrightarrow0.
$$

## 우세수렴이 요구하는 공통 지배

우세수렴정리는 가측 f_n이 거의 모든 곳에서 가측함수 f로 수렴하고 모든 n에 대해 |f_n|≤g인 하나의 적분가능 g가 있으면 L1 차이가 0으로 가며 적분도 수렴한다고 말합니다. x^n에는 g=1을 쓸 수 있습니다. 항마다 따로 적분가능하다는 사실이나 n에 따라 달라지는 경계만으로는 충분하지 않습니다. 슬라이더가 보여주는 유한 항보다 전체 수열의 가정이 중요합니다.

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

## 단조수렴에서 Fatou 부등식을 유도하기

비음수 가측함수 f_n에 대해 u_m=inf{f_n:n≥m}으로 둡니다. 가산 개 가측함수의 하한도 가측이고, u_m은 liminf f_n으로 단조 증가하며 모든 n≥m에 대해 u_m≤f_n입니다. 단조수렴을 적용하면 확장된 비음수 값까지 허용하는 다음 부등식을 얻습니다.

$$
\int\liminf_n f_n\,d\mu
=\lim_m\int u_m\,d\mu
\le\lim_m\inf_{n\ge m}\int f_n\,d\mu.
$$

마지막 항은 적분들의 liminf입니다. 따라서 점별 수렴이나 적분의 유한성을 가정하지 않고 Fatou 부등식을 얻습니다. 이 단계에서 사용하는 기반 정리는 단조수렴정리입니다.

## Fatou로 L1 우세수렴을 증명하기

이제 f_n과 f가 실수값 가측함수이고 f_n→f가 거의 모든 곳에서 성립하며 비음수 L1 함수 g가 |f_n|을 지배한다고 가정합니다. 하나의 영집합을 제외하면 모든 경계와 극한이 함께 성립하므로 |f|≤g입니다. 따라서 f와 각 f_n은 적분가능합니다. d_n=|f_n−f|로 두면 0≤d_n≤2g이므로 비음수 함수 2g−d_n에 Fatou를 적용할 수 있습니다.

$$
2\int g\,d\mu
\le\liminf_n\left(2\int g\,d\mu-\int d_n\,d\mu\right)
=2\int g\,d\mu-\limsup_n\int d_n\,d\mu.
$$

g의 적분이 유한하므로 소거할 수 있고, 비음수인 L1 거리가 0으로 가야 합니다. 마지막으로 |∫f_n−∫f|≤∫|f_n−f|가 적분의 수렴을 줍니다. 유한한 공통 지배가 핵심이며 무한값 두 개를 빼서는 증명이 되지 않습니다.

## 균등 적분가능성으로 꼬리 제어하기

확률공간에서 적분가능한 X_n의 모음이 균등 적분가능하다는 것은 큰 값의 기여가 n에 대해 균등하게 사라진다는 뜻입니다.

$$
\lim_{K\to\infty}\sup_n E[|X_n|\mathbf1_{\{|X_n|>K\}}]=0.
$$

가정을 명시한 유용한 충분조건은 다음과 같습니다. X가 적분가능하고 X_n→X로 확률수렴하며 위 모음이 균등 적분가능하면 L1 수렴합니다. A_n={|X_n−X|>δ}로 두고 두 크기를 K에서 절단하면 다음 상계를 얻습니다.

$$
E|X_n-X|\le\delta+2K P(A_n)
+E[|X_n|\mathbf1_{\{|X_n|>K\}}]
+E[|X|\mathbf1_{\{|X|>K\}}].
$$

목표 오차에 맞춰 먼저 δ를 작게 고르고, K를 크게 하여 두 꼬리를 제어한 다음, 고정한 δ와 K에서 n을 크게 하여 확률 항을 줄입니다. 이 선택 순서가 증명을 완성합니다. 확률수렴만으로는 희귀 사건의 큰 기대 손실을 놓칠 수 있는 이유도 설명합니다.

## 공통 적분가능 지배 없이도 L1 수렴할 수 있다

별도 예제로 Ω={1,2,…}, P({k})=2⁻ᵏ에서 X_n=(2ⁿ/n)1{ω=n}을 정의합니다. 고정한 각 결과는 결국 지지집합 밖에 있고 E|X_n|=1/n→0입니다.

| n | 높이 | 점 확률 | 기대값 |
|---|---|---|---|
| 2 | 2 | 1/4 | 1/2 |
| 4 | 4 | 1/16 | 1/4 |
| 8 | 32 | 1/256 | 1/8 |

공통 지배 g라면 g(k)≥2ᵏ/k여야 하므로 E[g]≥∑1/k=∞입니다. 예를 들어 N+1부터 2N까지의 각 묶음은 이 조화급수에 적어도 1/2을 기여합니다. 따라서 L1 수렴하더라도 공통 적분가능 지배는 없습니다. 대신 이 모음은 균등 적분가능합니다. K≥2 위에서 가장 큰 꼬리 기대값은 2ⁿ/n>K인 첫 지표 n(K)의 1/n(K)이고 n(K)→∞이기 때문입니다. 우세수렴의 가정은 충분조건이며 모든 수렴 수열에 필요한 조건은 아닙니다.

## 정확한 분수로 이산 모음을 검산하기

다음 코드는 개별 기대값, 유한 envelope 적분과 균등 꼬리를 계산합니다. 발산·수렴은 앞의 논증으로 확인하며 유한 실행만으로 증명되지 않습니다. 이 이산 확장은 기존 조절기의 세 Lebesgue 측도 곡선과 별도입니다.

```python
from fractions import Fraction as Q

def amplitude(n):
    return Q(2**n, n)

for n in [2, 4, 8]:
    mass = Q(1, 2**n)
    assert mass*amplitude(n) == Q(1, n)
    print("n, height, atom mass, expectation:", n, amplitude(n), mass, Q(1, n))
for N in [2, 4, 8, 16]:
    envelope_integral = sum(Q(1, k) for k in range(1, N+1))
    next_block = sum(Q(1, k) for k in range(N+1, 2*N+1))
    assert next_block >= Q(1, 2)
    print("finite envelope:", N, envelope_integral, "next block:", next_block)
for K in [2, 4, 16, 256]:
    n = 2
    while amplitude(n) <= K:
        n += 1
    # Amplitudes increase for n >= 2; tail expectations 1/n decrease.
    ui_tail = Q(1, n)
    assert all((Q(1, j) if amplitude(j) > K else 0) <= ui_tail
               for j in range(1, n+10))
    print("threshold, first tail index, uniform tail:", K, n, ui_tail)
```

## 추가 문제와 연구 연결

1. 우세수렴 증명에서 왜 2∫g를 소거할 수 있나요? **공통 지배가 적분가능하여 그 값이 유한하기 때문입니다.**
2. E|X_n|이 유계이면 균등 적분가능한가요? **기존 높이 n 봉우리는 기대값 1이지만 임의로 큰 꼬리에 그 기대값이 남으므로 아닙니다.**
3. 기대 손실의 극한에서 무엇을 확인해야 하나요? **수렴 방식과 공통 지배 또는 타당한 꼬리 제어입니다. 희귀 사건의 확률이 작다는 것만으로는 부족합니다.**

기반 수렴정리는 [Tao, §1.4, 따름정리 1.4.47·정리 1.4.49](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf), 균등 적분가능성은 [Durrett, §4.6](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf)을 참고하세요. 위 이산 모음과 꼬리는 직접 계산한 예제입니다.

## 확인 문제와 해설

1. x^n의 점별 극한은 모든 점에서 0인가요? **아니요. 1에서는 1입니다.**
2. 집중 반례가 Fatou를 위반하나요? **아니요. 0≤1입니다.**
3. 세 번째 예제 n=4의 적분은? **1.75입니다.**

추가 읽기: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
