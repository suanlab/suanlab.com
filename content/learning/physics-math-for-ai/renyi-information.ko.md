## 하나의 엔트로피에서 차수별 정보량으로

Shannon 엔트로피 외에도 질량의 거듭제곱을 이용하는 정보량이 있습니다. Rényi 엔트로피는 차수 α에 따라 큰 질량과 희귀 질량의 상대적인 강조를 바꿉니다. 이 실습은 유한 이진 분포와 α∈[0,8]을 사용하며 값의 단위는 bits입니다.

$$
H_\alpha(P)=\frac1{1-\alpha}\log_2\sum_i p_i^\alpha,\qquad \alpha\ne1.
$$

## 0·1·2·무한 차수를 구분하기

0차수는 양의 질량을 가진 지지 크기의 로그입니다. 1차수는 연속 확장으로 Shannon 엔트로피이고, 2차수는 두 독립 표본이 같은 결과일 충돌확률의 음의 로그입니다. α가 무한대로 가면 가장 큰 질량만 남아 min entropy가 됩니다. 화면은 무한 차수의 값도 별도 수치로 보여주지만 곡선 범위는 8까지만입니다.

$$
H_0=\log_2|\operatorname{supp}P|,\quad H_1=H(P),\quad
H_2=-\log_2\sum_i p_i^2,\quad H_\infty=-\log_2\max_i p_i.
$$

p=.8이면 H₀=1, H₁≈.72193, H₂≈.55639, H∞≈.32193 bits입니다. 차수가 커질수록 엔트로피가 감소하는 경향을 확인하세요. 균등 분포는 모든 차수에서 같고 결정적 분포는 모두 0입니다. 아주 작은 양수 질량은 0이 아니므로 H₀의 지지에 포함됩니다.

## Rényi 발산도 함께 비교하기

Q와의 차이는 다음 Rényi 발산으로 계산합니다. 이 단원은 Q의 두 질량을 모두 양수로 제한해 지지 특이점과 차수 극한을 분리합니다. α=1에서는 KL, α=0에서는 P의 지지에 놓인 Q 질량의 음의 로그입니다.

$$
D_\alpha(P\Vert Q)=\frac1{\alpha-1}\log_2\sum_i p_i^\alpha q_i^{1-\alpha},
\qquad D_0=-\log_2Q(\operatorname{supp}P).
$$

p=.8,q=.5,α=2이면 D₂≈.44361 bits입니다. 파랑은 차수별 엔트로피이고 주황은 Shannon 기준선입니다. 발산은 같은 차수의 표와 수치에서 비교합니다. 엔트로피와 발산을 같은 양으로 혼동하지 마세요.

## 극한과 계산의 주의점

α=1에 원래 식을 직접 넣으면 0/0입니다. 구현은 정확한 Shannon/KL 값을 사용하고 가까운 차수에서는 expm1·log1p 형태로 상쇄 오차를 줄입니다. α=0에서 0⁰을 모든 상태에 1로 넣는 대신 양의 지지만 셉니다. 여기서는 Rényi 엔트로피와 발산을 소개하며 여러 서로 다른 Rényi 상호정보량의 정의까지 동일하다고 주장하지 않습니다.

## 2차 발산은 중요도 가중치의 두 번째 모멘트다

유한 알파벳에서 p_i,q_i>0이고 둘 다 정규화되어 있다고 합시다. Q에서 iid 표집하고 w(i)=p_i/q_i로 보정하면 E_Q[w]=1입니다. 2차 Rényi 발산을 bits로 정의했으므로

$$
E_Q[w^2]=\sum_i\frac{p_i^2}{q_i}=2^{D_2(P\Vert Q)},\qquad
\operatorname{Var}_Q(w)=2^{D_2(P\Vert Q)}-1.
$$

따라서 D₂는 가중치 두 번째 모멘트의 로그입니다. 정규화 상수의 일반 중요도 추정 N^{-1}Σw는 비편향이고 분산은 (2^D₂−1)/N입니다. 이는 함수 f의 평균 추정 분산과 다릅니다. f가 들어간 오차는 앞선 중요도 표집 단원의 V와 A를 사용해야 합니다.

## 표본 ESS와 모집단 계수를 구분하기

실현된 N개 가중치의 ESS=(Σw)²/Σw²는 1 이상 N 이하입니다. 고정한 P,Q에서 대수법칙으로 표본평균 w와 w²가 각각 1과 E_Q[w²]로 수렴하므로 다음 극한을 얻습니다.

$$
\frac{\operatorname{ESS}_N}{N}
=\frac{(N^{-1}\sum_s w_s)^2}{N^{-1}\sum_s w_s^2}
\longrightarrow 2^{-D_2(P\Vert Q)}\quad\text{almost surely}.
$$

유한 알파벳·양의 Q에서는 두 모멘트가 유한합니다. 일반 연속 상태에서는 이 조건을 다시 확인해야 합니다. N·2^(−D₂)를 유한 N의 기대 ESS라고 부르면 안 됩니다. 표본에서 빠진 영역과 특정 f의 정확도를 이 가중치 지표 하나가 보증하지도 않습니다.

## 독립 좌표의 작은 차이가 누적되는 방식

각 좌표에서 P=(4/5,1/5), Q=(1/2,1/2)를 쓰고 d개 좌표가 두 분포 아래 각각 독립이라고 합시다. 결합 가중치는 좌표 가중치의 곱입니다. Q 아래 두 번째 모멘트도 곱으로 분리되므로

$$
E_{Q^{\otimes d}}[w^2]=(34/25)^d,\qquad
D_2(P^{\otimes d}\Vert Q^{\otimes d})=d\log_2(34/25).
$$

아래 마지막 열은 d를 고정하고 N→∞로 보낸 ESS/N의 극한입니다. 작은 N에서의 예상 표본 ESS가 아닙니다.

| d | D₂ bits | 극한 비율 |
|---|---|---|
| 1 | 0.44361 | 0.73529 |
| 10 | 4.43607 | 0.04620 |
| 20 | 8.87213 | 0.00213 |

이 예제에서는 좌표마다 같은 불일치를 반복해 지수적 모멘트 증가를 얻었습니다. 차원이 크다는 이유만으로 항상 이 식의 악화가 생기는 것은 아닙니다. P=Q이면 어떤 d에서도 발산은 0입니다. d와 함께 제안의 일치도가 변하는 경우도 별도 분석해야 합니다.

## 두 표본만으로 기대 ESS를 정확히 구하기

d=1,N=2의 네 순서쌍을 열거하면 가중치가 같을 때 ESS=2, 다를 때 ESS=25/17입니다. 각각 전체 확률이 1/2이므로 기대 ESS는 59/34≈1.73529입니다. 반면 모집단 극한 계수에 N을 곱하면 25/17≈1.47059로 다릅니다. N=2에서는 항상 ESS/N≥1/2이므로 d=10의 모집단 비율 약 0.0462를 그대로 대입할 수도 없습니다.

코드는 d=1,2,3의 전체 상태와 두 표본을 분수로 열거하고, 더 큰 d의 모멘트는 정확한 거듭제곱으로 계산합니다. 기존 조절기는 한 이진 좌표와 차수 α를 보여주는 별도 예제입니다.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod, log2

p, q = [Q(4, 5), Q(1, 5)], [Q(1, 2), Q(1, 2)]
base = sum(p[i]**2/q[i] for i in range(2))
assert base == Q(34, 25)
for d in [1, 2, 3]:
    states = list(product(range(2), repeat=d))
    masses = [prod(q[i] for i in state) for state in states]
    weights = [prod(p[i]/q[i] for i in state) for state in states]
    assert sum(a*w for a, w in zip(masses, weights)) == 1
    second = sum(a*w*w for a, w in zip(masses, weights))
    assert second == base**d
    expected_ess = sum(masses[i]*masses[j]*(weights[i]+weights[j])**2
                       /(weights[i]**2+weights[j]**2)
                       for i, j in product(range(len(states)), repeat=2))
    if d == 1:
        assert expected_ess == Q(59, 34)
        assert expected_ess != 2/second
    print("d, exact expected N=2 ESS, population coefficient times two:",
          d, expected_ess, 2/second)
for d in [1, 10, 20]:
    print("d, D2 in bits, limiting ESS/N:", d, d*log2(float(base)), float(1/base**d))
```

## 연구에 적용하기 전에 묻기

1. D₂로 모든 관심 함수의 오차를 결정하나요? **아니요. 가중치 모멘트와 함수별 오차를 구분합니다.**
2. N·2^(−D₂)는 유한 표본 기대 ESS인가요? **아니요. 고정 분포의 큰 N 극한에서 나온 계수입니다.**
3. 고차원 실험에 무엇을 기록하나요? **좌표 의존성·제안 불일치·지지·모멘트 조건과 표본 수를 명시합니다.**

Rényi 발산·χ² 관계의 배경은 [van Erven–Harremoës, §I-B](https://arxiv.org/html/1206.2459v2#S1.SS2)를 참고하세요. 가중치 항등식·곱 법칙·유한 ESS 열거는 위 분포에서 직접 유도했습니다.

## 확인 문제와 해설

1. 두 상태 중 하나만 양수이면 H₀는? **log₂1=0입니다.**
2. H₂에 쓰는 충돌확률은? **두 독립 표본이 같은 상태일 확률입니다.**
3. α=8은 정확한 무한 차수인가요? **아니요. 유한 차수와 극한을 구분합니다.**

추가 읽기: [van Erven & Harremoës, Rényi Divergence and Kullback–Leibler Divergence](https://arxiv.org/abs/1206.2459).
