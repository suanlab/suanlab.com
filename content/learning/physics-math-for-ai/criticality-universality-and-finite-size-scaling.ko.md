## 1. 유한계의 요동에서 임계현상으로

유한 이징 격자의 분배함수는 양의 유한 온도에서 양의 가중치를 유한 번 더한 값입니다. 그 자유에너지는 매끄럽습니다. 계가 커지면 요동이 더 먼 격자점까지 이어지고 완만한 봉우리가 뾰족해질 수 있습니다. 임계 특이성은 열역학적 극한에서의 성질입니다. 유한계의 높은 봉우리는 계 크기에 따른 관측값입니다.

## 2. 임계 기준점과 상관 길이

가로 결합 $K_x=J_x/T$, 세로 결합 $K_y=J_y/T$, 외부장 0인 정사각 격자 강자성 이징 모형을 사용합니다. 무한 격자의 전이선은 다음 식을 만족합니다.

$$
\sinh(2K_x)\sinh(2K_y)=1.
$$

두 결합이 같으면 $K_c=\tfrac12\log(1+\sqrt2)\approx 0.440687$입니다. 유한 격자의 결과를 해석할 때 무한 격자의 이 결과를 외부 기준점으로 사용합니다. 1차원 최근접 이징 사슬의 상관 길이는 온도가 0에 접근할 때 커지며, 양의 유한 온도에서는 유한합니다. 두 모형은 같은 스핀 대칭을 가지며 공간 차원이 다릅니다. [Onsager, Crystal Statistics I](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117) · [Tong, Statistical Physics §5](https://davidtong.org/teaching/statistical-physics/statmechhtml/S5)

## 3. 보편성에서 비교하는 양

$t=(T-T_c)/T_c$라 둡니다. 임계점 근처의 상관 길이는 $\xi\sim |t|^{-\nu}$로 증가합니다. 질서가 형성되는 쪽의 자발 자화는 $m_{\rm sp}\sim(-t)^\beta$, 연결 감수율은 $\chi\sim|t|^{-\gamma}$의 거동을 보입니다. 임계점에서 연결 상관함수는 일반적으로 $r^{-(d-2+\eta)}$로 감소합니다.

보편성은 기하와 정규화 조건을 맞춘 뒤 임계 지수와 척도 함수를 비교합니다. 임계 온도와 진폭은 미시 결합에 따라 달라질 수 있습니다. 2차원 정사각 격자 이징 모형의 지수는 $\beta=1/8$, $\gamma=7/4$, $\nu=1$, $\eta=1/4$입니다. 평균장 이론의 지수는 $\beta=1/2$, $\gamma=1$, $\nu=1/2$, $\eta=0$입니다. 평균장 계산으로 근사가 어떤 값을 예측하는지 확인하고 정확한 2차원 값과 비교해 요동의 영향을 읽습니다. [Tong, Statistical Field Theory §§1.2 및 2.2](https://davidtong.org/teaching/statistical-field-theory/)

## 4. 분포의 모양을 읽는 무차원 관측량

전체 자화를 $M=\sum_i s_i$, 격자점 수를 $N=L^2$라 둡니다. 외부장 0인 유한계는 부호 반전 대칭으로 $\langle M\rangle=0$입니다. 자화 분포는 서로 부호가 다른 두 질서 상태에 집중될 수 있습니다. Binder 비율은 2차와 4차 모멘트로 분포의 모양을 읽습니다.

$$
U_L=1-\frac{\langle M^4\rangle}{3\langle M^2\rangle^2}.
$$

독립인 무편향 스핀에서는 $\langle M^2\rangle=N$, $\langle M^4\rangle=3N^2-2N$이므로 $U_L=2/(3N)$입니다. $M=\pm N$ 두 값에 집중한 분포에서는 $U_L=2/3$입니다. 유한계의 분포가 두 봉우리로 뚜렷하게 모일수록 이 비율이 커지는 이유입니다. [Binder, Finite Size Scaling Analysis of Ising Model Block Distribution Functions](https://doi.org/10.1007/BF01293604)

## 5. 유한 크기에서 교점이 이동하는 이유

종횡비와 경계 조건을 고정하면 임계점 근처의 유한 크기 척도 관계의 선도항은 $U_L(t)\approx f(tL^{1/\nu})$입니다. $t=0$에서 선도항은 $L$에 의존하지 않습니다. 보정항 때문에 두 유한 크기의 곡선이 만나는 곳은 열역학적 극한의 임계점에서 벗어날 수 있습니다. 여러 크기 쌍을 비교하고 사용한 기하를 함께 적으세요.

실습의 원통 격자는 행 내부에 주기 경계, 행이 쌓이는 방향에 열린 경계를 둡니다. 크기를 바꿀 때 정사각형 종횡비를 유지합니다. 원통 전체 자화의 Binder 비율을 계산합니다.

## 6. 행을 쌓으며 유한계 모멘트 계산하기

행 상태 $b$에는 스핀 $L$개가 있고 행 자화는 $m_b$입니다. 마지막 행이 $a$일 때 누적한 비정규화 $k$차 모멘트를 $S_k(a)$라 둡니다. 새 행 $b$를 붙이면 전체 자화가 $m_b$만큼 증가합니다. 거듭제곱을 전개하면 다음 갱신식을 얻습니다.

$$
S_k^{\rm new}(b)=\sum_a w_{ab}\sum_{j=0}^k {k\choose j} S_j(a)m_b^{k-j}.
$$

전달 가중치 $w_{ab}$에는 새 행 $b$의 가로 결합과 $a$, $b$ 사이의 세로 결합이 들어갑니다. 마지막 행에 대해 합한 뒤 모멘트를 $S_0$으로 나눕니다. 지수에서 가능한 최대 결합 합을 빼면 공통 인자가 생기고 정규화한 모멘트에서는 상쇄됩니다.

```python
from math import comb
import numpy as np

def binder(length, kx, ky):
    bits = np.arange(1 << length)[:, None] >> np.arange(length)
    spins = 2 * (bits & 1) - 1
    horizontal = (spins * np.roll(spins, -1, axis=1)).sum(axis=1)
    m = spins.sum(axis=1)
    row = np.exp(kx * (horizontal - length))
    transfer = np.exp(ky * (spins @ spins.T - length)) * row[None, :]
    moments = np.array([row * m**k for k in range(5)])
    for _ in range(length - 1):
        incoming = moments @ transfer
        moments = np.array([
            sum(comb(k, j) * incoming[j] * m**(k-j) for j in range(k+1))
            for k in range(5)
        ])
    z, _, m2, _, m4 = moments.sum(axis=1)
    return 1 - m4 * z / (3 * m2**2)

def crossing(left, right):
    lo, hi = 0.4, 0.6
    assert binder(left, lo, lo) > binder(right, lo, lo)
    assert binder(left, hi, hi) < binder(right, hi, hi)
    for _ in range(25):
        mid = (lo + hi) / 2
        if binder(left, mid, mid) > binder(right, mid, mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

for length in [3, 4, 6, 8]:
    np.testing.assert_allclose(binder(length, 0, 0), 2/(3*length**2), atol=1e-14, rtol=0)
print('infinite Kc', np.arcsinh(1)/2)
print('finite crossings', crossing(3, 4), crossing(6, 8))
```

## 7. 유한 크기 추론을 점검하기

- $K=0$에서 모든 시험 크기의 $U_L=2/(3L^2)$를 재현하세요. 정규화와 모멘트 갱신을 점검하는 기준입니다.
- 크기 쌍 $(3,4)$와 $(6,8)$의 교점을 비교하세요. 알려진 등방성 $K_c$와 각각 얼마나 차이가 나는지 기록합니다.
- 가로·세로 결합을 함께 바꾸고 정확한 임계선과 비교하세요. 유한 크기의 교점은 종횡비, 경계 조건, 크기 보정항에도 영향을 받습니다.
- 표본을 사용한다면 시간 상관을 반영할 만큼 긴 블록으로 불확실성을 추정하세요. 이 행 계산에는 주어진 유한 격자의 표본 오차가 없습니다.
