## 커널을 유한 특징 내적으로 바꾸기

Gaussian RBF k(x,y)=exp(−(x−y)²/(2ℓ²))는 두 점이 가까울수록 1에 가깝습니다. 전체 커널 행렬을 저장하는 대신 유한한 주파수 특징을 만들 수 있습니다. 이 단원은 1차원 입력에서 cos/sin을 함께 쓰는 paired map을 사용합니다.

$$
z(x)=\frac1{\sqrt m}(\cos\omega_1x,\sin\omega_1x,\ldots,\cos\omega_mx,\sin\omega_mx).
$$

주파수는 m개지만 좌표는 2m개입니다. 무작위 위상 cos 하나를 쓰는 다른 정규화·분산 공식과 혼용하지 마세요.

## 삼각함수와 Gaussian 특성함수

cos a cos b+sin a sin b=cos(a−b)이므로 내적은 cos(ωⱼr)의 평균, r=x−y입니다. 이상적인 독립 ωⱼ~N(0,ℓ⁻²)에서 Gaussian 특성함수의 실수부를 사용하면 기대값은 정확히 RBF입니다.

$$
z(x)^\top z(y)=\frac1m\sum_j\cos(\omega_jr),\qquad E[\cos(\omega r)]=e^{-r^2/(2\ell^2)}.
$$

## 점별 분산을 정확히 계산하기

cos²u=(1+cos2u)/2를 사용하면 한 거리 r의 이상적인 표집 분산을 닫힌식으로 구합니다. s=r/ℓ라 하면 다음과 같습니다. m이 커지면 분산은 1/m로 줄지만 한 seed의 실제 오차가 모든 r에서 단조 감소하는 것은 아닙니다.

$$
\operatorname{Var}(\hat k(r))=\frac{(1-e^{-s^2})^2}{2m}.
$$

r=0이면 모든 cos가 1이어서 근사는 정확히 1, 분산도 0입니다. 단일 실현의 표준오차는 현재 실제 오차와 같은 숫자가 아닙니다. 또한 한 점의 표집 분산은 전체 입력 구간에서의 동시 오차 보장이 아닙니다.

## 재현 가능한 실현을 그림으로 보기

파랑은 정확 RBF, 주황은 seed로 고정한 유사 Gaussian 주파수의 근사입니다. 구현은 재현성을 위해 유사난수와 Box–Muller를 사용하며 이상적인 iid Gaussian 표집 가정의 정리와 분리해 해석합니다. m을 늘리면 같은 seed의 주파수 접두 부분은 유지됩니다.

주황 값이 음수가 될 수 있지만 자르지 않습니다. 유한 특징의 내적이므로 근사 Gram 행렬은 양의 준정부호여도 개별 원소가 모두 양수일 필요는 없습니다. RBF의 양수 원소를 유한 표본마다 강제하는 방법은 다른 추정량입니다.

## AI에서의 활용과 한계

커널 근사, explicit feature 학습, 표현 공간 비용을 이해하는 출발점입니다. 길이 척도 ℓ가 작으면 커널이 빠르게 변하고 같은 m에서 근사가 더 까다로울 수 있습니다. 데이터 차원·관측 범위·오차 확률까지 포함한 균일 보장에는 추가 논증이 필요하며 이 1차원 그림만으로 일반 성능을 선언하지 않습니다.

## 한 쌍의 분산에서 고정한 유한 집합의 보장으로

투영 전에 고정한 N≥2개의 1차원 점, 길이 척도 ℓ>0과 이상적인 독립 ω_j~N(0,ℓ⁻²)를 사용합시다. 여기서는 m개 주파수의 paired cos/sin map을 유지합니다. 오차 문턱 η>0에서 한 쌍의 추정량은 [−1,1] 안의 cos 값 m개의 평균이므로 Hoeffding으로

$$
P(|\widehat k(x,y)-k(x,y)|\geq\eta)
\leq2\exp(-m\eta^2/2).
$$

서로 다른 L=N(N−1)/2쌍에 union bound를 적용하면 최대 원소 오차가 η를 넘을 확률은 2L exp(−mη²/2) 이하입니다. 충분한 주파수 수는 m≥(2/η²)ln(2L/δ)이며 0<δ<1입니다. 쌍들은 같은 주파수를 공유하므로 서로 의존해도 됩니다. 독립성을 요구하는 대상은 주파수 표본입니다. 이 유한 집합 보장은 전체 연속 구간의 균일 보장과 다릅니다.

## 원소 오차가 Gram 연산자 오차로 바뀌는 과정

K와 K̂를 같은 N개 점의 정확·근사 Gram 행렬이라 하고 E=K̂−K로 둡니다. Paired map에서 각 특징의 제곱 노름은 1이므로 K_ii=K̂_ii=1입니다. 모든 비대각 원소 오차가 η 이하라면 단위 벡터 v에 대해

$$
|v^\top Ev|\leq\eta\sum_{i\ne j}|v_i||v_j|
=\eta\left[\left(\sum_i|v_i|\right)^2-1\right]
\leq(N-1)\eta.
$$

따라서 ||E||₂≤(N−1)η입니다. 같은 원소 오차라도 점 개수가 많으면 전체 선형 연산의 오차는 커질 수 있습니다. 두 Gram 행렬은 모두 PSD이며, K̂의 성질은 cᵀK̂c=||Σc_i z(x_i)||²≥0에서 직접 나옵니다.

## ridge 계수는 정규화 크기에 민감하다

λ>0에서 α=(K+λI)⁻¹y, α̂=(K̂+λI)⁻¹y로 정의합니다. 여기에는 Nλ를 넣지 않는 관례를 사용합니다. 역행렬 항등식과 PSD 때문에 두 역행렬의 노름이 각각 1/λ 이하라는 사실로

$$
(\widehat K+\lambda I)^{-1}-(K+\lambda I)^{-1}
=-(\widehat K+\lambda I)^{-1}E(K+\lambda I)^{-1},
$$
$$
\|\widehat\alpha-\alpha\|_2
\leq\frac{\|E\|_2\|y\|_2}{\lambda^2}
\leq\frac{(N-1)\eta\|y\|_2}{\lambda^2}.
$$

이는 학습점에서의 계수 오차 경계입니다. 새로운 점의 예측에는 그 점의 커널 벡터 오차도 추가로 필요하며, 이 식만으로 일반화 오차를 선언하지 않습니다. λ가 작으면 같은 커널 근사 오차가 크게 증폭될 수 있습니다.

## 작은 고정 예제에서 두 오차 비교하기

x=(0,.5,1,1.5), ℓ=1, y=(1,−1,1,−1)을 고정합니다. 아래 Entry는 최대 Gram 원소 오차, Coeff는 λ=1의 ||α̂−α||₂입니다. Python의 별도 고정 주파수 접두열을 사용하므로 화면의 JavaScript seed와 값이 같다는 뜻은 아닙니다.

| m | Entry | Coeff |
|---|---|---|
| 2 | 0.49891 | 0.19455 |
| 8 | 0.33936 | 0.10495 |
| 32 | 0.12520 | 0.02672 |

이 한 실현의 감소를 모든 seed·모든 입력에서의 단조 감소 보장으로 읽지 마세요. 코드는 특징 내적과 cos 평균을 대조하고 λ=.1,1,10에서 선형계를 풀어 위 결정적 경계 안에 있는지 수치적으로 확인합니다. 부동소수점 검산이며 모든 행렬의 엄밀한 반올림 오차 인증은 아닙니다.

```python
from math import cos, sin, exp, sqrt
from random import Random

def solve(matrix, vector):
    n = len(vector)
    a = [list(row)+[value] for row, value in zip(matrix, vector)]
    for j in range(n):
        pivot = a[j][j]
        assert pivot > 0  # These unpivoted pivots are positive for the SPD systems below.
        a[j] = [v/pivot for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [v-factor*w for v, w in zip(a[i], a[j])]
    return [row[-1] for row in a]

def norm(v):
    return sqrt(sum(x*x for x in v))

x, y = [0, .5, 1, 1.5], [1, -1, 1, -1]
N = len(x)
K = [[exp(-.5*(a-b)**2) for b in x] for a in x]
rng = Random(7)
omega = [rng.gauss(0, 1) for _ in range(32)]
for m in [2, 8, 32]:
    z = [[v/sqrt(m) for w in omega[:m] for v in [cos(w*a), sin(w*a)]] for a in x]
    approx = [[sum(a*b for a, b in zip(zi, zj)) for zj in z] for zi in z]
    cosine = [[sum(cos(w*(a-b)) for w in omega[:m])/m for b in x] for a in x]
    assert max(abs(approx[i][j]-cosine[i][j]) for i in range(N) for j in range(N)) < 1e-12
    error = max(abs(approx[i][j]-K[i][j]) for i in range(N) for j in range(N))
    for lam in [.1, 1, 10]:
        exact_system = [[K[i][j]+lam*(i == j) for j in range(N)] for i in range(N)]
        approx_system = [[approx[i][j]+lam*(i == j) for j in range(N)] for i in range(N)]
        exact, estimated = solve(exact_system, y), solve(approx_system, y)
        difference = norm([a-b for a, b in zip(exact, estimated)])
        bound = (N-1)*error*norm(y)/lam**2
        assert difference <= bound+1e-12
        if lam == 1:
            print('m, max entry error, coefficient error at lambda=1, upper bound:',
                  m, error, difference, bound)
```

## 연구에 적용하기 전에 묻기

1. m은 특징 좌표 수인가요? **여기서는 주파수 수이고 좌표는 2m개입니다.**
2. 작은 원소 오차만으로 같은 크기의 계수 오차가 보장되나요? **아니요. N과 λ의 증폭을 확인해야 합니다.**
3. 평가점이 주파수를 본 뒤 선택되면? **고정 유한 집합 보장의 가정을 다시 확인해야 합니다.**

배경은 [Rahimi–Recht, Random Features for Large-Scale Kernel Machines §3](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)를 참고하세요. 여기서는 paired map의 [−1,1] 평균에 맞추어 상수를 계산하고 유한 Gram·ridge 경계를 직접 유도했습니다.

## 확인 문제와 해설

1. m=32일 때 좌표 수는? **64입니다.**
2. r=0에서 오차는? **정확히 0입니다.**
3. 근사 원소가 음수면 PSD가 불가능한가요? **아니요. 행렬의 PSD와 원소별 양수는 다른 조건입니다.**

추가 읽기: [Rahimi & Recht, Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf).
