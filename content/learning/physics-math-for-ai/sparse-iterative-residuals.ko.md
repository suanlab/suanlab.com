## 0을 저장하는 대신 연산을 표현한다

경계값이 0인 일차원 Poisson 연산자는 대각 2, 이웃 성분 -1의 삼중대각 행렬입니다. Reaction c를 더하면 대각이 2+c가 됩니다. 내부 미지수가 N개이면 nonzero는 3N-2개뿐입니다. 실험은 반복 곱셈에서 dense 행렬을 만들지 않고 세 점 stencil을 직접 적용합니다.

우변은 모두 1입니다. 별도의 pivoted 직접 해법으로 기준 해를 계산합니다. c=0에서는 1부터 시작하는 index i에 대해 i(N+1-i)/2라는 독립적인 해도 있습니다. 복잡한 시뮬레이션 전에 검산하기 좋은 이산 포물선입니다.

## Jacobi의 오차를 고유값으로 읽는다

0에서 시작하는 weighted Jacobi는 다음과 같습니다.

$$
x_{k+1}=x_k+\frac\omega{2+c}(b-Ax_k),\qquad
\lambda_j=2+c-2\cos\frac{j\pi}{N+1}.
$$

오차 전파 행렬의 spectral radius는 max_j |1-omega lambda_j/(2+c)|입니다. 1보다 작으면 이 대칭 모형의 모든 초기 오차가 수렴합니다. 1보다 크면 증가하는 mode가 가능하지만 특정 초기 오차가 그 mode를 곧바로 크게 드러낸다는 뜻은 아닙니다.

c=0에서 N이 커지면 최소 고유값이 작아지고 매끄러운 오차 mode가 천천히 줄어듭니다. 각 sparse matvec의 비용은 N에 비례해도 충분한 정확도를 얻기 위한 반복 수가 커질 수 있습니다.

## CG는 실제 conjugate 방향을 만든다

CG는 SPD 행렬에서 잔차와 이전 방향으로 새로운 탐색 방향을 구성합니다. 실험은 실제 step length·잔차·방향 계수를 계산합니다. 정해진 감쇠 곡선을 대신 그리지 않습니다. 새 stencil 곱으로 다시 계산한 상대 잔차가 10^-12 이하이면 갱신을 멈추고 실제 update 수를 표시합니다.

N=8, c=0, 대칭 우변에서는 이 예의 CG가 네 번에 수렴합니다. 대칭성이 활성 부분공간을 줄이기 때문입니다. 정확 산술에서 N회 이내 종료한다는 결과를 모든 부동소수점 시스템의 무조건적 보장으로 읽으면 안 됩니다.

## 잔차가 오차 인증이 되려면

오차 e=x-xstar, 잔차 r=b-Ax에 대해 SPD 성질로 다음 상한을 얻습니다.

$$
\|e\|_2\le\|r\|_2/\lambda_{\min}(A).
$$

작은 최소 고유값의 역수가 중요합니다. 잔차가 작다는 사실만으로 해가 충분히 정확하다고 단정할 수 없습니다. 실험은 실제 해 오차·다시 계산한 true residual·그 상한을 함께 보여줍니다. 재귀적으로 갱신한 CG 잔차는 반올림으로 달라질 수 있어 중단 판단에는 새 곱셈을 사용합니다.

## Krylov 최적성에서 조건수별 CG 상계로

정확한 산술과 SPD A에서 CG는 x₀+span(r₀,Ar₀,…,Aᵏ⁻¹r₀)의 A-노름 오차를 최소화한다. 각 탐색 방향의 이차 목적 미분이 최소점에서 0이며, conjugate 방향은 제한된 이차식을 대각화하므로 순차 정확 선 최소화가 이 성질을 준다. 경쟁 오차는 차수 k 이하, p(0)=1인 p(A)e₀다. A를 대각화하면

$$
\|e_k\|_A\le\min_{p(0)=1,\deg p\le k}
\max_{\lambda\in[\lambda_{min},\lambda_{max}]}|p(\lambda)|\,\|e_0\|_A
\le2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k\|e_0\|_A.
$$

마지막에는 스펙트럼 구간에 맞춘 Chebyshev 다항식을 고른다. [−1,1]에서 절댓값은 1 이하이고, 0을 변환한 점의 값은 표시한 비율의 역 k제곱의 절반 이상으로 커져 계수 2를 얻는다. κ=1이면 A가 identity의 스칼라 배수라 정확한 한 보폭으로 풀린다.

고유값 (1,4,9), 우변 ones, 초기값 0의 대각 예에서는 κ=9, 비율 1/2다. 유리수 CG로 A-오차 제곱을 비교하자.

| k | 보장된 오차 제곱 배율 |
|---|---:|
| 1 | 1 |
| 2 | 1/4 |
| 3 | 1/16 |

```python
from fractions import Fraction as F
A = (F(1), F(4), F(9))
x, r = [F(0)]*3, [F(1)]*3
p = list(r)
initial_error = sum(1/a for a in A)
for k in range(1, 4):
    Ap = [a*v for a, v in zip(A, p)]
    rr = sum(v*v for v in r)
    alpha = rr/sum(v*w for v, w in zip(p, Ap))
    x = [v+alpha*d for v, d in zip(x, p)]
    next_r = [v-alpha*w for v, w in zip(r, Ap)]
    beta = sum(v*v for v in next_r)/rr
    p = [v+beta*d for v, d in zip(next_r, p)]
    r = next_r
    error = sum(a*(v-1/a)**2 for a, v in zip(A, x))
    bound = 4*F(1, 4)**k*initial_error
    assert error <= bound
    print(k, error, bound)
assert error == 0
```

실제 세 번째 오차는 0이지만 상계는 양수다. 유한 정밀도에서는 conjugacy를 잃을 수 있다. 기존 실습은 true residual을 다시 계산하고 실제 갱신 수를 표시하며 이 정확 산술 상계를 관측 경로로 대신 그리지 않는다.

## 직접 풀기: Stencil과 비용 확인

N=4, c=0의 해는 (2,3,3,2)입니다. 삼중대각 행렬을 곱하면 모두 1이 되는지 확인하세요. Nonzero는 10개로 dense의 16개보다 적습니다. N을 늘려 최소 고유값과 Jacobi 오차를 비교하고 omega=1.2에서 초기 오차가 작아 보여도 수렴 보장을 주장할 수 없는 이유를 설명하세요.

더 읽기: [SciPy CG 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
