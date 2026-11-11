## 어떤 간선을 둘지와 값은 얼마인지는 다른 문제

모수 학습은 구조를 고정한 뒤 확률표를 추정합니다. 구조 학습은 간선 자체를 비교합니다. 이 예제는 완전 관측된 이진 A,B에 대해 독립 M0:p(A)p(B)와 한 간선 M1:p(A)p(B|A)만 비교합니다. 자유 모수 수는 각각 2와 3입니다. 임의 그래프 전체의 구조 탐색 알고리즘을 구현한 것은 아닙니다.

## 열여섯 자료의 MLE

기본 count는 (5,3,3,5)입니다. 두 주변이 모두 .5라 독립 MLE는 네 질량 .25이고, 간선 모형의 MLE는 경험 결합 (5,3,3,5)/16입니다. 양의 count에만 count×log 확률을 합하며 영 count는 0으로 기여합니다.

$$
\ell(\theta)=\sum_{a,b}n_{ab}\log p_\theta(a,b).
$$

간선 모형은 독립 모형을 포함하므로 최적 우도는 작아지지 않습니다. 그러나 더 큰 최적 우도만으로 일반화 성능이나 올바른 구조가 증명되지는 않습니다.

## BIC의 추가 모수 비용

작을수록 좋은 BIC=−2ℓMLE+k ln N을 사용합니다. 작은 기본 자료에서는 추가 모수 비용 때문에 독립 모형이 선택됩니다. 같은 비율을 8번 반복한 count에서는 우도 이득이 N에 비례하고 penalty는 ln N이므로 간선 모형이 선택됩니다. 실제 자료를 재사용해 독립 표본 수를 부풀리는 행위가 정당하다는 뜻은 아닙니다. 화면은 명시한 전체 count를 입력으로 사용하는 모형 예제입니다.

## Smoothing과 점수 계산 분리

α는 조건부 표의 두 값에 같은 pseudocount를 더하는 Laplace형 smoothing입니다. P(B=1|A=1)=(n11+α)/(n10+n11+2α)입니다. 이 표는 예측을 비교하기 위한 것이고 BIC는 항상 MLE로 계산합니다. α를 바꾸어도 BIC가 바뀌지 않는지 확인하세요. Smoothed fit을 MLE인 것처럼 공식에 넣지 않습니다.

## 방향과 인과의 한계

완전한 두 변수 관측 분포는 A→B와 B→A 양쪽으로 모두 인수분해할 수 있습니다. 연관이 있다는 사실은 인과 방향을 식별하지 못합니다. BIC는 후보 모형과 정규성·표본 가정에 의존하는 점근 근사 기준이며 모든 경계·미관측 잠재 모형에서 자동 정당화되지 않습니다.

## 우도 이득은 경험 상호정보량이다

완전 관측된 두 이진 변수의 count n_ab, 총합 N>0, 경험 결합 p̂_ab=n_ab/N를 둡니다. 독립 모형의 MLE는 주변 count를 따로 맞춘 p̂_a p̂_b이고, 한 간선의 완전한 두 변수 모형은 경험 결합 자체를 맞출 수 있습니다. 양의 주변을 가정하면 두 로그우도의 차이는 다음과 같습니다. 영 count 항은 0으로 정의합니다.

$$
\ell_1-\ell_0
=\sum_{a,b}n_{ab}\ln\frac{\widehat p_{ab}}{\widehat p_a\widehat p_b}
=N\widehat I_{\mathrm{nats}}(A;B).
$$

따라서 k₁−k₀=1인 이 후보 쌍의 BIC 차이는 ln N−2NÎ입니다. 간선 모형이 이기려면 2NÎ>ln N이어야 합니다. bit 단위 Î를 넣을 때는 ln 2를 곱해 nats로 바꿉니다. 조건부 표를 반대 방향으로 인수분해해도 같은 양의 결합·같은 세 자유 모수이므로 우도와 BIC는 같습니다. 이 비교로 방향을 고를 수 없는 이유도 보입니다.

| N, 비율 (5,3,3,5) | 2NÎ, nats | BIC₁−BIC₀ |
|---|---|---|
| 16 | 1.010686 | 1.761903 |
| 32 | 2.021372 | 1.444364 |
| 128 | 8.085489 | −3.233459 |

아래 코드는 균등 주변인 예제와 주변이 다른 count (1,2,3,7)를 모두 검사합니다. 일반 자료에서 독립 MLE를 항상 .25로 두면 이 항등식과 점수가 틀립니다. 화면의 세 count 패턴은 모두 주변이 공정하도록 제한한 예제입니다.

## BIC 근사의 범위를 확인하기

정규적인 k차원 모형에서 내부 MLE 근방의 로그우도가 −N배 양의 이차형식으로 감소하고 prior 밀도가 매끄럽고 양수라면, 그 근방의 적분 폭은 각 방향 N⁻¹/²입니다. 모형 evidence의 로그는 ℓMLE−(k/2)ln N+O(1)이 되고, −2배하면 BIC가 나옵니다. 이는 고정 차원·비퇴화 곡률·표본 및 prior 가정을 쓰는 Laplace 설명입니다. 특이한 잠재 모형이나 경계에서 이 차원 비용이 그대로 유지된다고 결론낼 수 없습니다. 화면의 완전 일치 count는 경계 MLE이므로 표시한 공식을 계산하는 예제와 점근 Bayes 근사의 정당성을 구분하세요.

**직접 풀기:** bit MI=.1과 N=100을 그대로 2NÎ에 넣어도 되나요? **해설:** 아닙니다. 우도는 자연로그이므로 이득은 2×100×.1×ln 2입니다.

추가 근거: [Schwarz (1978), 모형 차원 선택의 점근 전개](https://sites.stat.washington.edu/courses/stat527/s14/readings/ann_stat1978.pdf). 이 두 후보의 상호정보량 항등식은 직접 전개했습니다.

```python
from math import log

def scores(counts):
    N = sum(counts)
    p = [n/N for n in counts]  # index a+2*b
    pa = [sum(p[a+2*b] for b in [0,1]) for a in [0,1]]
    pb = [sum(p[a+2*b] for a in [0,1]) for b in [0,1]]
    q = [pa[i%2]*pb[i//2] for i in range(4)]
    ll1 = sum(n*log(v) for n,v in zip(counts,p) if n)
    ll0 = sum(n*log(v) for n,v in zip(counts,q) if n)
    mi = sum(v*log(v/w) for v,w in zip(p,q) if v)
    assert abs((ll1-ll0)-N*mi) < 1e-11
    forward = [pa[i%2]*(p[i]/pa[i%2]) for i in range(4)]
    reverse = [pb[i//2]*(p[i]/pb[i//2]) for i in range(4)]
    assert max(abs(a-b) for a,b in zip(forward,reverse)) < 1e-15
    return N, 2*N*mi, log(N)-2*N*mi

for repetitions in [1,2,8]:
    N,gain,delta = scores([n*repetitions for n in [5,3,3,5]])
    print('N, twice log-likelihood gain, BIC(edge)-BIC(independent):',
          N, round(gain,6), round(delta,6))
scores([1,2,3,7])  # Unequal marginals: the independent MLE is not uniform.
```


## 확인 문제와 해설

1. 간선 모형의 자유 모수는? **3개입니다.**
2. α는 이 BIC 계산을 바꾸나요? **아니요. 별도 smoothing 표만 바꿉니다.**
3. 완전일치 자료가 인과 방향을 정하나요? **아니요. 관측만으로는 추가 가정이 필요합니다.**

추가 읽기: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
