## 가설이 고정된 점들에서 만들 수 있는 Label

VC 차원은 parameter 수를 단순히 세는 개념이 아닙니다. 같은 점 집합에서 가능한 모든 label 배치를 표현할 수 있는지를 묻습니다. 현재 가설은 실수 직선의 증가 threshold이고 n개의 서로 다른 점을 순서대로 놓습니다. 왼쪽은 -1, threshold 이후는 +1입니다. 위 행렬은 t=0부터 n까지 실제 가설 row를 모두 보여 줍니다.

열은 순서대로 놓인 점, row는 threshold 위치입니다. +1과 -1을 색뿐 아니라 숫자로 표시합니다. Query mask의 낮은 n개 bit만 label로 사용합니다. 아래 파랑은 각 row와 query의 mismatch 수이므로 구현이 표현할 수 없는 label도 직접 확인할 수 있습니다.

## Shattering과 VC 차원의 논증

Threshold가 이동하면 가능한 label은 정확히 n+1개입니다. Shattering은 같은 점에서 모든 2^n개 label을 표현해야 합니다.

$$
\Pi_H(n)=n+1,\qquad
\mathrm{VC}(H)=1.
$$

한 점은 threshold를 앞뒤로 옮겨 양쪽 label을 만들 수 있습니다. 하지만 어떤 두 점 x₁<x₂에서도 (+1,-1)은 만들 수 없습니다. 따라서 어떤 두 점도 shatter하지 못하고 더 큰 집합도 shatter하지 못합니다. 이 논증은 현재 화면의 한 점 배치만 검사한 것이 아니라 임의의 순서 있는 두 점에 적용됩니다.

## Rademacher 복잡도는 다른 평균이다

고정한 점에서 독립적인 공정 sign sigmaᵢ∈{-1,+1}을 상상합니다. 각 sign에 대해 가장 잘 상관되는 가설을 고른 뒤 그 최대값을 평균냅니다.

$$
\widehat{\mathfrak R}(H)=E_\sigma\left[\max_{h\in H}\frac1n\sum_i\sigma_i h(x_i)\right]
\le\min\left(1,\sqrt{\frac{2\log(n+1)}n}\right).
$$

현재는 n≤10이라 2^n sign을 실제로 모두 계산합니다. 가설 row 수 n+1과 sign 개수 2^n은 다른 대상입니다. Upper bound는 유한 class의 [-1,1] 함수에 대한 계산이며 일반화 bound로 옮길 때에는 loss class, IID와 confidence 조건도 필요합니다. 이 화면의 숫자를 모든 손실의 오차 상한으로 직접 부르지 않습니다.

## Random Walk의 최솟값과 Loss 복잡도

부호열의 처음 t개 합을 S_t, S_0=0이라 하자. Threshold t의 correlation은 S_n−2S_t이므로 정확한 empirical 복잡도는

$$
\widehat{\mathfrak R}(H)=\frac1nE_\sigma[S_n-2\min_{0\le t\le n}S_t]=-\frac2nE_\sigma\min_t S_t.
$$

기대 안에서 threshold를 최대화하는 식이며 평균 correlation을 먼저 구해 최대화한 값이 아니다. 아래 코드는 모든 random walk에서 두 표현을 유리수로 대조한다. 유한 가설 행렬의 최적화를 경로 최솟값으로 바꾸는 연결이다.

고정한 binary label y_i도 ±1로 쓰면 0–1 loss는 (1−y_i h_i)/2이다. 가설과 무관한 상수의 Rademacher 기대는 0이고 −sigma_i*y_i도 독립 공정 부호이다. 따라서 이 점들에서 loss class의 empirical 복잡도는 classifier 복잡도의 정확히 절반이다. 일반화에 연결하는 대상은 그 loss class다.

| 서로 다른 순서 점 n | 정확한 Classifier 복잡도 |
|---|---:|
| 1 | 1 |
| 2 | 3/4 |
| 4 | 19/32 |

```python
from itertools import product
from fractions import Fraction as F
for n in (1,2,4):
    hs=[[-1 if i<t else 1 for i in range(n)] for t in range(n+1)]
    total=walk=loss_total=F(0);labels=[(-1)**i for i in range(n)]
    for signs in product((-1,1),repeat=n):
        path=[0]
        for sign in signs:path.append(path[-1]+sign)
        best=max(sum(a*b for a,b in zip(signs,h)) for h in hs)
        assert best==path[-1]-2*min(path)
        total+=F(best,n);walk+=F(-2*min(path),n)
        loss_total+=max(F(sum(s*(1-y*v) for s,y,v in zip(signs,labels,h)),2*n) for h in hs)
    rad=total/2**n
    assert total==walk and loss_total==total/2
    print(n,rad)
```

IID 표본에서는 ghost sample의 symmetrization으로 E sup_h(R−Rhat)를 기대 loss 복잡도의 두 배로 제어한다. Empirical ERM과 고정 population oracle을 끼우면 기대 ERM 초과도 같은 양 이하이다. Oracle의 signed empirical gap 기대가 0이기 때문이다. Confidence 보장은 추가 집중 논증이 필요하다. 무작위 입력에 중복이 있어도 loss 변환은 유지되지만 서로 다른 점의 n+1 labeling 수를 그대로 쓸 수 없다. 화면의 완전 열거는 고정한 서로 다른 점들에 조건부이다.

## 손계산과 직접 풀기

n=2의 가설은 (++),(-+),(--)입니다. Sign (++),(-+),(--)에서는 완전히 맞는 가설이 있어 최대 상관이 1입니다. Sign (+-)에서는 최댓값이 0입니다. 따라서 정확한 empirical Rademacher 복잡도는 (1+1+1+0)/4=0.75입니다. Query mask 1이 바로 만들 수 없는 (+-) label입니다.

직접 풀기: n=1의 가설과 Rademacher 평균을 구하세요. 풀이: +와 - 두 가설이 있고 어느 sign이 나와도 일치하는 가설을 골라 상관 1을 얻습니다. 따라서 값은 1이며 한 점은 shatter됩니다. VC 차원 1과 현재 n=1의 복잡도 1은 단위와 정의가 다른 숫자입니다.

## AI 표현력과 일반화

가설의 label 표현력, 임의 sign에 맞출 능력, 실제 학습 데이터에서의 위험은 서로 다른 진단입니다. Parameter가 많아도 제약과 데이터 geometry에 따라 유효 class가 달라질 수 있습니다. 유한 label 행렬, 일반 shattering 논증, 위험 bound의 가정을 연결해 읽으세요. 이 threshold 예제를 곧바로 모든 neural architecture의 VC 차원으로 일반화할 수는 없습니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
