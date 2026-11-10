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

## 손계산과 직접 풀기

n=2의 가설은 (++),(-+),(--)입니다. Sign (++),(-+),(--)에서는 완전히 맞는 가설이 있어 최대 상관이 1입니다. Sign (+-)에서는 최댓값이 0입니다. 따라서 정확한 empirical Rademacher 복잡도는 (1+1+1+0)/4=0.75입니다. Query mask 1이 바로 만들 수 없는 (+-) label입니다.

직접 풀기: n=1의 가설과 Rademacher 평균을 구하세요. 풀이: +와 - 두 가설이 있고 어느 sign이 나와도 일치하는 가설을 골라 상관 1을 얻습니다. 따라서 값은 1이며 한 점은 shatter됩니다. VC 차원 1과 현재 n=1의 복잡도 1은 단위와 정의가 다른 숫자입니다.

## AI 표현력과 일반화

가설의 label 표현력, 임의 sign에 맞출 능력, 실제 학습 데이터에서의 위험은 서로 다른 진단입니다. Parameter가 많아도 제약과 데이터 geometry에 따라 유효 class가 달라질 수 있습니다. 유한 label 행렬, 일반 shattering 논증, 위험 bound의 가정을 연결해 읽으세요. 이 threshold 예제를 곧바로 모든 neural architecture의 VC 차원으로 일반화할 수는 없습니다.

참고: [Understanding Machine Learning 저자 페이지](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/).
