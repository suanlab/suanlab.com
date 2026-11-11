## 속도는 접공간에 놓입니다

매끄러운 다양체의 점 p에서 접벡터는 p를 통과하는 곡선의 속도입니다. chart에서는 함수에 방향 미분으로 작용할 수도 있습니다. 다양체 자체가 벡터 공간이 아니어도 각 점의 접공간은 선형 공간입니다. 원의 두 점을 더하면 대개 원 밖으로 나가지만 같은 점의 접벡터 둘을 더하면 그 접공간 안에 남습니다.

p(θ)=(cos θ,sin θ)를 미분하면 단위 접기저가 나옵니다.

$$
\partial_\theta p=(-\sin\theta,\cos\theta),\qquad
v=a\partial_\theta.
$$

주황 선분은 p에서 시작해 p+a∂θp에서 끝납니다. 원 위에 제한된 이동 경로가 아니라 주변 평면에서 그린 접벡터입니다. 단위원 metric에서 내재 제곱 노름은 a²입니다.

## 여접벡터는 벡터를 입력받습니다

여접공간은 접공간의 쌍대입니다. 여접벡터 covector는 접벡터를 스칼라로 보내는 선형 함수입니다. α=b dθ, v=a∂θ이면 dθ(∂θ)=1이므로 α(v)=ab입니다. 함수의 미분 df도 covector이며 df(v)는 v 방향으로 함수가 변하는 속도입니다.

covector를 보통 화살표로 그리면 벡터와의 식별을 암묵적으로 사용합니다. metric이 이 식별을 제공할 수 있지만 추가 구조입니다. 따라서 그림에는 접벡터만 그리고 covector 성분은 수치로 읽습니다.

## 국소 좌표 배율을 바꾸기

양수 s에 대해 u=sθ로 둡니다. 연쇄법칙으로 ∂u=(1/s)∂θ, du=s dθ입니다. 같은 기하적 대상을 유지하려면 성분은 반대로 변환되어야 합니다.

$$
v^u=sa,\qquad \alpha_u=b/s,\qquad
\alpha_uv^u=ab,\qquad g_{uu}=1/s^2.
$$

metric 제곱 노름도 gᵤᵤ(vᵘ)²=a²로 유지됩니다. a=1,b=2,s=2이면 새 성분은 2와 1이고 pairing은 2, 제곱 노름은 1입니다. 새 벡터 성분만 제곱하면 4가 되어 틀립니다.

## 연쇄법칙이 반대 방향의 성분 변환을 설명합니다

가역 Jacobian J를 가진 매끄러운 국소 좌표 변화에서 곡선의 속도는 J로 변환됩니다. 모든 속도에 대한 평가값을 유지하려면 covector는 역전치로 변환되어야 합니다. 길이 보존을 요구하면 metric의 규칙도 나옵니다.

$$
v'=Jv,\qquad \alpha'=J^{-T}\alpha,\qquad
g'=J^{-T}gJ^{-1},\qquad
\operatorname{grad}'f=J\operatorname{grad}f.
$$

모두 국소 명제입니다. 다른 점의 벡터를 더하려면 여전히 운반 규칙이 필요합니다. 1차원에서 u=θ+(1/2)sinθ로 두면 J=1+(1/2)cosθ는 1/2에서 3/2 사이입니다. 따라서 각도 이웃에서 유효한 증가 좌표 변화입니다. a=b=1, θ=0이면 J=3/2, 벡터 성분 3/2, covector 성분 2/3, metric 4/9입니다. pairing과 길이 제곱은 모두 1입니다.

코드는 세 국소 Jacobian을 정확분수로 검사합니다. covector 성분을 그대로 최적화 이동으로 쓰면 metric을 반영하지 않는 한 gradient 변환 법칙에 어긋납니다. 보통 Euclidean gradient descent가 매개화에 의존하는 기하적 이유입니다.

| J | 벡터 성분 | Covector 성분 | Metric |
|---:|---:|---:|---:|
| 3/2 | 3/2 | 2/3 | 4/9 |
| 1 | 1 | 1 | 1 |
| 1/2 | 1/2 | 2 | 4 |

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
for j in [F(3, 2), F(1), F(1, 2)]:
    v, alpha, g = j, 1/j, 1/j**2
    grad = alpha/g
    assert alpha*v == 1 and g*v*v == 1 and grad == j
    print(j, v, alpha, g)
```

## 직접 풀기와 풀이

a=−2,b=0.5,s=2에서 vᵘ=−4, αᵤ=0.25, pairing −1, 제곱 노름 4를 예상한 뒤 수치로 확인하세요. θ를 옮기면 물리적 접화살표는 회전하지만 이 스칼라들은 유지됩니다. 이어 a=0이면 b와 무관하게 모든 covector가 영벡터에 0을 주는지도 확인하세요.

## 학습 알고리즘과의 연결

자동 미분이 반환하는 편미분은 일반적으로 covector 성분입니다. 이를 바로 gradient 벡터 성분으로 쓰는 것은 특정 metric, 보통 해당 Euclidean 좌표를 가정합니다. 다양체나 비정규직교 매개화에서는 inverse metric으로 index를 올려야 합니다. natural·Riemannian 최적화를 유도할 때 미분·metric·gradient 벡터와 각각의 좌표 성분을 구별하세요.

추가 읽기: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
