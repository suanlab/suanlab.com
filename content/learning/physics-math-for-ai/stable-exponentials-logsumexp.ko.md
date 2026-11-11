## 답은 유한해도 계산식은 넘칠 수 있다

세 class의 logit을 (c,c-g,c-2g)로 두고 양의 온도 T로 나눈 뒤 확률을 계산합니다. c=1000, g=T=1이면 정상적인 확률분포가 존재합니다. 그러나 exp(1000)을 직접 만들면 binary64 범위를 넘습니다. 무한대끼리 나눈다고 원하는 확률이 복원되지는 않습니다.

c=-1000 부근에서는 반대 문제가 생깁니다. 단순 지수 세 개가 모두 0으로 반올림되어 정규화가 0/0이 됩니다. 확률 모형이 정의되지 않은 것이 아니라 사용한 표현식의 산술이 실패한 것입니다.

## 지수 계산 전에 기준을 옮긴다

온도로 나눈 logit을 z, 그 최댓값을 m이라고 하면 공통 인자를 약분할 수 있습니다.

$$
\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m},\qquad
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}.
$$

Shift된 지수의 인자는 모두 0 이하이고 적어도 하나는 exp(0)=1입니다. 따라서 분모가 0이 되지 않습니다. 거대한 partition function 자체를 만들지 않고 그 로그를 계산하는 것이 핵심입니다.

모든 underflow가 사라지는 것은 아닙니다. 최댓값보다 아주 작은 항의 shifted 지수는 여전히 0으로 반올림될 수 있습니다. 실험은 이 경우를 그대로 표시합니다. 단순 계산이 실패했을 때는 확률을 임의의 0으로 채우지 않고 정의되지 않은 수치로 구별합니다.

## 세 class를 손으로 계산하기

간격과 온도가 1이면 shifted 지수는 (1,e^-1,e^-2)이고 합은 약 1.503215입니다. 확률은 약 (0.665241,0.244728,0.090031)입니다. 공통 이동이 0, 1000, -1000이어도 이 확률은 같습니다.

하지만 log-sum-exp는 c+log(1+e^-1+e^-2)로 바뀝니다. 공통 이동 불변성은 정규화한 확률의 성질이며 log partition 자체의 성질은 아닙니다. 구현을 비교할 때 표시한 값이 확률인지 로그 정규화 상수인지 먼저 확인하세요.

## 온도와 공통 이동은 다르다

작은 온도는 상대 간격을 키워 분포를 집중시킵니다. 공통 이동과 달리 확률을 바꿉니다. 반대로 간격이 0이면 모든 양의 온도에서 확률이 각각 1/3입니다. Classifier의 안정성을 점검할 좋은 불변식입니다.

관측 class k의 cross-entropy는 LSE(z)-z_k로 계산할 수 있습니다. 먼저 작은 확률을 만들고 log를 취하면 이미 0으로 반올림된 정보를 되찾을 수 없습니다. 안정화는 잘못된 값이 나온 뒤 덧붙이는 처리가 아니라 정보를 잃기 전에 식을 다시 쓰는 작업입니다.

## 확률 가중치에서 볼록성과 섭동 상계 얻기

L(z)=logΣexp zᵢ를 미분하면 gradient는 p, Hessian은 diag(p)−ppᵀ다. 임의 벡터 v에서

$$
v^T\nabla^2L\,v=\sum_i p_iv_i^2-(\sum_i p_iv_i)^2
=\operatorname{Var}_p(v_i)\ge0,\qquad \nabla^2L\,\mathbf1=0.
$$

따라서 L은 볼록하고 공통 이동 방향의 곡률은 0이다. L 자체가 이동 불변이라는 뜻은 아니다. L(z+c1)=L(z)+c다. |Δzᵢ|≤ε이면 exp(±ε)를 인수로 제한해 |L(z+Δz)−L(z)|≤ε를 얻는다. 안정적인 산술과 수학적 조건성은 신뢰성의 서로 다른 근거다.

이 세 structured logit에서는 r=g/T로 두고 거의 같은 절대 logit을 빼기 전에 상대값 (0,−r,−2r)을 계산한다. 첫 class의 손실은 log1p(exp(−r)+exp(−2r))다. 표시된 LSE와 최대 logit이 같은 수로 반올림되어도 양의 손실을 유지한다. 확률도 요청한 상대 간격에서 평가하고 raw 절대 logit은 별도 진단으로 남긴다.

| r | 첫 class 손실 |
|---|---|
| 0 | log3 |
| 1 | log(1+e⁻¹+e⁻²) |
| 50 | 약 e⁻⁵⁰ |

```python
from math import exp, log1p, fsum
for r in (0., 1., 50.):
    weights = (1., exp(-r), exp(-2*r))
    p = tuple(x/fsum(weights) for x in weights)
    loss = log1p(weights[1]+weights[2])
    v = (1., -2., .5)
    mean = fsum(a*b for a, b in zip(p, v))
    quadratic = fsum(a*(b-mean)**2 for a, b in zip(p, v))
    assert quadratic >= 0 and loss > 0
    if r == 50:
        assert 1000.+loss == 1000.
        assert abs(loss/exp(-50)-1) < 1e-14
    print(r, loss, "Hessian quadratic:", quadratic)
```

Hessian 이차식을 분산으로 검산하고 offset 1000에서 작은 손실을 보존한다. 이미 같은 절대 logit으로 반올림된 데이터에서 정보를 복구했다는 뜻은 아니다. 개별 지수들이 유한해도 합이 무한대일 수 있으며 그 단순 정규화를 0 확률 벡터로 표시하면 안 된다.

## 직접 풀기: 확률과 정규화 상수

세 logit을 모두 1000으로 설정하세요. 확률은 각각 1/3이고 log-sum-exp는 1000+log(3)입니다. 모든 logit에 10을 더하면 확률은 같고 log-sum-exp는 10 증가합니다. 온도 2에서 원래 logit에 10을 더하면 증가량은 5입니다. 온도로 나누는 순서를 식으로 설명하세요.

더 읽기: [SciPy logsumexp 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html).
