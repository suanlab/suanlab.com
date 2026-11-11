## 함수 하나 대신 함수값들의 일관된 확률 모형

Gaussian process(GP)는 임의의 유한 입력 집합에서 함수값 벡터가 다변량 Gaussian이 되는 확률 과정입니다. 평균 함수 m과 공분산 kernel k가 필요하며 모든 유한 Gram 행렬이 PSD여야 합니다. 여기서는 평균 0, amplitude 1의 RBF 사전과 독립 Gaussian 관측 잡음을 사용합니다.

$$
f(X)\sim N(0,K),\qquad y=f(X)+\epsilon,\quad\epsilon\sim N(0,\sigma^2I).
$$

슬라이더의 σ²는 **분산**입니다. 표준편차를 대각에 그대로 넣으면 다른 모형이 됩니다. 세 관측만으로 모든 함수를 동일하게 가능하다고 보는 것이 아니라 kernel 사전의 구조로 관측 사이를 연결합니다.

## Gaussian 조건화로 평균과 분산 얻기

질의 x*에서 k*=k(X,x*), k**=k(x*,x*)라고 합시다. 공동 Gaussian의 조건부 분포에서 다음을 얻습니다.

$$
\begin{aligned}
m_*&=k_*^T(K+\sigma^2I)^{-1}y,\\
v_*&=k_{**}-k_*^T(K+\sigma^2I)^{-1}k_*.
\end{aligned}
$$

단위 amplitude와 앞 단원의 합 손실 관례에서는 GP 평균이 λ=σ²인 kernel ridge와 같습니다. 그러나 GP는 함수 분산도 제공하고, 모형 선택·prior 해석은 단순한 점 추정과 다릅니다.

## 잠재 함수와 새 관측을 분리하기

v*는 잡음 없는 잠재 함수의 불확실성입니다. 독립 잡음을 더해 새 y*를 관측한다면 분산은 v*+σ²입니다. σ²=.1이면 표의 두 분산 차이는 항상 .1입니다. σ²=0이고 질의가 관측 위치이면 잠재 분산은 0입니다. 관측 밖으로 이동하면 보통 kernel 사전의 분산 쪽으로 돌아가지만 유한 구간에서 반드시 단조로 증가한다는 보장은 아닙니다.

$$
\operatorname{Var}(y_*\mid y)=v_*+\sigma^2.
$$

코드는 선형 해법과 아래에 설명한 안정적인 조건부 공분산 차를 사용합니다. 계산한 분산이 음수이면 오류로 처리하며 대각 jitter를 더하지 않습니다.

## 그림의 구간이 보장하는 범위

파랑은 posterior 평균, 주황은 평균 ±2√v*입니다. 올바른 Gaussian 모형 아래 고정한 한 질의의 조건부 확률 질량은 약 95.45%입니다. 모든 x에서 동시에 함수를 덮는 95% band가 아니며 모형이 틀리거나 분포가 바뀌었을 때의 보장도 아닙니다.

관측 y를 바꿔도 kernel과 잡음 분산을 고정하면 posterior 분산은 변하지 않습니다. 실제 이분산 자료나 학습한 kernel hyperparameter의 불확실성을 다루려면 현재 모형을 확장해야 합니다.

## 독립인 잔차를 만들어 조건화를 유도하기

A=K+σ²I가 양정이라고 합시다. 공동 Gaussian 벡터 (y,f*)에서 z=f*−k*ᵀA⁻¹y를 정의하면 Cov(z,y)=0입니다. 공동 Gaussian에서는 비상관이 독립을 주므로 y를 조건으로 해도 z의 분포는 그대로입니다. 평균 0, 분산 k**−k*ᵀA⁻¹k*를 계산하면 두 예측 공식을 얻습니다. 공동 공분산의 PSD 성질도 Schur complement의 비음수를 줍니다.

여러 질의에는 k* 대신 교차 공분산 행렬을 쓰며 비대각 posterior 공분산도 중요합니다. 점별 구간은 자동으로 동시 구간이 되지 않습니다. 독립 표준정규 두 개조차 각 ±2 구간에 함께 들어갈 확률은 .9545가 아니라 erf(√2)²≈.9111입니다. 이 독립 두 변수 예제는 설명용이며 실제 GP 두 질의가 독립이라는 주장이 아닙니다.

## 작은 양의 분산과 정확한 보간을 구분하기

수학 공식이 맞아도 거의 같은 두 단위량의 차에서는 분산 전체를 잃을 수 있습니다. 실습의 length scale 1에서 아래 값은 450자리 Decimal로 독립 계산할 수 있습니다.

| 잡음 분산 | 질의 x | 잠재 분산 |
|---|---|---|
| 0 | 0 | 정확히 0 |
| 10⁻³⁰ | 0 | 약 10⁻³⁰ |
| 0 | 10⁻⁸ | 약 1.49081872×10⁻¹⁷ |
| 0 | 10⁻¹⁰⁰ | 약 1.49081872×10⁻²⁰¹ |

구현은 가장 가까운 잡음 없는 관측을 먼저 조건화하고 expm1으로 작은 공분산 차를 계산한 뒤 나머지 두 점을 조건화합니다. 잡음 효과에는 resolvent 항등식을 씁니다.

$$
v_\lambda(x)=v_0(x)+\lambda k_x^T K^{-1}(K+\lambda I)^{-1}k_x,
\qquad\lambda=\sigma^2.
$$

단위 대각에 작은 잡음 분산을 더해 반올림으로 사라져도 이 식은 명시한 잡음을 유지합니다. 고정된 서로 다른 세 입력의 양정 선형계를 풀며 jitter를 추가하지 않습니다. 두 역행렬의 차에서 나온 항등식으로 별도의 확률 가정이 아닙니다.

수학적으로 양수지만 binary64 범위 아래인 분산은 —와 수치 범위 안내로 표시하며 구간 반폭과 새 관측 분산도 함께 보류합니다. 정확한 무잡음 훈련 위치의 0은 유지합니다. 양의 구간 폭이 kernel·잡음 가정과 실제 자료의 적합성을 증명하지는 않습니다.

**확인 문제:** 증명에서 공분산 0이 왜 독립을 주나요? **해설:** 잔차와 관측이 공동 Gaussian이기 때문입니다. 일반적인 비Gaussian 변수에는 이 함의가 성립하지 않습니다.

```python
from decimal import Decimal as D, localcontext

def solve(A,b):
    a=[list(row)+[v] for row,v in zip(A,b)]
    n=len(a)
    for j in range(n):
        pivot=a[j][j]
        a[j]=[v/pivot for v in a[j]]
        for i in range(n):
            if i!=j:
                scale=a[i][j]
                a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    return [row[-1] for row in a]

with localcontext() as ctx:
    ctx.prec=450
    nodes=[D(-1),D(0),D(1)]
    def kernel(x,z): return (-(x-z)**2/2).exp()
    for noise,x in [(D(0),D(0)),(D('1e-30'),D(0)),(D(0),D('1e-8')),(D(0),D('1e-100'))]:
        A=[[kernel(u,v)+(noise if i==j else 0) for j,v in enumerate(nodes)] for i,u in enumerate(nodes)]
        k=[kernel(x,u) for u in nodes]
        w=solve(A,k)
        variance=1-sum(v*t for v,t in zip(k,w))
        if noise==0 and x==0: assert abs(variance)<D('1e-440')
        else: assert variance>0
        print('noise, x, latent variance:',noise,x,format(variance,'.8E'))
```

## 확인 문제와 해설

1. 평균이 kernel ridge와 같으면 모든 예측 정보도 같나요? **아니요.** GP의 조건부 분산과 모형 가정이 추가됩니다.
2. 잡음 분산 .1을 표준편차 .1로 읽으면? **실제 의도한 분산 .01과 .1을 혼동합니다.** 단위를 확인하세요.
3. 평균을 바꾸면 분산도 항상 바뀌나요? **현재 kernel과 σ²를 고정한 조건화에서는 아닙니다.** 분산 식은 y를 포함하지 않습니다.

추가 읽기: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
