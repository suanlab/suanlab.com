## 1. 외부장을 조금 바꾸면 무엇이 달라질까

사차 깁스 장 단원에서는 고정한 외부장에서 평균과 공분산을 구했습니다. 이번에는 외부장을 바꾸면서 하나의 함수로 두 관측량을 읽습니다. 물질의 평균이 목표값에 이르도록 외부장을 조절한다고 생각해 보세요. 외부장 좌표에서는 제어값을 정하고 평균을 관측합니다. 르장드르 변환은 평균을 지정하고 그에 필요한 외부장을 찾게 해 줍니다.

## 2. 연결 생성함수

유한 격자의 외부장 없는 무차원 작용을 $A_0(x)$라 둡니다. 외부장을 $h\cdot x$로 결합하면 다음과 같습니다.

$$
Z(h)=\int e^{-A_0(x)+h\cdot x}\,dx,\qquad W(h)=\log Z(h)-\log Z(0).
$$

따라서 $e^{W(h)}$는 외부장이 0인 깁스 측도의 적률생성함수입니다. 사차항이 양수인 모형에서는 외부장의 콤팩트 범위 전체에서 피적분함수와 다항식이 곱해진 미분항을 사차 감쇠 함수로 제한할 수 있습니다. 이 상계가 적분 안에서 미분하는 근거입니다. [Tong, Statistical Field Theory §2.2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2)

## 3. 곡률은 응답을 나타낸다

$W$를 한 번 미분하면 $x_i$가 적분에 들어갑니다. 그 비율을 다시 미분하면 분모도 함께 미분됩니다.

$$
\partial_i W=m_i=\mathbb E_h[x_i],\qquad
\partial_jm_i=\partial_{ij} W
 =\mathbb E_h[x_i x_j]-m_i m_j=\chi_{ij}.
$$

평균의 곱을 빼면 평균 자체가 만드는 기여가 제거됩니다. 모든 방향 $v$에 대해 $v^\mathsf T\chi v=\operatorname{Var}_h(v\cdot x)$이므로 $W$는 볼록합니다. 유한 차원 실수 공간 전체에서 밀도가 양수이면 $v\ne0$인 선형 관측량은 지지집합 위에서 변하므로 분산이 양수입니다. 한 방향의 사차 미분은 4차 중심 모멘트에서 분산 제곱의 세 배를 뺀 값입니다.

## 4. 외부장 좌표를 평균 좌표로 바꾸기

르장드르–펜셸 변환을 다음과 같이 정의합니다.

$$
\Gamma(m)=\sup_h\{h\cdot m-W(h)\}.
$$

$\nabla W$의 치역에 속하는 내부 평균에서 정류 조건은 $m=\nabla W(h)$입니다. 엄밀한 볼록성으로 대응하는 외부장은 하나입니다. 해당 외부장의 접평면은 $W$의 그래프를 지지하고, 변환값은 그 평면의 절편에 음수를 붙인 값입니다.

$\Gamma=h(m)\cdot m-W(h(m))$을 미분해 보세요. $\nabla W(h)=m$이므로 $dh$가 포함된 항이 상쇄됩니다. 한 번 더 미분하고 역함수 정리를 적용하면 다음을 얻습니다.

$$
\nabla_m\Gamma=h,\qquad
\nabla_m^2\Gamma=\chi(h)^{-1}.
$$

이 역행렬은 평균 좌표계에서의 곡률입니다. 미시 헤시안 $\nabla_x^2 A_0$은 개별 배치를 바꾸는 방향을 평가합니다. 유효 곡률이 양수여도 미시 헤시안에는 음의 고윳값이 있을 수 있습니다.

## 5. 정확히 풀리는 두 예제

정밀도 행렬 $K$가 양의 정부호인 중심 가우시안에서는 $W(h)=h^\mathsf T K^{-1}h/2$, $\Gamma(m)=m^\mathsf T K m/2$입니다. 역응답은 일정합니다.

편향이 없는 스핀 하나 $s\in\{-1,1\}$에서는 $W(h)=\log\cosh h$, $m=\tanh h$, $\chi=1-m^2$입니다. $(-1,1)$ 내부에서 $h=\operatorname{artanh}m$이고 다음 식을 얻습니다.

$$
\Gamma(m)=\tfrac{1+m}{2}\log(1+m)+\tfrac{1-m}{2}\log(1-m).
$$

양 끝점에서 연속적으로 접근하면 변환값은 $\log 2$입니다. 유한한 외부장으로는 내부 평균에 도달합니다. 포화에 접근할수록 $\chi$는 0에 가까워지고 유효 곡률은 발산합니다. $[-1,1]$ 밖에서는 변환값이 무한합니다.

## 6. 사차 모형 수치 실습

다음 Python 실습은 평균을 입력으로 받습니다. 목표 평균을 내는 외부장을 이분법으로 탐색합니다. 유한 구간의 가우스–르장드르 적분으로 값을 추정합니다. 앞 단원과 같이 적분 범위와 해상도를 바꾸어 결과를 비교하세요.

```python
import numpy as np

def response(radius, panels):
    nodes, weights = np.polynomial.legendre.leggauss(panels)
    x, w = radius * nodes, radius * weights
    base = np.log(w) - 0.25*x**4 + 0.5*x**2

    def observe(h):
        logw = base + h*x
        offset = np.max(logw)
        p = np.exp(logw - offset)
        total = p.sum()
        p /= total
        mean = p @ x
        return offset + np.log(total), mean, p @ (x-mean)**2

    return observe

def invert(target, observe):
    lo, hi = -2.0, 2.0
    assert observe(lo)[1] <= target <= observe(hi)[1]
    for _ in range(50):
        mid = (lo + hi)/2
        if observe(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    h = (lo + hi)/2
    logz, mean, chi = observe(h)
    return h, h*mean - logz + observe(0)[0], 1/chi

observe = response(6, 96)
target, step = 0.6, 1e-3
h, gamma, curvature = invert(target, observe)
slope = (invert(target+step, observe)[1] - invert(target-step, observe)[1])/(2*step)
np.testing.assert_allclose(slope, h, atol=2e-7, rtol=0)
np.testing.assert_allclose(
    [h, gamma, curvature], invert(target, response(7, 128)), atol=2e-8, rtol=0
)
print(h, gamma, curvature)
```

## 7. 확인할 항목

- 짝대칭인 사차 모형의 외부장이 0이면 평균은 0이고 감수율은 양수입니다. $x$와 $x^2$의 대칭성이 어떻게 다른지 설명해 보세요.
- 목표 평균을 고정하고 $\Gamma$의 기울기에서 필요한 외부장을 읽어 보세요. 두 목표값의 순서가 외부장의 순서와 어떻게 연결되는지 계산 전에 예측하세요.
- 사차 모형의 미시 곡률 $A_0''(0)=-1$과 $\Gamma''(0)>0$을 비교해 보세요. 미시 곡률은 배치 공간의 한 점을 봅니다. 유효 곡률에는 두 우물에 걸친 요동이 들어갑니다.
- 기준 측도가 고정된 에너지 기반 모델에서도 외부장 매개변수가 분포를 기울입니다. 평균은 로그 정규화 함수의 기울기이고 공분산은 그 응답 행렬입니다.
