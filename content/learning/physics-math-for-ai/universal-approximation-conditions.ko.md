## 보편 근사는 존재 명제다

전형적인 보편 근사 결과는 compact 영역 K의 연속 함수 f와 ε>0에 대해, 적절한 활성화·bias·충분한 폭을 허용하면 균등 오차가 ε보다 작은 유한 네트워크가 존재한다는 내용입니다. 연속 비다항 활성화인 ReLU는 적절한 bias와 폭을 허용하는 이론의 대표적인 예입니다. 정확한 정리를 사용할 때 활성화의 조건과 공간·노름을 함께 명시해야 합니다.

$$
\forall f\in C(K),\ \forall\varepsilon>0,\ \exists g_\theta:\|f-g_\theta\|_\infty<\varepsilon.
$$

이는 고정한 폭 하나로 모든 f와 모든 ε를 처리한다는 말이 아닙니다. 필요한 크기나 계수, 이를 찾는 학습 알고리즘의 성공도 이 존재 명제만으로 얻지 못합니다.

## ReLU hinge를 실제로 조립하기

[-1,1]을 n개 구간으로 나누고 x²의 선형 보간을 만듭니다. 처음 값은 1, 첫 기울기는 s₀=−2+2/n이며 내부 매듭 tⱼ=−1+2j/n에서 기울기가 4/n씩 증가합니다.

$$
g_n(x)=1+s_0(x+1)+\sum_{j=1}^{n-1}\frac4n\max(0,x-t_j).
$$

Affine skip과 n−1개 ReLU hinge를 쓰는 명시적 구성입니다. n=4이면 세 hinge로 균등 오차 1/16을 얻고 n=8이면 1/64입니다. 구간 길이 h=2/n에서 보간 오차 h²/4를 사용했습니다. 이는 x² 하나에 대한 구성이지 모든 연속 함수의 일반 정리를 증명한 것은 아닙니다.

## 선형 활성화가 실패하는 이유

Affine 사상을 아무리 합성하거나 넓게 합해도 결과는 affine입니다. [-1,1]의 x²에 대한 최선 균등 affine 근사는 상수 1/2이며 오차는 1/2입니다. 대칭점 −1과 1의 평균은 affine의 상수항이므로, 중앙 값과 양끝을 동시에 맞추려면 적어도 절반의 오차가 필요합니다.

$$
\inf_{a,b}\sup_{|x|\le1}|x^2-(a+bx)|=1/2.
$$

실습의 선형 모드는 이 최선 기준을 보여줍니다. ReLU n=1 구성은 상수 1이라 오차가 1입니다. 모순이 아닙니다. 하나는 명시한 보간 구성, 다른 하나는 affine 족 전체의 최선 해입니다.

## 연구에서 분리해야 할 질문

표현 존재, 적은 수의 유닛으로 표현, 유한 표본에서 일반화, 유한 시간에 최적화는 서로 다른 질문입니다. 깊이가 효율을 바꿀 수 있다는 주장도 대상 함수족과 계산 비용의 정의를 필요로 합니다. 현재의 작은 실험에서 폭을 늘려 오차가 줄었다고 모든 실제 학습 문제의 효율을 결론 내리지 않습니다.

## 1차원 연속 함수의 보편 근사를 구성으로 증명하기

[0,1]의 임의의 연속 함수 f와 매듭 0=t₀<⋯<t_n=1을 잡고 최대 간격을 h라 합시다. 구간 안 보간값은 양 끝 함수값의 볼록결합이므로 f(x)와의 차이는 ω_f(h) 이하입니다. 이 값이 ε보다 작게 h를 고르면 됩니다. j번째 구간의 할선 기울기를 s_j라 하면 전체 보간 함수는

$$
I_hf(x)=f(0)+s_0x+\sum_{j=1}^{n-1}(s_j-s_{j-1})\operatorname{ReLU}(x-t_j),
\qquad\|I_hf-f\|_\infty\le\omega_f(h).
$$

첫 기울기에서 시작해 각 hinge가 다음 기울기와의 차이만큼 바꾼다는 사실로 표현을 확인합니다. 0에서 값이 같고 연속성이 구간 사이를 연결합니다. Affine skip도 ReLU(x)−ReLU(−x)와 출력 상수로 나타낼 수 있습니다. Bias와 가변 폭을 허용한 입력 구간 하나의 조밀성을 증명한 것이며, 다차원 일반 정리에는 별도 논증이 필요합니다.

L-Lipschitz 함수에는 끝점 볼록결합으로 Lh/2까지 얻습니다. 가중 거리의 합이 2(x−a)(b−x)/(b−a)≤h/2이기 때문입니다. 이 계수는 최적화 알고리즘으로 찾은 것이 아닙니다. 선택한 함수값을 알고 있다는 가정 아래 직접 구성했습니다.

## 균등 폭보다 매듭 위치가 더 중요할 수 있다

별도 대상 f(x)=|x−1/3|에는 모서리가 있습니다. 2ReLU(x−1/3)−x+1/3으로 정확히 표현되지만, 포함 관계가 없는 균등 격자를 차례로 쓰면 보간 오차가 매번 줄지는 않습니다. 매듭과 모서리 밖에서 두 함수 모두 affine이므로 최대 차이는 이 유한 점들 중에서 나옵니다.

| 구간 수 n | 균등 격자의 정확 오차 | 이유 |
|---|---|---|
| 2 | 2/9 | 셀 내부의 모서리 |
| 3 | 0 | 모서리가 매듭 |
| 4 | 1/9 | 다시 셀 내부의 모서리 |
| 6 | 0 | 모서리가 매듭 |

코드는 이 별도 대상을 분수로 검산합니다. 기존 x² 실습에 모서리 위치 제어를 추가한 것은 아닙니다. 출력 bias까지 모든 bias를 제거한 ReLU 네트워크는 g(0)=0이므로 0을 포함하는 구간의 상수 1을 균등 근사할 수 없습니다. 비선형성만 남겼다고 바뀐 구조에 모든 버전의 정리가 유지되지는 않습니다.

**확인 문제:** n=3에서 n=4로 오차가 늘면 조밀성이 반박되나요? **해설:** 아닙니다. 충분히 정확한 네트워크가 존재한다는 내용이며, 지정한 모든 계수 선택 규칙의 매 단계 개선을 보장하지 않습니다.

```python
from fractions import Fraction as F

corner=F(1,3)
def target(x): return abs(x-corner)
for n in [2,3,4,6]:
    knots=[F(j,n) for j in range(n+1)]
    values=[target(x) for x in knots]
    slopes=[n*(values[j+1]-values[j]) for j in range(n)]
    def network(x):
        return values[0]+slopes[0]*x+sum(
            (slopes[j]-slopes[j-1])*max(F(0),x-knots[j])
            for j in range(1,n))
    j=min(n-1,int(corner*n))
    lo,hi=knots[j:j+2]
    interpolated=values[j]+(corner-lo)*slopes[j]
    error=interpolated-target(corner)
    assert network(corner)==interpolated
    # On each side of the corner both functions are affine;
    # their maximum absolute difference is attained at a knot or the corner.
    assert max(abs(network(x)-target(x)) for x in knots+[corner])==error
    assert error<=F(1,2*n)
    assert 2*max(F(0),corner-corner)-corner+corner==target(corner)
    print(n,str(error),float(error))
```

## 확인 문제와 해설

1. 정확한 x²를 ReLU 유한 spline으로 전 구간에서 표현하나요? **이 구성은 조각선형이므로 근사합니다.** 폭을 늘려 오차를 줄입니다.
2. bias를 빼도 같은 정리를 자동 적용하나요? **아니요.** 매듭의 위치를 옮기는 능력 등 가정이 바뀝니다.
3. 오차 ε를 얻으려면 이 구성의 n은? **n>1/√ε를 택하면 됩니다.** 이 특정 이차 함수 구성에 한정한 계산입니다.

추가 읽기: [Leshno et al., Nonpolynomial Activation and Approximation](https://doi.org/10.1016/S0893-6080(05)80131-5).
