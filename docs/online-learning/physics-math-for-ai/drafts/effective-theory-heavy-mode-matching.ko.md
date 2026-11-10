## 1. 보존할 응답 정하기

유효 이론은 관측하려는 척도에 필요한 자유도와 관측량을 남깁니다. 가벼운 좌표가 단단한 좌표와 결합한 상태에서 천천히 구동한다고 생각해 보세요. 단단한 좌표도 가벼운 좌표의 응답에 영향을 줍니다. 이 좌표를 적분하면 그 영향이 가벼운 자유도만 쓰는 이론의 매개변수와 미분항에 들어갑니다. [Burgess, Introduction to Effective Field Theory](https://arxiv.org/abs/hep-th/0701053)

## 2. 가우시안 무거운 모드 적분하기

유클리드 주파수 $\omega$마다 다음 실수 이차형식을 둡니다.
$$
Q(\omega)=
\begin{pmatrix}
\omega^2+m^2 & g\\
g & \omega^2+M^2
\end{pmatrix},
\qquad 0<m<M,\quad g^2<m^2M^2.
$$

안정성 조건에 따라 주파수가 0일 때도 행렬이 양의 정부호입니다. 무거운 변수에 대해 완전제곱을 만들고 가우시안 적분을 하면 가벼운 모드의 진폭과 무관한 정규화 인자와 다음 커널을 얻습니다.
$$
Q_{\rm light}(\omega)=\omega^2+m^2-\frac{g^2}{\omega^2+M^2}.
$$
해당 모드의 분산 또는 외부장 응답은 $1/Q_{\rm light}$입니다. 행렬식에서 온 정규화 인자는 절대 분배함수에 기여합니다.

## 3. 국소 미분 전개의 계수 맞추기

$x=\omega^2/M^2$라 두면 무거운 모드의 전파함수에서 $(1+x)^{-1}$이 나옵니다. $\omega=0$ 주변에서 전개하면 상수항과 $\omega^2$ 항에 이어 4차 이상의 미분항이 나옵니다. 가벼운 커널의 첫 두 계수는 다음과 같습니다.
$$
Q_{\rm light}(0)=m^2-g^2/M^2,\qquad
\left.\frac{dQ_{\rm light}}{d(\omega^2)}\right|_0=1+g^2/M^4.
$$
정합은 원하는 차수까지 전체 이론을 재현하도록 유효 이론의 저주파 계수를 정하는 일입니다. 정합한 이론으로 다른 저주파 응답을 예측하고 가능한 경우 정확한 결과와 비교합니다.

## 4. 적용 전에 오차 추정하기

무거운 모드의 항에서 $x^j$를 $j=p$까지 남깁니다. 유한 등비급수 항등식으로 정확한 커널 오차를 구할 수 있습니다.
$$
|Q_{\rm light}-Q_p|=\frac{g^2}{M^2}\frac{x^{p+1}}{1+x}.
$$
$0<x<1$이면 차수를 하나 올릴 때마다 오차가 $x$배로 줄어듭니다. $x>1$이면 차수를 올릴수록 오차가 증가합니다. 양의 유클리드 주파수에서는 특이점이 없고, 전개 반경은 $x=-1$의 극점이 정합니다.

두 커널이 양수일 때 응답의 차이는 $|Q_{\rm light}^{-1}-Q_p^{-1}|=|Q_{\rm light}-Q_p|/(Q_{\rm light}Q_p)$입니다. 모드가 부드러워져 분모가 작으면 작은 커널 오차도 큰 응답 차이를 만들 수 있습니다. 절단한 커널이 양수가 아니면 가우시안 확률분포를 정의할 수 없습니다.

## 5. 척도 차이와 정합 조건 확인하기

$m=1$, $M=4$, $g=2$인 안정한 모형을 사용합니다. $Q_{\rm light}(0)=0.75$입니다. $\omega=M/2$에서 차수를 하나 올리면 커널 오차가 $1/4$배가 되고, $\omega=2M$에서는 4배가 됩니다. 혼합을 바꿀 때 $g/(mM)$을 쓰면 안정성을 쉽게 유지할 수 있습니다. $g$를 고정하고 $M$을 키우면 정적 보정이 줄어듭니다. $g/(mM)$을 고정하고 $M$을 키우면 정적 보정이 남습니다.

```python
from math import isclose

m, M, g = 1.0, 4.0, 2.0
def kernels(omega, order):
    x = (omega / M)**2
    strength = (g / M)**2
    exact = omega**2 + m**2 - strength/(1+x)
    matched = omega**2 + m**2 - strength*sum((-x)**j for j in range(order+1))
    return exact, matched, strength*x**(order+1)/(1+x)

assert kernels(0, 0)[0] == 0.75
for ratio in [0.25, 0.5, 1, 2]:
    for order in range(5):
        exact, approx, error = kernels(ratio*M, order)
        assert isclose(abs(exact-approx), error, rel_tol=1e-8, abs_tol=1e-14)
        if approx > 0:
            assert isclose(abs(1/exact-1/approx), error/(exact*approx), rel_tol=1e-8, abs_tol=1e-14)
    print(ratio, kernels(ratio*M, 2))
```

## 6. 새 유효 모형에서 반복할 검사

- 저주파에서 보존할 관측량과 필요한 정규화를 지정하세요.
- 남길 계수를 정합하고 의도한 범위에서 다음 차수의 오차를 확인하세요.
- 안정성 경계에 더 가까운 혼합비로 반복하고 커널과 응답의 오차를 비교하세요.
- 축소한 통계 모형을 적합할 때 정합 자료에 사용한 매개변수와 다른 척도에서 검증할 예측을 구분하세요.
