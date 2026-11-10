## 1. 사이트마다 실수값을 갖는 장

앞 단원에서는 공간 블록마다 이산 자화를 남겼다. 여기서는 각 사이트에 실수 좌표를 부여한다. 격자가 위치와 이웃 관계를 정하고 배치 공간은 연속이다. 두 사이트 예제의 적분 영역은 두 장 값을 좌표로 갖는 평면이다.

## 2. 에너지와 기준 측도

무차원 장 좌표를 사용하고 작용 A를 E/T로 둔다. 각 무방향 edge를 한 번씩 세면

$$
A(x)=\sum_i (a x_i^4+b x_i^2-h_i x_i)+\frac{c}{2}\sum_{\langle i,j\rangle}(x_i-x_j)^2,
\qquad
Z=\int_{\mathbb R^N} e^{-A(x)}\prod_i dx_i .
$$

양의 c는 이웃 사이트 값의 차이에 벌점을 준다. 원천 hᵢ는 해당 사이트의 양 또는 음의 값을 선호하게 한다. 기준 측도에 대한 깁스 밀도는 exp(−A)/Z이며 F/T=−log Z다. 사차항과 기울기 항은 Landau–Ginzburg 자유에너지에도 나타난다. [Tong, Statistical Field Theory §1.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1)

## 3. 사차항이 적분을 안정화하는 이유

dᵢ=max(0,−b)+|hᵢ|/2로 놓는다. |x|≤(x²+1)/2로 원천을 이차항과 상수로 제한한다. x²에 대해 완전제곱을 만들면

$$
a x_i^4+b x_i^2-h_i x_i\ge
\tfrac a2 x_i^4-C_i,\qquad C_i=\frac{d_i^2}{2a}+\frac{|h_i|}{2}.
$$

a>0, c≥0이면 사차 감쇠의 곱은 모든 다항 가중치에 대해 적분 가능하다. b가 음수여도 정규화와 모멘트가 유한하다. 로그 가중치에서 ΣCᵢ를 빼면 이동한 작용이 음이 아니게 된다. 이동한 적분 Zs에서 log Z=log Zs+ΣCᵢ로 원래 정규화를 복원한다.

## 4. 대칭과 요동

원천이 모두 0이면 전체 반전이 작용과 측도를 보존한다. 각 사이트 평균과 총차수가 홀수인 단항식의 평균이 0이다. x₀x₁은 두 좌표를 함께 반전해도 값이 같다. c=0이면 사이트별 분포가 곱으로 분리되어 서로 다른 사이트의 공분산은 0이다. 양의 c는 비슷한 장 값을 갖는 배치에 더 큰 가중치를 준다.

## 5. 국소 Hessian과 전체 분포

두 사이트에서 a=1/4, b=−1/2, c=1, h=0을 사용한다. 균일한 두 배치 (1,1), (−1,−1)이 작용을 최소화한다. 원점의 Hessian 고윳값은 −1과 1이다. 두 최솟값에서는

$$
H=\begin{pmatrix}3&-1\\-1&3\end{pmatrix},
\qquad H^{-1}=\frac18\begin{pmatrix}3&1\\1&3\end{pmatrix}.
$$

역 Hessian은 한 최솟값 중심의 이차 근사 공분산이다. 전체 적분에는 두 최솟값과 그 사이의 영역이 들어간다. R=4와 축마다 64셀을 사용한 전체 비대각 공분산의 포함 구간은 [0.50,0.54] 안에 들어간다. 한 최솟값 근사의 값은 0.125이며 전체 평균은 대칭으로 0이다. [Tong, Statistical Field Theory §2](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S2)

## 6. 유한 적분 상자 안의 오차

다항식 P에 대해 f=P exp(−A−ΣCᵢ)로 둔다. 중점법에 필요한 이차 미분은

$$
\partial_i^2 f=e^{-A-\sum_j C_j}
\left(\partial_i^2P-2\partial_i P\,\partial_i A+
P[(\partial_i A)^2-\partial_i^2 A]\right).
$$

각 셀에서 이 식의 값을 구간으로 제한한다. 셀 평균과 중점값의 차이는 Σᵢ Δᵢ² sup|∂ᵢ²f|/24 이하이다. 각 축의 일차원 중점 오차를 차례로 더한다. 셀 전체의 함수값 범위로도 평균을 제한할 수 있다. 두 구간을 교차시키고 셀 부피를 곱해 적분 구간을 얻는다.

## 7. 상자 바깥의 꼬리 기여

절댓값이 R보다 큰 좌표를 R+u로 놓고 (R+u)⁴≥R⁴+4R³u를 적용한다.

$$
\int_{|x|>R}|x|^p e^{-\alpha x^4}dx
\le 2e^{-\alpha R^4}\sum_{k=0}^p
\binom pk\frac{R^{p-k}k!}{(4\alpha R^3)^{k+1}},\qquad \alpha=a/2 .
$$

상자 밖에서는 한 좌표 이상이 R을 넘는다. 좌표마다 해당 축의 꼬리 상한과 나머지 축의 전체 상한을 곱하고 더한다. 이 값이 생략한 적분 기여의 절댓값 상한이다. P가 음이 아니면 꼬리 기여의 하한을 0으로 둘 수 있다.

## 8. 모멘트를 구간으로 정규화하기

각 전체 모멘트 적분을 양의 하한을 가진 Zs 구간으로 나눈다. 공분산은 이차 곱 모멘트에서 두 평균 구간의 곱을 뺀다. 질량 구간이 넓으면 나눗셈과 뺄셈으로 최종 구간이 넓어진다. 상자 안 적분, 꼬리 상한과 정규화 구간을 함께 읽는다.

## 9. 세 가지 정련

R을 고정한 채 적분 셀을 늘리면 중점 계산을 정련한다. 셀 수를 고정하고 R을 키우면 꼬리와 셀 폭이 함께 바뀐다. 물리 격자를 정련하면 이웃 그래프와 작용이 바뀐다. 적분 오차를 비교할 때는 물리 격자를 고정한다. 물리 격자 정련에서는 간격에 따른 계수와 측도의 변환도 지정한다.

## 10. 해석적으로 풀리는 경우와 모멘트 항등식

A(x)=a x⁴인 경우 양쪽 반직선에서 u=a x⁴로 치환한다. 짝수 p의 정규화 전 모멘트는 Γ((p+1)/4)/(2a^((p+1)/4))이므로 E[x⁴]=1/(4a)다. 결합 모형에서는 ∂ᵢ(xᵢ exp(−A))를 전체 공간에서 적분한다. 사차 감쇠로 경계항이 사라져 E[xᵢ∂ᵢA]=1이다. 이 식은 2차·4차 모멘트, 원천 응답, 이웃 곱을 한꺼번에 확인하는 데 쓰인다.

## 11. 두 사이트 계산 실행

NumPy의 Gauss–Legendre 절점으로 이차원 적분을 근사한다. 절점 수와 반경을 각각 바꿔 결과를 포함 구간과 비교한다.

```python
import numpy as np

def moments(a=.25, b=-.5, c=1., h=(0., 0.), radius=5., nodes=96):
    t, w = np.polynomial.legendre.leggauss(nodes)
    x, y = radius * t[:, None], radius * t[None, :]
    action = a*(x**4+y**4) + b*(x*x+y*y) - h[0]*x - h[1]*y + c*(x-y)**2/2
    weights = radius**2 * w[:, None] * w[None, :] * np.exp(-action)
    z = weights.sum()
    ex, ey = (weights*x).sum()/z, (weights*y).sum()/z
    ex2, ex4, exy = [(weights*p).sum()/z for p in (x*x, x**4, x*y)]
    identity = 4*a*ex4 + 2*b*ex2 - h[0]*ex + c*(ex2-exy)
    return np.log(z), ex, exy-ex*ey, identity

uncoupled = moments(c=0.)
coupled = moments()
assert abs(uncoupled[2]) < 1e-10
assert 0.50 < coupled[2] < 0.54
assert abs(coupled[3]-1) < 1e-9
print("uncoupled", uncoupled)
print("coupled", coupled)
```

## 12. 이해 확인

1. b를 음수로 두어도 사차 감쇠가 적분 가능하므로 분배함수는 유한하다.
2. 원천 0에서 두 좌표를 함께 반전하면 곱의 부호가 유지되므로 공분산이 남을 수 있다.
3. R을 고정하고 셀 수를 늘리면 상자 안의 계산 오차가 바뀐다. 꼬리 공식은 반경과 감쇠 상한에 의존한다.
4. 국소 공분산 1/8과 전체 공분산 약 0.52를 비교하자. 전체 분포는 두 정렬된 영역 사이의 변동도 포함한다.
5. A에 상수 K를 더하면 정규화 모멘트는 같고 log Z가 K만큼 감소한다.
6. c를 0에서 키우면 c(x₀−x₁)²/2가 큰 차이를 억제한다. 두 사이트 예제에서 그에 따른 공분산을 측정할 수 있다.
