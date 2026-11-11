## 다양체 위에 남아 최적화하기

단위원 S¹에서 f(x)=−x₁을 최소화하자. 최소점은 (1,0), 반대편 (−1,0)은 극대점이다. 주변 공간의 제약 없는 보폭은 원을 벗어날 수 있다. Riemannian 최적화는 미분을 접공간 벡터로 표현하고 그 보폭을 다양체로 되돌린다.

여기 내적은 주변 유클리드 내적을 제한한 것이다. 기하라는 말을 함께 쓴다고 Fisher metric과 같은 것은 아니다.

## 주변 gradient의 접공간 사영

x=(cos θ,sin θ)에서 접벡터는 xᵀv=0을 만족한다. 주변 gradient −e₁을 사영하면

$$
\operatorname{grad}_{S^1}f=(I-xx^T)(-e_1)
=(-\sin^2\theta,\cos\theta\sin\theta),
\qquad \|\operatorname{grad}f\|=|\sin\theta|.
$$

화면의 직교 잔차가 접공간 여부를 검사한다. 각도 좌표에서 음의 gradient 속도는 θ̇=−sin θ다. 목적 초과 1−cos θ는 최소점 가까이에서 안정적인 반각 공식으로 계산한다.

## 원으로 돌아가는 두 방법

v=−ηgrad f에 대해 정규화 retraction은 Rₓ(v)=(x+v)/‖x+v‖다. xᵀv=0이므로 분모는 √(1+‖v‖²)여서 0이 되지 않는다. 각도 갱신은 θ⁺=θ−atan(η sin θ)다.

같은 접벡터의 circle exponential은 θ⁺=θ−η sin θ다. 정확한 geodesic 보폭이지 전체 시간 구간의 비선형 gradient-flow ODE를 정확히 적분한 것은 아니다. 두 복귀는 국소적으로 일치하지만 유한 보폭에서는 다르다.

θ=π/2,η=0.5이면 정규화 결과는 (1/√5,2/√5)≈(0.447214,0.894427), exponential은 (sin0.5,cos0.5)≈(0.479426,0.877583)다. 둘 다 가능점이지만 끝점 거리는 약 0.036350이다.

## 정지성만으로 최소를 인증하지 않기

θ=π에서는 접 gradient가 정확히 0이라 두 방법 모두 극대점에 남는다. 목적 초과는 2다. 구현은 알려진 정확한 각도를 처리하여 부동소수점 sin π 잡음을 가짜 탈출로 해석하지 않는다.

작은 접 잔차는 최소점뿐 아니라 극대점에서도 나온다. 곡률·감소 거동·목적값이 추가 정보를 준다. 일반 다양체에는 여러 정지점이 있을 수 있고 수렴 가정도 문제에 맞춰 확인해야 한다.

## 유한 기하 보폭의 감소 증명

각도 좌표에서 F(θ)=−cos θ이고 모든 곳에서 |F″|≤1이다. Taylor 상계로 exponential 갱신은

$$
F(\theta-\eta\sin\theta)\le F(\theta)
-\eta(1-\eta/2)\sin^2\theta.
$$

0<η<2이면 이 목적에서 비정지 보폭은 엄격히 감소한다. 합을 취하면 Σsin²θₖ<∞여서 sin θₖ→0이다. 정지성에 접근한다는 보장이고, 정확히 극대점에 초기화하는 경우는 무조건 최소에 도달한다는 주장의 반례로 남는다.

정규화 retraction도 유효 보폭 a(θ)=atan(ηsin θ)/sin θ를 쓰면 같은 형태다. sin θ=0에서는 극한 η를 쓴다. 0이 아닌 sine에서 atan z는 z와 부호가 같고 절댓값은 |z| 이하이므로 0<a≤η다. 따라서 η 대신 a를 넣은 같은 상계가 성립한다. 고정 η>0에서는 원 전체에서 유효 보폭의 양의 하한도 존재한다. 결과적으로 0<η<2에서 같은 정지성 결론을 얻는다. 끝값 η=2는 여기서 사용한 엄격히 양의 감소 계수로 다루지 않으며 별도 분석이 필요하다.

| θ | F″(θ) | 기울기 노름 | 초과 |
|---|---:|---:|---:|
| 0 | 1 | 0 | 0 |
| π/2 | 0 | 1 | 1 |
| π | −1 | 0 | 2 |

```python
from math import sin, cos, atan
eta = 0.5
for mode in ("exponential", "normalization"):
    for initial in (-2., .8, 2.8):
        theta = initial
        for _ in range(20):
            sine = sin(theta)
            effective = eta if sine == 0 else (
                eta if mode == "exponential" else atan(eta*sine)/sine)
            next_theta = theta-effective*sine
            upper = -cos(theta)-effective*(1-effective/2)*sine*sine
            assert -cos(next_theta) <= upper+1e-14
            theta = next_theta
        print(mode, initial, "final excess:", 2*sin(theta/2)**2)
```

세 초기점에서 두 복귀 방법의 실제 20회 경로를 검산한다. 증명은 이 원 목적의 전역 곡률 상계를 쓰며 모든 다양체나 retraction에 유클리드 smoothness를 그대로 주장하지 않는다. 최소점 근처의 각도 선형 배율은 1−η, 극대점 근처는 1+η지만 정확한 극대점 자체는 정지한다.

## 직접 풀기와 풀이

한 보폭 예제 둘을 비교한 뒤 η를 절반으로 하자. atan a=a−a³/3+…로 이 정규화 복귀의 선도 끝점 차이가 세제곱임을 예상할 수 있다. 노름은 계속 1인데 끝점은 다른지 보자. 해석적 끝점 차이는 a−atan(a)의 안정적인 나머지로 계산하여 거의 같은 각도의 차감이 세제곱 항을 지우지 않게 한다.

반대편 예제에서 가능성과 접 gradient 0만으로 왜 전역 최소라고 할 수 없는지 설명한다. 파란 원과 주황 반복점을 동일 축척으로 그리며, 점 사이 직선은 시각적 연결이지 geodesic이나 연속 시간 궤적이 아니다.

더 읽기: [Boumal의 Smooth Manifold Optimization 교재](https://www.nicolasboumal.net/book/).
