## 최적화의 해가 다시 목적함수를 정의할 때

내부 변분 에너지를 E(y,λ)=yᵀH(λ)y/2−bᵀy, b=(1,−1), H=[[1+λ,c],[c,2+λ]]로 두자. 제어 범위에서 H는 대칭 양의 정부호이므로 상태 y는 Hy=b의 유일한 해다. 변분 상태 방정식의 작은 Galerkin 모형으로 볼 수 있다.

외부 목적 V(y)=‖y−t‖²/2, t=(r,0)는 검증 손실이다. λ가 상태를 바꾸어 손실도 바뀐다. 상태는 해석적으로 풀고 반복 조절기는 adjoint 계산만 바꾼다.

## 직접 민감도와 adjoint

∂H/∂λ=I이고 b가 고정이므로 정상성을 미분하면

$$
H\frac{dy}{d\lambda}=-y,\qquad
\frac{dV}{d\lambda}=(y-t)^T\frac{dy}{d\lambda}.
$$

대신 Hᵀv=y−t를 풀면 dV/dλ=−vᵀy다. 큰 문제에서 초매개변수가 많고 외부 목적이 스칼라 하나이면 전치 풀이로 전체 상태 민감도 행렬의 저장을 피할 수 있다.

일반 g(y,λ)=0과 V(y,λ)에서는 직접 의존성도 포함하여 dV/dλ=V_λ−vᵀg_λ, g_yᵀv=V_y다. 여기의 단순식은 명시한 의존 구조 때문에 성립한다.

## Adjoint 풀이만 근사하기

v₀=0에서 실제 Richardson 보폭 v⁺=v+(y−t−Hv)/6을 수행한다. 이 보폭은 표시한 SPD 범위에서 안정하다. 주황 곡선은 −v_Kᵀy, 파란 곡선은 정확한 adjoint 미분이다.

근사 미분이 작거나 0이라고 외부 정지점인 것은 아니다. λ=c=r=0에서 상태 (1,−0.5), 정확한 adjoint (1,−0.25), 정확한 미분 −1.125다. K=0의 근사 미분 0은 단지 v₀=0이라 나온다.

## 잔차와 조건수로 오차 인증

r_adj=y−t−Hᵀv_K이면 v*−v_K=H⁻ᵀr_adj다. 최소 고유값이 양수이므로

$$
|(-v_K^Ty)-(-v^{*T}y)|
\le\frac{\|y\|\,\|r_{adj}\|}{\lambda_{min}(H)}.
$$

화면은 실제 오차와 계산 가능한 상한을 비교한다. 조건 정보 없이 잔차만 작다고 충분하지 않다. 비대칭 문제에서는 SPD 고유값 논리 대신 특이값이나 역연산자 상계가 필요하다.

## 상태 오차까지 포함한 hypergradient 인증

H는 정확한 SPD이고 λmin(H)≥m>0이라고 하자. 근사 상태 ỹ의 잔차를 s=b−Hỹ, 근사 adjoint ṽ의 잔차를 ρ=ỹ−t−Hṽ라고 쓰자. εy=‖s‖/m이면 ‖y−ỹ‖≤εy다. 정확한 adjoint는 Hv=y−t이므로 식을 빼면 ‖v−ṽ‖≤(εy+‖ρ‖)/m이다. 곱 −vᵀy의 오차를 분해하면

$$
|{-\tilde v^T\tilde y}-({-v^Ty})|
\le(\|\tilde y\|+\varepsilon_y)
\frac{\varepsilon_y+\|\rho\|}{m}
+\|\tilde v\|\varepsilon_y.
$$

앞의 정확한 상태 인증과 달리 두 잔차를 모두 포함한다. 유도에서 모든 노름은 유클리드다. 여전히 H가 정확하고 b,t가 고정이며 ∂H/∂λ=I라는 가정이 필요하다. 연산자 근사·모형 오차·외부 목적의 직접 의존성에는 추가 항을 넣어야 한다.

H=diag(1,2), b=(1,−1),t=0에서 상태를 ỹ=(1+ε,−1/2)로 바꾸고 그 상태의 adjoint를 정확히 풀자. ρ=0이어도 hypergradient 오차는 |2ε+ε²|다. Adjoint 잔차 0이 상태 오차를 없애지는 않는다. 유클리드 벡터 노름을 L1 상계로 바꾸면 다음의 보수적인 유리수 인증을 얻는다.

| ε | 실제 오차 | 상계 |
|---|---:|---:|
| 0 | 0 | 0 |
| 1/100 | 201/10000 | 139/5000 |
| 1/10 | 21/100 | 61/200 |

```python
from fractions import Fraction as F
true_hyper = -F(9, 8)
for eps in (F(0), F(1, 100), F(1, 10)):
    state = (1+eps, -F(1, 2))
    adjoint = (state[0], state[1]/2)
    state_error_bound = abs(eps)  # m=1; state residual is (-eps,0).
    adjoint_residual = (state[0]-adjoint[0], state[1]-2*adjoint[1])
    assert adjoint_residual == (0, 0)
    approximate = -sum(v*y for v, y in zip(adjoint, state))
    error = abs(approximate-true_hyper)
    # L1 upper bounds on the Euclidean norms in the proved certificate.
    bound = ((sum(map(abs, state))+state_error_bound)*state_error_bound
             +sum(map(abs, adjoint))*state_error_bound)
    assert error == abs(2*eps+eps*eps) and error <= bound
    print(eps, error, bound)
```

추가 프로그램은 상태 자체를 섭동한다. 위 인터랙티브 실습은 해석적인 상태를 유지하고 Richardson adjoint 작업량만 바꾼다. 근사 상태의 adjoint를 정확히 풀었다고 실제 내부 최적해의 정확한 미분이 되는 것은 아니다.

## 직접 풀기와 풀이

λ=c=r=0에서 adjoint를 8회 반복하면 미분 약 −0.887555, 오차 0.237445, 상한 0.260932다. 오차가 상한 안에 있는지 확인하자. 결합 예제에서도 직접 민감도와 adjoint가 같은 정확한 전체 미분을 주는지 검산한다.

이 인증은 상태와 모형 연산자가 정확하다는 가정 아래다. 실제 PDE의 근사 상태·이산화 오차·근사 Hessian에는 추가 오차를 포함해야 한다. Adjoint 풀이 성공만으로 과학 모형 전체가 인증되는 것은 아니다.

더 읽기: [MIT의 행렬 미분·변분 강의](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
