## 무엇을 미분하는지 먼저 정하기

ODE의 연속 정확해로 정의한 손실과 컴퓨터가 실제로 N번 적분한 끝값의 손실은 다릅니다. 이산 adjoint는 후자의 계산 그래프를 정확한 chain rule로 미분합니다. 이산 gradient가 연속 gradient와 다르다는 이유만으로 구현 오류라고 판단하면 안 됩니다. 같은 함수의 독립 미분과 비교하는 것이 먼저입니다.

상태는 y′=θy(1-y), y(0)=0.2이고 forward Euler를 사용합니다. J=(y_N-target)²/2로 목적함수를 정의합니다. 파랑은 실제 Euler 상태, 주황은 연속 logistic 정확해입니다. 저장된 상태, forward tangent, backward adjoint를 같은 시각 행에서 비교할 수 있습니다.

## Forward Tangent 유도

$$
F(y,\theta)=y+h\theta y(1-y),\quad
F_y=1+h\theta(1-2y),\quad F_\theta=hy(1-y).
$$

초기 상태는 θ에 의존하지 않으므로 s₀=0입니다. Chain rule로 s_next=F_y s+F_theta를 반복하고 마지막에 (y_N-target)s_N을 곱합니다. 많은 모수에 각각 tangent를 전파하면 비용이 커집니다. 현재는 한 모수라 두 방식을 투명하게 비교할 수 있습니다.

## Reverse Adjoint 유도

$$
a_N=y_N-\mathrm{target},\qquad
a_n=F_y(y_n,\theta)a_{n+1},\qquad
\frac{dJ}{d\theta}=\sum_{n=0}^{N-1}a_{n+1}F_\theta(y_n,\theta).
$$

현재 step의 gradient 항에는 a_n이 아니라 a_next가 들어갑니다. 모수는 모든 step에서 공유되므로 각 항을 합해야 합니다. 이 예제는 저장한 forward 상태를 뒤에서 읽습니다. 큰 문제에서는 checkpoint·재계산·메모리 비용을 고려해야 하며 이를 단순히 무료라고 가정하면 안 됩니다.

## 유한차분과 연속 Gradient

중앙 유한차분은 같은 Euler 반복을 θ+ε와 θ-ε에서 다시 실행합니다. ε=10^-5는 현재 작은 부드러운 모형의 검산값이며 모든 정밀도와 문제에 최적인 상수는 아닙니다. 큰 ε의 절단 오차와 작은 ε의 cancellation을 구분하세요.

연속 정확해는 y(t)=1/(1+4 exp(-θt))이고 dy(T)/dθ=T y(T)(1-y(T))입니다. 따라서 연속 J의 gradient도 계산할 수 있지만 이것은 다른 유한 h 목적함수의 미분입니다. N을 늘리며 차이가 줄어드는지 확인하는 일은 이산 adjoint 내부 검산과 구분합니다.

## 반복 프로그램의 미분과 그 극한의 미분은 다르다

암시적 logistic step은 F(z,y,θ)=z−y−hθz(1−z)=0을 푼다. F_z≠0인 곳에서 암묵 미분하면

$$
\frac{dz}{d\theta}=\frac{dy/d\theta+h z(1-z)}{1-h\theta(1-2z)}.
$$

미분 가능한 근사 상태가 F=r(θ)를 만족하면 분자에 r′(θ)가 추가된다. 작은 잔차값만으로 미분 오차를 제한할 수 없다. r(θ)=εsin(θ/ε²)는 진폭 ε지만 도함수 진폭은 1/ε다. 고정 횟수 Newton을 미분하면 그 반복 프로그램의 도함수다. 상태와 민감도의 수렴에는 적절한 추가 제어가 필요하다.

| 미분 대상 | 한 모수의 상태 민감도 |
|---|---|
| 연속 Logistic 흐름 | T y(T)(1−y(T)) |
| 고정 초기값의 수렴한 암시적 Step | h z(1−z)/F_z |
| 고정 Newton 프로그램 | 실제 갱신을 통한 Chain Rule |

```python
from decimal import Decimal as D,localcontext
with localcontext() as ctx:
    ctx.prec=60
    h,theta,y=D('.25'),D('.5'),D('.2')
    a,b=h*theta,1-h*theta
    root=2*y/(b+(b*b+4*a*y).sqrt())
    exact_derivative=h*root*(1-root)/(1-h*theta*(1-2*root))
    x,dx=y,D(0)
    for k in range(1,5):
        residual=x-y-h*theta*x*(1-x)
        denominator=1-h*theta*(1-2*x)
        total_derivative=denominator*dx-h*x*(1-x)
        denominator_derivative=2*h*theta*dx-h*(1-2*x)
        dx-= (total_derivative*denominator-residual*denominator_derivative)/(denominator*denominator)
        x-=residual/denominator
        if k in (1,2,4):print(k,float(abs(x-root)),float(abs(dx-exact_derivative)))
    assert abs(x-root)<D('1e-30') and abs(dx-exact_derivative)<D('1e-28')
```

Decimal 예제는 Newton 갱신을 미분하고 그 극한을 암묵 도함수와 비교한다. 실제 명시적 Euler adjoint와는 별도의 보조 step이다. 연속 기준 gradient도 초기 상태에 더하기 전 작은 목표 차이를 expm1으로 평가하도록 보완했다. 저장 Euler 상태가 전혀 변하지 않아도 연속 민감도는 작게 남을 수 있다. 유한 계산 그래프를 다른 목적함수로 바꾸어 검증하는 것은 아니다.

## 손계산과 직접 풀기

θ=0이면 모든 Euler 상태는 0.2입니다. 각 step의 F_y=1, F_theta=0.16h이므로 s_N=0.16T, reverse a_n=0.2-target입니다. 전체 gradient는 (0.2-target)0.16T로 두 방식이 같습니다.

직접 풀기: T=1, target=0.8인 이 경우 gradient의 부호와 값을 구하세요. 풀이: -0.6×0.16=-0.096입니다. θ를 조금 키우면 logistic 상태가 증가해 목표에 가까워지므로 손실이 감소한다는 직관과 맞습니다. N이 달라도 θ=0에서는 이 특별한 계산이 동일합니다.

## AI 연구에서의 Gradient 검증

Differentiable simulator, Neural ODE와 optimal control에서 어떤 forward 해법을 미분했는지 기록하세요. 가변 step 선택, event, 수렴하지 않은 implicit solve를 미분할 때는 추가 비매끄러움과 오차가 생길 수 있습니다. 현재 실험은 고정 step·부드러운 scalar 모형이며 checkpointing 전체 구현이나 연속 adjoint solver를 제공한다고 주장하지 않습니다.

참고: [SciPy IVP 정의와 적분기 설정](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
