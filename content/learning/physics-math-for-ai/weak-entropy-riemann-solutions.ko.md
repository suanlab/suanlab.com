## Jump에는 적분형 해석이 필요하다

Shock의 불연속점에는 보통 의미의 공간 도함수가 없습니다. 보존 법칙을 분포·약한 의미로 써야 조각별 매끄러운 장에도 적용할 수 있습니다. Burgers flux는 f(u)=u²/2이고 Riemann 초기값은 0 왼쪽 uL, 오른쪽 uR입니다.

속도 s의 jump가 약한 보존식을 만족하려면 delta 계수가 사라져야 합니다.

$$
s[u]=[f(u)],\qquad
s=\frac{u_L+u_R}{2},\qquad [u]=u_R-u_L.
$$

같은 상태일 때도 속도 식은 연속적으로 정의되지만 실제 jump는 없습니다. RH defect 0은 계면의 보존을 검사할 뿐 물리적으로 허용되는 진화를 혼자 선택하지 못합니다.

## 압축과 팽창은 Entropy 부호가 다르다

uL>uR이면 빠른 왼쪽 특성이 느린 오른쪽에 모여 압축 shock를 만듭니다. 속도는 두 상태 사이입니다. uL<uR의 expansion jump도 같은 대수 속도로 약한 식을 만족하지만 특성은 양쪽으로 흩어집니다.

Convex quadratic entropy eta=u²/2, entropy flux q=u³/3의 jump 계수는 다음입니다.

$$
[q]-s[\eta]=\frac{(u_R-u_L)^3}{12}.
$$

Entropy 부등식은 이 계수가 0 이하일 것을 요구합니다. 압축은 만족하고 팽창은 어깁니다. Candidate mode는 보존적인 모든 jump를 허용 해로 취급하지 않고 이 반례를 직접 보여줍니다.

## Rarefaction이 벌어지는 틈을 채운다

uL<uR의 양의 시간 entropy 해는 fan입니다.

$$
u(x,t)=\begin{cases}
u_L&x\le u_Lt,\\
x/t&u_Lt<x<u_Rt,\\
u_R&x\ge u_Rt.
\end{cases}
$$

상태 0과 1, t=1, x=1/4이면 fan 값은 1/4이지만 expansion jump는 0입니다. 둘 다 약한 flux를 보존해도 entropy 선택은 fan입니다. t=0의 그림은 원래 step 초기값을 표시합니다.

## 열린 관측 구간에는 경계 Flux가 있다

수치는 [-2,2]의 질량과 양 끝 flux의 시간 적분을 각각 구합니다. 수지는 M(t)-M(0)+Fout-Fin=0입니다. 파동이 구간을 빠져나가면 질량이 바뀌어도 보존 위반이 아닙니다. 이 예제의 shock·fan 경계 통과를 구간별로 정확히 적분합니다.

Jump는 열린/닫힌 끝점을 둔 별도 수평 선분으로 그립니다. 대각 연결선으로 이어 그리면 실제로 없는 매끄러운 전이와 중간 상태를 암시하기 때문입니다.

## 소멸 점성이 압축 jump를 선택하는 이유

움직이는 step은 분포 의미에서 uₜ=−s[u]δ, f(u)ₓ=[f]δ이므로 Rankine–Hugoniot 조건을 준다. Entropy 쌍에도 같은 계산을 적용하면 ηₜ+qₓ=([q]−s[η])δ다. η=u²/2, q=u³/3에서 팽창 jump의 계수가 양수이므로 보존만으로 충분하지 않은 이유를 알 수 있다.

양의 점성을 더한 uₜ+(u²/2)ₓ=εuₓₓ, ε>0에서 L>R을 잇는 U(z), z=x−st를 찾자. 한 번 적분하고 양 끝 극한을 적용하면

$$
s=(L+R)/2,\quad \varepsilon U'=\tfrac12(U-L)(U-R),
\quad U(z)=s-\tfrac{L-R}{2}\tanh\!\left(\frac{(L-R)z}{4\varepsilon}\right).
$$

U는 L에서 R로 감소하며 폭은 ε/(L−R)에 비례한다. L<R인 증가 연결에서는 같은 ODE의 두 상태 사이 U′가 음수라 증가하는 연결 profile이 없다. 양의 점성은 팽창 shock를 배제한다. 일반 볼록 entropy에서는 ηₜ+qₓ=ε(η′uₓ)ₓ−εη″uₓ²로 소산 부호가 드러난다. 정당한 극한을 위해 compactness와 초기 데이터 가정이 필요하므로 이 진행파 계산만으로 완전한 소멸 점성 수렴 정리를 증명한 것은 아니다.

| L=1,R=0의 ε | z=0의 profile | 중앙 기울기 |
|---|---:|---:|
| 1/2 | 1/2 | −1/4 |
| 1/4 | 1/2 | −1/2 |
| 1/8 | 1/2 | −1 |

```python
from math import tanh
L, R = 1., 0.
s, jump = (L+R)/2, L-R
for eps in (.5, .25, .125):
    k = jump/(4*eps)
    for z in (-1., 0., 1.):
        h = tanh(k*z)
        U = s-jump*h/2
        first = -jump*k*(1-h*h)/2
        second = jump*k*k*h*(1-h*h)
        assert abs(eps*first-(U-L)*(U-R)/2) < 1e-14
        assert abs((U-s)*first-eps*second) < 1e-14
    print(eps, s, -jump*jump/(8*eps))
```

매끄러운 점성 profile은 보조 해석 예제이고 실제 비점성 jump 실습은 분리된 두 선분을 유지한다. 아래 계산은 적분한 profile ODE와 매끄러운 PDE 잔차를 검산하며, 불연속을 가짜 대각 연결선으로 대체하지 않는다.

## 직접 풀기: 보존하지만 틀린 선택

uL=0, uR=1에서 두 mode 모두 RH defect가 0이지만 candidate의 entropy 계수는 +1/12입니다. 상태 2와 1, t=2에서는 shock가 관측 창을 지나 질량이 6에서 8이 됩니다. 누적 유입-유출이 2인지 확인하고 창의 질량이 일정해야 한다는 검사가 왜 잘못인지 설명하세요.

더 읽기: [Clawpack Riemann 교재의 Burgers 단원](https://www.clawpack.org/riemann_book/html/Burgers.html).
