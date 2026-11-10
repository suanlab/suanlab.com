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

## 직접 풀기: 보존하지만 틀린 선택

uL=0, uR=1에서 두 mode 모두 RH defect가 0이지만 candidate의 entropy 계수는 +1/12입니다. 상태 2와 1, t=2에서는 shock가 관측 창을 지나 질량이 6에서 8이 됩니다. 누적 유입-유출이 2인지 확인하고 창의 질량이 일정해야 한다는 검사가 왜 잘못인지 설명하세요.

더 읽기: [Clawpack Riemann 교재의 Burgers 단원](https://www.clawpack.org/riemann_book/html/Burgers.html).
