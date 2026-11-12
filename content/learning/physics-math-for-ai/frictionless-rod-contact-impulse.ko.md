## 충돌 여부는 접촉점에서 판단한다

강체는 질량 중심이 이동하면서 동시에 회전합니다. 따라서 가장 낮은 끝점의 속도는 질량 중심 속도와 다를 수 있습니다. 질량 중심이 올라간다고 모든 점이 바닥에서 멀어지는 것은 아닙니다. 충격량을 적용할지 결정하려면 실제 접촉점의 법선 속도를 먼저 계산해야 합니다.

이 단원은 균일한 가는 막대가 고정 수평 바닥과 만나는 한 번의 순간 충돌을 다룹니다. 강체의 호환 속도, 선충격량, 각충격량과 에너지 손실을 연결합니다. 이후 비행이나 영 법선 속도에서의 지속 접촉은 아직 풀지 않습니다.

## 초기 접촉점이 하나인 기하를 고정한다

질량 중심 C를 (0, h), 아래 끝점 O를 (a, 0), 위 끝점을 (−a, 2h)에 둡니다. a≥0, h>0, m>0이며 균일 막대의 길이와 관성모멘트는

$$
\ell=2\sqrt{a^2+h^2}, \qquad I=\frac{m(a^2+h^2)}3
$$

입니다. 아래 끝점만 바닥에 닿고 나머지 선분은 바닥 위에 있습니다. a나 h를 바꾸면 선언한 막대의 기하와 관성도 바뀝니다. 두 모서리가 동시에 닿는 수평 직사각형에는 다른 접촉 모델이 필요합니다.

순간 충돌 사상은 속도가 뛰는 동안 위치와 방향을 고정합니다. 충돌 전 중심 속도는 (Vx, Vy), 각속도 ω는 반시계 방향이 양수입니다. 바닥 법선은 n=(0, 1), 중심에서 접촉점까지의 벡터는 r=(a, −h)입니다. 접선 충격량은 없고, 중력 같은 유한한 힘은 이상적인 영 지속시간에서 유한 충격량을 만들지 않습니다.

## 강체 운동과 호환되는 접촉점 속도

평면 강체 운동학으로

$$
v_O=V+\omega\, \hat z\times r
=(V_x+h\omega, \;V_y+a\omega)
$$

입니다. 충돌 전 법선 성분을 c⁻=Vy⁻+aω⁻로 둡니다. c⁻<0이면 접근하는 접촉이므로 양의 법선 충격량을 적용합니다. c⁻≥0이면 이 사상은 충격량을 적용하지 않습니다. 양수는 분리 속도이고 0은 접근하지 않는 경계입니다.

영 법선 속도에서도 이후 동역학에는 접촉 지지력이 필요할 수 있습니다. 충돌 법칙이 비활성이라는 사실만으로 그 힘이나 정적 평형을 판정하지 않습니다.

## 선충격량과 각충격량을 함께 쓴다

위쪽 법선 충격량을 jn이라 하고 단위는 N s입니다.

$$
V_x^+=V_x^-, \qquad V_y^+=V_y^-+\frac{j_n}{m},
$$

$$
\omega^+=\omega^-+\frac{a j_n}{I}.
$$

레버암 a를 곱하면 각충격량이 됩니다. 바닥은 막대의 외부이므로 막대의 벡터 운동량과 질량 중심에 대한 회전 각운동량은 일반적으로 바뀝니다. 이는 외부 충격량에 대한 물리적 반응입니다.

무마찰 조건은 질량 중심의 접선 운동량이 보존된다는 뜻입니다. 회전이 달라지면 끝점 접선 속도 Vx+hω는 변할 수 있습니다. 접선 충격량 0과 끝점 접선 속도 일정은 같은 조건이 아닙니다.

## 반발계수는 접촉점 법선 속도에 적용한다

접근하는 접촉에서 e∈[0, 1]을 고르고

$$
c^+=-e c^-
$$

를 요구합니다. 충격량 식을 대입하면

$$
c^+=c^-+Wj_n, \qquad W=\frac1m+\frac{a^2}{I}>0,
$$

$$
j_n=-\frac{(1+e)c^-}{W}
$$

입니다. W의 단위는 1/kg이며 역수는 이 접촉에서 보이는 유효 법선 질량입니다. 법선 충격량이 병진과 회전을 함께 만들기 때문에 전체 질량과 다를 수 있습니다.

일반적인 막대에 Vy⁺=−eVy⁻를 적용하면 안 됩니다. 여기서 그 간단한 관계는 a=0처럼 법선 충격량의 회전 결합이 없을 때 성립합니다. 현재 법칙이 제한하는 것은 끝점 법선 속도입니다.

## 에너지 변화를 직접 유도한다

전체 운동에너지는

$$
K=\frac12m(V_x^2+V_y^2)+\frac12I\omega^2
$$

입니다. 속도 점프를 대입하면 활성 충돌에서

$$
K^+-K^-=j_nc^-+\frac12Wj_n^2
=-\frac{(1-e^2)(c^-)^2}{2W}
$$

를 얻습니다. e=1이면 에너지가 병진과 회전 사이에서 재분배되며 합은 보존됩니다. e<1이면 음수가 아닌 손실이 생깁니다. 비활성 분기에서는 속도와 에너지가 모두 그대로입니다.

이 손실은 선언한 반발 사상의 에너지 예산입니다. 온도, 변형의 시간 이력, 내부 손실 기구 사이의 분배까지 예측하지 않습니다. 그런 질문에는 추가 구성 모델이 필요합니다.

## 보존되는 각운동량의 기준점을 찾는다

고정된 충돌 위치 O에 대한 중심 상대 위치는 C−O=(−a, h)입니다.

$$
L_O=I\omega+(C-O)\times mV
=I\omega-a mV_y-hmV_x.
$$

충격량은 O를 지나므로 O에 대한 충격 토크는 0이고 LO⁺=LO⁻가 정확하게 성립합니다. 질량 중심의 회전 각운동량은 달라도 이 전체 각운동량은 일정합니다. 궤도 항을 빼고 Iω만 검사하면 다른 양을 검사하는 셈입니다.

이는 고정 충돌점에서의 순간 관계입니다. 이후 접촉점이 움직이거나 기준점이 가속하면 미분과 수송 항을 다시 검토해야 합니다.

## 탄성·비탄성 경계를 주의 깊게 읽는다

탄성 충돌이어도 질량 중심이 위로 튀어야 하는 것은 아닙니다. 회전이 끝점을 위로 열면서 중심 Vy⁺는 여전히 음수일 수 있습니다. 보장되는 것은 접촉점의 출구 법선 속도와 전체 운동에너지 보존입니다.

e=0에서는 c⁺=0입니다. 접선 병진, 각속도와 양의 에너지가 남을 수 있습니다. 법선 속도 하나가 0이라는 조건은 모든 속도가 사라진다는 뜻이 아닙니다. 이후 지지·분리 여부는 지속 접촉 모델의 문제입니다.

수직 막대 a=0은 관성 mh²/3이 양수지만 법선 충격량이 중심을 지나므로 각속도가 바뀌지 않습니다. 이때 입자와 같은 수직 반발 관계를 회복합니다.

## 접촉 Jacobian으로 일반화한다

일반화 속도를 ν=(Vx, Vy, ω), 접촉 Jacobian을 J=(0, 1, a), 질량행렬을 M=diag(m, m, I)로 두면

$$
c=J\nu, \qquad M(\nu^+-\nu^-)=J^\mathsf Tj_n,
\qquad W=JM^{-1}J^\mathsf T
$$

입니다. 이 식은 유효 접촉 질량의 구조를 보여 줍니다. 현재는 스칼라 법선 접촉 하나와 양의 질량행렬이어서 법선 풀이가 명확합니다. 마찰이나 여러 동시 접촉, 탄성 표면을 더하면 문제를 닫는 법칙도 달라집니다. 접선 법칙을 임의로 덧붙이기 전에 일관성과 에너지 성질을 따로 검토해야 합니다.

## 기하와 전후 비교를 읽는 방법

이상적인 충돌 전후에서 막대의 위치와 방향은 같습니다. 속도만 바뀌므로 그림을 나중 시각의 궤도로 해석하지 않습니다. 속도 화살표는 m/s, 충격량은 N s이며 각속도와 각운동량도 별도 단위를 사용합니다.

제어값에서 관성·충격량·속도·에너지는 정확한 유리수이고 길이는 수치 제곱근입니다. 반발 잔차는 접근 분기에서만 정의합니다. 비활성 충돌을 반발식의 실패로 표시하지 않습니다.

## 연구 확장: 속도를 사영한 뒤 반발계수의 식별 가능성을 묻는다

### 질량 행렬이 비교할 거리를 정한다

충돌은 병진과 회전을 함께 바꾼다. 후보 속도 변화 두 개를 비교할 때는 좌표 숫자 사이의 보통 유클리드 거리보다 운동에너지 계량이 적절하다. 기존 순간 고정 기하·마찰 없는 단일 바닥 접촉을 유지하고 m>0,a≥0,h>0,I=m(a²+h²)/3으로 둔다. 일반화 속도 v=(Vx,Vy,ω)에 대해

$$
M=\operatorname{diag}(m,m,I),\quad J=(0,1,a),\quad c=Jv,
\qquad W=JM^{-1}J^{\mathsf T}>0,\quad
n=\frac{M^{-1}J^{\mathsf T}}{W},\quad Jn=1,\quad P=\operatorname{Id}-nJ.
$$

로 정의한다. 행 J는 일반화 속도를 접촉점 법선 속도로 바꾼다. n은 법선 속도를 1만큼 바꾸는 응답이며 보통 공간의 단위벡터가 아니다. 병진·회전 성분이 서로 다른 물리 단위를 갖는다. 질량 행렬이 양의 정부호이고 J가 0이 아니므로 W>0이다.

닫히는 접촉 c<0에서 e=0의 결과 Pv는 다음 문제의 유일한 해다.

$$
\min_{Jz\ge0}\frac12(z-v)^{\mathsf T}M(z-v).
$$

허용 속도는 바닥을 파고들지 않는다. v를 후보 z로 바꾸는 비용을 에너지 크기의 이차형식으로 비교한다. 모든 허용 후보를 다음처럼 써서 최소해를 증명한다.

$$
z=Pv+t+\alpha n,\quad Jt=0,\quad\alpha\ge0,
\qquad\frac12\|z-v\|_M^2=\frac12\|t\|_M^2+\frac{(\alpha-c)^2}{2W}
\ge\frac{c^2}{2W}\quad(c<0).
$$

여기서 t는 접선 방향의 일반화 속도이며 실제 시간이 아니다. tᵀMn=Jt/W=0이므로 교차항이 사라진다. c<0이므로 α≥0를 선택해서 (α−c)²를 c²보다 줄일 수 없다. 등호에는 t=0,α=0이 필요해 z=Pv가 나온다. c≥0이면 원래 v가 이미 허용되고 비용 0이므로 충격을 가하지 않는다.

P는 **닫히는 가지**에서 경계 초평면으로의 선형 사영이다. 전체 비탄성 법칙은 허용 반공간으로의 사영이다. 분리 중인 속도에도 P를 무조건 적용하면 충돌이 필요 없는 운동까지 바꿔 버린다.

### 반사와 축소로 에너지 수지를 해석한다

다음 항등식이 성립한다.

$$
P^2=P,\quad P^{\mathsf T}M=MP,\qquad
R=2P-\operatorname{Id},\quad R^2=\operatorname{Id},\quad R^{\mathsf T}MR=M.
$$

앞의 두 식은 질량 계량에서 직교 사영임을 나타낸다. R은 그 계량의 반사로 운동에너지를 보존하며 닫히는 가지의 e=1 갱신이다. R²=Id는 선형 연산자의 성질이다. 실제 충돌 뒤에는 접촉이 분리 중이므로 물리 법칙을 곧바로 다시 적용하면 충격이 0이지 이전 속도로 자동 복원되는 것이 아니다.

v를 질량 계량에서 직교하는 접선·법선 성분으로 나누면 c<0,0≤e≤1에서

$$
v_T=Pv,\quad v_N=nJv,\quad v^+=v_T-ev_N,\qquad
K^+=K_T+e^2K_N,\quad K_N=\frac{c^2}{2W},\quad
K^--K^+=(1-e^2)\frac{c^2}{2W}.
$$

이다. 접선 모드는 유지하고 법선 모드만 반사·축소한다. 법선 운동에너지는 e²배가 된다. 따라서 완전 비탄성 접촉도 접선 모드 에너지를 없애지 않아 물체 전체가 정지할 필요가 없다. 일반화 법선 모드에는 질량중심 운동과 회전이 함께 들어가므로 COM 수직 속도만 뒤집으면 일반적으로 틀린다.

실제 입력 `[1,2,2,0,-10,0,5]`를 넣자. m=a=h=1,I=2/3,W=5/2,c=−1,e=1/2다. 법선 충격량은 3/5 N s다. 이후 Vy=−2/5 m/s,ω=9/10 rad/s, 접촉점 법선 속도는 +1/2 m/s다. 에너지는 1/2에서 7/20 J로 줄어 3/20 J를 잃는다. 질량중심은 내려가도 접촉점은 올라갈 수 있다.

### 상태 갱신은 안정적이어도 역문제는 불안정할 수 있다

고정 기하·e에서 실제 충격량은

$$
j(c)=-\frac{1+e}{W}\min(c,0).
$$

이다. c에 대해 연속이고 Lipschitz 상수는 (1+e)/W다. 0에서 왼쪽 미분은 −(1+e)/W, 오른쪽 미분은 0이다. 매끄러운 한 가지의 미분을 경계 너머로 연장하면 접촉 활성화의 변화를 놓친다. 접촉 사건을 통과하는 시뮬레이터를 미분할 때 중요한 차이다.

이번에는 측정 속도로 e를 추정하자. 실제 입사 속도는 −δ,δ>0, 이후 속도는 eδ로 두고 절대 오차 ζ 이내의 d₋,d₊를 더한다. 비율 추정량은

$$
\widehat e=\frac{e\delta+d_+}{\delta-d_-},\qquad
|\widehat e-e|\le\frac{\zeta(1+e)}{\delta-\zeta}
\quad(0\le\zeta<\delta,\;|d_-|,|d_+|\le\zeta).
$$

를 만족한다. e를 빼면 분자는 d₊+ed₋이고 분모는 δ−ζ 이상이므로 상계가 나온다. 충격량이 작게 0으로 가더라도 반발계수 식별은 나쁠 수 있다. e=1/2,ζ=0.001 m/s에서 다음과 같다.

| 닫힘 속도 δ (m/s) | 속도 오차 ζ (m/s) | e의 오차 상계 | 닫힘 부호 인증? |
|---|---:|---:|---|
| 0.1 | 0.001 | 1/66 | 가능 |
| 0.01 | 0.001 | 1/6 | 가능 |
| 0.001 | 0.001 | 비율 상계 사용 불가 | 불가 |

δ≤ζ이면 입사 속도 구간이 0에 닿거나 넘어간다. 사전 제약 0≤e≤1로 추정을 제한할 수는 있지만 측정 비율이 식별 정보를 준다는 뜻은 아니다. 분모 상계를 쓸 수 없다는 말이 물리 제약을 넣은 모든 추정량의 값도 무한히 커진다는 주장은 아니다.

![연속적인 충격 갱신과 나빠지는 반발계수 식별](/assets/learning/classical-contact-grazing.ko.svg)

위 패널은 m=a=h=1,e=1/2다. 아래는 ζ=0.001 m/s,δ>ζ만 사용하며 가로·세로 모두 로그축이다. 종축은 서로 다른 양이다. 비율 오차 상계의 발산을 실제 충격 갱신의 발산으로 읽지 않는다.

실제 입력 `[1,2,2,0,-11,10,5]`는 c⁻=−1/10, 충격량 3/50, c⁺=1/20, 손실 3/2000이다. `[1,2,2,0,-10,10,5]`는 정확히 비접근 상태라 충격이 없다. 더 미세한 δ=0.01·0.001은 별도 측정 오차 분석이며 실제 속도 입력 간격을 확장한 것이 아니다.

### 정확한 사영과 측정 오차를 실행해 검산한다

다음 표준 라이브러리 셀은 27기하의 사영·반사 항등식과 324반발 상태, 허용 사영 후보, 속도 오차 상계를 분수로 검사한다. 사이트 계산기를 가져오지 않고 독립적인 질량 행렬 계산을 사용한다.

```python
from fractions import Fraction as Q
from itertools import product


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, v):
    return [dot(row, v) for row in a]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def kinetic(mass_diag, v):
    return dot(mass_diag, [x * x for x in v]) / 2


def operators(mass, a, h, restitution):
    inertia = mass * (a * a + h * h) / 3
    mass_diag = [mass, mass, inertia]
    j = [Q(0), Q(1), a]
    w = sum(x * x / m for x, m in zip(j, mass_diag))
    normal = [x / m / w for x, m in zip(j, mass_diag)]
    identity = [[Q(int(i == k)) for k in range(3)] for i in range(3)]
    projection = [[identity[i][k] - normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    reflection = [[identity[i][k] - 2 * normal[i] * j[k]
                   for k in range(3)] for i in range(3)]
    update = [[identity[i][k] - (1 + restitution) * normal[i] * j[k]
               for k in range(3)] for i in range(3)]
    return mass_diag, j, w, normal, identity, projection, reflection, update


def restitution_error_bound(delta, noise, restitution):
    if not 0 <= noise < delta:
        raise ValueError('Incoming speed must be separated from zero.')
    return noise * (1 + restitution) / (delta - noise)


def impact_checks():
    cases = 0
    for mass, a, h in product(map(Q, [1, 2, 4]), [Q(0), Q(1, 2), Q(3, 2)],
                              [Q(1, 2), Q(1), Q(2)]):
        d, j, w, n, identity, p, r, _ = operators(mass, a, h, Q(0))
        m = [[d[i] if i == k else Q(0) for k in range(3)] for i in range(3)]
        assert mm(p, p) == p
        assert mm(transpose(p), m) == mm(m, p)
        assert mm(r, r) == identity
        assert mm(mm(transpose(r), m), r) == m
        for e in [Q(0), Q(1, 2), Q(1)]:
            *_, update = operators(mass, a, h, e)
            for velocity in [[Q(0), Q(-1), Q(0)], [Q(1), Q(1, 2), Q(-1)],
                             [Q(-1), Q(0), Q(1)], [Q(0), -a, Q(1)]]:
                c = dot(j, velocity)
                after = mv(update, velocity) if c < 0 else velocity[:]
                impulse = -(1 + e) * min(c, Q(0)) / w
                assert impulse >= 0
                assert dot(j, after) == (-e * c if c < 0 else c)
                loss = kinetic(d, velocity) - kinetic(d, after)
                assert loss == ((1 - e * e) * c * c / (2 * w) if c < 0 else 0)
                if e == 0:
                    # Feasible competitors differ by a tangent and a nonnegative
                    # normal component; compare the projection objective exactly.
                    optimum = kinetic(d, [x - y for x, y in zip(after, velocity)])
                    for tangent_x, tangent, opening in product(
                            [Q(-1), Q(0), Q(1)], [Q(-1), Q(0), Q(1)],
                            [Q(0), Q(1, 2), Q(1)]):
                        offset = [tangent_x, -a * tangent, tangent]
                        candidate = [x + y + opening * z for x, y, z in zip(after, offset, n)]
                        assert dot(j, candidate) >= 0
                        cost = kinetic(d, [x - y for x, y in zip(candidate, velocity)])
                        assert cost >= optimum
                        if tangent_x == tangent == opening == 0:
                            assert cost == optimum
                cases += 1
    e, noise = Q(1, 2), Q(1, 1000)
    for delta in [Q(1, 10), Q(1, 100)]:
        bound = restitution_error_bound(delta, noise, e)
        for incoming_error, outgoing_error in product([-noise, Q(0), noise], repeat=2):
            estimate = (e * delta + outgoing_error) / (delta - incoming_error)
            assert abs(estimate - e) <= bound
        print('Incoming speed, restitution error bound:', delta, bound)
    try:
        restitution_error_bound(noise, noise, e)
    except ValueError:
        pass
    else:
        raise AssertionError('Grazing interval needs rejection')
    print(f'27 geometries: projection/reflection identities; {cases} restitution states passed')


impact_checks()
```

### 연구 질문과 풀이

1. **e=0이면 이후의 바닥 접촉도 해결되는가?** 아니다. 순간적으로 허용되는 속도만 준다. 지속 접촉·중력·변하는 기하·후속 충돌은 시간 진화 모델이 필요하며 이 단원은 이를 추론하지 않는다.
2. **상태 예측은 안정적인데 e의 추정은 왜 어려울 수 있는가?** 정방향 충격은 접선 접촉 부근에서도 Lipschitz이지만 역문제는 작은 두 속도를 나눈다. 정방향 연속성이 역방향 조건수를 보장하지 않는다.
3. **탄성 반사를 두 번 적용하면 두 실제 충돌을 되돌릴 수 있는가?** 행렬의 제곱은 항등이지만 물리 법칙은 가지를 바꾼다. 첫 갱신 후에는 분리 중이므로 다른 동역학이 다시 닫히는 사건을 만들기 전에는 두 번째 충격이 없다.

일반 질량 행렬과 접촉 Jacobian은 [MIT Underactuated Robotics의 다물체 동역학](https://underactuated.mit.edu/multibody.html)을 참고한다. 위 사영 증명·오차 상계·단일 접촉 가정은 이 예제를 정의한다.

## 예제와 해설이 있는 연습문제

1. **상승하는 중심과 접근 끝점.** m=a=h=1, V=(0, 1/2), ω=−1, e=1/2입니다. **해설:** I=2/3, c⁻=−1/2, jn=3/10입니다. 충돌 후 Vy=4/5, ω=−11/20, c⁺=1/4이며 손실은 3/80입니다. 중심이 올라가도 끝점은 충돌합니다.

2. **탄성 에너지 재분배.** m=a=h=1, V=(1, −1), ω=0, e=1입니다. **해설:** W=5/2, jn=4/5, Vy⁺=−1/5, ω⁺=6/5, c⁺=1입니다. 중심은 내려가지만 접촉은 열립니다. 전후 K는 1로 같고 병진·회전 사이 분배가 달라집니다.

3. **법선 속도의 비탄성 정지.** 앞 예제에서 e=0으로 두세요. **해설:** jn=2/5, Vy⁺=−3/5, ω⁺=3/5입니다. 끝점 법선 속도는 0이어도 K⁺=4/5와 Vx=1이 남습니다. 손실은 1/5이고 이후 지속 접촉은 아직 풀지 않았습니다.

4. **정확한 접근 경계.** a=1, Vy=−1, ω=1입니다. **해설:** c⁻=0이므로 허용되는 모든 e에서 jn=0입니다. 질량 중심은 내려가지만 끝점 법선 속도는 0입니다. 이후의 지지력은 연속 접촉 계산에서 구해야 합니다.

5. **수직 극한.** m=4, a=0, h=3, Vy=−2, ω=2, e=1입니다. **해설:** I=12, W=1/4, jn=16이며 Vy⁺=2, ω⁺=2입니다. 레버암이 0이므로 법선 충격량의 회전 토크가 없습니다.

6. **끝점 접선 속도.** 두 번째 탄성 예제로 돌아가세요. **해설:** 접선 충격량 0이므로 Vx=1은 유지됩니다. 그러나 끝점 접선 속도는 hω 때문에 1에서 11/5로 바뀝니다. 바닥 접촉점에 대한 각운동량은 회전·궤도 항이 보상되어 0을 유지합니다.

## 연구 연결과 원자료

학습한 시뮬레이터가 중심 속도만으로 충돌을 분류하면 잘못된 분기를 고를 수 있습니다. 다른 기준점의 각운동량을 보존시키거나 반발계수를 다른 속도에 적용하거나 e=0에서 모든 운동을 지우는 오류도 구분해야 합니다. 접촉 운동학·충격량·에너지·올바른 기준점의 각운동량을 따로 평가하세요.

일반적인 질량행렬과 접촉 Jacobian 구조는 [MIT Underactuated Robotics의 접촉·충돌 설명](https://underactuated.mit.edu/multibody.html)을 참고하세요. 단일 막대 기하·무마찰 법칙·정확 제어값과 위 예제는 이 단원에서 별도로 선언한 실험입니다. 다음 단원은 명시적인 접선 마찰 법칙으로 바퀴의 연속 구름·미끄럼을 풉니다.
