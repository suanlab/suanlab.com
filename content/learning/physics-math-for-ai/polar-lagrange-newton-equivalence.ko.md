## 하나의 운동을 두 좌표로 설명하기

카메라는 직교 좌표를, 거리·방향 센서는 반지름과 각도를 보고한다. 센서가 바뀌어도 물체의 운동은 같다. 그러나 직교 좌표 가속도와 극좌표를 두 번 미분한 값은 다른 숫자일 수 있다. Lagrange 방정식은 에너지에서 올바른 좌표 운동방정식을 얻는 체계적인 방법이다.

이번에는 순간적인 좌표 변환을 넘어 실제 시간 궤적을 계산한다. 먼저 방정식을 유도하고, 같은 중심력 운동을 직교 좌표에서 푼 뒤 극좌표로 관찰한다. 원점을 지나는 실험에서는 위치·속도·힘이 유한해도 극좌표 역변환이 정의되지 않는다. 물리적 운동의 실패와 표현의 실패를 구분하자.

## 정상 작용에서 방정식이 나오는 이유

시작·끝 시각과 위치를 고정한다. 양 끝에서 0인 변화 η에 대해 S=∫L(q,qdot)dt를 미분하고 부분적분하면

$$
\delta S=\left[\frac{\partial L}{\partial\dot q_i}\eta_i\right]_0^T
-\int_0^T\left[\frac{d}{dt}\frac{\partial L}{\partial\dot q_i}-\frac{\partial L}{\partial q_i}\right]\eta_i\,dt.
$$

위 식은 좌표 i에 대해 합한 표현이다. 고정 경계에서는 첫 항이 사라진다. 충분히 매끄러운 모든 내부 변화에 대해 일차 변분이 0이면 변분법의 기본 보조정리로 대괄호 안의 각 성분이 0이어야 한다. 이것이 Euler–Lagrange 방정식이다. 이 논증은 정상성을 주며 최솟값까지 보장하지 않는다. 앞 단원의 평탄 방향과 안장점 사례를 기억하자.

일반 역학의 정식화는 Russ Tedrake의 MIT [Multi-Body Dynamics](https://underactuated.mit.edu/multibody.html)를 참고할 수 있다. 아래 식은 우리 극좌표 모델의 정의역과 단위를 포함해 직접 유도한다.

## 미분하기 전에 물리계를 정하기

질량 m>0, 중심 위치에너지 U=kr²/2, 강성 k≥0으로 둔다. 직교 좌표 힘은 −kx다. 정규 극좌표 영역 r>0에서는

$$
\boldsymbol x=r\begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix},\qquad
L=\frac m2(\dot r^2+r^2\dot\theta^2)-\frac k2r^2.
$$

반지름은 m, 각도는 rad, 라그랑지안은 J 단위다. 원점은 이 좌표 영역에 포함되지 않는다. 각도의 반복과 분기선 문제도 별도로 존재한다. 주각이 π에서 −π로 바뀌었다고 물체가 충격을 받았다고 해석하지 않는다.

L의 편미분에서는 다른 좌표와 속도 인자를 독립적으로 유지한다. 반면 시간에 대한 전미분에서는 실제로 함께 변하는 양을 모두 미분한다. 두 미분을 혼동하면 기하학적인 관성 항을 빠뜨리기 쉽다.

## 반지름 방정식 유도

∂L/∂rdot=m rdot이므로 시간 미분은 m rddot이다. 좌표에 대한 편미분은 ∂L/∂r=m r thetadot²−kr이다. 따라서

$$
m\ddot r-mr\dot\theta^2+kr=0,
\qquad \ddot r=r\dot\theta^2-\frac{k}{m}r.
$$

첫 식의 각 항은 힘 N 단위다. 좌표 방정식의 오른쪽에 r thetadot²가 있다고 해서 용수철이 바깥으로 민다는 뜻은 아니다. 반지름 방향 기저가 회전하면서 좌표 자체의 변화율이 달라지는 효과다. 실제 힘의 방향은 가속도 벡터를 복원하여 확인한다.

## 각도 방정식 유도

L은 θ에 명시적으로 의존하지 않지만 각속도의 계수는 반지름에 의존한다.

$$
\frac{\partial L}{\partial\dot\theta}=mr^2\dot\theta,\qquad
\frac{d}{dt}(mr^2\dot\theta)
=m(r^2\ddot\theta+2r\dot r\dot\theta).
$$

그러므로 r>0에서

$$
\ddot\theta=-\frac{2\dot r\dot\theta}{r}.
$$

계수 2는 곱의 미분에서 나온다. 이를 생략하면 토크가 없는데도 mr²thetadot가 변하게 된다. 각도 Euler–Lagrange 방정식은 토크 N m 단위이며 반지름 방정식의 N과 다르다. 여기서는 보존 항등식을 직접 확인한다. 대칭과 보존량을 일반적으로 연결하는 Noether 정리 전체는 후속 주제다.

## 뉴턴 가속도를 복원하기

기저의 시간 미분은 erdot=thetadot etheta, ethetadot=−thetadot er다. 위치를 두 번 미분하면

$$
\boldsymbol a=(\ddot r-r\dot\theta^2)\boldsymbol e_r
+(r\ddot\theta+2\dot r\dot\theta)\boldsymbol e_\theta.
$$

두 Lagrange 방정식을 대입하면 실제 반지름 가속도는 −kr/m이고 접선 가속도는 0이다. 따라서 a=−(k/m)x로 직교 좌표의 뉴턴 방정식과 일치한다. 좌표 가속도 rddot와 물리 가속도의 투영 a·er는 같지 않다.

속도에 대한 L의 헤시안은 대각 성분 m, mr²를 가진다. r>0에서는 가역이므로 가속도를 결정할 수 있다. r=0에서는 이 좌표의 정규성 논증이 실패한다. 직교 좌표의 질량 m이 사라지는 것은 아니다.

## 실제 해석해를 따라가기

Ω=√(k/m)로 쓰면 초기 직교 위치 x₀와 속도 v₀에 대한 해는

$$
\boldsymbol x(t)=\boldsymbol x_0\cos(\Omega t)
+\frac{\boldsymbol v_0}{\Omega}\sin(\Omega t),\qquad
\boldsymbol v(t)=-\Omega\boldsymbol x_0\sin(\Omega t)+\boldsymbol v_0\cos(\Omega t).
$$

k=0에서는 별도로 x=x₀+t v₀, v=v₀를 사용하여 0으로 나누지 않는다. 이는 해석해 공식이지만 컴퓨터에서 사인·코사인을 평가한 소수값까지 정확하다는 뜻은 아니다. 에너지 잔차를 강제로 0으로 덮지 않는다.

정규 위치에서는 r=|x|, rdot=x·v/r, thetadot=(x×v)/r²로 극좌표 상태를 얻는다. 각속도는 물리 벡터에서 계산하며, 분기선을 넘는 각도 차이를 나머지 연산한 값으로 계산하지 않는다.

## 조절값과 시간축 읽기

자연 진동수는 정수 f=0,…,4로 Ω=πf/4 s⁻¹로 정하고 k=mΩ²로 둔다. f를 유지하고 질량을 바꾸면 강성도 함께 바뀐다. 초기 반지름은 코드/2 m, 초기 반지름 속도는 코드/2 m/s, 초기 방향은 π/2의 정수배다.

각속도 코드 W는 thetadot₀=W Ωref/2를 뜻한다. f>0이면 Ωref=Ω이고 자유 입자 분기에서는 Ωref=1 s⁻¹이다. 이 약속으로 소수 입력을 π와 같게 맞추지 않아도 W=±2인 정확한 원운동 조건을 설정할 수 있다. 조회 시각은 0~2초의 같은 궤적 위 한 점을 선택한다. ε로 후보 경로를 바꾸거나 적분 간격을 바꾸는 조절값이 아니다.

## 예제: 직선 운동인데 극좌표 가속도는 비영

m=1, k=0, r₀=1, θ₀=0, rdot₀=1, thetadot₀=1로 두자. 초기 위치는 (1,0), 속도는 (1,1)이고 뉴턴 가속도는 0이다. 그러나 초기 rddot=1, thetaddot=−2다. 기저 회전에 따른 항이 좌표 가속도와 상쇄되어 실제 가속도 벡터가 0이 된다.

1초 뒤 위치는 (2,1), 반지름은 √5, 반지름 속도는 3/√5, 각속도는 1/5이다. 직교 속도는 계속 (1,1), 운동에너지는 1 J, 각운동량은 1 kg m²/s다. 반지름이 바뀌면 토크가 없어도 각속도는 바뀔 수 있다.

## 예제: 원운동의 구심 가속도

m=1, f=2, r₀=1, rdot₀=0, W=2이면 Ω=π/2이고 초기 각속도가 Ω와 같다. 궤적은 원이므로 rddot=0이다. 하지만 실제 반지름 방향 가속도는 −π²/4 m/s²다.

운동에너지와 위치에너지는 각각 π²/8 J, 전체 에너지는 π²/4 J, 각운동량은 π/2 kg m²/s다. 1초 뒤 위치는 (0,1), 속도는 (−π/2,0)이다. 반지름이 일정하다고 외력이 없는 운동은 아니다.

## 예제: 원점을 지나도 운동은 계속된다

m=1, f=2를 유지하되 두 초기 속도를 0으로 두고 (1,0)에서 시작하자. 경로는 x=(cos(πt/2),0)이다. t=1에서 원점을 속도 (−π/2,0)으로 통과한다. 그 순간 힘과 가속도는 0이고 전체 에너지는 π²/8 J다.

원점에서는 극좌표의 각도·속도 역변환이 정규 상태를 만들 수 없다. 구현은 극좌표 상태를 미정으로 반환하고 해당 속도 곡선을 원점 앞뒤로 나눈다. 유한한 직교 궤적은 계속 표시한다. 정규 샘플 사이에 생기는 원점 통과 시각도 해석적으로 구해 삽입하므로 극좌표 곡선이 미정 지점을 가로질러 이어지지 않는다. 조회 시각의 알려진 사분면 위상은 항등식으로 판정한다. 다른 사건 시각은 해석적 근의 소수 평가이며 원점 위치는 그 근의 항등식에 따라 0으로 둔다. 작은 양의 반지름을 임의로 삽입하지 않는다. 원점 가까이에서 정의는 되지만 조건이 나빠지는 경우는 정확한 원점 실패와 구분한다.

## 연구 확장: 궤적만으로 Lagrangian을 식별할 수 있는가

일정한 속도로 움직이는 자유 수레에서 위치와 시간은 측정할 수 있다. 그러나 그 운동을 만든 변분 함수가 직접 측정되는 것은 아니다. 속도 Hessian이 양의 정부호인 서로 다른 Lagrangian 후보들도 이런 궤적을 모두 맞출 수 있다. 적합 결과가 물리적으로 식별되었다고 판단하기 전에 가설의 범위를 명시하자.

### 더 넓은 운동 Lagrangian 후보 만들기

학습 예제를 기존 실습의 자유 분기 $f=0$으로 제한해 퍼텐셜이 없는 경우를 보자. 직교 좌표에서 다음 후보를 생각할 수 있다.

$$
s=\boldsymbol v\cdot\boldsymbol v,\qquad
L_\lambda(\boldsymbol v)=\frac m2s+\frac\lambda4s^2,\qquad m>0,\quad\lambda\ge0.
$$

$m$은 kg, $s$는 m²/s², $\lambda$는 kg s²/m² 단위이므로 $L_\lambda$의 두 항은 모두 에너지 단위다. 0이 아닌 $\lambda$는 기존의 자연적인 이차 운동에너지 모델 밖에 있는 비이차 후보다. 추가 계수는 보충 Python 실험에서 사용한다. 기존 조작에 계수를 추가하거나 진동수가 0이 아닌 조화 분기까지 동등하다고 주장하는 것은 아니다.

### 양의 곡률만으로 후보를 고를 수는 없다

물리적 속도에 대해 미분하면 다음과 같다.

$$
\boldsymbol p_\lambda=(m+\lambda s)\boldsymbol v,\qquad
H_\lambda=\frac{\partial^2L_\lambda}{\partial\boldsymbol v\,\partial\boldsymbol v}
=(m+\lambda s)I+2\lambda\boldsymbol v\boldsymbol v^T.
$$

0이 아닌 벡터 $u$에 대해 $u^TH_\lambda u=(m+\lambda s)|u|^2+2\lambda(v\cdot u)^2>0$이다. 0이 아닌 속도에서 수직·평행 방향의 고유값은 $m+\lambda s$, $m+3\lambda s$이며, 속도가 0이면 둘 다 $m$이다. 모든 후보의 속도·운동량 사상은 정칙하다. 다만 $\lambda>0$일 때 이 응답 행렬은 속도에 의존하므로 일정한 Newton 질량 행렬과는 다르다.

후보는 위치에 의존하지 않는다. 따라서 Euler–Lagrange 방정식과 정칙한 극좌표로 끌어온 식은 다음과 같다.

$$
\mathcal E_x=H_\lambda\ddot{\boldsymbol x}=0,\qquad
\mathcal E_q=J^TH_\lambda(J\ddot q+\dot J\dot q)=0,\qquad
\frac{\partial^2\widetilde L_\lambda}{\partial\dot q\,\partial\dot q}=J^TH_\lambda J,\quad r>0.
$$

$r>0$에서 $J$와 $H_\lambda$가 가역이므로 모든 자유 해는 $\ddot x=0$을 만족한다. 극좌표 방정식도 앞에서 유도한 회전 기저 식과 같다. 끌어온 속도 Hessian 역시 양의 정부호다. 원점에서는 좌표가 여전히 유효하지 않으며 운동 Lagrangian 후보를 바꿔도 랭크 결함을 고칠 수 없다.

### 관측량과 전체 끝점 고정 최적해 비교하기

위치가 같더라도 운동량과 시간에 독립적인 에너지는 같지 않을 수 있다.

$$
E_\lambda=\boldsymbol p_\lambda\cdot\boldsymbol v-L_\lambda
=\frac m2s+\frac{3\lambda}{4}s^2.
$$

보존되는 후보 에너지가 곧 보정된 에너지 측정값은 아니다. 더 나아가 엄격한 볼록성과 Jensen 부등식으로 $T>0$의 매끄러운 끝점 고정 경로에 대해 다음을 얻는다.

$$
S_\lambda[\boldsymbol x]\ge
T L_\lambda\!\left(\frac{\boldsymbol x(T)-\boldsymbol x(0)}{T}\right),\qquad
\text{equality iff }\boldsymbol v(t)\text{ is constant a.e.}
$$

따라서 같은 직선 경로가 모든 후보에서 유일한 작용 최소다. 단지 미분방정식의 해만 공유하는 것이 아니다. 최적 경로를 맞추거나 각 후보의 자체 에너지가 보존됨을 확인하는 것만으로는 $\lambda$를 고를 수 없으며 작용값 자체는 다르다.

### 속도를 명시하고 보정된 힘으로 탐침하기

물리적 힘의 관례를 고정하고 강제 Euler–Lagrange 방정식에 $\boldsymbol F$를 가한다. 초기 응답은 다음과 같다.

$$
H_\lambda\boldsymbol a=\boldsymbol F,\qquad
H_1\big|_{m=1,\,v=(1,1)}=\begin{pmatrix}5&2\\2&5\end{pmatrix},\qquad
\boldsymbol a\big|_{F=(1,0)}=\frac1{21}\begin{pmatrix}5\\-2\end{pmatrix}.
$$

예제 행렬의 수치는 SI 단위이며 $H$는 kg, 가속도는 m/s²다. $m=1$ kg, $v=(1,1)$ m/s, $F=(1,0)$ N에서 다음과 같이 계산한다.

| $\lambda$ / (kg s²/m²) | $E_\lambda$ / J | 초기 $a_x$ / (m/s²) | 초기 $a_y$ / (m/s²) |
|---|---|---|---|
| 0 | 1 | 1 | 0 |
| 1/2 | 5/2 | 3/8 | -1/8 |
| 1 | 4 | 5/21 | -2/21 |

![같은 자유 궤적과 서로 다른 보정된 초기 힘 응답](/assets/learning/analytical-lagrange-kinetic.ko.svg)

이 후보에서 $F_y=0$은 $p_y$를 일정하게 유지한다. $p_y=(m+\lambda s)v_y$의 속력 의존 계수가 두 속도 성분을 결합하므로 표처럼 y 방향 가속도가 생긴다.

이는 동일한 0이 아닌 속도에서의 초기 가속도이며 강제 궤적을 적분한 결과가 아니다. 속도가 0인 순간에 한 번만 탐침하면 모든 후보에서 $H_\lambda=mI$이므로 사차 항을 식별할 수 없다. 측정 정확도뿐 아니라 어떤 상태를 자극했는지도 중요하다.

### 방정식 대칭과 작용의 준대칭 구분하기

$dF(x,t)/dt$는 속도에 대한 아핀 식이므로 속도 Hessian이 0이다. 따라서 0이 아닌 Hessian 변화는 전미분 게이지가 아니다. 물리적 대칭 가정으로도 후보를 구분할 수 있다. x 방향의 일정한 Galilean boost $u$에 대해 다음이 성립한다.

$$
L_0(\boldsymbol v+u\boldsymbol e_x)-L_0(\boldsymbol v)
=\frac d{dt}\left(mu x+\frac12mu^2t\right),\qquad
H_\lambda(u\boldsymbol e_x)-H_\lambda(0)
=\lambda u^2\begin{pmatrix}3&0\\0&1\end{pmatrix}.
$$

$\lambda>0$, $u\ne0$이면 뒤의 차이가 0이 아니므로 boost에 따른 변화를 위치·시간 함수의 전미분으로 쓸 수 없다. 자유 방정식은 boost에 불변이어도 모든 후보 작용이 통상적인 Galilean 준불변성을 갖는 것은 아니다. 자연적인 이차 후보나 이 작용 원리를 강제하는 학습기는 추가 물리 가정을 사용한다. 그 가정을 궤적 자료와 함께 명시해야 한다.

### 실행하고 추론의 조건 점검하기

표준 라이브러리 코드로 표와 boost Hessian 검사를 재현한다.

```python
from fractions import Fraction as Q

def candidate(lam, v):
    m = Q(1)
    s = sum(x*x for x in v)
    p = [(m+lam*s)*x for x in v]
    H = [[(m+lam*s)*int(i == j)+2*lam*v[i]*v[j]
          for j in range(2)] for i in range(2)]
    energy = m*s/2+3*lam*s*s/4
    return p, H, energy

# SI: m=1 kg, v=(1,1) m/s, F=(1,0) N. Responses are initial accelerations.
for lam in map(Q, ['0', '1/2', '1']):
    p, H, energy = candidate(lam, [Q(1), Q(1)])
    det = H[0][0]*H[1][1]-H[0][1]*H[1][0]
    acceleration = [H[1][1]/det, -H[1][0]/det]
    assert H[0][0] > 0 and det > 0
    assert [sum(H[i][j]*acceleration[j] for j in range(2))
            for i in range(2)] == [1, 0]
    print(lam, energy, *acceleration)
    zero_H = candidate(lam, [Q(0), Q(0)])[1]
    boost_H = candidate(lam, [Q(1), Q(0)])[1]
    assert boost_H[0][0]-zero_H[0][0] == 3*lam
# All candidates produce free straight paths; a calibrated forced response
# and the usual Galilean action principle supply additional information.
```

**연구 질문과 답**

- 양의 속도 Hessian과 일치하는 자유 궤적으로 자연 운동에너지를 식별하는가? **답:** 아니다. 더 넓은 이 후보 가족은 두 검사를 모두 통과한다.
- 경계항의 모호성만 설명한 것인가? **답:** 아니다. 전미분은 이런 속도 의존 Hessian 변화를 만들 수 없다.
- 정지 상태의 힘 측정 한 번으로 $\lambda$를 식별하는가? **답:** 아니다. 그 순간 Hessian은 $mI$다. 유익한 0이 아닌 속도 탐침이나 명시적인 물리 모델 가정이 필요하다.

## 해설이 있는 연습 문제

1. 각도 방정식의 2는 어디서 나오는가? **해설:** r²thetadot를 곱의 미분법으로 미분한다. r²의 미분은 2r rdot이므로 r²thetaddot+2r rdot thetadot가 된다.
2. 자유 운동 예제의 1초 뒤 각운동량은? **해설:** m(x vy−y vx)=1(2−1)=1 kg m²/s다. mr²thetadot=1·5·1/5로 계산해도 같다. 각속도만 보존되는 것은 아니다.
3. 원운동에서 rddot=0이면 물리적 반지름 가속도도 0인가? **해설:** 아니다. rddot−r thetadot²=−rΩ²이며 안쪽 중심력이 필요하다.
4. 원점에서 작은 반지름으로 나누면 되는가? **해설:** 그러면 존재하지 않는 좌표 상태를 만들어 낸다. 직교 위치·속도를 유지하고 극좌표 역변환을 미정으로 표시한다.
5. f와 초기 좌표 데이터를 유지하고 질량을 두 배로 하면? **해설:** 강성·에너지·각운동량은 두 배이고 실제 궤적은 같다. 이 실험은 강성이 아니라 고유 진동수를 고정한다.
6. Euler–Lagrange 방정식을 얻으면 모든 시간 구간에서 작용 최소인가? **해설:** 아니다. 주어진 경계 변화에 대한 정상 조건이다. 앞의 진동자 작용 예제에서는 곡률이 평탄하거나 양·음 방향을 가질 수 있었다.

## 연구 연결과 검증의 범위

동역학을 학습한 AI 모델은 정규 좌표 사이에서 물리적으로 일치해야 한다. 좌표의 이계 미분을 그대로 같게 두기보다 실제 가속도·에너지·각운동량을 비교한다. 좌표의 정의역 정보를 상태 표현에 포함하여 특이한 역변환을 임의의 유한 상태로 대체하지 않도록 하자.

계산은 독립적인 60자리 직교 해석해와 반지름 미분 기준값으로 검증한다. 이는 해석해의 수치 평가다. 다음 단원에서는 Lagrange 방정식으로 결합계를 유도하고 실제 궤적을 수치 적분하므로, 적분 간격과 오차도 함께 확인해야 한다.
