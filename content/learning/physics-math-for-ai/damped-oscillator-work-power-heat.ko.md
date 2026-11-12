## 진동이 줄어들어도 에너지 수지는 남는다

용수철에 연결한 수레를 놓으면 점성 저항 때문에 진폭이 점점 줄어듭니다. 역학적 에너지가 감소하는 것은 계산의 실패가 아닙니다. 이 모형은 그 에너지를 소산된 열이라는 양으로 옮깁니다. 외부 구동이 함께 작용하면 그 일이 양인지 음인지까지 포함해야 합니다.

이 단원은 에너지 수준만 그리지 않고 실제 시간 초기값 문제를 풉니다. 질량과 용수철 상수는 양수, 감쇠 계수는 음수가 아니며 구동은 일정합니다. 저장 역학적 에너지, 구동 일, 적분한 열을 구분합니다. 온도·열용량·열저장고·주기 구동은 이 모형에 들어 있지 않습니다.

## 저장과 전달을 담당하는 힘을 구분한다

x는 용수철의 자연 길이 위치에서 잰 변위이고 v는 속도입니다.

$$
m\ddot x+c\dot x+kx=F,\qquad x(0)=x_0,\quad \dot x(0)=v_0.
$$

용수철 퍼텐셜은 V=kx²/2, 운동에너지는 K=mv²/2입니다. 일정 구동은 저장 에너지 바깥에 두어 외부 일이 명시적으로 나타나게 합니다. 다른 규약에서는 −Fx를 퍼텐셜에 넣을 수도 있지만 그때는 같은 구동 일을 다시 더하지 않아야 합니다.

점성 힘은 −cv이며 계수의 단위는 kg/s로 용수철 상수 N/m와 다릅니다. 점성 힘의 일률은 −cv²이므로 속도의 방향과 관계없이 0 이하입니다. 수레가 왼쪽으로 움직여도 저항은 속도의 반대 방향입니다.

## 평형 위치로 좌표를 옮겨 운동을 푼다

정적 평형은 xe=F/k입니다. y=x−xe, y0=x0−xe, α=c/(2m)로 두면 다음을 얻습니다.

$$
\ddot y+2\alpha\dot y+\omega_0^2y=0,\qquad \omega_0^2=k/m.
$$

c²−4mk의 부호가 감쇠 영역을 결정합니다. 일정 구동은 평형 위치를 옮기지만 구동 주파수를 만들지 않습니다. 따라서 이 실습을 주기 공명 실험이라고 부르면 안 됩니다. 물리적 초기 위치를 고정하고 구동을 바꾸면 y0도 바뀌므로 좌표와 초기 조건을 함께 변환해야 합니다.

## 부족 감쇠에서는 진동이 남는다

c²<4mk일 때 ω=√(ω0²−α²)로 두면 다음 초기값 해를 얻습니다.

$$
y(t)=e^{-\alpha t}\left[y_0\cos(\omega t)+\frac{v_0+\alpha y_0}{\omega}\sin(\omega t)\right],
$$

$$
v(t)=e^{-\alpha t}\left[v_0\cos(\omega t)-\frac{\alpha v_0+\omega_0^2y_0}{\omega}\sin(\omega t)\right].
$$

c>0이면 포락선이 줄어듭니다. c=0이면 같은 식이 무감쇠 조화 운동으로 환원되고 열은 정확히 0입니다. 실습은 이 극한을 별도로 표시합니다. 감쇠가 없어도 일정 구동이 일을 할 수 있으므로 현재 규약의 역학적 에너지는 바뀔 수 있습니다.

## 임계 감쇠에는 중근 해가 필요하다

c²=4mk이면 두 특성근이 같습니다. 서로 다른 두 근의 차이로 나누는 식을 분모가 0인 채 사용할 수 없습니다.

$$
y(t)=e^{-\alpha t}[y_0+(v_0+\alpha y_0)t],
$$

$$
v(t)=e^{-\alpha t}[v_0-\alpha(v_0+\alpha y_0)t].
$$

지수함수에 곱해지는 t 항이 반드시 필요합니다. 위치와 속도의 두 초기 조건을 모두 유지하세요. 임계라는 말은 근의 구조를 뜻하며 모든 초기 조건과 모든 응답 지표에서 가장 빠르다는 일반적인 보장은 아닙니다.

## 과감쇠에는 서로 다른 두 감쇠율이 있다

c²>4mk에서 느린 근과 빠른 근을 정의합니다. 수치 계산에서는 비슷한 양수끼리 빼는 상쇄를 피하도록 느린 근을 다음과 같이 계산합니다.

$$
r_{\mathrm{fast}}=-\alpha-\sqrt{\alpha^2-\omega_0^2},\qquad
r_{\mathrm{slow}}=-\frac{\omega_0^2}{\alpha+\sqrt{\alpha^2-\omega_0^2}}.
$$

이때 y=Cs exp(rslow t)+Cf exp(rfast t)입니다. 계수는 Cs+Cf=y0, rslow Cs+rfast Cf=v0를 풀어 구하고 속도는 미분합니다. 양의 질량·강성·감쇠에서는 두 근이 음수입니다. 초기 조건에 따라 지수 항의 합이 특정 위치를 지나거나 방향을 바꿀 수 있으므로 과감쇠라는 이름만으로 모든 궤적 모양을 단정하지 않습니다.

## 순간 일률 수지를 유도한다

원래 운동 방정식에 속도를 곱합니다.

$$
\frac{d}{dt}\left(\frac12mv^2+\frac12kx^2\right)=Fv-cv^2.
$$

역학적 에너지 E=K+V, 구동 일률 Pdrive=Fv, 감쇠 일률 Pdamp=−cv²입니다. 두 일률의 합은 저장 역학적 에너지의 변화율입니다. 구동 일률은 양수도 음수도 될 수 있습니다. 감쇠 일률은 양수가 될 수 없고 대응하는 열 발생률은 +cv²입니다.

힘만으로 일률을 정할 수 없으며 현재 속도도 필요합니다. 순간 정지에서는 힘과 가속도가 0이 아니어도 일률은 0입니다. 이후에는 가속도로 속도가 변하여 일이 다시 쌓일 수 있습니다.

## 구동 일과 열을 별도로 적분한다

일정 구동의 일은 F[x(t)−x0]입니다. 열은 다음 독립 적분으로 계산합니다.

$$
Q(t)=\int_0^t c\,v(s)^2ds\ge0,
\qquad E(t)-E(0)+Q(t)=W_{\mathrm{drive}}(t).
$$

구현은 잔차를 0으로 만드는 나머지 양을 열이라고 정의하지 않습니다. 해석적으로 구한 속도의 제곱을 직접 적분합니다. 진동 영역에서는 지수와 삼각함수의 적분, 중근에서는 지수에 이차 다항식을 곱한 적분, 실근 영역에서는 세 지수 항의 적분을 사용합니다.

짧은 시간의 지수 적률은 거의 같은 수의 뺄셈을 피하도록 수렴 급수를 사용합니다. 초월함수는 부동소수점으로 평가하므로 잔차는 수치 진단이고 정확한 유리수 증명은 아닙니다. 운동 방정식과 열 발생률을 별도로 수치 적분한 기준값으로 운동과 열을 함께 검사합니다.

## 위치·에너지·일률 그림 읽기

가로 축은 모두 시간 s입니다. 첫 그림은 해석적 시간 운동의 위치 m입니다. 에너지 그림은 E, Q, E0+Wdrive를 J로 비교하며 E+Q가 마지막 양과 같아야 합니다. 일률 그림은 구동, 음의 감쇠, 합계 일률을 W로 표시합니다.

고정된 10초 범위는 같은 초기값 해를 보여 줍니다. 조회 시각은 현재 수치를 바꾸고 시간 범위는 유지합니다. 무감쇠, 임계 감쇠, 강한 과감쇠, 이미 평형에 놓인 상태, 초기 속도의 반대 방향 구동을 비교하세요. 이 가정에서는 구동 일이 음수가 될 수 있지만 소산 열은 음수가 될 수 없습니다.

## 연구 확장: 지수 수렴 인증식과 힘 오차 상계 만들기

에너지가 증가하지 않는다는 사실은 유용하지만 수렴률에는 추가 논증이 필요합니다. 변위가 0이 아니고 속도는 0인 순간에는 에너지가 양수인데 미분은 0입니다. 따라서 H'=−cv²만으로 모든 상태에서 양의 상수에 대해 H'≤−gamma H라고 할 수 없습니다. 복원력도 추정에 참여하도록 교차 항을 추가하겠습니다.

### 구동 평형을 기준으로 에너지 고르기

m,k,c>0이고 구동 F는 일정하다고 가정합니다. 원래 해와 같이 평형 위치로 좌표를 옮깁니다.

$$
y=x-\frac Fk,\quad H=\frac12mv^2+\frac12ky^2
=E_{\mathrm{mech}}-Fx+\frac{F^2}{2k},\qquad
\dot H=-cv^2.
$$

H는 평형에서의 변위를 재며 평형에서 0입니다. 구동이 일을 하면 증가할 수도 있는 기존의 용수철·운동 저장 에너지 E_mech와 다릅니다. 기존 에너지 규약이나 독립 열 적분을 바꾸지 않습니다. 여기에는 양의 감쇠가 필요합니다. 기존 c=0 계산은 여전히 타당하지만 아래의 수축을 갖지는 않습니다.

### 정지한 변위도 보는 항 추가하기

단위가 1/s인 epsilon을 골라 다음 식을 미분합니다.

$$
0<\varepsilon<\frac cm,\qquad
L=H+\varepsilon m yv+\frac{\varepsilon c}{2}y^2,
\qquad
\dot L=-(c-\varepsilon m)v^2-\varepsilon k y^2=-D.
$$

상쇄를 확인해 봅시다. epsilon m yv의 미분은 epsilon m v²−epsilon c yv−epsilon k y²입니다. epsilon c y²/2의 미분이 혼합 항을 소거합니다. 남은 D의 두 계수는 양수입니다. H'와 달리 변위만 있고 정지한 경우를 포함해 영이 아닌 모든 상태에서 미분이 엄격히 음수입니다.

감소하는 식이라고 자동으로 상태를 제어하지는 못하므로 L이 양수인지도 증명해야 합니다. 에너지에 맞춘 좌표를 사용합니다.

$$
\mathbf z=\begin{pmatrix}\sqrt{k}\,y\\\sqrt{m}\,v\end{pmatrix},\quad
L=\frac12\mathbf z^TS\mathbf z,\quad
S=\begin{pmatrix}1+\varepsilon c/k&\varepsilon\sqrt{m/k}\\
\varepsilon\sqrt{m/k}&1\end{pmatrix},\quad
\det S=1+\frac{\varepsilon(c-\varepsilon m)}{k}>0.
$$

두 좌표의 단위는 J의 제곱근이며 S는 무차원입니다. 첫 대각 원소와 행렬식이 양수이므로 대칭 행렬은 양의 정부호입니다. 두 양의 고유값을 lambda_min,lambda_max라 하면 Rayleigh 몫으로 다음을 얻습니다.

$$
\lambda_{\min}H\le L\le\lambda_{\max}H,\quad
d=\min\left(\varepsilon,\frac cm-\varepsilon\right)>0,
\quad D\ge2dH\ge\frac{2d}{\lambda_{\max}}L,
\quad \gamma=\frac{2d}{\lambda_{\max}}.
$$

D도 같은 좌표 제곱을 포함하며 가중치는 양수인 c/m−epsilon과 epsilon입니다. L의 미분 부등식을 적분하거나 exp(gamma t)L을 미분하면 다음을 증명합니다.

$$
L(t)\le e^{-\gamma t}L(0),\qquad
H(t)\le\frac{\lambda_{\max}}{\lambda_{\min}}e^{-\gamma t}H(0).
$$

앞계수 lambda_max/lambda_min이 중요합니다. 이를 빼면 처음 정지한 반례까지 포함해 H 자체가 모든 초기 상태에서 즉시 지수적으로 줄어든다고 잘못 주장합니다. 충분 상계이며 반드시 가장 날카로운 감쇠율은 아닙니다. 초기 H가 0이면 정지 평형 해입니다.

### 임계 진동자의 상수 계산하기

일반적으로 epsilon=c/(2m)을 편리하게 사용할 수 있습니다. m=k=1,c=2이면 S의 원소는 [[3,1],[1,1]], 고유값은 2±sqrt(2), gamma는 초당 2−sqrt(2)입니다. 에너지 앞계수는 3+2sqrt(2), 약 5.828427입니다.

같은 질량·강성에서 이 인증식의 상수는 다음과 같습니다.

| c (kg/s) | epsilon (1/s) | gamma (1/s) | 에너지 앞계수 |
|---|---|---|---|
| 1 | 0.5 | 0.552786 | 2.618034 |
| 2 | 1 | 0.585786 | 5.828427 |
| 10 | 5 | 0.194193 | 101.990195 |

보장값을 적은 행들이며 최적 감쇠 선택이나 모든 정확한 궤적의 속도 순서를 증명한 것은 아닙니다. 임계 감쇠에서 F=0,y0=1,v0=0이면 y=(1+t)exp(−t),v=−t exp(−t)입니다. 이를 대입하면 ODE 수치 적분 없이 정확한 에너지와 상계를 비교합니다.

### 알려진 힘 예산이 궤적 오차에 작용하게 하기

m,k,c가 같고 구동만 r(t)만큼 다른 두 정확한 흐름을 비교합니다. 위치 차이 e는 다음을 만족합니다.

$$
e=\widehat x-x,\qquad m\ddot e+c\dot e+ke=r(t),\qquad
\dot L_e=-D_e+r(\dot e+\varepsilon e).
$$

L_e,D_e에는 y,v 대신 e와 그 속도를 넣습니다. H_e는 J 단위의 궤적 오차 척도이며 실제 장치에 추가로 저장된 에너지가 아닙니다. 초기 상태 차이는 L_e(0)에 남습니다.

가중 Cauchy–Schwarz로 (e'+epsilon e)²을 D_e와 계수 C_r의 곱으로 감싼 뒤 Young 부등식으로 r과의 곱을 제어합니다.

$$
C_r=\frac1{c-\varepsilon m}+\frac\varepsilon k,
\quad (\dot e+\varepsilon e)^2\le C_rD_e,
\quad r(\dot e+\varepsilon e)\le\frac12D_e+\frac12C_rr^2,
\quad \dot L_e\le-\eta L_e+\frac12C_rr^2,
\quad \eta=\frac d{\lambda_{\max}}.
$$

계수 1과 epsilon을 가중치 c−epsilon m,epsilon k에 맞춰 제곱하면 역가중 합이 C_r입니다. C_r의 단위는 s/kg여서 C_r r²은 일률입니다. exp(eta t)를 곱해 적분하면 결정론적 입력 상계를 얻습니다.

$$
L_e(t)\le e^{-\eta t}L_e(0)+\frac{C_r}{2}
\int_0^t e^{-\eta(t-s)}r(s)^2\,ds.
$$

전체 구간에서 |r(s)|≤rho이면 지수 커널을 적분하고 오차 에너지에서 위치를 추정합니다.

$$
L_e(t)\le e^{-\eta t}L_e(0)
+\frac{C_r\rho^2}{2\eta}(1-e^{-\eta t}),\qquad
|e(t)|\le\sqrt{\frac{2L_e(t)}{k\lambda_{\min}}}.
$$

초기 과도 기여는 줄어들고 보장은 유한한 상계로 접근합니다. 시간에 따라 변하는 유계 입력의 오차가 반드시 상수로 수렴하는 것은 아닙니다. 전체 시간의 힘 예산이 정당화되어야 하며 힘 표본만으로는 부족합니다. 계산한 이산 궤적에서는 이산화 오차도 별도 기여입니다.

### 알려진 일정 오차 해와 비교하기

m=k=1,c=2,r=0.1 N이고 두 초기 상태가 모두 0이라고 합시다. 정확한 위치 오차는 0.1[1−(1+t)exp(−t)] m이며 0.1 m로 갑니다. 이 인증식에서는 eta=1/(2+sqrt(2)),C_r=2 s/kg이고 점근적 위치 상계는 약 0.341421 m입니다. 실제 오차와의 간격은 부등식 실패가 아니라 보수성의 크기입니다.

![임계 진동자의 정확한 에너지는 앞계수가 있는 지수 상계 아래에 있고, 일정한 힘 차이가 만드는 위치 오차는 보수적인 입력 상계 아래에서 유계입니다.](/assets/learning/classical-damped-certificate.ko.svg)

위쪽은 외력 없는 변위 진동자의 로그 에너지 비율 축입니다. 아래쪽은 별도의 일정 힘 차이 예제이며 위치 오차 m와 정확한 극한 0.1 m를 보여 줍니다. 시간 축은 모두 s입니다. 독립 계산이며 기존 실습은 선언한 물리 에너지·일·열·일률을 유지합니다.

### 대수적 상수와 정확한 예제 재현하기

코드는 양의 이차형식 상계·미분 추정·임계 닫힌 해를 확인합니다. 부동소수점 상태 표본은 모든 상태에 대한 대수 증명의 예시이며 표본만으로 증명한 것은 아닙니다.

```python
from math import sqrt, hypot, exp, expm1

def certificate(m,k,c):
    assert m>0 and k>0 and c>0
    epsilon = c/(2*m)
    s11 = 1+epsilon*c/k
    off = epsilon*sqrt(m/k)
    largest = (s11+1+hypot(s11-1,2*off))/2
    determinant = 1+epsilon*(c-epsilon*m)/k
    smallest = determinant/largest  # avoids subtracting close eigenvalues
    d = min(epsilon,c/m-epsilon)
    gamma, eta = 2*d/largest,d/largest
    Cr = 1/(c-epsilon*m)+epsilon/k
    return epsilon,smallest,largest,gamma,eta,Cr

for m,k,c in [(1.,1.,1.),(1.,1.,2.),(1.,1.,10.),(4.,3.,2.)]:
    epsilon,lo,hi,gamma,eta,Cr = certificate(m,k,c)
    for y,v in [(1.,0.),(.3,-.5),(-2.,3.)]:
        H = (m*v*v+k*y*y)/2
        L = H+epsilon*m*y*v+epsilon*c*y*y/2
        D = (c-epsilon*m)*v*v+epsilon*k*y*y
        assert lo*H-1e-12 <= L <= hi*H+1e-12
        assert D+1e-12 >= gamma*L
        for r in [-.2,0.,.3]:
            Ldot = -D+r*(v+epsilon*y)
            assert Ldot <= -eta*L+Cr*r*r/2+1e-12
    print("m,k,c,gamma,prefactor:",m,k,c,
          round(gamma,6),round(hi/lo,6))

# Independent critical closed solutions; m=k=1,c=2.
epsilon,lo,hi,gamma,eta,Cr = certificate(1.,1.,2.)
rho = .1  # constant force mismatch in N, identical zero initial states
for t in [0.,.1,.5,1.,2.,5.,10.]:
    y,v = (1+t)*exp(-t),-t*exp(-t)  # unforced, y0=1,v0=0
    H = (y*y+v*v)/2
    L = H+y*v+y*y
    assert L <= 1.5*exp(-gamma*t)+1e-12
    assert H <= (hi/lo)*exp(-gamma*t)*.5+1e-12
    error = rho*(1-(1+t)*exp(-t))
    error_v = rho*t*exp(-t)
    Le = (error_v**2+error**2)/2+error*error_v+error**2
    bound = Cr*rho*rho/(2*eta)*(-expm1(-eta*t))
    assert Le <= bound+1e-12
    assert abs(error) <= sqrt(2*bound/lo)+1e-12
print("exact position-error limit:",rho)
print("certified position-error limsup:",sqrt(Cr*rho*rho/(eta*lo)))
```

### 풀이가 있는 연구 질문

1. **H'≤0만으로 표시한 지수율을 얻지 못하는 이유는?** y=1,v=0이면 H=k/2>0인데 H'=0입니다. H에 음의 상수를 곱한 값은 이미 엄격히 음수입니다. 보조 L과 노름 비교가 빠진 변위 항을 제공합니다.
2. **c=0에도 같은 오차 상계 증명을 적용하나요?** epsilon은 0과 c/m 사이에 엄격히 있어야 하는데 감쇠 0에서는 그 구간이 비어 있습니다. 무감쇠 입력은 다르게 분석해야 하며 유계 공명 외력은 응답을 누적시킬 수 있습니다.
3. **학습한 힘 평가에서 무엇을 보고해야 하나요?** 공통 계수·초기 상태 차이·전체 구간 힘 예산·노름/앞계수/감쇠율을 명시하고 물리 저장 에너지와 오차 척도를 구분하세요. 에너지 수지와 궤적 예측도 함께 확인해야 합니다.

선형 식과 양의 이차형식에서 직접 유도한 인증식입니다. 열 법칙을 바꾸거나 감쇠 계수의 실험적 타당성을 주장하지 않으면서 명시적인 충분 감쇠율을 제공합니다.

## 풀이가 있는 여섯 연습

1. **연습: 무감쇠 극한.** m=k=1,c=0,F=0,x0=1,v0=0이면? **풀이:** x=cos t, v=−sin t이며 모든 시간에 E=1/2J, Q=Wdrive=0입니다. 위치가 바뀌어도 역학적 에너지는 보존됩니다.
2. **연습: 근 분류.** m=k=1에서 c=1,2,3을 비교하세요. **풀이:** 판별식은 −3,0,5로 부족·임계·과감쇠입니다. 임계에서는 중근 해를 사용합니다.
3. **연습: 임계 일정 구동.** m=k=1,c=2,F=1,x0=v0=0에서 1초 상태는? **풀이:** x=1−2/e, v=1/e입니다. 구동 일은 약 0.264241J이고 저장 에너지는 약 0.102579J입니다.
4. **연습: 열 직접 적분.** 앞 예제의 열을 구하세요. **풀이:** v=t exp(−t)이므로 Q(1)=2∫0¹t²exp(−2t)dt=(1−5exp(−2))/2≈0.161662J입니다. 저장 에너지에 더하면 구동 일과 같습니다.
5. **연습: 이미 평형인 상태.** F=k=1,x0=1,v0=0이고 감쇠는 임의이면? **풀이:** y0=v0=0이어서 x=1로 유지됩니다. 용수철 에너지 1/2J는 존재하지만 구동 일·열·일률은 0입니다. 비영 구동이 항상 에너지를 계속 넣는 것은 아닙니다.
6. **연습: 제동하는 입력.** m=k=c=1,F=−1,x0=0,v0=2에서 0.1초를 보세요. **풀이:** 아직 속도가 양수여서 구동 일률과 누적 일은 음수입니다. cv²는 양수라 열은 증가합니다. 역학적 에너지는 구동이 제거한 에너지와 생성된 열의 합만큼 줄어듭니다.

## 에너지 검사로 인증할 수 있는 범위

학습 모형이 관측되지 않은 소산 항을 잔차로 정의하면 궤적 오류를 숨길 수 있습니다. 선언한 음이 아닌 열 발생률을 독립 적분하는 편이 더 강한 검사입니다. 그래도 수지가 맞는다는 사실만으로 마찰이나 구동 법칙을 식별할 수는 없습니다. 가정을 기록한 뒤 관측 궤적, 힘, 에너지 예산을 함께 비교하세요. 실제 열 모형에는 누적 Q 외에도 상태 변수와 구성 관계가 필요합니다.

## 배경과 범위

특성근에 따른 해의 배경은 [MIT OCW의 감쇠 조화 진동자](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/damped-harmonic-oscillators/)를 참고하세요. 일정 구동의 평형 이동, 부호 있는 일, 별도 열 적분은 선언한 모형에서 유도했습니다. 주기 공명과 통계적 열저장고는 별도 주제로 다룹니다.
