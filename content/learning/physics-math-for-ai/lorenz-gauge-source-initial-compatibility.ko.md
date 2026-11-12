## 전위의 파동식을 맞추면 충분할까

네 개의 전위 성분이 각각 파동식을 만족하면 Maxwell 방정식도 모두 만족할까요? Lorenz 표현을 선택하고 초기·경계 조건까지 맞추었다면 유용한 방법입니다. 하지만 전위 파동식만 검사하면 물리적으로 맞지 않는 장이 남을 수 있습니다. 이 단원에서는 스칼라 파동식·벡터 파동식·원천 보존이 모두 맞으면서도 지정 원천에 대한 Gauss 법칙을 실패하는 구체적인 예제를 만듭니다.

## 기호와 국소 가정을 고정한다

매끄러운 V·A, 일정 양의 ε·μ, c²=1/(εμ)를 사용합니다. 진공 또는 이와 같은 선형 매질의 국소 모형입니다. 앞 단원의 ξ=x/L,τ=ct/L 정규화를 따르며 무차원 a·v와 실제 SI 전위를 구분합니다. 이 단원의 종방향 예제는 전위의 제약을 읽기 위한 것이며, 비영 종방향 전기장을 자유 공간의 횡파라고 부르지 않습니다.

## G가 남기는 물리 잔차

□는 Laplacian에서 시간 이차 미분/c²를 뺀 연산자이고 G는 Lorenz 조합입니다. 전위로부터 만든 E/B를 Maxwell 식에 대입하면 아래 관계가 나옵니다. 특히 분리된 두 파동식을 만족할 때도 −∂tG와 ∇G가 남습니다. Faraday와 발산 B는 전위 항등식이지만 전하·전류를 포함하는 두 법칙에는 추가 조건이 필요합니다. 두 잔차의 단위도 서로 다릅니다.

$$
\Box=\Delta-c^{-2}\partial_t^2,\quad G=\nabla\cdot\mathbf A+c^{-2}\partial_tV,\quad \nabla\cdot\mathbf E-\rho/\epsilon=-\Box V-\rho/\epsilon-\partial_tG,\quad \nabla\times\mathbf B-c^{-2}\partial_t\mathbf E-\mu\mathbf J=-\Box\mathbf A-\mu\mathbf J+\nabla G.
$$

## 원천 보존이 G의 파동식을 만든다

분리된 전위 파동식의 발산·시간 미분을 더하면 G의 식을 얻습니다. ρ와 J가 전하를 보존하면 G의 식은 동차입니다. 그러나 동차 파동식에는 비영 해도 많습니다. 따라서 원천 보존은 중요한 적합성 조건이지만 그것만으로 G=0을 선택하지 않습니다. 학습 모형에서 보존 손실항만 추가했다고 게이지의 초기 제약까지 자동 해결된 것은 아닙니다.

$$
\Box\mathbf A=-\mu\mathbf J,\quad \Box V=-\rho/\epsilon\quad\Longrightarrow\quad \Box G=-\mu(\nabla\cdot\mathbf J+\partial_t\rho).
$$

## 초기값은 두 개이며 경계도 필요하다

G는 시간에 이차인 파동식을 따릅니다. 적절한 유일성이 성립하는 초기·경계 문제에서 초기 G=0과 ∂tG=0, 그리고 호환되는 경계 조건을 함께 주어야 G=0을 유지할 수 있습니다. 한 관측점에서의0은 초기 함수 전체가0이라는 뜻이 아니고, 초기 함수가0이어도 시간 미분 함수는 비영일 수 있습니다. 아래 조건은 Lorenz 표현을 유지하기 위한 충분한 설정이지 모든 표현에서 G=0만 물리적으로 허용한다는 주장이 아닙니다.

$$
G(\mathbf x,0)=0,\qquad \partial_tG(\mathbf x,0)=0,\qquad \text{compatible boundary data}.
$$

## 종방향 조화 전위를 지정한다

이제 정규화 연산자 □̃=Δξ−∂ττ를 사용합니다. z 방향 a만 있고 v와 같은 위상을 가집니다. k와 ω는 비음수 무차원 값입니다. 위상은 kζ−ωτ입니다. k=0이면 공간 좌표 의존성이 없어 공간적으로 균일하고, ω=0이면 시간 의존성이 없어 정적입니다. 두 값이 모두0이면 공간과 시간에 모두 일정합니다. 이 모형을 항상 파동으로 해석할 필요는 없습니다. 두 진폭 v₀,a₀를 독립적으로 정해 누락된 제약이 어떤 차이를 만드는지 보겠습니다.

$$
v=v_0\cos\phi,\qquad \mathbf a=(0,0,a_0\cos\phi),\qquad \phi=k\zeta-\omega\tau,\qquad D=k^2-\omega^2.
$$

## 실제 E와 G의 계수를 비교한다

두 계수 Q와 C를 정의합니다. 전기장 계수는 kv₀−ωa₀이고 Lorenz 계수는 ωv₀−ka₀입니다. 비슷해 보여도 같은 식이 아닙니다. 자기장은0입니다. C=0이면 이 가족의 Lorenz 조건을 만족하고, Q=0이면 이 가족의 E가0입니다. 특정 위상에서 sin이0이라고 전체 가족의 E나 G가0이라고 판단해서는 안 됩니다.

$$
Q=kv_0-\omega a_0,\quad C=\omega v_0-ka_0,\quad \mathbf e=(0,0,Q\sin\phi),\quad \mathbf b=0,\quad \widetilde G=C\sin\phi.
$$

## 전위 파동식이 지정한 원천

각 전위에 □̃를 적용하면 −D가 곱해집니다. 따라서 파동식이 지정하는 무차원 원천은 아래와 같습니다. 이 원천은 전위 파동식으로 정의한 값이며, 실제 장이 Maxwell 법칙으로 요구하는 원천과 아직 비교하지 않았습니다. 연속성 잔차는 D C sinφ입니다. D=0이면 C가 비영이어도 이 원천은0이고 보존됩니다.

$$
\widetilde\rho_{\rm wave}=Dv_0\cos\phi,\quad \widetilde j_{{\rm wave},z}=Da_0\cos\phi,\quad \partial_{\tau}\widetilde\rho_{\rm wave}+\partial_{\zeta}\widetilde j_{{\rm wave},z}=DC\sin\phi.
$$

## 실제 장이 요구하는 원천

E와 B에서 다시 원천을 구하면 ρ̃field=kQcosφ, j̃field,z=ωQcosφ입니다. 이 둘은 서로 전하를 보존하지만 wave 원천과 다를 수 있습니다. 두 원천 집합의 차이가 바로 Gauss·Ampere 잔차입니다. 이 예제에서는 파동식으로 지정한 원천과 장으로 역산한 원천을 표의 별도 열로 놓고 비교하는 것이 좋습니다.

$$
\widetilde\rho_{\rm field}=kQ\cos\phi,\quad \widetilde j_{{\rm field},z}=\omega Q\cos\phi,\quad R_G=\omega C\cos\phi=-\partial_{\tau}\widetilde G,\quad R_{A,z}=kC\cos\phi=\partial_{\zeta}\widetilde G.
$$

## 공명인데 진공 해는 아닌 반례

k=ω=1,v₀=1,a₀=0이면 D=0입니다. 따라서 두 전위 파동식은 영원천에 대해 성립하고 그 원천은 보존됩니다. 하지만 Q=C=1이어서 E가 비영이고, 예를 들어 φ=0에서 E는0이어도 공간 미분과 Gauss 잔차는1입니다. 진공 Gauss·Ampere 법칙을 만족하지 않습니다. 한 점의 장 값0과 잔차0도 다르다는 것을 보여줍니다.

$$
k=\omega=v_0=1,\quad a_0=0:\quad D=0,\quad Q=C=1,\quad \widetilde\rho_{\rm wave}=\widetilde j_{{\rm wave},z}=0,\quad R_G=R_{A,z}=\cos\phi.
$$

## 비공명에서 호환되는 원천 예제

k=2,ω=1,v₀=1,a₀=1/2이면 C=0,Q=3/2,D=3입니다. 파동 원천과 장 원천이 모두 ρ̃=3cosφ,j̃z=1.5cosφ로 일치합니다. B가0이고 종방향 E가 비영이어도 이 명시된 전하·전류를 공급하면 국소 Maxwell 해가 될 수 있습니다. 이를 영원천 자유 복사라고 부르지 않고 강제된 종방향 모형으로 설명합니다.

$$
k=2,\ \omega=1,\ v_0=1,\ a_0=\tfrac12:\quad C=0,\quad Q=\tfrac32,\quad D=3,\quad \widetilde\rho=3\cos\phi,\quad \widetilde j_z=\tfrac32\cos\phi.
$$

## 초기 G만0인 전위 파동

V=0,az=a sin(kζ)sin(kτ)를 선택합니다. 벡터 전위의 자유 파동식은 성립하며 지정 원천은0입니다. 초기에는 a와 G가 모두0이지만 전기장은 시간 미분에서 나오므로 일반적으로0이 아닙니다. 이 초기 E가 진공 Gauss 조건을 실패할 수 있습니다. 전위를 초기화할 때 A의 값만 맞추고 시간 미분을 빠뜨리는 실수를 드러내는 예제입니다.

$$
v=0,\quad a_z=a\sin(k\zeta)\sin(k\tau),\quad e_z=-ak\sin(k\zeta)\cos(k\tau),\quad \widetilde G=ak\cos(k\zeta)\sin(k\tau).
$$

## 누락된 초기 시간 미분을 찾는다

초기 ∂τG̃=a k²cos(kζ)는 일반적으로 비영입니다. a=k=1,ζ=0이면 초기 G̃=0이지만 시간 미분은1이고 Gauss 잔차는−1입니다. 이 시점의 E 자체는0입니다. 호환되는 초기 전기장 제약을 확인하지 않으면 전위 값과 G 값이 모두0이라는 사실이 잘못된 안도감을 줄 수 있습니다. 경계에서도 외부에서 G를 주입하지 않도록 조건을 맞추어야 합니다.

$$
\widetilde G(\zeta,0)=0,\qquad \partial_{\tau}\widetilde G(\zeta,0)=ak^2\cos(k\zeta),\qquad R_G(\zeta,0)=-ak^2\cos(k\zeta).
$$

## 일정한 비영 G의 예외를 이해한다

v=γτ,a=0이면 e=b=0이고 G̃=γ입니다. 전위 파동식과 영원천 Maxwell 식은 모두 성립합니다. 따라서 G≠0이라고 모든 물리 해를 거부하면 이 유효한 표현을 잘못 분류합니다. 핵심은 ∂tG와 ∇G가 남기는 잔차입니다. Lorenz G=0은 표준적인 충분 조건이며 특정 표현을 고르는 선택입니다. 표현 조건과 물리 적합성을 독립 판정으로 기록하세요.

$$
v=\gamma\tau,\quad \mathbf a=0:\quad \mathbf e=\mathbf b=0,\quad \widetilde G=\gamma,\quad \partial_{\tau}\widetilde G=0,\quad \nabla_{\xi}\widetilde G=0.
$$

## 근접 공명과 작은 계수의 계산

D는 두 제곱을 뺄 수도 있지만 (k−ω)(k+ω)로 계산하면 근접 공명의 작은 차이를 더 잘 유지할 수 있습니다. Q와 C는 곱 두 개의 차이이므로 각 곱이 먼저 반올림되면 비영 차이를 잃을 수 있습니다. 보상 곱 계산과180자리 정확 입력 기준을 비교하되 입력 자체가 이미 같아진 상황은 복구할 수 없음을 밝힙니다. 표현 범위가 부족하면 미정으로 표시하며 허용 오차로0을 강제하지 않습니다.

$$
D=(k-\omega)(k+\omega),\qquad Q=kv_0-\omega a_0,\qquad C=\omega v_0-ka_0.
$$

## AI 검증을 여러 질문으로 나눈다

학습한 전위에는 전위 파동식, 원천 연속성, G, 초기 G·∂tG, 초기 장의 제약, 경계 조건, 실제 E/B의 Maxwell 잔차를 각각 기록하세요. 이들 값의0을 한 개 총점으로 합치면 작은 단위의 중요한 결함이 가려질 수 있습니다. 정규화한 잔차와 SI 관측을 함께 표시하고, 해석적 가족 항등식을 독립 수치 인증으로 부르지 않습니다. 배경 전위 식은 [Feynman II §18–6](https://www.feynmanlectures.caltech.edu/II_18.html), 초기 제약 예제는 위 식에서 직접 유도한 구성입니다.

## 연구 확장: 제약 에너지와 경계 유입

### 제약 값에서 전파 문제로

이미 정의한 정규화 좌표·퍼텐셜 척도에서 한 공간 좌표 zeta와 정규화 속도 1을 씁니다. g는 정규화 Lorenz 조합, s는 분리된 퍼텐셜 파동식이 지정하는 원천의 연속성 결함입니다. 이 식을 만족하는 매끄러운 퍼텐셜은 구동 제약 파동식을 따릅니다. g의 미분은 그 지정 원천에 대한 Gauss·종방향 Ampere 잔차입니다. 비영 s는 Maxwell 이론에서 허용되는 전하 생성이 아니라 일관되지 않은 원천 요청·수치 근사의 진단입니다.

$$
g=\partial_\zeta a_z+\partial_\tau v,\quad s=\partial_\tau\widetilde\rho+\partial_\zeta\widetilde j_z,\quad g_{\tau\tau}-g_{\zeta\zeta}=s,\quad R_G=-g_\tau,\quad R_A=g_\zeta.
$$

### 제약 에너지 유도하기

파동식에 g_tau를 곱해 공간 적분하고 공간 항을 한 번 부분 적분하면 경계 선속과 아래 에너지 항등식을 얻습니다. 이 양의 함수형은 표현 제약의 미분을 측정하며 joule 단위의 전기·자기장 에너지가 아닙니다. 두 성분은 정규화된 물리 방정식 결함의 제곱을 직접 측정합니다. 값이 작아도 공간적으로 일정한 g는 결정할 수 없습니다. 매끄러움·공간 영역·경계 자료도 명제의 일부입니다.

$$
\mathcal E_g=\tfrac12\int_0^\ell(g_\tau^2+g_\zeta^2)\,d\zeta,\qquad\frac{d\mathcal E_g}{d\tau}=\int_0^\ell g_\tau s\,d\zeta+[g_\tau g_\zeta]_0^\ell.
$$

### 어느 특성이 어느 경계에서 들어오는가

오른쪽 진행 함수 f(tau−zeta)에서는 w_minus만 비영입니다. 따라서 왼쪽의 w_minus와 오른쪽의 w_plus가 유입 자료이고 다른 두 값은 유출입니다. 경계 기여는 유입 진폭 제곱에서 유출 제곱을 빼고 4로 나눈 것입니다. 주기 경계에서는 상쇄됩니다. 시간적으로 호환되는 동차 Dirichlet g도 선속 0이며 유입 특성을 0으로 두면 기여가 양수가 되지 않습니다. 초기 자료가 0이어도 나중의 경계 입력이 제약 파동을 보낼 수 있습니다.

$$
w_\pm=g_\tau\pm g_\zeta,\qquad[g_\tau g_\zeta]_0^\ell=\tfrac14\left(w_+^2(\ell)+w_-^2(0)-w_-^2(\ell)-w_+^2(0)\right).
$$

### 원천의 전체 이력이 들어가는 상한

경계의 양의 기여가 없으면 Cauchy–Schwarz로 체적 항을 sqrt(2 energy)와 s의 L2 노름의 곱으로 제한합니다. 이를 적분하면 아래 상한입니다. 에너지가 0인 순간은 양의 정규화 뒤 극한으로 처리합니다. 순간적인 연속성 결함이 작다는 정보만으로 시간 적분 없이 긴 진화를 제한할 수 없습니다. 감쇠 항이 없으므로 오차 감소를 약속하지도 않습니다. 이산 해법은 대응하는 부분 적분 항등식과 안정한 경계 구현을 필요로 하며 연속체 상한만으로 인증되지 않습니다.

$$
\sqrt{\mathcal E_g(\tau)}\le\sqrt{\mathcal E_g(0)}+\frac1{\sqrt2}\int_0^\tau\|s(\cdot,u)\|_{L^2}\,du\qquad\text{if the boundary contribution is nonpositive}.
$$

### 공간 평균은 별도 자유도다

주기 구간의 평균은 구동 상미분방정식을 따릅니다. 주기 Poincare 부등식은 에너지로 평균을 뺀 g를 제한하지만 일정한 평균 자체는 보지 못합니다. 평균과 초기 평균 속도를 별도로 기록합니다. 원천 결함과 에너지가 0이면 g는 공간·시간 상수입니다. Lorenz 표현 g=0을 선택하려면 초기 상수도 0이어야 합니다. 물리 잔차는 g의 미분을 포함하므로 비영 상수 g도 유효한 Maxwell 장을 나타낼 수 있습니다. 표현 준수와 물리 호환성은 다른 판정입니다.

$$
\overline g=\ell^{-1}\int_0^\ell g\,d\zeta,\quad\overline g_{\tau\tau}=\overline s\ \text{(periodic)},\qquad\|g-\overline g\|_{L^2}\le\frac{\ell}{2\pi}\|g_\zeta\|_{L^2}.
$$

### 감쇠 없는 초기 오차는 지속된다

길이 2pi의 주기 구간과 양의 정수 k에서 초기 g는 0이지만 그 시간 미분은 일반적으로 비영인 정지파를 만듭니다. 제약 에너지는 일정합니다. 기존 초기 자료 실습은 g=a k cos(k zeta) sin(k tau)이므로 A=a k이고 에너지는 길이 곱하기 a 제곱·k 네제곱의 1/4입니다. 파동 방정식 손실과 지정 원천 보존이 성립해도 Maxwell 잔차가 지속될 수 있습니다. 공간 해상도를 높여도 호환되지 않는 연속체 초기 모드는 사라지지 않습니다.

$$
g=A\cos(k\zeta)\sin(k\tau),\quad\ell=2\pi,\quad k\in\mathbb N_{>0}:\qquad g(\zeta,0)=0,\quad g_\tau(\zeta,0)=Ak\cos(k\zeta),\quad\mathcal E_g=\ell A^2k^2/4.
$$

### 퍼텐셜에서 경계 유입 반례 구성하기

r_plus=max(r,0)로 둡니다. 아래 4차 퍼텐셜은 표시한 미분에 충분히 매끄럽고 자유 퍼텐셜 파동식을 풉니다. tau=0에서 구간의 두 퍼텐셜과 초기 속도는 모두 0이며 지정 원천과 그 결함도 0입니다. 그러나 왼쪽 경계가 파면을 주입하면 그 뒤에서 g=e_z가 비영이고 Gauss·Ampere 잔차는 −3(tau−zeta) 제곱입니다. 파면이 오른쪽을 통과하기 전 직접 적분한 에너지 증가는 전부 왼쪽 경계에서 옵니다. 이는 진공 장 제약과 호환되지 않는 퍼텐셜 파동 해이며 진공 Maxwell 복사파가 아닙니다.

$$
v=\tfrac14(\tau-\zeta)_+^4,\quad a_z=0,\quad g=e_z=(\tau-\zeta)_+^3,\quad s=0:\qquad\mathcal E_g=9\tau^5/5,\quad\dot{\mathcal E}_g=9\tau^4\quad(0\le\tau\le\ell).
$$

### 작은 균일 원천 결함의 누적

주기 구간에서 공간적으로 균일한 아래 스칼라 퍼텐셜을 택합니다. 전기·자기장은 0이지만 분리된 스칼라 식은 전하 delta 곱하기 tau와 전류 0을 지정합니다. 연속성 결함은 일정한 delta이고 Gauss 불일치는 선형, 제약은 이차적으로 커집니다. 이 예는 원천 이력 상한에 등호를 이룹니다. 표는 길이 1·delta=0.1로 원천 구동 예와 별개의 경계 파면 예를 시간 1까지 비교합니다. 둘은 물리 장 에너지가 아닌 제약 진단이며 같은 경계 문제의 두 해가 아닙니다.

$$
v=\delta\tau^3/6,\quad a_z=0,\quad\widetilde\rho_{\rm wave}=\delta\tau,\quad\widetilde j_{\rm wave}=0,\quad s=\delta,\quad g=\delta\tau^2/2,\quad R_G=-\delta\tau,\quad R_A=0,\quad\mathcal E_g=\ell\delta^2\tau^2/2.
$$

### 에너지 0도 Lorenz 상수를 정하지 못한다

마지막 정확한 가족은 시간에 선형인 스칼라 퍼텐셜·영장·일정한 비영 g를 가집니다. 제약 에너지와 두 Maxwell 잔차는 0입니다. 이를 물리적 실패로 거부하면 표현 선택과 장 방정식을 혼동합니다. 연구 검증은 g와 평균, 초기 제약 두 함수, 지정 원천 연속성의 시간 이력, 유입 경계 자료와 미분 잔차를 각각 보고해야 합니다. 이 보충 에너지 모형은 기존 처방 가족 실습에 원천 주입·경계 파동 제어를 추가하지 않습니다.

$$
v=\gamma\tau,\quad a_z=0:\qquad g=\gamma,\quad\mathcal E_g=0,\quad R_G=R_A=e_z=b_z=0.
$$

| tau | 균일 제약 에너지 | 균일 Gauss 결함 크기 | 파면 제약 에너지 | 파면 유입 |
| --- | --- | --- | --- | --- |
| 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 0.25000000 | 0.00031250 | 0.02500000 | 0.00175781 | 0.03515625 |
| 0.50000000 | 0.00125000 | 0.05000000 | 0.05625000 | 0.56250000 |
| 1.00000000 | 0.00500000 | 0.10000000 | 1.80000000 | 9.00000000 |

```python
from fractions import Fraction as F
length, delta = F(1), F(1, 10)
for tau in [F(0), F(1, 4), F(1, 2), F(1)]:
    uniform_energy = length*delta**2*tau**2/2
    gauss_defect = abs(delta*tau)
    # Incoming polynomial front; this formula requires 0 <= tau <= length.
    incoming_energy = 9*tau**5/5
    incoming_power = 9*tau**4
    values = [tau, uniform_energy, gauss_defect, incoming_energy, incoming_power]
    print(" ".join(f"{float(x):.8f}" for x in values))
```

[Feynman II 18–6절](https://www.feynmanlectures.caltech.edu/II_18.html)은 고전 퍼텐셜·게이지의 배경입니다. 여기의 유한 격자 exactness·투영과 1차원 제약 에너지 예제는 명시한 영역·경계 가정에서 직접 유도합니다.

![지속 초기 제약·유입 파면·누적 원천 오차](/assets/learning/electromagnetism-lorenz-constraint-energy.ko.svg)

### 해설이 있는 연구 질문

**지정 원천을 보존해도 영 초기 제약이 나중에 실패할 수 있는가?** **답:** 그렇다. 유입 경계 자료가 비영 제약 파동을 주입할 수 있다. 다항 파면은 명시적인 퍼텐셜 파동 예다.

**연결 시공간 영역에서 제약 에너지 0은 무엇을 보이는가?** **답:** g의 두 미분이 사라져 상수임을 보인다. 그 값은 초기 평균이나 표현 조건으로 별도 지정해야 한다.

**순간 연속성 손실이 작으면 장시간 결함도 작은가?** **답:** 아니다. 상한은 원천 노름의 시간 적분과 경계 유입에 의존한다. 균일 예에서는 작은 일정 결함도 누적된다.

## 풀이가 있는 연습 문제

### 연습 1

영원천이면 □G=0입니다. 이것만으로 G=0인가요?

**해설:** 아닙니다. 비영 초기·경계 데이터를 가진 동차 해가 있습니다. 적절한 초기 G·∂tG와 경계 조건을 함께 지정해야 Lorenz 표현의0을 유지합니다.

### 연습 2

k=ω=v₀=1,a₀=0,φ=0에서 전기장과 Gauss 잔차는?

**해설:** 정규화 전기장은0이지만 Gauss 잔차는1입니다. 전위 자유 파동식과 원천 보존만으로 진공 적합성은 충분하지 않습니다.

### 연습 3

k=2,ω=1,v₀=1에서 C=0이 되는 a₀와 원천은?

**해설:** a₀=1/2입니다. Q=3/2,D=3이므로 ρ̃=3cosφ,j̃z=1.5cosφ로 두 원천 계산이 일치합니다.

### 연습 4

정상파 반례에서 a=k=1,ζ=0,τ=0이면 초기 G·시간 미분은?

**해설:** G̃=0,∂τG̃=1입니다. Gauss 잔차는−1이므로 초기 G만 확인하는 검사는 실패를 놓칩니다.

### 연습 5

G̃=2가 일정하면 진공 Maxwell 식을 반드시 실패하나요?

**해설:** 아닙니다. v=2τ,a=0이면 E=B=0이며 ∂τG̃와 gradient G̃도0입니다. Lorenz 표현은 아니지만 물리적으로 유효합니다.

### 연습 6

L=1m,U=2V일 때 ρ̃=3의 실제 전하 밀도는?

**해설:** ρ=(εU/L²)ρ̃=6ε C/m³입니다. 매질의 ε를 사용하며 정규화 수3에 전하 단위를 직접 붙이지 않습니다.
