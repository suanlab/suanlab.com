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
