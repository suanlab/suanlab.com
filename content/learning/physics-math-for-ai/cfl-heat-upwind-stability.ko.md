## 시간과 공간 격자를 함께 선택하기

PDE 수치 해법의 시간 step은 공간 간격과 분리해서 고를 수 없는 경우가 많습니다. 열 확산의 명시적 차분은 h²에 비례하는 dt를, 양의 속도 수송의 upwind는 h에 비례하는 dt를 요구합니다. 이 단원에서 CFL r은 모형별로 다른 정의를 갖습니다. 같은 숫자 r이라도 같은 물리 시간을 뜻하지 않습니다.

주기 [0,1)의 N개 점에 cos(2πx)+ε cos(πNx)를 놓습니다. 두 번째 작은 성분은 격자가 표현하는 최고 주파수입니다. 불안정한 모드가 실제로 관측되도록 의도적으로 넣은 초기 성분이지 매 step 추가하는 잡음이 아닙니다. 모든 상태는 이전 배열에서 동시에 갱신합니다.

## 두 Stencil과 두 안정성 조건

$$
u_i^{n+1}=u_i^n+r(u_{i-1}^n-2u_i^n+u_{i+1}^n),\quad r=\nu\Delta t/h^2
$$

는 열 방정식 u_t=νu_xx의 명시적 중심 차분입니다. Fourier multiplier는 1-4r sin²(θ/2)이므로 모든 모드가 비증폭이려면 0≤r≤1/2입니다. 여기서 식의 u와 확산계수 ν를 구별하세요.

$$
u_i^{n+1}=u_i^n-r(u_i^n-u_{i-1}^n),\quad r=c\Delta t/h
$$

는 c>0인 수송 u_t+cu_x=0의 upwind입니다. Multiplier는 1-r+r exp(-iθ)이고 0≤r≤1에서 크기가 1 이하입니다. 음의 속도에는 반대 이웃을 사용해야 하며 현재 실험은 양의 c만 허용합니다.

## 실제 시간·보존·유한 관측

표의 현재 시간은 T=n dt입니다. r이나 N을 바꾸면 dt와 T도 달라지므로 최종 그림을 같은 물리 시간의 다른 실험으로 착각하지 마세요. 각 상태에서 주황 기준해는 현재의 동일한 T로 계산합니다. 열의 두 모드는 각각 exp(-νk²T)로 감쇠하고 수송의 두 모드는 x-cT로 이동합니다.

주기 경계에서 stencil의 이웃 교환은 전체 합에서 상쇄되어 h∑uᵢ를 보존합니다. 이 보존은 안정성을 보장하지 않습니다. 부호가 교대하는 큰 값도 합이 0일 수 있기 때문입니다. Nyquist multiplier와 최대 nodal 오차, 질량 변화를 함께 보세요.

## 손계산과 직접 풀기

열에서 r=0.75이면 Nyquist multiplier는 1-4r=-2입니다. 초기 크기 ε=0.001은 20 step 뒤 0.001·(-2)²⁰=1048.576이 됩니다. 물리 열 방정식은 오히려 고주파를 가장 빠르게 감쇠하므로 이 성장은 수치 현상입니다.

직접 풀기: upwind에서 r=1이면 한 step은 어떤 연산인가요? 풀이: uᵢ_next=uᵢ-(uᵢ-uᵢ₋₁)=uᵢ₋₁이므로 정확히 한 셀을 이동합니다. dt=h/c라 실제 이동 거리 cdt도 h입니다. 주기 샘플을 정수 셀만큼 이동하는 이 특별한 경우를 일반 r의 무확산 정확성으로 확대하면 안 됩니다.

## AI Simulator의 실패 진단

학습된 PDE rollout에서 폭발을 보았을 때 모델 오차, 시간 stencil 불안정, 미해상 입력과 경계 오류를 구분하세요. 질량만 맞는다고 해가 맞는 것도 아닙니다. CFL 조건은 현재 선형 1차원 stencil의 조건이며 모든 비선형 PDE solver에 동일한 숫자를 적용하지 않습니다. 다차원, 비균일 격자, 고차 reconstruction에는 별도의 증폭·에너지 분석이 필요합니다.

참고: [MIT 수치 PDE](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/), [Clawpack 유한체적 설명](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html).
