# 보존형 이산화 구현 전 자체 유도

후속 상태: [두 단원의 홈페이지 통합 검증](continuum-conservative-authoring-status.md)을 마쳤다. 아래 내용은 자료·설계·기반 단계의 기록이며 현재 다음 순서는 연속체 전체 영역 종합 점검이다.

2026-11-06. [자료 검토](continuum-conservative-source-audit.md)를 바탕으로 한 자체 특수화다. 이 문서는 구현 전 유도 기록이다. 현재 [두 단원 계산·원고 기반](continuum-conservative-godunov-foundation.md)을 검증했고 화면 통합은 미완료다.

## 공유 면과 비균일 저장 체적

주기 ring의 양의 셀 체적 Vi,일정 단면적A,면 거리de>0를 사용한다. q는 kg/m³인 수동 농도다. 면e는 셀i에서j로 향한다. incidence 행렬B는 출발 셀에+1,도착 셀에−1을 가진다. 질량 행렬M=diag(Vi)로 두면 다음과 같다.

$$
M\dot q=-BF,\quad F_e=aAq_{\mathrm{up},e}+\gamma_e(q_i-q_j),\quad
\gamma_e=DA/d_e\ge0,\qquad \boldsymbol1^TB=0.
$$

따라서 실제 질량의 시간 변화는0이다. 공유 면이 내부에서 상쇄되기 때문에 임의의 면 플럭스에도 성립하지만,일관성·양성·안정성이 자동으로 따라오지는 않는다. 확산만 있으면 L=B diag(γ)Bᵀ가 대칭 비음이고 이차 노름은 다음을 만족한다.

$$
\frac{d}{dt}(\boldsymbol1^TMq)=0,\qquad
E=\tfrac12q^TMq,\qquad
\dot E=-\sum_e\gamma_e(q_i-q_j)^2.
$$

E의 단위는kg²/m³이며 에너지J가 아니다. 일정 면적의 주기 upwind 이류를 더하면−|a|A∑jump²/2의 수치 노름 손실이 더해진다. 실제 Euler 단계에서는 양의 이차 항도 남는다.

$$
\Delta E=\Delta t\,q^TMr+\frac{\Delta t^2}{2}r^TMr,\qquad r=\dot q.
$$

국소 양성의 충분 조건은 dt(|a|A+γleft+γright)/Vi≤1이다. 실제 조건을 계산해야 하며 비균일 격자에 평균 셀 크기 하나만 대입하지 않는다. 비음 갱신 계수·상수 보존·질량 가중 불변성에서 Jensen 수축을 별도로 확인한다. 모든 흐름·확산이0이면 자동 시간 척도의 역수가 미정이므로 임의의 유한dt로 채우지 않는다.

## 평균 체적을 쓰는 음성 대조

Vi 대신 Vref를 모든 셀에 사용하면 단순 합은 보존돼도 물리 질량은 바뀔 수 있다.

$$
\dot q_{\mathrm{wrong}}=-B F/V_{\mathrm{ref}},\qquad
\frac{dM_{\mathrm{physical}}}{dt}
=-\sum_{e:i\to j}\frac{V_i-V_j}{V_{\mathrm{ref}}}F_e.
$$

Vi=(1,2,1),γe=1,q=(1,0,0),a=0이면 F=(1,0,−1),BF=(2,−1,−1)이다. 실제 갱신률은(−2,1/2,1)이고 질량 변화는0이다. Vref=4/3의 잘못된 갱신률은(−3/2,3/4,3/4)이며 질량 변화율은3/4다. dt=1/10에서 올바른 이차 노름 변화는−69/400=−0.1725,잘못된 질량은1에서1.075로 변한다. 이 예제는 독립 유리수 점검의 후보이며 제품 검증을 대신하지 않는다.

학습한 면 플럭스에도B구조는 보존성을 부여하지만,주기 cycle의 모든 면에 같은 상수를 더하면BF는 같다. 셀 변화만으로 모든 순환 플럭스의 값을 식별할 수 있다는 주장도 피한다.

## Burgers의 Godunov 인터페이스

f(q)=q²/2다. qL≤qR이면 희박파의0속도 ray를 사용한다. 양수 쪽이면qL,음수 쪽이면qR,0을 가로지르면0이다. qL>qR이면 충격 속도s=(qL+qR)/2의 부호로 상류 추적값을 고른다. s=0에서는 동등한 보존 플럭스를 주는 어느 추적값을 사용할지 명시한다.

미보정 Roe의 qL=−1,qR=1은 평균 특성속도0이라 보존 플럭스1/2를 주지만,정확한 sonic 희박파의 Godunov 플럭스는0이다. 두 방법 모두 하나의 면 플럭스를 공유할 수 있으므로 질량 보존만으로 entropy 적합성이 증명되지는 않는다.

수학적 entropy η=q²/2,그 플럭스ψ=q³/3을 쓰면 충격파의 분포적 결손은 다음과 같다.

$$
[\psi]-s[\eta]=-\frac{(q_L-q_R)^3}{12}.
$$

압축 충격파에는 비양수,팽창 충격파에는 양수다. 이것은 정확한 파의 선택 조건이며 수치 한 단계 entropy 부등식과는 다른 주장이다. 정지 충격파의 수치 entropy 플럭스와 충분한 단계 조건은 아래 후속 유도에서 구분한다.

양의 주기 상태에 비보존형 qnew_i=qi−(dt/dx)qi(qi−qprev)를 적용하면 물리 적분은 다음만큼 변한다.

$$
\Delta M=-\frac{\Delta t}{2}\sum_i(q_i-q_{i-1})^2.
$$

같은 초기 자료의 보존형 공유 플럭스는 내부 합이 상쇄된다. 실제 충격 위치·경계 수지·독립 적분·격자 오차까지 대조한 후 새 단원의 교육 주장으로 사용한다.

## 구현 전 정확한 산술 점검

`.runtime/learning/conservative-preparation/derivation-sanity.py`에서 위 ring 예제를 기호·유리수로 대조했다. 실제 질량은1,평균 체적 대조의 질량은43/40,올바른 노름 변화는−69/400이었다. dt=1의 과도한 단계는 질량을 유지하지만 음의 농도와 증가한 노름을 만든다. 주기 upwind 제곱 항등식,충격 entropy 인수분해,양·음·정지·sonic·영 상태의Godunov 인터페이스7사례와 미보정Roe의 sonic반례도 확인했다.

이는 자체 유도 점검이며 제품 모델·체계적인 독립 참조·학생 원고·브라우저 검증을 대신하지 않는다. 이 점검 당시 남았던 수치 entropy 플럭스와 시간 이산 보정은 아래 후속 유도에서 대조한다.

## 국소 수치 entropy 플럭스와 시간 이산 보정

[Tadmor, Acta Numerica 2003](https://math.umd.edu/~tadmor/pubs/files/Tadmor_Acta2003.pdf)의 식3.8–3.11과7절을 검토한 뒤 Burgers의 이차 entropy에 직접 특수화했다. 실제 entropy 플럭스는ψ=q³/3이고,잠재함수는Ψ=qf−ψ=q³/6으로 서로 다르다. 면 상태a,b와 보존 플럭스F에 대해 다음과 같이 정의한다.

$$
\widehat\psi(a,b;F)=\frac{a+b}{2}F-\frac{a^3+b^3}{12},\qquad
D(a,b;F)=\frac{b^3-a^3}{6}-(b-a)F.
$$

셀 가중치Vi=Δxi,갱신률ri=(Fleft−Fright)/Vi인 스칼라 정규화에서 직접 대수로 다음 항등식을 얻는다.

$$
V_i\frac{d\eta_i}{dt}+\widehat\psi_{i+1/2}-\widehat\psi_{i-1/2}
=-\frac{D_{i+1/2}+D_{i-1/2}}{2},
$$
$$
V_i\Delta\eta_i+\Delta t(\widehat\psi_{i+1/2}-\widehat\psi_{i-1/2})
=-\frac{\Delta t}{2}(D_{i+1/2}+D_{i-1/2})
+\frac{\Delta t^2}{2}V_i r_i^2.
$$

따라서 공간의D≥0만으로 유한 전진Euler 단계의 부등식을 선언할 수 없다. 실제 양의 시간 보정을 함께 계산한다. dt Vi ri²≤Dleft+Dright는 이 수치 entropy 플럭스에서의 충분한 국소 조건이며,보편적 최적CFL 상수라고 부르지 않는다. 정지 충격파에서 위 플럭스는 선언한 수치 플럭스이며 유일한 물리적 점 추적값이라는 뜻이 아니다.

Godunov가 왼쪽 물리 플럭스a²/2를 선택하면D=(a−b)²(2a+b)/6,오른쪽b²/2를 선택하면D=−(a−b)²(a+2b)/6,sonic 희박파의0을 선택하면D=(b³−a³)/6이다. 각 선택 영역의 부호로D≥0을 확인할 수 있다. Rusanov의α=max(|a|,|b|)에서는D=(b−a)²[α/2−(b−a)/12]≥0이다.

미보정Roe의−1→1 면에서는D=−2/3이고반대1→−1 면에서는D=2/3이다. 두 셀 주기의 전체 합은0이지만 한 면은 국소 부등식을 어긴다. 질량뿐 아니라 전체 entropy 합만 검사해도 잘못된 국소 선택을 놓칠 수 있다.

`.runtime/learning/conservative-preparation/entropy-flux-sanity.py`에서 국소·전진Euler 항등식을 기호식으로 확인하고 Godunov49쌍·Rusanov49쌍을 유리수로 대조했다. 위Roe의 전역 상쇄 반례도 통과했다. 이는 구현 전 유도 점검이며 제품 모델·독립 기준·학생 원고·화면의 검증은 아직 아니다.
