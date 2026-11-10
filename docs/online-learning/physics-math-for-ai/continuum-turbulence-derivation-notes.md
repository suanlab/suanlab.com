# 난류·폐쇄 문제 구현 전 자체 유도

2026-11-06 후속 상태: [홈페이지 통합 검증](continuum-turbulence-authoring-status.md)을 마쳤다. 아래는 해당 단계 당시의 설계·검증 기록이다.

2026-11-06. [자료 검토](continuum-turbulence-source-audit.md)에서 출발한 자체 가족 설계다. 이후 [두 가족의 코드·독립 참조·한영 원고](continuum-turbulence-foundation.md)를 검증했다. 두 실습의 화면 검증은 아직 남았다. 아래는 구현 전 유도와 사전 점검 당시 기록이다.

## 발산0 앙상블과 실제 공분산

높이H,ξ=y/H∈[0,1],양의 streamwise 파수k를 사용한다. 균등 위상α를 평균하며 실제 구현의 유한 위상 수도 선언한다.φ=ξ²(1−ξ)²,χ=φ(2ξ−1),섭동 streamfunction ψ=aφ sin(kx+α)+bχ cos(kx+α)로 둔다. a,b의 단위는m²/s다.

$$
u=U(y)+\psi_y,\quad v=-\psi_x,\qquad
U=6U_b\xi(1-\xi)[1+c(2\xi-1)].
$$

벽에서φ=φ_y=χ=χ_y=0이므로 평균과 섭동 모두 정지 벽 조건을 만족한다. 미분을 실제y좌표로 수행하면 발산이0이다. 위상 평균으로 다음 공분산을 직접 얻는다.

$$
R_{uu}=\tfrac12(a^2\phi_y^2+b^2\chi_y^2),\quad
R_{vv}=\tfrac{k^2}2(a^2\phi^2+b^2\chi^2),\quad
R_{uv}=\tfrac{abk}2(\phi_y\chi-\chi_y\phi)
=-\frac{abk}{H}\phi^2.
$$

실제 속도 공분산은 비음 행렬이다. 작은 determinant를 두 큰 곱의 차로만 정의하지 않고 다음 Gram 인자도 대조한다.

$$
\det R=\frac{k^2}{4}(a^2\phi\phi_y+b^2\chi\chi_y)^2\ge0.
$$

순간장은 제조한 정상 강제장이다. p=0을 선택한다면 전체2차원 강제 벡터를 f=ρ(u·∇)u−μ∆u로 복원해야 한다. x평균 강제는 G=ρ(Ruv)_y−μU_yy이며,y방향 평균 강제도 일반적으로0이 아니다. x식 하나의 평형을 전체 외력 없는NS해로 표현하지 않는다.

총 평균 전단 응력은τ=μU_y−ρRuv이며G=−τ_y다. 벽에서U=0을 써서 실제 길이 적분으로 얻는 일 수지는 다음과 같다. 단면 폭·주기 길이를 곱할 때 실제 기하를 선언한다.

$$
\int_0^H GU\,dy=\int_0^H\mu U_y^2\,dy
-\int_0^H\rho R_{uv}U_y\,dy.
$$

평균 전단0이고Ruv≠0이면 유한한 scalar νt로Ruv=−νt U_y를 맞출 수 없다. 둘 다0이면 비는0/0으로 미정이다. 일정한 비음 후보νm을 사용할 때 후보Ruv,m=−νm U_y와 실제 공분산·강제·일 수지의 결손을 별도 계산한다. 같은U와 다른a,b는 다른응력과 일반적으로 다른 강제를 만든다. 이를 같은 전체 물리 문제의 비유일성으로 확대하지 않는다.

## 주기 Gaussian 필터와 두 Burgers 모드

주기L=2π/k,표준편차ℓ≥0를 가진 정규화 Gaussian의 주기 convolution을 사용한다. s=kℓ,gm=exp(−m²s²/2)다. u=A(t)sin(kx)+B(t)sin(2kx),A=A0exp(−νk²t),B=B0exp(−4νk²t)로 두면u_t=νu_xx다. Burgers식u_t+(u²/2)_x=νu_xx+f를 만족시키는 실제 제조 강제는f=u u_x다.

필터한 장은ū=g1 A sin(kx)+g2 B sin(2kx),잔여 응력은τ=overline(u²)−ū²다. 상수 폭·주기라서 미분과 필터가 교환된다. 필터 잔차의 필터값을0으로 놓는 Reynolds 규칙은Gaussian에 적용하지 않는다.

$$
\bar u_t+(\bar u^2/2)_x
=\nu\bar u_{xx}+\bar f-\tfrac12\tau_x,\qquad
\Pi=-\tfrac12\tau\bar u_x.
$$

여기서Π>0은 해상 에너지에서 잔여 쪽으로의 전달로 정의한다. 단위는m²/s³이며 실제 전력을 구하려면ρ와단면적,길이 적분을 포함해야 한다. 점성 소산과는 다른 항이다. Fourier 전개로 얻는τ의 상수·cos(kx)·cos(2kx)·cos(3kx)·cos(4kx)계수는 차례대로 다음과 같다.

$$
C_0=\tfrac12[A^2(1-g_1^2)+B^2(1-g_2^2)],\quad
C_1=ABg_1(1-g_2),\quad C_2=\tfrac12A^2(g_1^2-g_2),
$$
$$
C_3=AB(g_1g_2-g_3),\quad C_4=\tfrac12B^2(g_2^2-g_4),\qquad
\int_0^L\Pi\,dx=-\frac{LkA^2B}{4}(g_1^2-g_2^2).
$$

위상·진폭 부호에 따라 국소/전체 전달이 달라진다. 양의 Gaussian 커널의τ는 분산이므로 비음수지만Π까지 항상양수라는 뜻은 아니다. 각 방향의 제조 강제 일도 함께 적분해야 해상 에너지 수지가 맞는다. Burgers의1/2인자를NS응력 발산에 그대로 적용하지 않는다.

## 작은 폭에서의 양의 표현 검토

작은s에서1−exp(−s²)류는expm1과로그를 사용한다. 그러나 각Fourier 계수를 안정화해도 특정 관측점의 분산을 큰 항의 합/차로 계산하면 상쇄가 남는다. Gaussian 이동Z의 Hermite 전개는 다음 양의 급수를 제공한다.

$$
\tau(x)=\sum_{n\ge1}\frac{\ell^{2n}}{n!}
\left[\partial_x^n\bar u(x)\right]^2.
$$

유한 Fourier 장에서 이 항등식을 독립 convolution으로 대조하고,구현 전 절단 오차를 검증해야 한다. 단일sine의 정확한 극한은τ=A²[(1−e^(−s²))²sin²(kx)+(1−e^(−2s²))cos²(kx)]/2이므로 극값에서는4차,일반점에서는2차로 시작한다. 실제0폭과 작은 비영 폭을 구분한다. 정확한주기분수 좌표와 그 부동소수점 근사에 대한 삼각함수 관례도 선언해야 한다.

에너지 전달·분산·비선형 항·강제·점성·경계 주기 및 필터 폭을 각각 독립적으로 검사한다. 물리 필터 폭,수치 격자,aliasing을 같은 것으로 취급하지 않으며 작은 자기 잔차를 보편적 폐쇄나 실제 난류 예측 성능으로 포장하지 않는다.

## 구현 전 산술 점검

`.runtime/learning/turbulence-preparation/derivation-sanity.py`에서80자리로 위상8개의 실제 평균·공분산,전체2차원 강제의 평균과 실제y적분의 일 수지를 대조했다. Gaussian 부분은 정규분포의 실제 무한 구간 convolution을 두 점에서 계산해 Fourier분산과 맞추고,공간 전달 적분도 위 식과 대조했다. 평균 전단0·비영 공분산의 중앙점도 확인했다. 결과는 같은 폴더의`derivation-sanity.json`에 있다. 이는 유도 메모의 사전 점검이며 제품 모델·체계적인 독립 참조·학생 원고·브라우저 검증을 대신하지 않는다.

## 두 Fourier 모드의 유한 양의 분산 표현

후속 유도에서 무한 Hermite 급수 대신 정확한 유한 표현을 얻었다. θ=kx,t=(kℓ)²,z=exp(−t)로 두고 Gaussian 이동의 짝·홀 부분을 분리한다. cos와 sin의 공분산은 짝·홀 대칭으로0이다. 각2×2 공분산을 완전제곱으로 분해하면 다음을 얻는다.

$$
V_e=\tfrac12(1-z)^2,\quad r_e=\sqrt z(1+z)^2,\quad
D_e=\tfrac12(1-z)^4(1+z)^2(1+z+z^2),
$$
$$
V_o=\tfrac12(1-z^2),\quad r_o=\sqrt z(1+z^2),\quad
D_o=\tfrac12(1-z)^3(1+z)(1+z^2)(1+z+z^2),
$$
$$
\tau=V_e(A\sin\theta+r_eB\sin2\theta)^2+D_e(B\sin2\theta)^2
+V_o(A\cos\theta+r_oB\cos2\theta)^2+D_o(B\cos2\theta)^2.
$$

0≤z≤1이므로 네 계수는 비음수다. cos 두 모드의 두번째 조건부 분산이De,sin 두 모드의 두번째 조건부 분산이Do임을 기호 인수분해로 확인했다. z=1에서도 나눗셈 없이 각 항이 정확히0이 된다. 무한 급수 절단이나 음수 결과의 사후0처리가 필요 없다.

작은 양의 t에서1−z는expm1 또는 로그로 계산한다. re≈4,ro≈2라서 진폭 조합까지 상쇄될 수 있다. 따라서 조합을 기준값과 보정으로 분리한다. 다음 차이를 먼저 계산해야 작은 폭에서 남는ℓ6·ℓ8 항을 버리지 않는다.

$$
r_e-4=\operatorname{expm1}(-t/2)+2\operatorname{expm1}(-3t/2)+\operatorname{expm1}(-5t/2),
$$
$$
r_o-2=\operatorname{expm1}(-t/2)+\operatorname{expm1}(-5t/2).
$$

`.runtime/learning/turbulence-preparation/positive-gaussian-variance.py`에서 위 공분산 인수분해를 기호식으로 증명하고,80자리 실제Gaussian convolution4사례와 대조했다. 폭10^-6의 단일 모드 극값과 A=1,B=−1/2,θ=0의1차 기울기 상쇄를 포함한다. 이 결과는 구현 전 유도 검증이며 제품 모델과 극한 참조·화면 검증은 아직 없다.

## Burgers 필터 예제의 전력 예산을 먼저 고정하자

w=ū,e=w²/2로 쓰면 필터식에w를 곱한 실제 국소 예산은 다음과 같다. 전체 전력에서는ρ와단면적S,공간 길이 적분을 포함한다.

$$
e_t+\partial_x\left(\frac{w^3}{3}-\nu w w_x+\frac{w\tau}{2}\right)
=-\nu w_x^2+w\bar f-\Pi.
$$

현재 제조 가족은 각 모드가 열방정식으로 감쇠하므로 필터한 운동에너지와 그 미분은 다음과 같다.

$$
K_r=\frac{\rho SL}{4}(g_1^2A^2+g_2^2B^2),\qquad
\dot K_r=-\frac{\rho SL\nu k^2}{2}(g_1^2A^2+4g_2^2B^2)=-D_r.
$$

따라서 이 특수한 제조 가족에서는 실제 필터 강제 일이 전달과 정확히 상쇄한다. 주기 적분으로 F_r=ρS∫w fbar dx=ρS∫Π dx를 확인해야 한다. Π만 보고 해상 에너지 감쇠율이라고 해석하지 않는다. 원래 강제는 ∫u f dx=∫u²u_x dx=0이지만, 필터한 강제와 필터 속도의 곱의 적분은 일반적으로0이 아니다.

잔여 에너지는 ρS∫τ/2 dx=ρSL[A²(1−g1²)+B²(1−g2²)]/4다. 해당 강제 공분산의 일은−F_r이고, 잔여 점성 손실은 전체 분자 손실과 Dr의 차다. 전체·해상·잔여 예산을 같은 부호 관례로 확인할 수 있다. 국소 플럭스·실제 공간 적분·독립 미분으로 위 식을 검증하는 단계는 아직 남았다.

상수 속도 추가에 따른 응력 불변성과 전체 동역학의 Galilean 변환을 구분한다. 고정 시각의 분산은 상수 추가에 불변이지만, 같은 x에서 속도에 상수만 더하면 필요 외력도 달라질 수 있다. 좌표 이동까지 포함한 비교를 제공할 때는 시간 미분·이송 위상·변환 외력을 함께 선언해야 한다.
