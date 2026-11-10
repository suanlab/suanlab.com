# POVM·instrument·backaction: 유도와 독립 검산

2026-11-08. `qubit-povm-instrument-backaction`의 수치 계약이다. 두 상태 불확정성/측정 가족의 한영 본문·Worker·reader 등록이나 연구 심화 완료 보고가 아니다.

## 입력과 하나의 POVM

22개 소유 정수의 절댓값은10⁶ 이하이다. 유리수 Bloch 상태4개, A/B 방향의 비영 복소 raw spinor 각4개, 강도 κ의 분자/양의 분모2개, 결과별 비영 quaternion 각4개다. 정확한 Bloch ball과0≤κ≤1을 검사하며 잘라 넣지 않는다. 관측량과 pointer 값은 무차원이다.

정규화 spinor의 사영에서 유리수 단위 방향 n,b를 만들고 A=n·σ,B=b·σ라 둔다. 효과는 E_s=(I+sκA)/2, s=±1이며 p_s=(1+sκm)/2,m=n·r다. 효과의 determinant는(1−κ²)/4이고 합은I다.0<κ<1에서는 날카로운 사영 측정이 아니다. 이 모형은 unbiased binary qubit 효과 가족이며 모든 POVM을 입력받는 기능은 아니다.

## 같은 효과의 세 instrument

h=√(1−κ²), r⊥=r−(n·r)n라 쓰자. 양의 제곱근 M_s=√E_s를 사용한 Lüders 갱신의 비정규화 상태는 τ_s=(p_s I+w_s·σ)/2이며 w_s=[h r⊥+(m+sκ)n]/2다. κ=0에는 상태를 바꾸지 않고 독립적인 결과를 기록하며, κ=1에는 사영 갱신이 된다.

피드백 갱신은 K_s=U_s M_s다. U_s=(aI−i v·σ)/√(a²+|v|²)의 유리수 Bloch 회전 R_s를 직접 구성한다. 효과 K_s†K_s=E_s는 같고 w_s만 R_s w_s로 바뀐다. 재준비 갱신은 J_s(ρ)=p_sP_s,P_s=(I+sA)/2다. 실제 Kraus 분해는 입력 A 고유방향별 K_sr=√((1+srκ)/2)|n_s⟩⟨n_r|를 사용한다.

세 갱신의 조건부 상태는 p_s>0에서만 정의한다. 확률0에는 null이며, 상태를0행렬로 정규화하지 않는다. 비선택 상태는 모든 τ_s의 합이다. 후속 sharp B의 joint law는 Tr(P_t^B τ_s)다. 이것은 기록 후 후속 측정의 확률이며 초기 sharp A/B의 공동 분포가 아니다.

기록 s와 이후 sharp A 결과가 다른 확률도 계산한다. 이 의미의 준비/반복성과 같은 unsharp POVM 자체를 반복했을 때의 확률은 다르다. 재준비는 모든 κ에서 sharp A의 해당 고유상태를 만들지만 unsharp 장치의 재측정 결과가 반드시 같다는 뜻은 아니다.

## 읽기값 분산·잡음·편향·오차

Pointer 값을 x_s=t s로 지정하면 O1=tκA,O2=t²I다. 출력 평균은 tκm, 분산은 t²−t²κ²m², intrinsic POVM noise는 O2−O1²=t²(1−κ²)I다. 상태의 준비 분산1−m²와 구별한다.

이 장치의 RMS 오차는 명시한 isometry V=Σ_sα K_sα⊗|s,α⟩와 pointer로 정의한다. ε²=ΣTrρ(K_sα(x_s−A))†K_sα(x_s−A)=t²+1−2tκ다. 이는 intrinsic noise와 편향 연산자(O1−A)²의 기여의 합이다. Raw t=1에서는 ε²=2(1−κ), calibrated t=1/κ에서는 ε²=1/κ²−1이다. κ=0에는 calibrated 기록을 정의하지 않아 null로 둔다.

## RMS 교란과 상태 거리

η²(B)=ΣTrρ(BK_sα−K_sαB)†(BK_sα−K_sαB)다. Dual 관측량 D*(B)=cI+d·σ를 계산하면 η²=2−2d·b−2c(r·b)다. Lüders에서는 η²=2(1−h)(1−(n·b)²), 재준비에서는2(1−κ(n·b)²)다. 피드백은 실제 U_s에 따라 달라진다.

별도로 trace distance=|rout−r|/2, 후속 B 분포의 total variation=|(rout−r)·b|/2와 제곱 수축 계정을 계산한다. 선택한 I/2가 그대로여도 sharp Z 갱신의 η(X)=√2다. 상태 거리0은 관측량의 RMS 교란0과 같지 않다.

C=|Trρ[A,B]|/2에 대해 준비 Robertson 하한과 εη+εσB+σAη≥C를 따로 보고한다. 후자는 위 isometry/pointer의 Ozawa RMS 정의에 대한 관계다. 다른 운용적 오차 정의와 교환하지 않는다. ε²η²−C²는 정확한 수체에서 판정하므로 단순 곱 하한의 실패와 준비 불확정성 위반을 혼동하지 않는다. 보정 관계의 수치 slack은 수정하지 않은 구간이며 미해결 상태도 허용한다.

## 정확 수체·채널·정밀도

상태·조건부 확률·dual 관측량·제곱 교란·Choi는 Q(h)의 a+b h로 계산한다. h가 유리수이면 접어 넣고, 제곱 비교로 정확한 부호를 판정한다. 반대 부호가 상쇄할 때는(a²−b²h²)/(a−bh)로 먼저 유리화한다. 출력용768bit 작업·512bit 경계와 중첩 제곱근을 계산하며 미해결 관측을0으로 바꾸지 않는다.

Affine channel과 Kraus 작용을 대조한다. Choi는 output⊗input·인덱스2*out+in의 unnormalized trace2 행렬이며 출력 부분추적이I다. 정확 수체 Gaussian rank와 실제 제공한 Kraus 개수·최소 개수를 구별한다. Complete positivity의 근거는 명시한 Kraus 구성이다. 임의 공동 입력이나 일반 channel 입력 기능으로 표시하지 않는다.

## 독립 참조

참조는 생산 코드를 import하지 않는다. SymPy의 효과 고유공간에서 양의 행렬 제곱근을 만들고 실제 KρK†·K†BK·commutator norm·basis matrix 작용에서 상태·RMS·Choi를 계산한다. 피드백은3차원 회전식 대신 SU(2) 행렬을 직접 곱한다. 재준비에는 다른 Kraus 분해 |n_s⟩⟨j|√E_s를 사용해 같은 channel과 RMS를 검산한다.

정확 수체 계수와 부호를 대조하며 복소 Kraus·제곱근·Ozawa 합은250dps·210자리 참조를 사용한다. 방향성 구간 폭은 max(1,|참조|)×10⁻¹²⁵를 유지한다. 복소 효과는 고유공간 계산 전에 정확히 정리하고, SymPy 부울은 명시적으로 JSON 부울로 바꾼다. 계산 실패를 참조 사례 삭제나 정밀도 완화로 해결하지 않는다.

배경은 [IBM의 POVM 정의](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/formulations-of-measurements), [Preskill §3.2.4](https://www.preskill.caltech.edu/ph229/notes/chap3.pdf), [Ozawa의 measurement-operator formalism·식80](https://arxiv.org/abs/quant-ph/0510083)에서 확인했다. 위 유한 입력·세 비교 갱신·정확 수체·검산 구성은 이 교재의 설계다.
