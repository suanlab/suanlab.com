# Preparation covariance: 유도와 독립 검산

2026-11-08. 첫 수치 가족 `finite-state-covariance-uncertainty`의 계약이다. 불확정성·측정 전체나 실제 교재 읽기의 완료 보고가 아니다.

## 정확한 상태와 임의 Hermitian 관측량

75개 소유 정수의 절댓값은10⁶ 이하이다.37개는 d=1…4·네 비음 weight와 복소 raw 벡터,34개는 두 Hermitian4×4 윗삼각의 실수/허수 성분과 양의 분모,4개는 실수 선형 결합 u,v의 분자/분모다. 차원 밖 좌표는0이며 점유한 벡터만 비영이어야 한다. 관측량은 무차원 수치 행렬로 제공한다.

rho=Σpᵢzᵢzᵢ†/Nᵢ이며, DA=A−Tr(rho A)I·DB도 같은 방식으로 정의한다. vA=Tr(rho DA²),vB=Tr(rho DB²),c=Tr(rho DA DB)=s+ik다. 여기서 s=Tr(rho{DA,DB})/2, k=Tr(rho[A,B])/(2i)다. G=[[vA,c],[c*,vB]]는 DA√rho·DB√rho의 Hilbert–Schmidt Gram이므로 양성이다.

따라서 vAvB≥s²+k²≥k²이며, 강한 slack은 detG, 약한 slack은 detG+s²다. 교환자 행렬 자체와 그 상태의 기대가0인 경우를 구별한다. [X,P]=iℏI는 유한 행렬에서 trace 때문에 성립할 수 없으므로 정준 x,p를 이 입력으로 구현했다고 주장하지 않는다.

## 분산 분해와 복소 투영

준비별 중심화 공분산의 평균과 준비별 평균의 공분산을 더하면 전체 G다. 사이 항은 실수이고 내부 항에는 교환자 기대가 포함된다. 같은 rho의 다른 ensemble에서 전체 G는 같지만 두 항은 다를 수 있다.

Xᵢ=√(pᵢ/Nᵢ) DA zᵢ, Yᵢ=√(pᵢ/Nᵢ) DB zᵢ의 열을 모으면 c=ΣXᵢ†Yᵢ다. vA>0일 때 최적 복소 계수는 c/vA, 잔차의 최소 norm 제곱은 detG/vA이며 ΣX†(Y−cX/vA)=0이다. vA=0이면 계수는 비유일하여 null이지만 최소 잔차 vB는 정의된다. Gram 포화와 Y가 영벡터 X의 배수라는 주장을 혼동하지 않는다.

실수 V=[[vA,s],[s,vB]]는 Var(uA+vB)=(u,v)V(u,v)ᵀ를 설명한다. 이 타원은 비가환 A,B의 공동 측정 법칙이 아니다. 각 입력의 C=uA+vB를 직접 곱해 같은 분산이 나오는지 검산한다.

## 작은 고유값·퇴화·정밀도

G와 V의2×2 스펙트럼만 계산한다. 임의 d의 A/B 고유값 계산기는 아니다. 작은 고유값은 det/high를 사용하고 정확한 det0은 먼저 분기한다. gap²=0이면 같은 고유공간을 합치며 임의의 축을 선택하지 않는다. 영행렬의 양의 rank0과 영 고유값의 중복도2는 다르다.

사영의 성분 x/√gap²는 부호(x)√(x²/gap²)로 계산하여 작은 gap으로 반올림한 수를 나누지 않는다. 표준화 공분산/교환자도 같은 방식이며 vAvB=0일 때 null이다. 정확 유리수·정확 영 판정·768bit 작업/512bit 방향성 보고를 유지한다. 음의 determinant를 잘라 복구하거나 norm을 재조정하지 않는다. 별도 helper 검사에서는 사용자 입력 범위와 구별하여10⁻³⁰⁰ 크기의 gap도 구성하고, 고유공간의 정규화된 방향이 바뀌지 않음을 확인한다.

## 독립 참조와 설명 사례

참조는 생산 코드를 import하지 않는다. SymPy의 직접 복소 행렬 trace·Gaussian rank와 mpmath250dps 일반 Hermitian eigensolver/고유벡터 사영을 사용한다. 표준편차·사영·가중 열은210자리 숫자로 대조하고 구간 폭은 max(1,|참조|)×10⁻¹²⁵ 이하로 확인한다.

Pauli X/Y·순수 Bloch(3/5,4/5,0)에서는 vA=16/25,vB=9/25,s=−12/25,k=0이다. 약한 하한은0이나 강한 하한144/625는 포화한다. 같은 쌍의 I/2는 detG=1이다. 순수 qutrit에서 A가 e₀↔e₁, B가 e₀↔e₂를 잇는 경우에도 detG=1이어서 순수 qubit의 포화를 일반 차원으로 확대할 수 없다.

[Trevisan §4.1](https://web.dm.unipi.it/trevisan/teaching/PhD/2022-qinfo/2022-Qinfo-notes.pdf)의 covariance/commutation 행렬을 확인했다. 그 문서의 Com=iTr(rho[A,B])/2는 여기의 k와 부호가 반대다. 제곱 하한은 같으며 이 구현은 상단 c=Tr(rho DA DB)=s+ik 관례로 일관되게 직접 검산한다. Gram·투영·손계산과 유한 입력 경계는 위의 독립 구성이다.
