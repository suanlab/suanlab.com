# 유한 ensemble density: 유도와 독립 검산

2026-11-08. 첫 수치 가족 `finite-ensemble-density-gram-tomography`의 계약이다. 전체 density matrix 주제·연구 심화·실제 읽기의 완료 보고가 아니다.

## 정확 density와 양성

정규직교 C^d에서 양의 점유 성분의 raw norm N_i와 고전 확률 p_i로 rho=Σp_i z_i z_i†/N_i를 만든다. 임의 x에 대해 x†rho x=Σp_i|z_i†x|²/N_i≥0이고 trace는1이다. 영 weight·영 raw 벡터에는 정규화 상태를 만들지 않는다. 그 weighted column은 정의에 따라0이다.

Hermitian positive semidefinite 행렬의 rank는 양수인 principal minor의 최대 크기다. d≤4에서 모든 minor를 확인하며 수치 tolerance로 rank를 줄이지 않는다. 순수 조건은 Tr(rho²)=1이며 점유한 모든 ray가 같은 조건과 일치한다. 최대 혼합의 순도1/d와 비정규화 linear entropy1−Tr(rho²)를 구별한다.

## Gram과 spectrum 불변량

B_i=√(p_i/N_i)z_i, S_ij=z_i†z_j, T_ii=p_i/N_i다. 영 weight에서는 T_ii=0으로 정의한다. rho=BB†와 G=B†B의 nonzero spectrum이 같다. G의 principal minor는 det(S_J)∏_{i∈J}T_ii이므로 제곱근 없이 계산한다. elementary symmetric sums에서 density와 Gram의 특성다항식을 얻는다. d<4이면 χ_G(λ)=λ^(4−d)χ_rho(λ)다.

Tr(rho²)=Σ_ij p_i p_j |⟨ψ_i,ψ_j⟩|²다. 두 점유 벡터가 같은 raw phase/scale을 가질 필요는 없다. 최대 혼합을 Z 또는 X 또는 Y의 직교 ensemble로 준비해도 rho와 state-only 측정은 같으며 원래 준비 label은 rho에 남지 않는다.

## Tomography와 분산 계정

기준 확률로 rho_ii를 얻는다. pair의 합 d_ij=rho_ii+rho_jj라 하면 X± 확률은 d_ij/2±Re rho_ij, Y± 확률은 d_ij/2∓Im rho_ij다. Y+의 상태는 (e_i+i e_j)/√2다. 따라서 Re=(pX+−pX−)/2, Im=(pY−−pY+)/2다. 나머지 부분공간의 확률1−d_ij도 기록한다.

A=aI+b(2P−I)에 대해 Tr(rho A), Tr(rho A²)를 직접 계산한다. 전체 분산은 Σp_i Var_i(A)+Σp_i(μ_i−μ)²다. rho=I/2의 Z 혼합과 X 혼합은 Z 측정의 전체 분산1이 같지만 내부/사이 분산은0/1과1/0으로 다르다.

고유공간의 Lüders 갱신은 PρP/Tr(Pρ)다. 확률0에는 null, 퇴화한 결과에는 혼합 조건부 상태도 허용한다. 고유값 합치기와 d=1의 단일 결과를 먼저 처리한다. pinching에서는 순도 감소가 ||rho−ΣPρP||HS²와 같으며 일반 channel의 entropy 증가 정리로 확대하지 않는다.

## 독립 참조와 정밀도

생산 코드는 유리수 rank-one 혼합·minor 기반 rank·성분별 tomography를 사용한다. 참조는 SymPy 복소 행렬·Gaussian rank·Berkowitz determinant/characteristic polynomial·A의 독립 eigenspace에서 만든 V(V†V)^−1V†와 실제 pair projectors의 trace를 사용한다. Gram의 정확 불변량은 rational AB/BA companion T S에서 별도로 계산한다. 제곱근 좌표는 mpmath250dps·210자리로 대조하며 생산 코드를 import하지 않는다.

코드의 Laplace minor와 SymPy Berkowitz/Gaussian 연산을 구별한다. B/Gram 보고 구간의 폭 기준은 max(1,|참조|)×10⁻¹²⁵이며 참조 decimal의 끝자리 오차를 별도 허용한다. 정수 raw 입력의 범위·영 확률·가까운 rank 경계·큰 복소 좌표·퇴화 측정·위상/재척도 사례를 포함한다.

배경 정의는 [Preskill §2.3·2.5](https://www.preskill.caltech.edu/ph219/chap2_15.pdf)와 [IBM convex combinations](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/convex-combinations)에서 확인했다. 위의 유한 계약·기호 유도·직접 검산과 예제는 이 교재의 설계이며 아직 entropy·부분추적·실제 Worker/화면을 구현했다는 뜻이 아니다.
