# Density matrix 집필·구현 계약

2026-11-08. spin·angular momentum 다음의 원래 주제다. 전체51영역509주제와 분량·주차 제한 없는 연구 깊이를 유지한다. 두 계산 가족과 일반 이론을 연결하며 유한 계산 범위를 전체 이론의 제한으로 바꾸지 않는다.

## 두 가족과 일반 이론

1. `finite-ensemble-density-gram-tomography`: d=1…4에서 최대 네 순수 상태의 확률적 혼합을 만든다. rho·Gram·rank·순도·특성다항식·완전한 행렬 성분 복원·측정 모멘트·Lüders 조건부/비선택 상태를 비교한다. 이 가족의 수치 기반을 먼저 구현한다.
2. `qubit-density-spectrum-entropy-dephasing`: 혼합 qubit의 Bloch ball·고유값·von Neumann entropy·기저 의존 coherence와 dephasing을 연결한다. 정확한 양성 조건을 확인하고 작은 고유값·log0·퇴화를 분리한다. 이 가족의 수치 기반과 한영 본문을 검증했으며 Worker·실제 읽기는 남는다.

한영 본문은 일반 밀도 연산자의 Hermiticity·양성·trace1·스펙트럼·convexity·극점·ensemble 비유일성을 유도한다. 단순한 진폭 평균이 왜 실패하는지, 대각 성분만으로 상태를 알 수 없는 이유, 같은 rho에서 달라질 수 있는 준비별 분산 분해를 손계산한다.

부분추적과 purification의 정의·Bell 상태의 축약 상태도 이론과 손계산으로 소개한다. 실제 복합계·Schmidt·entanglement 계산은 해당 원래 주제에서 더 확장한다. 일반 d의 entropy·정리·증명·실험과 연구 깊이는 qubit 계산만으로 완료 처리하지 않는다.

## 첫 가족의 정확 계약

49개 소유 정수는 차원·4개 비음 weight·4개의 복소 raw 벡터·비영 probe·관측량의 중심/간격 유리수다. 정수 절댓값은10⁶ 이하, 분모는 양수, 차원을 넘는 좌표는0이다. 점유한 성분만 비영 상태를 요구한다. weight0인 영벡터는 정규화 상태가 없지만 대응 weighted column은 정확히0이다.

p_i=w_i/Σw, N_i=z_i†z_i, rho=Σp_i z_i z_i†/N_i다. 평균 벡터를 정규화해 rho를 대신하지 않는다. 모든 principal minor를 정확히 계산하고 양성인 최대 principal minor의 크기로 rank를 확인한다. 순도 Tr(rho²)·비정규화 linear entropy1−Tr(rho²)와 최소 순도1/d를 구별한다.

B의 열은 √p_i ψ_i이며 rho=BB†, Gram=B†B다. Gram의 principal minor는 raw overlap minor와 양의 행/열 척도의 곱으로 정확 유리수가 된다. rho와 Gram의 nonzero spectrum·rank·순도가 같고 d<4에서는 Gram 특성다항식에 영 고유값의 인자가 추가된다. 일반 고유값을 실제로 대각화했다고 표시하지 않는다.

기준 기저 확률과 각 i<j의 (e_i±e_j)/√2, (e_i±i e_j)/√2 측정에서 모든 실수/허수 성분을 복원한다. 두 성분의 부분공간 밖 확률도 포함하므로 d>2에서 두 결과만을 전체 확률로 잘못 읽지 않는다.

## 측정과 수치 경계

A=aI+b(2P−I), P=vv†/(v†v)를 사용한다. 같은 고유값은 먼저 합치고 d=1에는 rank0 결과를 만들지 않는다. 조건부 상태 PρP/p는 p>0에서만 정의되며 퇴화한 결과의 조건부 상태는 혼합일 수 있다. 비선택 측정은 지정한 Lüders instrument에 따른다. 모든 quantum operation의 보편적 갱신 규칙으로 확장하지 않는다.

전체 분산은 준비 내부의 분산 평균과 준비별 평균의 분산의 합이다. 각 항은 ensemble 의존적이며 전체 rho만으로 원래 label 기록을 복원할 수 없다. squared Hilbert-Schmidt 차이·coherence 제곱은 trace distance나 측정 성공 확률이 아니다.

rho·Gram 불변량·tomography·측정 계정은 정확 복소 유리수다. 정규화 진폭과 B/Gram 그림만 제곱근 방향성 구간을 사용한다.768bit 작업·512bit 보고, max(1,|참조|)×10⁻¹²⁵의 폭 기준을 유지한다. 미해결 값을0으로 바꾸거나 출력 norm을 보정하지 않는다.

## 다음 순서

첫 수치 가족 → 스펙트럼/엔트로피 가족 → 상세 한영 본문 → 입력/Worker → 고정 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 교재 등록·실제 읽기·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 교재 카드를 넣지 않는다.

[Preskill Chapter2 §2.3·2.5](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), [IBM의 도입](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/introduction)과 [convex combinations](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/convex-combinations)을 열어 density·ensemble·축약 상태의 배경을 확인했다. 유한 입력·정확 불변량·tomography·조건부 계산·검산 계약은 이 교재의 구현 설계다.

## 첫 수치 가족의 진행

2026-11-08 진행: [density matrix의 유한 ensemble 수치 기반](quantum-density-ensemble-foundation-status.md)을 구현했다.1…4차원에서 rho·Gram·순도/rank·특성다항식·tomography·Lüders 조건부 상태·분산 분해를 비교하고475검사·타입·lint·독립43참조(39유일 입력)를 확인했다. 다음은 스펙트럼/엔트로피 가족 → 상세 한영 본문 → Worker/교수 화면 → 실제 읽기다. 아직 미등록이므로558단원·304/362입문 검증·58미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 두 수치 가족의 진행

2026-11-08 진행: [density matrix의 두 수치 가족](quantum-density-spectrum-foundation-status.md)을 검증했다. 새 혼합 qubit 가족은 안정적인 고유값·entropy·기저 coherence·부호 있는 dephasing·Kraus/Choi·거리/fidelity를 독립38사례/37유일 입력·221표본으로 대조했다.527검사·타입·lint·참조 재생성을 확인했고, 다음은 두 가족의 상세 한영 본문 → Worker/교수 화면 → 실제 읽기다. 아직 미등록이므로558단원·304/362입문 검증·58미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 상세 한영 원고의 진행

2026-11-08 진행: [density matrix의 두 가족과 상세 한영 원고](quantum-density-foundation-status.md)를 검증했다.4편·56절·48개 표시 수식·8개 표·32개 해설 문제에서 일반 density 조건·Gram·tomography·entropy·부분추적·purification·dephasing을 유도하고 손계산했다.543검사·타입·lint를 통과했으며, 다음은 Worker → 교수 화면 → 실제 읽기다. 아직 미등록이므로558단원·304/362입문 검증·58미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## Worker의 진행

2026-11-08 진행: [density matrix의 Worker 계약과 실제 브라우저 검증](quantum-density-worker-status.md)을 마쳤다. 새97개를 포함한640검사·타입·lint, 실제 Worker81입력·78175수치 관측과 취소·교체·오류/재시도·종료를 확인했다. 다음은 교수 화면 → Interactive Book 실제 읽기·생산 자산 검증이다. 아직 미등록이므로558단원·304/362입문 검증·58미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 교수 화면의 진행

2026-11-08 진행: [density matrix의 개념·실험·시각화 화면](quantum-density-interface-status.md)을 검증했다.32개 예제·개념/예측/안내 각6개와 전체 입력·저장·행렬/곡선·혼합 Bloch ball을 준비했다.649검사·타입·lint, 실제16조합/접근성16회·3D8회·WebGL 복구4회·키보드/터치·모션·Worker350개 종료를 확인했다. 다음은 Interactive Book reader 등록·생산 자산·실제 읽기 검증이다. 아직 미등록이므로558단원·304/362입문 검증·58미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 실제 교재 등록과 다음 주제

2026-11-08 최신: [density matrix 두 단원의 실제 교재 검증](quantum-density-integration-status.md)을 마쳐560단원·1680안내, 원래305/362입문 검증·57미완료다. 전체10728검사와 표지 수정 후14집중 검사·타입·lint·콘텐츠·빌드·1798 HTML, 실제 읽기/접근성16조합·3D16회·WebGL 복구4회·키보드/터치와 생산 Worker81입력·78175수치 관측을 확인했다. 다음은 uncertainty·measurement다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다. 아래는 이전 기록이다.
