# Uncertainty·measurement 집필·구현 계약

2026-11-08. density matrix 다음의 원래 주제다. 전체51영역509주제와 분량·주차 제한 없는 연구 깊이를 유지한다. 현재560단원·1680안내·305/362입문 검증·57미완료이며 이 준비를 등록이나 검증 완료로 세지 않는다.

## 두 계산 가족

1. `finite-state-covariance-uncertainty`:1…4차원 혼합 상태와 임의의 두 Hermitian 수치 행렬을 입력한다. 상태 평균·분산·복소 공분산 Gram·Robertson/Schrödinger 하한·포화·분산 분해·복소 투영·실수 선형 결합을 비교한다. 임의 차원 이론과 증명을 본문에서 다루며 qubit 포화 현상을 일반화하지 않는다.
2. `qubit-povm-instrument-backaction`: 양의 효과와 결과 확률, 지정한 instrument의 조건부/비선택 상태를 구별하는 계산 가족을 준비한다. 같은 POVM의 서로 다른 상태 갱신·측정 강도·읽기값 noise·상태 교란을 명시적으로 비교한다. 22개 소유 유계 정수로 정한 Bloch 상태·Pauli 방향·강도·결과별 SU(2) 피드백을 사용한다. 세 instrument와 raw/보정 pointer의 구체적 의미는 [측정 수치 계약](quantum-povm-instrument-source-audit.md)에 고정했다.

## 일반 이론과 설명

정규화 상태에서 중심화한 관측량을 Hilbert–Schmidt 벡터로 옮겨 Cauchy–Schwarz와2×2 Gram의 양성에서 불확정성을 유도한다. 두 관측량의 분산은 동일하게 준비한 여러 계에서 각각 측정한 결과의 폭이다. 비가환 관측량의 공동 확률분포나 기기의 오차를 자동으로 제공하지 않는다.

공분산 항을 버린 약한 하한·교환자 기대가0인 비가환 사례·상수 관측량의 영 분산·순수 고차원 상태의 엄격한 부등식·표본 추정 오차와 상태 분산을 분리한다. 정준 x,p의 무한 차원 정의역과 Gaussian 최소 불확정 예를 해석적으로 다룬다. 유한 행렬에서 [X,P]=iℏI를 정확하게 구현했다고 표시하지 않는다. 시간-에너지·엔트로피·상태 의존 오차/교란 부등식은 각 가정과 별도 의미를 설명하고 연구 심화로 연결한다.

POVM은 효과의 양성·합I와 Born 확률을 지정한다. 결과 확률만으로 측정 뒤 상태는 정해지지 않는다. Kraus/instrument·조건부 정규화·확률0/null·비선택 channel·순차 측정의 의미를 별도로 유도한다. 일반 다차원 POVM·Naimark 확장·복합계·연구 깊이는 유한 qubit 예제로 완료 처리하지 않는다.

## 수치 기반과 후속 순서

첫 가족은75개 소유 유계 정수로 차원·네 ensemble 가중치/복소 대표·두 임의 Hermitian 행렬의 윗삼각 성분/양의 분모·실수 선형 결합 계수를 정한다. 정확 유리수 계정을 먼저 확인하고 표준편차·그림 좌표·2×2 공분산 스펙트럼에만 방향성 구간을 쓴다.

수치 가족1 → POVM/instrument 가족2 → 상세 한영 본문 → Worker → 고정 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·실제 읽기·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다.

배경은 [Trevisan §4.1](https://web.dm.unipi.it/trevisan/teaching/PhD/2022-qinfo/2022-Qinfo-notes.pdf), [Preskill Chapter2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), [IBM general measurements](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/introduction)에서 확인했다. 수치 계약·입력·손계산과 독립 참조는 이 교재에서 구성한다.

## 첫 수치 가족의 진행

2026-11-08 진행: [uncertainty·measurement의 첫 공분산 수치 가족](quantum-uncertainty-covariance-foundation-status.md)을 구현했다. 임의1…4차원 Hermitian 관측량·혼합 상태의 Robertson/Schrödinger 하한·분산 분해·복소 투영·선형 결합과 공분산 스펙트럼을 검산했다.716검사·타입·lint·독립51참조(49유일 입력)·참조 재생성 일치를 확인했다. 다음은 POVM/instrument 수치 가족 → 상세 한영 본문 → Worker/교수 화면 → 실제 읽기다. 아직 미등록이므로560단원·305/362입문 검증·57미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 두 수치 가족의 진행

2026-11-08: [POVM/instrument 가족](quantum-povm-instrument-foundation-status.md)의 새52검사와 이전716검사, 합계768검사를 통과했다. 두 가족의 독립 참조91사례/88유일 입력·정확78128·방향성23176대조를 확인했다. 다음은 두 단원의 상세 한영 원고4개다. Worker·교수 화면·등록·실제 읽기는 미완료이며 등록560단원·입문305/362·57미완료를 유지한다.

## 상세 한영 본문 준비

2026-11-08: [두 단원의 한영 원고4개](quantum-uncertainty-foundation-status.md)를 작성했다. 전체48절·34표시 수식·8표·24해설이며 새17/합계785검사로 수식 렌더링·번역 수식 일치·본문 예제와 Gaussian 적분을 확인한다. 다음은 입력/Worker 수명주기 → 교수 화면 → Interactive Book 등록·실제 읽기다. 아직 등록 수와 원래 입문 검증 수는 늘리지 않는다.

## Worker 준비

2026-11-08: [두 입력 가족의 Worker 계약과 수명주기](quantum-uncertainty-worker-status.md)를 구현했다. 새107/합계892검사, 실제Chrome Worker91입력·102584관측과 React StrictMode 입력 변경·취소·실패/재시도·unmount를 확인했다. 다음은 고정 개념·자유 실험·그림·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·실제 읽기다. 별도 브라우저 harness의 검증을 등록 reader 검증으로 세지 않으며560단원·입문305/362·57미완료를 유지한다.

## 교수 화면 준비

2026-11-08: [두 가족의 개념·실험·시각화](quantum-uncertainty-teaching-status.md)를 구현했다.32예제·공분산 타원·사영·세 instrument의 Bloch/RMS 비교와 저장·모션·가이드·퀴즈를 연결했다. 새42/합계934검사,16한영/명암/모바일·데스크톱 화면·접근성·실제3D8회, Worker696종료와 추가 제어40종료를 확인했다. 다음은 원고·단원·가이드·퀴즈·표지·검색·선수 관계 등록 → 실제 reader·Next Worker·정적 출력 검증이다. 아직560단원·입문305/362·57미완료를 유지한다.

## 실제 교재 등록·검증

2026-11-08: [두 단원의 원고·가이드·퀴즈·표지·검색·선수 관계를 등록하고 실제 reader를 검증했다](quantum-uncertainty-integration-status.md). 전체11015검사·빌드·1800 HTML·읽기/접근성16조합·생산 Worker91입력102584관측을 확인했다. 현재562단원·1686안내·원래306/362입문 검증·56미완료이며 quantum 입문은6/11이다. 다음은 tensor product·entanglement다. 연구 깊이·전체51영역509주제·공용 그림 후속은 미완료다.
