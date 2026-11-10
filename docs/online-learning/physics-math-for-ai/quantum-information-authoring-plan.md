# 양자정보: 순차 집필과 실험 설계

2026-11-09. 양자 영역의 다음 입문 주제 `quantum information`을 아래 아홉 단원으로 확장한다. 주차·분량 제한은 두지 않으며 연구 심화는 별도로 계속한다. 초안 ID는 `learning-quantum-information-topics.ts`에서만 관리하고, 실제 reader 검증 전에는 공용 registry에 등록하지 않는다.

| 순서 | 단원 | 설명·계산·시각화의 중심 | 현재 상태 |
|---|---|---|---|
|1|`binary-qubit-state-discrimination`|사전확률·가중 차이 행렬·Helstrom 측정·성공률·trace distance·제곱 fidelity|한영 원고, 유리수 커널, 독립 참조325개|
|2|`quantum-entropy-compression-conditional-information`|Bell/고전 상관/독립 상태, 음의 조건부 엔트로피, IID 전형 부분공간과 유한 블록 차원|한영 원고·유한 블록 및 결합 상태 커널·독립 참조|
|3|`classical-quantum-ensembles-holevo-information`|cq 블록·측정 후 확률표·달성 가능한 상호정보량과 Holevo 상계|한영 원고·방향 로그 계산기·독립 행렬 참조|
|4|`quantum-relative-entropy-data-processing`|비가환 행렬 로그·지지집합·무한값·탈위상 전후 DPI·쌍의 회복 가능성|한영 원고·스펙트럼 로그 계산기·독립 행렬 참조|
|5|`no-cloning-bb84-information-disturbance`|내적에 의한 복제 불가·CNOT 반례·이상적 intercept-resend와 sifted QBER|한영 원고·80가지 정확 열거·독립 ket/사후분포 참조|
|6|`teleportation-superdense-coding-resources`|복소 입력·네 측정 가지·순서 있는 보정·통신 전 Bob 상태·자원 장부|한영 원고·정확 프로토콜·독립 조밀 회로 참조|
|7|`quantum-error-correction-syndrome-knill-laflamme`|3큐비트 반복 코드·syndrome·논리 coherence·Knill–Laflamme·참조계 검증|한영 원고·정확 반복 코드/오류 조건·독립 조밀 참조|
|8|`pure-state-entanglement-majorization-resources`|Schmidt 확률·주요화 방향·비교 불가능한 쌍·Vidal 꼬리비|한영 원고·정확 판정·독립 순열 볼록 껍질·Bell 희석|
|9|`quantum-channel-capacities-coding-limits`|C/CE/Q·보조 자원·탈위상 채널·한 번의 coherent information과 정규화의 구분|한영 원고·출력/환경/선택 입력·가족 용량 계산·독립 확장 참조|

## 계산과 설명의 기준

- 각 본문은 직관 → 정의와 유도 → 직접 계산 → 실패하는 해석 → 해설 문제 → AI/연구 연결 순서로 작성한다. 한영 의미·수식·예제 수를 맞춘다.
- 이진 판별은 $\Delta=\pi\rho_0-(1-\pi)\rho_1$의 양의 고유공간으로 증명한다. 영 고유값의 비유일성, 무정보 입력의 관례, 동일하게 반올림되는 양의 이득을 구분한다.
- 유한 블록에서는 정확한 고유값·중복도·사영 가중치와 정수 메모리 차원을 제공한다. 부호기를 구현하지 않고 가중치를 얽힘 충실도라고 부르지 않는다.
- cq 예제의 $|0\rangle$, $(3|0\rangle+4|1\rangle)/5$에서 평균 상태의 고유값은 $(4/5,1/5)$다. 선택한 Helstrom 측정의 상호정보량은 $1-h_2(1/10)$이며, 일반적인 accessible information 최적화와 동일하다고 주장하지 않는다.
- BB84의 이상적 전면 공격은 sifted QBER $1/4$를 준다. 현대적 유한 키·조합 가능 보안 증명이나 임계값은 이 장난감 계산으로 대체하지 않는다.
- 반복 코드의 IID X 오류는 논리 오류 $3p^2-2p^3$다. 일반 Pauli 오류 정정으로 확대하지 않는다. 논리 $|+\rangle$ 하나가 유지되는 것만으로 채널 복구를 주장하지 않는다.
- 순수 상태 결정론적 LOCC는 내림차순 Schmidt 확률에 대한 $\lambda_\psi\prec\lambda_\phi$ 방향이다. 엔트로피 하나만으로 전환 가능성을 판정하지 않는다.
- 용량의 단일 문자 식은 해당 채널 가족의 정리를 명시한다. 유한 입력 탐색으로 일반 양자 용량을 계산했다고 쓰지 않는다.

## 화면과 등록 순서

첫 실험은 Bloch 벡터 두 개·사전확률 → 가중 차이의 고유값 → 최적 측정축·조건부 결과표 → 사전확률만 쓰는 기준선의 흐름이다. 다음 실험은 두 부분계의 고정 엔트로피와 변하는 결합 스펙트럼, 보존 rank와 정확한 블록 가중치를 비교한다. 측정 축의 비유일성과 rank0은 명시적으로 설명한다.

계산별 입력 소유권·독립 응답 검증 → 취소/교체 가능한 Worker → 개념 그림·안내·예측·해설 퀴즈 → 키보드·축소 모션·저장·모바일 → 격리된 등록 후보와 생산 자산 → 실제 reader·정적 출력 검증 순서로 진행한다. 현재 아홉 단원의 수치 기반에는 공개 입력/Worker/화면이 없으므로 완료 단원으로 세지 않는다.

진입은 **도서 → Interactive Book**이다. Online Book 목록에 카드를 추가하지 않는다. 기존 reader URL은 유지한다. 전체51영역509주제, 영역 종합 점검·연구 깊이·공용 그림 보완은 계속 남아 있다.

## 입력 계약 진행

2026-11-09. [아홉 가족의 입력 소유권·계산 연결과44개 한영 예제](quantum-information-input-status.md)를 준비했다. 다음은 독립 응답 검증·실제 Worker이며, 교수 화면·reader 등록과 연구 심화는 아직 남아 있다.
