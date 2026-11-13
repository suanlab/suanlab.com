# Boson 동적 절단·MPS transfer 연구 보강

2026-11-13. 기존 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 본문과 풀이 32개를 정확히 보존했다. [검증 근거](quantum-representation-research-evidence.json)와 [자료·가정 대조](quantum-representation-research-source-audit.md)를 따른다.

## 동적 절단 오차

`boson-fock-cutoff-boundary-algebra`는 경계 잔차의 누적 적분으로 시간 전개 오차를 제어한다. 구동 oscillator의 전체 coherent 해와 유한 C=1 궤적을 직접 비교하고, top 확률이 0으로 돌아와도 전체 상태 오차가 사라지지 않는 예를 계산한다. 초기 투영 오차·전역 위상·유계 관측량 조건을 명시한다. 점 표본이나 수치 적분을 구간 상계 인증으로 표시하지 않는다.

## Transfer와 상관 길이

`matrix-product-state-contraction-gauge`는 bond dimension 2의 명시적 순수 MPS family에서 CZZ(r)=lambda^r을 유도한다. 고정된 작은 bond dimension과 발산하는 상관 길이가 양립하는 이유, GHZ·product endpoint, virtual 고정점과 physical 밀도행렬, 복소 gauge의 조건수를 설명한다. 표의 거리 4는 N>=5의 독립 예제이며 기존 reader N<=4 제한을 유지한다. Transfer gap에서 부모 Hamiltonian의 에너지 gap을 추론하지 않는다.

## 검증

- CI 선언 Python 3.12 격리 환경에서 관련 13파일·327검사가 통과했다. 한영 수식·코드·표와 영어 한글 잔류·KaTeX를 검사하고 원본문 hash·풀이 32개를 보존했다.
- Boson 경계 잔차와 실제 전개 96준비, coherent Schroedinger 방정식 24준비를 독립 비교했다. 정확한 유한 귀환에서도 전체 trace distance 약 0.9897을 확인했다. 수치 quadrature 비교는 방향성 적분 인증이 아니다.
- N=2…7 MPS 30상태의 직접 수축·모든 ZZ 쌍, 복소 gauge 상태 60비교·transfer power 24비교, 일반 Jordan 블록 24비교를 확인했다. Jordan 블록은 이 MPS family에 존재한다고 주장하지 않는다. SymPy로 정확한 각도·transfer 다항식 각 4경우를 확인했다.
- 빌드·타입·콘텐츠·정적 출력 검사의 실제 종료 코드는 0이다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG 바이트를 대조했다.
- 한영 320·768·1440px 밝은 화면과 1440px 어두운 화면 16상태를 확인했다. 모식도·예측 후 해설, 안내·퀴즈 각 6회와 저장·완료 복원 2그룹을 검증했다. 현재 입력과 실제 계산 완료 및 boson 에너지/분산·MPS norm/정의 여부를 비교했다. 접근성 위반·페이지 오류가 없고 브라우저를 종료했다.
- 원본 그림 4개와 정상 읽기 모바일 화면 4개를 직접 검토했다. SVG·캡션 경계·목차 앵커와 긴 코드·표·수식의 키보드 스크롤을 확인했다.

## 보존과 다음 순서

기존 boson 1…2모드·cutoff 0…3, MPS N=1…4 제어를 유지한다. 새 시간 적분·infinite MPS·Schmidt truncation UI를 추가한 것으로 설명하지 않는다. 검토 원장 935항목을 보존하고 4편을 더했다. 기존 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 순차 연구 대조는 **20/34단원**이다. 다음은 `schmidt-cuts-bond-rank-truncation`, 이어서 `finite-qubit-circuit-order-interference`다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**은 유지한다. 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 별도로 확인한다.
