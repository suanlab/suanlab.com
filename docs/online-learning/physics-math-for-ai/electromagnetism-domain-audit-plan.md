# 전자기학 영역 종합 점검 계획

2026-11-05. 전자기학의 입문 단원 연결 뒤 수행할 점검이다. 이 계획과 등록 목록은 완료 증거가 아니다. 전체51영역509주제·연구 심화 완료와도 구분한다.

## 원래 주제와 점검 대상

| 원래 주제 | 두 입문 단원 |
| --- | --- |
| 전하·전류·장 | `charge-superposition-test-field`<br>`charge-continuity-surface-current` |
| 정전·정자기 | `gauss-symmetry-electrostatic-sphere`<br>`ampere-symmetry-steady-current-cylinder` |
| 전위·벡터 퍼텐셜 | `electrostatic-potential-reference-path-work`<br>`magnetic-vector-potential-curl-gauge` |
| Maxwell 방정식 | `maxwell-plane-wave-local-integral-laws`<br>`maxwell-charge-displacement-continuity` |
| 유도·전자기파 | `faraday-induction-fixed-moving-loop`<br>`electromagnetic-wave-equation-transverse-initial-data` |
| 장의 에너지·Poynting 흐름 | `poynting-source-work-local-balance`<br>`transverse-pulse-energy-flux-balance` |
| 매질·경계 | `dielectric-interface-free-bound-charge`<br>`normal-incidence-impedance-interface` |
| gauge freedom | `time-dependent-gauge-field-invariance`<br>`lorenz-gauge-source-initial-compatibility` |
| 회로·센서와 연속장 | `rc-circuit-field-closure-approximation`<br>`finite-area-coil-sensor-observability` |
| 유효 모델 | `transmission-line-lumped-limit-energy`<br>`layered-medium-bloch-effective-limit` |

## 내용과 물리의 점검

모든 원래 주제에 실질적인 한영 이론·가정·직관·단위·손계산·풀이 문제가 있는지 실제 원고로 확인한다. 선수 관계가 앞 단원과 연결되는지, 개념 단계의 고정 예제·그림·통계가 같은 조건을 쓰는지 대조한다. 서로 다른 시간 규약·SI/정규화 척도·점/가족 판정·초기/경계 조건을 섞지 않는다.

전하 보존·정전/정자기·전위/게이지·Maxwell·고정/이동 유도·에너지 흐름·매질 경계·회로/센서·유효 모델 사이의 설명을 이어 읽는다. 국소 장과 전체 장치, 전위 조건과 실제 장, 선속과 전압, 해석적 항등식과 독립 검증을 구분한다. 입문 예제의 범위 밖인 증명·일반 기하·분산/손실 재료·연구 심화도 명시한다.

## 계산과 기존 단원 보존

제품식과 독립 ODE·구적·차분·행렬 지수·선형계·고정밀 기준이 무엇을 각각 검증하는지 확인한다. 정확한 영·미정·비호환·비유일·구조 rank·표현 손실을 점검하고 수치 잔차를 전역 인증으로 바꾸지 않는다. 예전 canonical 단원·안내·실험 그룹과 상대 순서를 보존한다. 관련·전체 테스트, 타입·lint·콘텐츠 검사와 최종 정적 빌드·출력 검사를 실제 종료 기록으로 확인한다.

## 전체 영역의 화면 점검

20개 단원 전부의 한영·모바일/데스크톱·명암 화면, 영어 UI, 접근성·키보드, 조작·예제·정밀 입력·저장·퀴즈·선수/앞뒤 링크·목차를 검사한다. 각 가족의 영입력·경계·작은 비영 입력·미정 상태를 포함하고 원본 크기 그림과 설명의 실제 캡처를 직접 검토한다. 일부 신규 단원의 통합 검사만으로 영역 전체를 통과 처리하지 않는다.

## 완료 원장 갱신 조건

발견한 결함을 고친 뒤 최종 소스·정적 출력을 동결하고 검사 전후 해시를 대조한다. 요구 범위 전체를 입증하는 보고서·실행·화면 증거가 있어야 전자기학 영역을 입문 검증 완료로 바꾼다. 그 뒤 다음 영역은 연속체·유체·탄성·수송의 질량·운동량·에너지 보존이다. 종합 점검 전에는 `wholeDomainAuditComplete`를 false로 유지한다.
