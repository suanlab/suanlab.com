# 비평형·확률열역학·복잡계 연구 진행표

2026-11-13. 이 영역에는 이미 **10원주제·17등록 단원·34한영 원고**가 있다. [기존 입문 종합 점검](nonequilibrium-introductory-audit.md)은 출발 근거이며 미집필로 다시 세지 않는다. 현재 추가 연구 절의 순차 보강 검증은 **3/17단원**, 영역 연구 종합 검토는 미완료다. 전체는 **44영역·437주제 대조 / 7영역·72주제 후속**이며 교육과정 완료 상태는 false다.

## 진행 원칙

기존 본문·풀이·독립 계산·Worker·실제 화면 근거를 먼저 읽는다. 가정·직관·수식 유도·실행 예제·그림·연습 해설을 더하고 원문 보존과 한영 일치를 확인한다. 새로운 경로 모형이나 추론을 기존 확률 법칙 reader의 조작으로 소개하지 않는다. 주차·분량 제한은 두지 않는다.

행/열 생성자 규약, 물리 시간·잠재 사건·실제 점프·기록 시점을 구분한다. 확률 흐름이나 경로 KL에 에너지·저장고·국소 상세평형 없이 물리적 열 단위를 붙이지 않는다. 유한 상태 계산과 일반 생성자 정리, 정상 법칙과 동적 응답, 확률적 초기화와 결정론적 숨은 강제를 구분한다.

## 기존 단원별 후속 순서

| 단원 ID | 원주제 | 상태 |
|---|---|---|
| `two-state-continuous-time-generator-euler` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `poisson-uniformization-potential-events` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `state-dependent-holding-time-jump-occupation` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `ornstein-uhlenbeck-density-euler-moments` | Fokker-Planck | 기존 근거 대조 후 연구 보강 |
| `fokker-planck-cell-mass-boundaries` | Fokker-Planck | 기존 근거 대조 후 연구 보강 |
| `stationary-gaussian-circulating-current` | 확률 흐름 | 기존 근거 대조 후 연구 보강 |
| `continuous-cycle-path-reversal` | detailed balance와 그 위반 | 기존 근거 대조 후 연구 보강 |
| `thermal-ou-fluctuation-response` | fluctuation-dissipation | 기존 근거 대조 후 연구 보강 |
| `markov-perturbation-linear-response` | linear response | 기존 근거 대조 후 연구 보강 |
| `stationary-jump-path-entropy-production` | entropy production | 기존 근거 대조 후 연구 보강 |
| `driven-two-state-heat-work` | stochastic thermodynamics | 기존 근거 대조 후 연구 보강 |
| `reservoir-resolved-two-state-heat-flow` | stochastic thermodynamics | 기존 근거 대조 후 연구 보강 |
| `hopf-amplitude-critical-relaxation` | synchronization·critical dynamics | 기존 근거 대조 후 연구 보강 |
| `two-oscillator-locking-phase-slips` | synchronization·critical dynamics | 기존 근거 대조 후 연구 보강 |
| `finite-kuramoto-collective-order` | synchronization·critical dynamics | 기존 근거 대조 후 연구 보강 |
| `markov-partition-lumpability-memory` | coarse graining·유효 기술 | 기존 근거 대조 후 연구 보강 |
| `linear-hidden-memory-markov-limit` | coarse graining·유효 기술 | 기존 근거 대조 후 연구 보강 |

## 검증과 다음 후속

[첫 세 단원의 경로 정보·관측 간격 별칭·재생 시간 평균](master-equation-research-status.md)을 한영 6원고·6그림·84검사·24실제 화면에서 검증했다. 원문과 풀이 36개를 보존했고 추가한 표시 수식은 56개다. [자료 대조](nonequilibrium-research-source-audit.md)와 JSON 근거를 따른다.

다음은 `ornstein-uhlenbeck-density-euler-moments`, `fokker-planck-cell-mass-boundaries`의 기존 원고·계산·입문 검증을 대조하고 연구 설명을 보강하는 것이다. 이후 확률 흐름 → 상세평형 위반 → 변동·소산 → 선형 응답 → 엔트로피 생성 → 확률열역학 → 동기화·임계 동역학 → coarse graining 순서를 따른다. 나머지 **14단원** 보강과 원래 **10주제의 영역 종합 검토**는 미완료다.
