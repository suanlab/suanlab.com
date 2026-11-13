# 비평형·확률열역학·복잡계 연구 진행표

2026-11-13. 이 영역에는 이미 **10원주제·17등록 단원·34한영 원고**가 있다. [기존 입문 종합 점검](nonequilibrium-introductory-audit.md)은 출발 근거이며 미집필로 다시 세지 않는다. 현재 추가 연구 절의 순차 보강 검증은 **0/17단원**, 영역 연구 종합 검토는 미완료다. 전체는 **44영역·437주제 대조 / 7영역·72주제 후속**이며 교육과정 완료 상태는 false다.

## 진행 원칙

기존 본문·풀이·독립 계산·Worker·실제 화면 근거를 먼저 읽는다. 가정·직관·수식 유도·실행 예제·그림·연습 해설을 더하고 원문 보존과 한영 일치를 확인한다. 새로운 경로 모형이나 추론을 기존 확률 법칙 reader의 조작으로 소개하지 않는다. 주차·분량 제한은 두지 않는다.

행/열 생성자 규약, 물리 시간·잠재 사건·실제 점프·기록 시점을 구분한다. 확률 흐름이나 경로 KL에 에너지·저장고·국소 상세평형 없이 물리적 열 단위를 붙이지 않는다. 유한 상태 계산과 일반 생성자 정리, 정상 법칙과 동적 응답, 확률적 초기화와 결정론적 숨은 강제를 구분한다.

## 기존 단원별 후속 순서

| 단원 ID | 원주제 | 상태 |
|---|---|---|
| `two-state-continuous-time-generator-euler` | master equation | 기존 근거 대조 후 연구 보강 |
| `poisson-uniformization-potential-events` | master equation | 기존 근거 대조 후 연구 보강 |
| `state-dependent-holding-time-jump-occupation` | master equation | 기존 근거 대조 후 연구 보강 |
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

## 첫 후속

두 상태 연속시간 생성자·Euler, Poisson uniformization, 상태 의존 holding/jump/occupation 세 단원의 기존 자료부터 대조한다. 첫 자료로 [Rao·Teh 2절 식 (2)와 3.1절](https://jmlr.org/papers/volume14/rao13a/rao13a.pdf)의 경로 밀도·마지막 대기 구간·virtual self event를 확인했다. 원문의 열 생성자와 교재의 행 생성자를 전치로 대조한다. 경로 관측 추론·관측 간격의 별칭·체류 평균 분산은 별도 지정 모형에서 직접 유도할 연구 예제이며 원문에 이미 있는 결과로 인용하지 않는다. 이후 Fokker–Planck → 확률 흐름 → 상세평형 위반 → 변동·소산 → 선형 응답 → 엔트로피 생성 → 확률열역학 → 동기화·임계 동역학 → coarse graining 순서를 따른다. 17단원 보강 뒤 원래 10주제의 영역 종합 검토를 별도로 수행한다.
