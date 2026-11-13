# 비평형·확률열역학·복잡계 연구 진행표

2026-11-13. 이 영역에는 이미 **10원주제·17등록 단원·34한영 원고**가 있다. [기존 입문 종합 점검](nonequilibrium-introductory-audit.md)은 출발 근거이며 미집필로 다시 세지 않는다. 현재 추가 연구 절의 순차 보강 검증은 **7/17단원**, 영역 연구 종합 검토는 미완료다. 전체는 **44영역·437주제 대조 / 7영역·72주제 후속**이며 교육과정 완료 상태는 false다.

## 진행 원칙

기존 본문·풀이·독립 계산·Worker·실제 화면 근거를 먼저 읽는다. 가정·직관·수식 유도·실행 예제·그림·연습 해설을 더하고 원문 보존과 한영 일치를 확인한다. 새로운 경로 모형이나 추론을 기존 확률 법칙 reader의 조작으로 소개하지 않는다. 주차·분량 제한은 두지 않는다.

행/열 생성자 규약, 물리 시간·잠재 사건·실제 점프·기록 시점을 구분한다. 확률 흐름이나 경로 KL에 에너지·저장고·국소 상세평형 없이 물리적 열 단위를 붙이지 않는다. 유한 상태 계산과 일반 생성자 정리, 정상 법칙과 동적 응답, 확률적 초기화와 결정론적 숨은 강제를 구분한다.

## 기존 단원별 후속 순서

| 단원 ID | 원주제 | 상태 |
|---|---|---|
| `two-state-continuous-time-generator-euler` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `poisson-uniformization-potential-events` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `state-dependent-holding-time-jump-occupation` | master equation | [경로·별칭·재생 연구 보강 검증](master-equation-research-status.md) |
| `ornstein-uhlenbeck-density-euler-moments` | Fokker-Planck | [관측 추론·최초 도달 연구 보강 검증](fokker-research-status.md) |
| `fokker-planck-cell-mass-boundaries` | Fokker-Planck | [관측 추론·최초 도달 연구 보강 검증](fokker-research-status.md) |
| `stationary-gaussian-circulating-current` | 확률 흐름 | [표본 시간 방향·유한 시간 요동 연구 검증](current-reversal-research-status.md) |
| `continuous-cycle-path-reversal` | detailed balance와 그 위반 | [표본 시간 방향·유한 시간 요동 연구 검증](current-reversal-research-status.md) |
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

[다음 두 Fokker–Planck 단원](fokker-research-status.md)도 한영 4원고·4그림·55검사·16실제 화면에서 검증했다. 기존 풀이 24개를 보존하고 관측 추론·Euler 계수 편향·방향별 최초 도달 시간·공간 오차를 보강했다.

[순환 확률 흐름과 경로 반전](current-reversal-research-status.md)도 한영 4원고·4그림·67검사·16실제 화면에서 검증했다. 표본화 KL·반 회전 가역성·유한 시간 요동·드문 지수 가중치를 보강했고 기존 풀이 24개를 보존했다.

다음은 `thermal-ou-fluctuation-response`의 변동·소산 연구 보강이다. 이후 선형 응답 → 엔트로피 생성 → 확률열역학 → 동기화·임계 동역학 → coarse graining 순서를 따른다. 나머지 **10단원** 보강과 원래 **10주제의 영역 종합 검토**는 미완료다.
