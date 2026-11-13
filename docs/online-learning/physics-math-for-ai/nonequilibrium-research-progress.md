# 비평형·확률열역학·복잡계 연구 진행표

2026-11-13. 이 영역에는 이미 **10원주제·17등록 단원·34한영 원고**가 있다. [기존 입문 종합 점검](nonequilibrium-introductory-audit.md)은 출발 근거이며 미집필로 다시 세지 않는다. 현재 추가 연구 절의 순차 보강 검증은 **17/17단원**, 영역 연구 종합 검토는 미완료다. 전체는 **44영역·437주제 대조 / 7영역·72주제 후속**이며 교육과정 완료 상태는 false다.

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
| `thermal-ou-fluctuation-response` | fluctuation-dissipation | [스펙트럼·제어 경로 정보 연구 검증](response-research-status.md) |
| `markov-perturbation-linear-response` | linear response | [스펙트럼·제어 경로 정보 연구 검증](response-research-status.md) |
| `stationary-jump-path-entropy-production` | entropy production | [엔트로피·확률열역학 연구 검증](entropy-thermal-research-status.md) |
| `driven-two-state-heat-work` | stochastic thermodynamics | [엔트로피·확률열역학 연구 검증](entropy-thermal-research-status.md) |
| `reservoir-resolved-two-state-heat-flow` | stochastic thermodynamics | [엔트로피·확률열역학 연구 검증](entropy-thermal-research-status.md) |
| `hopf-amplitude-critical-relaxation` | synchronization·critical dynamics | [등위상선·자극 응답 연구 검증](hopf-isochron-research-status.md) |
| `two-oscillator-locking-phase-slips` | synchronization·critical dynamics | [시간 점유·교란 스펙트럼 연구 검증](synchronization-statistics-research-status.md) |
| `finite-kuramoto-collective-order` | synchronization·critical dynamics | [시간 점유·교란 스펙트럼 연구 검증](synchronization-statistics-research-status.md) |
| `markov-partition-lumpability-memory` | coarse graining·유효 기술 | [조건부 상태·초기층 보정 연구 검증](coarse-predictive-research-status.md) |
| `linear-hidden-memory-markov-limit` | coarse graining·유효 기술 | [조건부 상태·초기층 보정 연구 검증](coarse-predictive-research-status.md) |

## 검증과 다음 후속

[첫 세 단원의 경로 정보·관측 간격 별칭·재생 시간 평균](master-equation-research-status.md)을 한영 6원고·6그림·84검사·24실제 화면에서 검증했다. 원문과 풀이 36개를 보존했고 추가한 표시 수식은 56개다. [자료 대조](nonequilibrium-research-source-audit.md)와 JSON 근거를 따른다.

[다음 두 Fokker–Planck 단원](fokker-research-status.md)도 한영 4원고·4그림·55검사·16실제 화면에서 검증했다. 기존 풀이 24개를 보존하고 관측 추론·Euler 계수 편향·방향별 최초 도달 시간·공간 오차를 보강했다.

[순환 확률 흐름과 경로 반전](current-reversal-research-status.md)도 한영 4원고·4그림·67검사·16실제 화면에서 검증했다. 표본화 KL·반 회전 가역성·유한 시간 요동·드문 지수 가중치를 보강했고 기존 풀이 24개를 보존했다.

[열적 OU 변동·소산과 Markov 선형 응답](response-research-status.md)도 한영 4원고·4그림·233검사·16실제 화면에서 검증했다. 주파수 응답·잡음·탐침 일률과 경로 점수·Fisher 정보·기준선 추정을 보강했고 기존 풀이 24개를 보존했다.

[엔트로피 생성·확률열역학 세 단원](entropy-thermal-research-status.md)도 한영 6원고·6그림·487검사·24실제 화면에서 검증했다. 전류 정밀도·초기 중요도 재가중·저장고 계수 요동을 보강했고 기존 풀이 36개를 보존했다.

[Hopf 등위상선·위상 응답](hopf-isochron-research-status.md)도 한영 2원고·2그림·188검사·8실제 화면에서 검증했다. 진폭 회복 뒤 남는 위상·유한 자극·임계점 부근의 강제 근사 조건을 보강했고 기존 풀이 12개를 보존했다. 미등록이라는 오래된 상태 문장만 수정했다.

[위상 시간 점유·유한 집단 스펙트럼](synchronization-statistics-research-status.md)도 한영 4원고·4그림·249검사·16실제 화면에서 검증했다. 시간 평균의 정보 손실·관측 창·균형 상태의 성장률을 보강하고 기존 풀이 24개를 보존했다.

[조건부 예측 상태·초기층 보정](coarse-predictive-research-status.md)도 한영 4원고·4그림·198검사·16실제 화면에서 검증했다. 정상 이력의 정보·확률 필터·느린 모드와 초기층 합성 근사를 보강했고 기존 풀이 24개를 보존했다.

순차 보강은 **17/17단원**이다. 다음은 전체 17단원·34한영 원고·원래 **10주제의 영역 종합 검토**다. 내용·선수 관계·독립 계산·같은 정적 출력의 전체 화면 대조를 마치기 전에는 영역 완료로 집계하지 않는다.
