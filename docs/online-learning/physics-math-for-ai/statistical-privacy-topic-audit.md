# 프라이버시·공정성·강건성 입문 범위 점검

2026-11-04. 원래 열 범위 주제를 아래 실제 단원에 대응한다. 이 점검은 명시된 유한 모형과 입문 설명의 범위를 기록한다. 임의의 배포 시스템 인증이나 연구 수준 전체 심화의 완료를 뜻하지 않는다. 실행 결과는 진행 원장의 `verification`과 `.runtime/learning/evaluation-*` 증거를 따른다.

| 원래 범위 주제 | 실제 단원 ID | 핵심 유도·반례와 적용 조건 |
| --- | --- | --- |
| differential privacy의 정의 | `finite-events-differential-privacy` | 한 비트 교체 인접성, 모든 방향 쌍·출력 사건의 최대 양의 원자 차이. 한 사건의 통과와 전역 정의를 구분한다. |
| epsilon·delta·구성 | `fresh-copy-privacy-composition` | 독립 새 응답, 동일 응답 복사, XOR의 조건부 누출을 비교한다. 주변 보장만으로 적응적 구성을 주장하지 않는다. |
| 민감도·잡음 | `bounded-sensitivity-laplace`, `gaussian-sensitivity-calibration` | 전역 민감도의 달성 예, 전체 실수 꼬리 사건, Laplace/Gaussian 우도비와 오차. 수치 언더플로는 순수 프라이버시의 증거가 아니다. |
| 정보 누출과 추론 | `posterior-odds-privacy-inference` | 가능한 관측의 Bayes 갱신, 사전/사후 오즈, 집단 거리, 희귀 공개. 강한 사전 예측과 새 누출을 분리한다. |
| fairness 지표·식별 조건 | `fairness-metrics-selection-identification` | 혼동행렬 분모, 선택 편향의 동일 관측 두 모집단, 알려진 양의 선택확률 역산. 0 분모·0 선택확률은 미정이다. |
| 지표 간 양립 조건 | `binary-parity-compatibility`, `calibration-class-balance` | 서로 다른 base rate에서 parity와 calibration 조건의 관계. 완벽 예측·퇴화·정의되지 않은 비율을 일반적인 불가능성 주장과 구분한다. |
| robust statistics | `replacement-contamination-location`, `influence-and-breakdown` | 유한 교체 표본과 모집단 오염, 평균·중앙값·Huber 점수, 영향함수와 breakdown. 국소 영향과 큰 오염의 붕괴는 다른 질문이다. |
| adversarial perturbation | `linear-norm-adversarial-margins`, `relu-attacks-and-certificates` | 쌍대 노름의 달성 교란, 정확 경계 동률, ReLU 유한 후보 최솟값과 interval 하한·실행 가능한 공격. 공격 실패는 보장이 아니다. |
| 분포적 강건성 | `kl-ambiguity-risk-envelope`, `finite-wasserstein-risk-certificates` | KL 방향·엔트로피 쌍대·온도 극한, 평균 운송비용·유한 원시/쌍대 일치. 내부 최악 위험의 계산과 현실 분포 포함은 구분한다. |
| 수학적 보장과 현실적 평가의 차이 | `stratified-evaluation-shift-bounds`, `selection-aware-benchmark-coverage` | 단측 이항 상한·동시 보정·미관측 집단 식별 불가·혼합 이동, 선택 후 포함률·고정 유한 후보 보정. 평균 위험 0과 전칭 정확성도 구분한다. |

## 검증을 해석하는 방법

각 단원은 한·영 원고, 네 개 개념 단계, 세 개 안내 실험과 풀이가 있는 퀴즈 두 개를 제공한다. 원고와 안내는 `content/learning/physics-math-for-ai/` 및 `src/data/interactive-book.ts`에 있다. 모델별 독립 검사는 `scripts/learning-*.test.ts`, 실제 조작·언어·반응형·접근성 검사는 진행 원장에 연결된 브라우저 스크립트에 있다. 원래 열 주제의 연결을 모두 확인하더라도 더 넓은 모형의 증명과 논문 재현까지 완료했다고 해석하지 않는다.

## 남는 심화 작업

연속·고차원 메커니즘과 더 강한 프라이버시 구성, 구조적/인과적 공정성, 일반 분포의 강건 추정, 큰 신경망의 형식 인증, 데이터로 정당화한 분포 불확실성 집합, 의존 표본과 순차·적응적 평가가 후속 심화다. 다음 정방향 집필 영역은 측정·단위·물리 모델링이며 기존 주제 순서를 유지한다.
