# 교육과정 원주제 대조 기록 색인

2026-11-12 기준. `src/data/learning-curriculum.ts`의 51영역·509개 원주제를 분모로 사용한다. 등록된 학습 단원 수나 개별 계산 모델 수를 원주제 수와 바꾸어 계산하지 않는다.

현재 교재에는 650개 학습 단원과 1,969개 안내 실험이 등록되어 있다. 요청 당시 남은 입문 48개와 후속 입문 목록 362/362의 집필·검증은 끝났다. 이는 아래 27영역을 모두 미집필로 본다는 뜻이 아니다.

## 대조 기록이 연결된 24영역

초기 15영역의 147개에 확률 그래프 모델 10개·부호와 알고리즘 정보 9개·함수해석과 연산자 9개·근사이론과 커널 10개·조화해석과 신호 처리 10개·미분기하와 위상 10개·군과 대칭 10개·볼록해석과 제약 최적화 10개·비볼록과 확률 최적화 10개를 더한 235개 원주제의 내용 대조 기록을 연결했다. 각 기록은 검토한 가정·논증·코드·화면과 후속 범위를 밝힌다. 원주제 대조가 해당 수학 분야의 모든 연구 이론을 완성했다는 뜻은 아니다.

| 영역 | 원주제 수 | 대조 기록 |
|---|---:|---|
| 논리·집합·증명 | 9 | [logic-proof](logic-elementary-topic-review.md) |
| 수·함수·좌표·기초 기하 | 11 | [elementary-math](logic-elementary-topic-review.md) |
| 이산수학과 조합 구조 | 9 | [discrete-math](discrete-topic-review.md) |
| 선형대수와 행렬 분석 | 11 | [linear-algebra](linear-algebra-topic-review.md) |
| 미적분과 실해석 | 9 | [analysis](analysis-topic-review.md) |
| 다변수·벡터·행렬 미적분 | 10 | [matrix-calculus](matrix-calculus-topic-review.md) |
| 복소수 해석과 복소 계산 | 9 | [complex-analysis](complex-analysis-topic-review.md) |
| 텐서와 다중선형대수 | 10 | [tensor-methods](tensor-methods-topic-review.md) |
| 추상대수와 대수적 구조 | 9 | [abstract-algebra](abstract-algebra-topic-review.md) |
| 확률과 확률변수 | 10 | [probability](probability-topic-review.md) |
| 측도론적 확률과 극한 | 10 | [measure-probability](measure-probability-topic-review.md) |
| 통계적 추정과 실험 | 10 | [statistics](statistics-topic-review.md) |
| Bayes 추론과 잠재변수 | 10 | [bayesian-inference](bayesian-inference-topic-review.md) |
| 정보이론과 압축 | 10 | [information-theory](information-theory-topic-review.md) |
| 고차원 확률과 무작위 행렬 | 10 | [high-dimensional](high-dimensional-topic-review.md) |
| 확률 그래프 모델과 구조적 추론 | 10 | [graphical-models](graphical-models-topic-review.md) |
| 부호·계산과 알고리즘 정보 | 9 | [coding-algorithmic-information](coding-algorithmic-information-topic-review.md) |
| 함수해석과 연산자 | 9 | [functional-analysis](functional-analysis-topic-review.md) |
| 근사이론·커널·표현력 | 10 | [approximation-kernels](approximation-kernels-topic-review.md) |
| 조화해석과 신호 처리 | 10 | [harmonic-analysis](harmonic-analysis-topic-review.md) |
| 미분기하·위상·데이터 형상 | 10 | [geometry-topology](geometry-topology-topic-review.md) |
| 군·대칭·표현론 | 10 | [symmetry-groups](symmetry-groups-topic-review.md) |
| 볼록해석과 제약 최적화 | 10 | [convex-optimization](convex-optimization-topic-review.md) |
| 비볼록·확률·대규모 최적화 | 10 | [stochastic-optimization](stochastic-optimization-topic-review.md) |

## 다음 27영역의 근거 통합

아래 274개 원주제는 기존 집필·개별 감사 기록을 영역별 원주제에 연결하고 남은 범위를 확인할 대상이다. 일부는 이미 상세 원고·독립 계산·Worker·화면 검증이 있다. 기존 근거를 먼저 읽고 부족한 내용만 보강하며, 기록이 아직 이 색인에 연결되지 않았다는 이유로 미집필·실패로 판정하지 않는다.

| 영역 ID | 영역 | 원주제 수 |
|---|---|---:|
| variational-geometry | 변분법과 기하적 최적화 | 10 |
| optimal-transport | 최적수송과 분포의 기하 | 8 |
| numerical-computing | 수치계산과 자동 미분 | 10 |
| differential-equations | 상미분·편미분방정식 | 10 |
| stochastic-processes | 확률과정·확률미분방정식 | 11 |
| numerical-simulation | 과학계산과 수치 시뮬레이션 | 11 |
| algorithms-complexity | 알고리즘·계산복잡도·계산 구조 | 10 |
| learning-theory | 통계학습·일반화·강건성 | 11 |
| causality | 인과추론과 식별 가능성 | 11 |
| control-decision | 제어·최적제어·강화학습의 수학 | 11 |
| discrete-optimization | 조합 최적화와 형식적 추론 | 9 |
| game-theory | 게임이론과 다중 행위자 학습 | 10 |
| statistical-privacy | 프라이버시·공정성·강건성의 수학 | 10 |
| physical-modeling | 측정·단위·물리 모델링 | 9 |
| classical-mechanics | 뉴턴 역학과 강체 운동 | 10 |
| analytical-mechanics | 해석역학과 구조 보존 | 10 |
| waves-optics | 파동·진동·음향·광학 | 9 |
| electromagnetism | 전자기학과 전자기장 | 10 |
| continuum-physics | 연속체·유체·탄성·수송 | 11 |
| thermo-statmech | 열역학과 평형 통계역학 | 11 |
| nonequilibrium | 비평형·확률열역학·복잡계 | 10 |
| quantum-physics | 양자역학과 양자정보 | 11 |
| field-multiscale | 장이론·임계현상·다중스케일 | 10 |
| relativity | 상대론과 시공간 기하 | 10 |
| matter-computation | 물질·계산 장치·물리적 자원 | 10 |
| inverse-problems | 역문제·동화·실험 설계 | 11 |
| research-methods | 수학·물리 기반 AI 연구 방법 | 10 |

## 확인 기준과 다음 순서

- 원주제의 정의·가정·논증, 대학원생을 위한 직관·수치 예제·개념 시각화와 연습 해설을 원고·등록 단원에 대조한다.
- 한영의 수식·의미·실행 코드, 실제 실습의 계산과 경계, 난수 실현·근사·정리·연구 근거의 범위를 구분한다.
- Interactive Book의 연결, 기존 홈페이지 스타일, 반응형·접근성·언어 전환·저장과 정적 빌드·내부 링크 검증을 확인한다.
- 기존 원고·요구 범위를 보존하고, 명시한 보존·변경 검증을 통과한 결과만 commit/push한다. 공개 반영 완료는 배포 결과로 별도 확인한다.

다음은 `variational-geometry`의 10개 원주제이며 이후 표의 교육과정 순서를 따른다. 기존 고차 집필·감사 근거를 통합하기 전에는 전체 교육과정 완료를 선언하거나 `curriculumComplete`를 true로 바꾸지 않는다.

현재 진행: [변분법 첫 두 주제의 한영 증명·실행 예제 보강](variational-geometry-topic-review.md). 남은 여덟 주제를 포함한 영역 대조는 진행 중이므로 위 완료 집계에는 더하지 않았다.
