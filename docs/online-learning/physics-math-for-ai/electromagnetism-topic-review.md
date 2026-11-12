# 전자기학과 전자기장 10원주제 연구 대조 순서

2026-11-13. 전체 51영역·509원주제를 유지하며 전자기학의 **10주제·20단원·40한영 원고**를 기존 등록 순서로 검토한다. [입문 영역 감사](electromagnetism-domain-audit.md)는 이미 존재한다. 전하·전류의 [탐침 교란·순환 전류](charge-research-status.md)에 이어 정전·정자기의 [내부 밀도 역문제·표피 효과](statics-research-status.md)까지 **4/20단원**을 검증했다. 나머지 16단원과 영역 전체 종합 대조는 후속이다. 순차 배치 통과를 전체 영역 완료로 집계하지 않는다.

## 기존 내용과 연구 대조 순서

각 행의 두 단원과 원주제는 실제 `src/data/learning-curriculum.ts`에서 확인했다. 기존 상세 원고·실습·독립 참조·입문 감사의 가정을 먼저 읽고 부족한 논증만 보강한다.

| 원주제 | 기존 단원 | 집필·검토 근거 |
|---|---|---|
| 전하·전류·장 | `charge-superposition-test-field`, `charge-continuity-surface-current` | [기존 집필](charge-current-fields-authoring-status.md) · [연구 보강](charge-research-status.md) |
| 정전·정자기 | `gauss-symmetry-electrostatic-sphere`, `ampere-symmetry-steady-current-cylinder` | [기존 집필](electro-magnetostatics-authoring-status.md) · [연구 보강](statics-research-status.md) |
| 전위·벡터 퍼텐셜 | `electrostatic-potential-reference-path-work`, `magnetic-vector-potential-curl-gauge` | [기존 집필](potentials-authoring-status.md) |
| Maxwell 방정식 | `maxwell-plane-wave-local-integral-laws`, `maxwell-charge-displacement-continuity` | [기존 집필](maxwell-authoring-status.md) |
| 유도·전자기파 | `faraday-induction-fixed-moving-loop`, `electromagnetic-wave-equation-transverse-initial-data` | [기존 집필](induction-waves-authoring-status.md) |
| 장의 에너지·Poynting 흐름 | `poynting-source-work-local-balance`, `transverse-pulse-energy-flux-balance` | [기존 집필](poynting-energy-authoring-status.md) |
| 매질·경계 | `dielectric-interface-free-bound-charge`, `normal-incidence-impedance-interface` | [기존 집필](material-interfaces-authoring-status.md) |
| gauge freedom | `time-dependent-gauge-field-invariance`, `lorenz-gauge-source-initial-compatibility` | [기존 집필](gauge-freedom-authoring-status.md) |
| 회로·센서와 연속장 | `rc-circuit-field-closure-approximation`, `finite-area-coil-sensor-observability` | [기존 집필](circuit-sensors-authoring-status.md) |
| 유효 모델 | `transmission-line-lumped-limit-energy`, `layered-medium-bloch-effective-limit` | [기존 집필](electromagnetic-effective-models-authoring-status.md) |

## 첫 두 단원의 보강 설계

`charge-superposition-test-field`는 원천을 고정한 Coulomb 장과 시험 전하의 힘을 이미 구분한다. 다음 보충 계산은 기계적으로 구속된 원천이 유한한 탐침에 반응하는 한 축 모형을 명시해, 탐침을 작게 하는 극한과 원천을 실제로 고정하는 준비를 구분한다. 힘의 평형·국소 안정·가능한 평형 가지와 붕괴 경계를 직접 유도하고, 관측 힘을 탐침 전하로 나눈 값이 언제 원래 장을 근사하는지 정량화한다. 이를 일반적인 자기 일관 Maxwell 해나 실제 장치의 안정성 인증으로 소개하지 않는다. 기존 고정 원천 실습의 입력은 보존한다.

`charge-continuity-surface-current`는 지정한 Gaussian 수송의 국소 보존과 면 전류·구간 수지를 이미 다룬다. 다음 보충 계산은 유한 닫힌 회로의 방향 있는 incidence 행렬을 사용해 전하 축적 자료가 결정하는 전류 발산과 관측되지 않는 순환 전류를 구분한다. 같은 전하 이력의 다른 전류, 하나의 추가 가지 관측이 주는 정보, 최소 노름 정규화의 선택을 직접 계산한다. 원래 Gaussian 이동 모델과 별도인 유한 회로 예제이며 새 네이티브 제어 기능으로 표시하지 않는다. 전하만 맞추는 AI 손실의 식별 한계를 해설한다.

위 설계는 [한영 원고·독립 계산·16화면 검증](charge-research-status.md)을 마쳤다. 기존 여섯 해설 문제를 각각 보존하고 설명·직관·단위·표·Python·그림·연구 질문을 추가했다. 선택 75검사와 새 빌드·1,888 HTML을 통과했다. 이후 [정전·정자기 보강](statics-research-status.md)도 선택 58검사·최종 16화면과 새 빌드로 검증했다. 다음은 전위·벡터 퍼텐셜의 `electrostatic-potential-reference-path-work`, `magnetic-vector-potential-curl-gauge`다.

## 검증과 보존 기준

- 보강 직전 Git 기준과 기존 40원고·검토 원장을 고정한다. 연구 절 하나를 제거해 원본문이 복원되도록 하고, 다른 영역·등록 순서·실습 입력을 보존한다.
- 번역의 가정·수식·예제·해설을 대조하고 표준 라이브러리 독립 계산과 원고 Python을 실행한다. 기존 모델의 실제 관측량과 보충 모델의 증명을 구분한다.
- 각 배치의 한영 그림과 기존 홈페이지 색·서체·명암·반응형을 검토하고 실제 reader의 안내·퀴즈·개념·저장·키보드·접근성을 검사한다.
- 20단원 순차 보강 뒤 같은 최종 빌드의 영역 전체 대조를 수행한다. 개별 배치 통과를 10원주제의 영역 완료로 합산하지 않는다. 전체 진행은 [영역 색인](domain-topic-review-index.md)을 따른다.

전자기학 다음은 연속체·유체·탄성·수송 11원주제다. 나머지 영역은 [남은 순서](remaining-next-steps.md)를 유지한다. 입문 362/362와 요청 당시 48개 완료는 연구 보강의 분모와 구분한다.
