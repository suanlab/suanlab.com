# Gauge freedom의 자료 대조와 독립 준비

2026-11-05. 매질·경계 다음 입문 주제의 준비 기록이다. 이 단계는 자료·수치 탐색 기록이다. 이후 [계산 모델·상세 한영 원고](gauge-freedom-foundation.md)를 작성·검증했다. 이후 [단원 등록·화면 통합 검증](gauge-freedom-authoring-status.md)을 마쳤다. 자료 탐색 자체로 완료 수를 올리지는 않는다.

## 1차 자료와 표기

[Feynman II §18–6](https://www.feynmanlectures.caltech.edu/II_18.html)의 식18.16–18.26에서 시간 의존 전위와 동시 게이지 변환, Lorenz 조건과 분리된 전위 파동 방정식을 확인했다. 같은 장 §18–1은 전하 보존과 Maxwell 방정식의 관계를 다룬다. 명칭은 Lorenz이며 Lorentz와 구분한다. 아래 예제·초기 조건 대조는 이 관계에서 직접 구성한 것이다.

## 후보 단원

| 후보 ID | 중심 질문 |
| --- | --- |
| `time-dependent-gauge-field-invariance` | 전위가 달라져도 E·B가 같으려면 무엇을 함께 바꾸어야 하는가? |
| `lorenz-gauge-source-initial-compatibility` | 전위 파동식과 보존 원천만으로 Maxwell 적합성이 충분한가? |

첫 단원은 Bz=b(t), A=(−b(t)y/2,b(t)x/2,0), V=0에서 시작한다. χ=a xyz g(t)로 A′=A+∇χ, V′=V−∂tχ를 함께 바꾼다. E=−∇V−∂tA와 B=∇×A는 유지된다. A 또는 V만 바꾸는 대조에서는 E가 달라진다. 닫힌 경로에서 ∇χ의 적분과 실제 전기·자기장을 따로 확인한다.

G=∇·A+c⁻²∂tV, □=Δ−c⁻²∂tt로 정의한다. G′=G+□χ이므로 시간 의존 변환이 항상 Lorenz 조건을 유지하지는 않는다. 이 다항식에서는 Δχ=0, G′−G=−a xyz g″/c²다. **Lorenz 조건을 벗어난 전위도 같은 유효한 물리장을 나타낼 수 있다.** 게이지 조건 위반과 Maxwell 물리 위반을 혼동하지 않는다.

두 번째 단원은 분리된 파동식 □A=−μJ, □V=−ρ/ε에서 □G=−μ(∇·J+∂tρ)를 유도한다. 원천 보존이면 G의 파동식은 동차지만, G=0을 유지하려면 적절한 경계 조건과 초기 G·∂tG가 모두 필요하다.

V=V₀cos(kz−ωt), Az=A₀cos(kz−ωt)의 종방향 예제에서 전위 파동식이 지정한 원천과 실제 E에서 역산한 원천을 비교한다. ω=ck이면 전위 파동식의 원천은0이어도 Lorenz 조건을 벗어난 선택의 E가 비영이 되어 진공 Gauss 법칙을 실패할 수 있다. 별도 V=0, Az=a sin(kz)sin(ckt)는 초기 G=0이지만 ∂tG≠0인 대조다. 단일 시점의 G 검사만으로 충분하다고 가르치지 않는다.

## 독립 계산과 한계

`.runtime/learning/prepare-gauge-freedom.py`는 제품 모듈을 가져오지 않는다. 다항식144조건에서 5점 공간·시간 차분으로 E/B/G를 계산하고 SciPy로 네 모서리의 게이지 추가분을 적분한다. 종방향24조건은 스칼라·벡터 파동식과 원천 연속성, 실제 장의 원천을 비교한다. 초기 제약3조건과 근접 공명180자리 기준3조건도 확인한다. 결과는 `.runtime/learning/gauge-freedom-preparation.json`, 실제 종료 기록은 `gauge-freedom-preparation-exit.json`이다.

차분은 c가1정도인 교육용 계수를 사용했다. 후속 제품에서는 SI 단위와 입력 표현 범위·미정 상태를 별도 설계해야 한다. 수치 잔차는 엄밀한 전역 오차 상한이 아니다. 매끄러운 국소 전위의 입문 예제이며 비단순연결 공간·위상·양자 게이지 이론·완전한 경계값 문제까지 완료했다고 주장하지 않는다.

[후속 계산·원고 설계](gauge-freedom-authoring-plan.md) · [남은 전체 순서](remaining-next-steps.md)

## 기초 집필로 확장한 검증

재현 가능한 독립 생성기를 `scripts/learning/generate-gauge-freedom-references.py`에 정리했다. 실제 전위의 차분·혼합 이차 미분과 폐경로 구적576조건, 조화36조건·초기 제약36조건,180자리 곱 차이5조건·근접 공명3조건을 추가했다. 원천과 장의 동일성을 분리하고 일정 비영 G의 유효한 진공 표현도 대조했다. [기초 집필 기록](gauge-freedom-foundation.md)은 초기 자료 탐색과 제품·원고 검증을 구분한다.
