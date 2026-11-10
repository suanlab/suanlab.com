# 실시간 위상: 입력 편집·저장·학습 화면 통합

2026-11-11. [안내와 해설 문제](real-time-phase-lessons-status.md)에 세 모형의 입력 form과 저장을 연결했다. 이 단계는 **저장·접근성 요건까지20/23검증**이다. 실제 Interactive Book reader 등록·Next 생산 자산, 기존 콘텐츠 보존, 단원 범위 감사는 후속이다. 미등록 초안을 공개 단원이나 완료한 원주제로 세지 않는다.

## 적용된 모형과 편집 중인 초안

사슬·정규화된 packet·canonical fold를 각각 편집한다. 모형 선택과 값 수정은 결과를 바꾸지 않는다. 유효한 입력을 **입력 적용**으로 제출했을 때만 계산과 독립 검산을 시작한다. 잘못된 입력이면 오류를 표시하고 마지막 적용 입력의 결과를 유지한다. 현재 적용한 모형을 별도로 확인할 수 있다.

기존32예제와 N2…32·12자리 정수/분수 범위를 유지한다. 구간 수·관측 첨자에는 소수·지수·부호 덧붙임·공백·선행0 표기를 허용하지 않는다. ω²의 음수, 정확 초점과 영 pivot, Abel 조절, 두 폭 족과 극소 ℏ/폭을 그대로 지원한다. 폭 선택에 따라 입력 이름을 σ² 또는 r=σ²/ℏ로 바꿔 단위를 구별한다. 예제32개의 한영 표시 이름도 추가했다.

## 복원하는 상태

- 적용 입력과, 적용하지 않았거나 잘못된 값이 남아 있는 편집 초안.
- 13개념 중 선택 단계, 6안내 선택·해설 열림·안내별 예측, 8문제 답변.
- 행렬 종류·복소 성분·척도·선택 성분·전체 표 열림·2D/3D 선택.
- 사슬 모드·영방향 표시 계수, packet 표본·비교 ℏ·분산 log 선택, fold의 scaled/raw/log 선택.

버전이 있는 독립 저장키를 사용한다. 읽기를 마치기 전에 저장값을 덮어쓰거나 계산하지 않는다. 복원할 입력·상태는 엄격한 schema·길이·범위 검사 뒤 자체 소유 불변 객체로 만든다. 손상된 저장값은 설명과 함께 기본값으로 대체한다. 저장이 차단되거나 quota 쓰기가 실패해도 현재 화면에서 계속 실험할 수 있다.

계산 결과·검산 receipt·실행/취소 상태는 저장하지 않는다. 복원한 입력은 실제 Worker에서 다시 계산·검산한다. 비교와33개 Airy sweep은 명시적으로 시작하며, 저장된 표시 선택만으로 자동 실행하지 않는다. GPU 자원·카메라의 순간 회전·수치 raw disclosure의 열림은 세션에 한정한다. 3D 선택은 복원하되 화면에 보일 때만 renderer를 준비한다.

## 검증 범위

Node20.20.2/24.13.0 각17검사로 원래32입력의 draft/저장 왕복, 잘못된 초안과 마지막 모형의 분리, 최대 학습 텍스트·행렬 성분960·표본64, schema/접근자 거부, 한영 SSR·실제 독립 gate를 통과한 세 모형의 controlled 화면·reader 의존성 분리를 확인했다. 수치 producer와 검산 엔진은 reader bundle에 포함하지 않는다. 기존 시각화 회귀63검사(원래32파동 예제·25행렬 예제의 전체 검산과 SSR/의존성·곡선 검사)도 Node24에서 통과했다.

Chromium의 한영/명암/320·1440px8조합에서 세 모형의 새로고침·입력 적용·13개념·6안내·문제24선택지, 편집 비재계산·잘못된 초안·언어 전환·저장 복원·접근성24회를 확인했다. 비교 표시를 복원해도 두 족 계산과 Airy sweep이 시작되지 않으며, 별도 명시 비교는 실제 계산/검산을 완료했다.

추가로 최대/경계4입력의 모바일 복원, 손상 JSON/잘못된 버전·읽기/쓰기 차단·quota 환경, 실제 Worker 오류 재시도·시작 신호 후 취소·실행 중 unmount와 재진입을 검사했다. 최대31×31 행렬의961행 전체 표·마지막 성분·3D 선택이 복원되며, 키보드와 실제 touch의 픽셀 변화·컨텍스트 손실/복구·접근성을 확인했다. 이 표시 조작은 수치 Worker를 만들지 않는다. 기본 흐름100개·추가 최대/저장/수명 흐름33개·3D 복원6개, 추적한 native Worker139개가 모두 종료되었다.

타입/lint 통과. 이번 단계는 실제 native Worker를 사용하는 authoring bundle 검증이다. Next 생산 자산·실제 reader 등록은 다음 단계에서 따로 증명한다. 기존 out과 다른 preview는 보존한다.

## 재현과 다음 순서

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-phase-state.test.ts scripts/learning-real-time-phase-lessons.test.ts scripts/learning-real-time-phase-concepts.test.ts
node --import tsx scripts/preview-real-time-phase-state.ts
node --import tsx scripts/check-real-time-phase-state-native.ts
node --import tsx scripts/check-real-time-phase-state-3d-native.ts
```

증거: `.runtime/learning/real-time-phase-state/verification.json`. 다음은 실제 Interactive Book의 선수·검색·도식·실험 연결과 생산 Worker 자산이다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362검증·47미완료, 전체51영역509주제와 연구 심화의 미완료 범위를 유지한다.
