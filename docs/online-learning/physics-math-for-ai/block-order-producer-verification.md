# 국소 질서변수: 계산 응답·독립 검산·Worker 검증

2026-11-11. `microscopic-blocks-and-local-order-parameters`는 **등록 전 준비 중**이다. 등록602단원, 검증된 입문 원주제316/362, 미완료46개를 유지한다. 활성 미등록 원고는 한 단원이며, 실제 reader·시각화 검증은 아직 남았다.

## 구현과 독립성

- `learning-block-order-input.ts`: reader에서 쓸 입력을 엔진과 분리했다. descriptor로만 복사하며 getter·사용자 iterator·toJSON을 실행하지 않는다. 양의 활동도를 기약분수로 만들고 입력과 배열을 동결한다.
- `learning-block-order-evaluator.ts`: Gray-code 기반 정확 계산과 생산측 로그 오차 구간을 반환한다. 프로필을 **128개씩 chunk**로 보내므로 625개 모두 기존 wire 한도 안에 들어간다. 오래된 wire 계약을 완화하거나 확률이 작은 행을 버리지 않는다.
- `learning-block-order-validation.ts`: 별도 행 transfer로 모든 상태 수 다항식을 재구성한다. 선택 미시 평균은 직접 이진 열거 후 위치별 다항식으로 평가한다. 생산 fraction·Gray 갱신·로그 모듈·참고 fixture를 import하지 않는다.
- `learning-block-order-controller.ts`: 계산 Worker 종료 후 독립 검산 Worker를 생성하고, 양쪽 응답을 양도받는 복사를 task로 나누어 처리한다. 취소·교체·dispose 시 복사도 중단한다. 요청 ID·canonical 입력 key의 일치를 수학 검산과 구별한다.
- `use-block-order-calculation.ts`: 렌더 시 입력이 달라진 이전 결과를 숨기고, hidden 상태에서 취소한다. SSR에서는 Worker를 만들지 않는다. 실제 reader에서의 hook·visibility·저장 복원 검사는 다음 단계다.

## 로그와 엔트로피

정확한 유리수 값과 초월함수 표시를 분리한다. 생산 계산은1024bit 내부 구간을512bit dyadic 구간으로 내보낸다. 독립 검산은 별도2048bit 산술로 포함 관계를 확인한다. canonical 분수, dyadic denominator, 순서, 최대6 quantum 폭을 검사한다.

표시는 상대/절대 log Z, F/T, 미시 엔트로피, 프로필 엔트로피, 평균 숨긴 엔트로피, 선택 프로필의 숨긴 엔트로피·F/T·평균 E/T, 평균곱 기준의 로그 가중치 잔차, 조건부 cross ratio 로그다. 마지막 값은 블록 쌍이 없을 때만 null이다. 단순 double 오차를 정확한 유리수 결과로 표시하지 않는다.

독립 로그 검산은 절대 log Z를 `ln(Z²)/2`로, 선택 F/T를 `−ln P(k)−ln Z`로, 선택 E/T를 원래 에너지의 조건부 평균으로 계산한다. 프로필 엔트로피도 생산측의 `−ΣP ln P` 대신 제한 다항식 기대값으로 재구성한다. 수학적으로 동등한 식과 서로 다른 산술을 대조한다.

## 검증 결과

| 항목 | 직접 확인한 증거 |
|---|---|
| 전체 응답 비교 | 새 Python 참고 계산24개 + 이름 붙인 예제9개 + 최대 활동도 모형1개 |
| 정확 결과 변조 | 작은 예제의 모든 scalar leaf187곳 거절 |
| 표시 구간 변조 | null·거짓0·큰 폭·비기약·비dyadic·역전 구간66건 거절 |
| 최대 응답 | 16스핀·625프로필, chunk 길이128/128/128/128/113, JSON980,986문자, 마지막 프로필 변조도 거절 |
| Node20 | 기반·원고·프로토콜·컨트롤러·공유 EventTarget 검사100개 통과 |
| Node24 | 같은 범위의100개 검사 통과; 입력 분리 후 기반·원고8개 재확인 |
| 타입·lint | 전체 TypeScript, 변경 뒤 native probe 타입 검사, 신규·변경 TS/TSX lint 통과 |

새 컨트롤러의 대표 입력을 `scripts/fixtures/worker-messageerror-inputs.json`에 추가했고 이전 객체들은 보존했다. 공유 EventTarget72개 검사는 실제 `addEventListener('messageerror')` 경로와 해제를 포함한다. property fallback만 통과한 것으로 native 경로를 대신하지 않는다.

## 실제 Chromium Worker

```bash
node --import tsx --test --test-concurrency=2 scripts/learning-block-order-foundation.test.ts scripts/learning-block-order-manuscripts.test.ts scripts/learning-block-order-protocol.test.ts scripts/learning-block-order-controller.test.ts scripts/learning-worker-message-error.test.ts
node --import tsx scripts/check-block-order-workers-native.ts
```

native probe는 intercepted localhost 페이지에서 실제 module Worker 두 종류를 로드한다. 34개 모형의 전체 응답 SHA-256이 Node와 같으며, 최초24개는 실행 때 생성한 Python 참고값과도 대조했다. 계산·표시·attribution 변조, wire 한도, Worker 예외, 양 단계 시간 초과, 양 단계 취소·decode error, 같은 컨트롤러의 재시도와 진행 중 입력 교체를 확인했다.

생성한 **90개 Worker가 모두 한 번 종료**되었고 handler와 messageerror listener가 해제됐다. 최종 활성0, 동시 최대1이었다. 계산 중 텍스트 입력이 유지되고942 animation frame이 진행됐다. frame 수는 이 실행의 관찰값이며 성능 보장은 아니다. 브라우저 pageerror는0이다.

재현 기록은 `.runtime/learning/block-order-protocol/`과 `.runtime/learning/block-order-workers/verification.json`에 있다. 브라우저는 이 probe가 생성한 것만 종료했다. 기존 preview 서버는 유지했다. 이 probe의 esbuild 자산을 Next 생산 자산이나 실제 홈페이지 reader 검증으로 주장하지 않는다.

## 다음 단계

한영20절 원고는 이미 실제 Markdown 렌더링·Python 예제 검사를 통과했다. 이제 기존 테마의 스핀/블록 공간 그림, 고정 프로필의 미시 배치 변화, 조건부 위치 평균, 정확 합과 국소 기준 비교, 경계에 따른 이웃 판정, 해설 문제를 연결한다. 실제 reader의 반응형·접근성·언어·저장 복원 및 Next Worker 자산 검증 후 등록한다. 통계장론 원주제와 전체 목표는 계속 진행 중이다.
