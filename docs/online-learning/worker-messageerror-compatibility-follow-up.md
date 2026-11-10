# Worker messageerror 공통 호환성 점검

2026-11-11. 기존 정적 후보 **45개 controller**와 신규 실시간 경로적분 controller의 이벤트 등록·해제를 공통화했다. 현재 **46개 controller / 66개 계산·검산 등록 경로**를 점검했으며 이 공통 후속 항목은 완료했다. 입문 주제의 완료 수에는 포함하지 않는다.

## 수정 내용

`src/lib/learning-worker-message-error.ts`는 native Worker에 `addEventListener('messageerror', …)`를 등록하고 port 종료 시 listener를 제거한다. EventTarget이 없는 최소 test port에는 기존 속성 fallback을 유지한다. 재등록은 이전 listener를 먼저 해제하며 계산·검산의 기존 요청 귀속 검사와 오류 분류는 유지한다.

Chromium에서 `onmessageerror` 속성 대입만으로 이벤트가 연결되지 않는 것을 확인한 것이 수정의 계기다. 재현 명령은 `node --import tsx scripts/check-worker-messageerror-capability.ts`다. 실제로 손상된 structured clone을 만들어 decode 실패를 유발한 검사는 아니다. [HTML Worker 인터페이스](https://html.spec.whatwg.org/multipage/workers.html#worker)의 EventTarget 경로로 등록한다.

## 검증

- Node.js 20.20.2/24.13.0 신규 각 68검사: 모든 controller의 계산 오류·재시도·교체·취소·dispose, 20개 이중 Worker controller의 검산 port 오류, 기한을 지원하는 19개 controller의 계산 기한 종료, 최소 port fallback과 재등록을 확인했다. many-body controller에는 기존 기한 계약이 없어 기한 검사를 적용하지 않았다.
- 기존 이중 Worker controller 415검사와 기존 단일 Worker의 오류·취소 등 선택 회귀 93검사를 통과했다. 새 검사를 합하면 576개다.
- 실제 Chromium module Worker에서 46개 controller의 계산 이벤트, 20개 검산 이벤트, 재시도·취소·늦은 이벤트를 확인했다. 생성 132개 / 종료 132개, page error 0개다. 오류 이벤트와 검산 단계 진입용 메타데이터는 synthetic이며 이 검사를 수치 검산의 증거로 쓰지 않는다.
- 실시간 경로적분의 기존 native 오류 6흐름을 다시 확인했다. 비동기 script 오류·native DataCloneError·생성 실패와 synthetic messageerror 이후 모두 정상 재시도했고 port 19개를 종료했다.
- ESLint·타입 검사, 별도 후보 checkout의 정적 build와 1,838 HTML의 링크·자산·fragment·사이트맵·canonical 검사를 통과했다. 최근 saddle 단원 영어/light의320·1440px reader2상태·개념20회·문제16회·접근성2회도 통과했다.
- 범용 reader 검사에서 실험 화면이 아직 없는 경우를 준비 완료로 처리하는 대기 조건을 발견해 수정했다. 기존 출력과 후보 출력의 저장/새로고침 복원은 각각4회 확인했다. 최초 실패 로그를 보존하고 전체 회귀를 재실행해 한영 레이아웃·접근성70상태와 상호작용17그룹을 모두 통과했다.

## 재현 및 증거

```bash
node --import tsx --test scripts/learning-worker-message-error.test.ts
node --import tsx scripts/check-worker-messageerror-controllers-native.ts
```

두 번째 명령의 기본 preview 주소는 `http://127.0.0.1:4204`다. 경로를 Playwright로 가로채 테스트하며 홈페이지 파일을 변경하지 않는다. 입력은 `scripts/fixtures/worker-messageerror-inputs.json`에 고정했다. `.runtime/learning/worker-messageerror-compatibility/verification.json`과 `native-controllers.json`, `.runtime/worker-messageerror-*.log`에 실행 결과와 hash를 보관한다.

후보 checkout은 `/tmp/suanlab-worker-eventtarget-candidate-20261111`이다. 기존 root `out`과 preview 서버는 보존했다. 다음은 실시간 경로적분의 개념도·학습 UI·reader 통합이다. **15/23 요건**, **315/362 입문 검증·47미완료**, 등록 **600단원**을 유지한다. 전체 51영역 509주제의 감사·연구 심화도 미완료다.
