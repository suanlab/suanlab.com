# 실시간 경로적분: 실제 Worker와 수명 관리

2026-11-11. [전체 응답 검산](real-time-phase-whole-response-validator-status.md)을 실제 계산/독립 검산 module Worker와 controller·React hook에 연결했다. 현재 단원은 **15/23 요건 검증**이며, 개념도·학습 화면·reader·생산 자산·범위 감사는 아직 미완료다.

## 메시지와 실행 분리

계산 adapter는 소유 입력을 읽고 started를 보낸 뒤 결과를 계산한다. 성공 결과는 여전히 unknown인 **미검산 응답**이다. 별도 검산 adapter는 verification-started를 보내고 bounded copy와 전체 수학 gate를 통과한 뒤에만 fullResponseValidated receipt를 발급한다. 입력 실패·요청 실패·수치 실패·복사 resource-limit을 구별한다.

reader hook/controller는 계산·검산 엔진이나 BigInt 연산을 import하지 않는다. SSR은 idle을 반환하며 Worker/document를 읽지 않는다. 두 Worker 번들과 hook의 esbuild dependency graph로 분리를 확인했다. 현재 증거는 실제 브라우저용 ESM 번들이며, Next 생산 자산과 reader 등록을 대신하지 않는다.

## 요청 소유권과 기한

요청마다 계산 port와 신뢰하는 검산 port를 소유한다. 계산 종료 시 계산 port를 해제하고, 검산 종료 시 검산 port도 해제한다. 검산 전·ready 반영 전의 큰 복사는 chunk마다 현재 token을 확인하고 yield한다. structured clone은 freeze를 보존하지 않으므로 검산 port에서 받은 값도 main thread에서 다시 소유·freeze한다.

취소·입력 변경·disabled·hidden 이벤트·unmount·오류·기한 종료는 이전 token과 두 port를 폐기한다. 잡힌 이전 callback이나 늦은 메시지는 ready를 만들 수 없다. render-time inputKey 비교로 effect 교체 전에도 이전 입력의 결과를 숨긴다. 중복 started와 검산 진행 신호가 기한을 초기화하지 않는다. 요청 sequence의 safe-integer 한도에서도 이전 job을 먼저 중단한다.

생산 기본 기한은 요청 전체120000ms다. 이는 실패/취소의 상한 계약이며 모든 기기의 완주 시간을 증명하는 수학적 상계가 아니다. 원래 N2…32·12자리 입력 범위와 wire 한도를 줄이지 않았다.

## 발견하고 수정한 이벤트 호환성

Chromium에서 속성 대입만 한 messageerror가 전달되지 않는 것을 추가 오류 검사로 재현했다. EventTarget listener 등록으로 바꾸고 port 해제 시 listener를 제거한다. 실제 EventTarget이 없는 최소 test port에만 속성 fallback을 쓴다. 양쪽 port에서 listener 제거 후의 늦은 이벤트도 무시함을 확인했다.

처음 실패한 native 오류 로그와 진단 상태를 보존한다. 수정 후 비동기 script 오류2개·실제 native DataCloneError1개·잘못된 URL constructor 실패1개·native port의 synthetic messageerror2개 모두 오류 상태와 정상 재시도로 복구했다. synthetic messageerror를 실제 decode 손상 유발로 부르지 않는다. [기존 controller45개 정적 후보의 공통 후속 점검](../worker-messageerror-compatibility-follow-up.md)은 별도로 남는다.

## 실행한 검사

- Node.js20.20.2/24.13.0의 새37검사: 세 mode adapter, 미검산/검산 분리, 잘못된 입력·귀속·getter·복사 한도, 취소/교체/오류/기한·재시도, reentrant 취소, ID 한도, EventTarget listener 해제, SSR과 bundle 의존성.
- Chromium의 실제 module Worker32입력: 전체 결과가 Node producer hash와 같고, 수치 변조32개·귀속 변조32개를 거부했다. calculation-started32회·verification-started96회를 확인했다. 반복 사용하는 native port2개를 모두 종료했다.
- 실제 hook: 최대 혼합 서로소 N32 입력의 계산 started·검산 started 후 타이핑과 취소, 늦은 ready 부재·재시도·Node hash 일치·main-thread freeze를 확인했다. mode 교체·키 순서·invalid/disabled·visibility 이벤트·post 실패 복구·pending unmount도 확인했다. 이 흐름의 port29개가 모두 종료되었다.
- 실제 무한 루프 계산/검산 Worker 각각을10000ms test deadline으로 종료하고 같은 controller에서 정상 재시도했다. visibility는 이벤트 시뮬레이션이며 실제 OS의 background-tab 전환 검증은 아니다.
- 별도 오류 probe의 native port19개를 모두 해제했다. 최종 typecheck/lint 결과와 source/번들 hash는 verification artifact에 기록한다.

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-phase-workers.test.ts scripts/learning-real-time-phase-controller.test.ts
node --import tsx scripts/check-real-time-phase-workers-native.ts
node --import tsx scripts/check-real-time-phase-hook-native.ts
node --import tsx scripts/check-real-time-phase-errors-native.ts
```

증거는 `.runtime/learning/real-time-phase-worker/verification.json`, native report·dependency graph·양쪽 Node 로그다. 기존 root out/preview 서버는 변경하지 않았다. 공통 오류 호환성 점검 후 기존 스타일의 개념도·위상/행렬·packet/fold 실험·설명·문제·접근성, Interactive Book reader 등록과 생산 build/보존 감사를 진행한다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362 검증·47미완료와 전체51영역509주제·영역 감사·연구 심화는 유지한다.
