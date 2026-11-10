# Worker messageerror 공통 호환성 후속 점검

2026-11-11. 실시간 경로적분 Worker의 추가 Chromium 오류 검사에서 `onmessageerror` 속성에 함수를 대입해도 EventTarget 이벤트가 연결되지 않는 환경을 확인했다. 같은 native module Worker에서 `addEventListener('messageerror', …)`는 동작했다. 확인한 것은 synthetic 이벤트의 **등록·전달 경로**이며 실제로 손상된 structured clone을 만들어 decode 실패를 유발했다고 주장하지 않는다.

재현은 `node --import tsx scripts/check-worker-messageerror-capability.ts`다. 결과와 browser version은 `.runtime/learning/worker-eventtarget-capability/probe.json`에 있다. [HTML Worker 인터페이스](https://html.spec.whatwg.org/multipage/workers.html#worker)는 EventTarget을 사용한다. 특정 브라우저의 IDL 속성 지원을 가정하기보다 listener 방식으로 등록하고 종료 시 해제한다.

새 `RealTimePhaseCalculationController`는 EventTarget listener를 사용한다. 최소 fake port에는 속성 fallback을 유지하며, 양쪽 port의 해제·재시도를 Node와 Chromium에서 검증했다.

기존 controller에는 같은 속성 대입 패턴이 **45개 파일**에서 발견되었다. 이는 정적 후보 목록이며 각 factory/adapter의 실제 동작까지 검증한 목록은 아니다. 인벤토리와 source hash는 `.runtime/learning/real-time-phase-worker/legacy-messageerror-followup.json`에 보관한다. 현재 단계에서는 해당 기존 파일들을 수정하지 않았다.

다음 순서로 점검한다.

1. 각 controller의 native port와 오류 등록/해제 경로를 확인한다.
2. EventTarget 등록을 공통 helper로 정리하고 fake port fallback을 유지한다.
3. 계산·검산 port의 decode-error 이벤트, 취소·교체·기한 뒤 listener 해제를 검사한다.
4. 기존 controller 검사와 reader bundle의 계산 모듈 분리를 확인한다.
5. 후보 checkout에서 정적 build·reader 회귀를 확인하고 기존 홈페이지 출력을 보존한다.

이 후속 점검은 입문 **47개 주제**와 별개의 공통 구현 항목이다. 전체 목표 완료 전 해결해야 하며, 기존 600단원의 저장된 검증 기록을 새 native 오류 검사의 증거로 대신하지 않는다.
