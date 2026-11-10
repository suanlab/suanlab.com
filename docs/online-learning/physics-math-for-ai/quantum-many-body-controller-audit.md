# 다체계 계산·검증 controller와 반응성 감사

2026-11-09. `ManyBodyCalculationController`와 `useManyBodyCalculation`을 구현했다. 계산·독립 수학 검증·큰 응답의 소유 복사 중에도 취소·입력 교체가 가능하게 한다. 실제 교재 등록과 Next Worker 자산 검증은 별도로 남아 있다.

## 실행 경계와 신뢰 모델

한 요청은 계산 Worker와 검증 Worker를 차례로 사용한다. 계산 Worker의 raw success는 화면에 공개하지 않는다. 검증 Worker는 여섯 독립 검증기 중 입력 종류에 맞는 검증기를 실행하며, 구조·입력 귀속·수학 검증을 통과한 응답만 receipt로 반환한다. Main controller/hook 번들에는 생산 수치 엔진과 수학 검증기가 포함되지 않는다. 계산을 main thread에서 대신 실행하는 fallback도 없다.

Controller는 애플리케이션이 생성한 **신뢰하는 검증 Worker factory**를 사용해야 한다. Main의 `manyBodySuccessHeader`는 이 receipt의 ID·종류·입력 key·입력 snapshot을 확인하는 transport 검사이며, 수학 검증기를 대체하지 않는다. 임의의 외부 포트가 보낸 receipt 문자열을 인증 토큰으로 취급하지 않는다. 계산 포트에서 위조한 결과가 실제 별도 검증 Worker를 거쳐 거부되는 것을 브라우저에서 확인했다.

## 요청 수명과 취소

각 run은 증가하는 안전한 request ID와 소유·동결 입력 snapshot을 가진다. 상태는 idle → starting → calculating → validating → ready 또는 error다. Validating에는 raw 복사·독립 검증·반환 복사가 포함된다. 새 요청·cancel·dispose는 이전 포트의 handler를 분리하고 Worker를 종료한다. 이전 token의 메시지·오류·복사 continuation은 현재 상태를 변경하지 못한다.

Factory·notify의 재진입, 시작 전 취소, 계산 중 입력 교체, raw 복사 중 취소, 검증 중 취소·재시도, 결과 복사 중 취소·늦은 receipt·clone 오류·postMessage 실패도 검사한다. React hook은 새 입력 key의 render에서 이전 ready 결과를 즉시 숨긴다. StrictMode cleanup과 SSR import/render도 Worker를 누출하지 않는다.

## 큰 응답의 소유 복사

`snapshotManyBodyWireInTasks`는 descriptor만 읽는 깊이 우선 generator다. 원래 동기 snapshot과 같은 plain record·dense array·symbol/상속/accessor 거부·alias 허용·cycle 거부·깊이64·노드500000·배열65536·record512·문자열8192 경계를 유지한다. 반환은 중첩 동결한 별도 소유 복사다. Getter/get trap을 실행하지 않고 `__proto__`도 소유 key로 보존한다.

한 작업은 최대2048노드 또는 약4ms를 처리한 뒤 일반 timer task에 양보한다. 모든 continuation에서 활성 token을 재확인한다. 초기 `scheduler.yield()` 기반 구현은 개별 긴 task가 없어도 timer가 뒤로 밀려 큰 boson 결과에서 약587–620ms heartbeat gap을 보였다. 일반 timer로 양보하도록 바꾼 뒤 같은 큰 입력에서 최대68.4ms로 줄었다. 이는 해당 로컬 harness 측정이며 모든 환경·실제 교재의 성능을 보장하지 않는다.

## 검증과 한계

새30검사·기존255회귀·타입·lint를 확인했다. 여섯 종류의 정상/위조 계산 dispatch, task copy의 경계·취소·일반 timer 실행과 SSR 렌더를 검사한다. React StrictMode harness의 큰 여섯 입력·위조 raw 거부·복사/검증 중 취소·재시도·교체에서 Worker21개 생성/종료가 일치했다. 이전 종류의 ready 결과가 새 입력 아래 보인 경우는0이었다.

동일 harness의 검증 중 타이핑13–19ms, raw 복사 취소57ms·검증 취소41ms, pending heartbeat 최대68.4ms를 기록했다. Harness의150ms 예산과 비교하며 약50–61ms task도 별도로 기록한다. 실제 Next 자산 URL 대신 preview-origin Blob Worker를 사용했다. 실제 reader·시각화·안내·퀴즈·저장·접근성·생산 자산·정적 출력 통합은 다음 단계다. 기존 홈페이지 출력과 Interactive Book 진입 경로를 보존했다.
