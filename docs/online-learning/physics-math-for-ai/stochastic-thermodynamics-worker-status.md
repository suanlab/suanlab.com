# 확률열역학: 두 가족의 입력 계약과 Worker 기반

최신: [두 단원의 실제 Interactive Book 통합과 검증](stochastic-thermodynamics-integration-status.md)을 마쳤다. 아래는 해당 준비 단계의 기록이며 전체 연구 범위 완료를 뜻하지 않는다.

2026-11-08. [두 수치 모형과 한영 원고](stochastic-thermodynamics-foundations-status.md)에 계산 전송 기반을 연결했다. **교수 화면·교재 등록·실제 읽기 화면 검증은 아직이다.** 등록543단원·1629안내, 원래297/362입문 검증·65미완료 및 전체51영역509주제·연구 깊이 범위를 유지한다.

## 구현

- `src/lib/learning-thermal-input.ts`: 기존18필드·16필드의 매개변수 계약을 수치 모형에서 분리했다. 모형의 기존 타입·검증기 export와 계산은 보존한다.
- `learning-thermal-protocol.ts`: 두 가족을 구별하는 입력, 고정 순서의 불변 스냅샷·키, JSON 초안/저장 복원, 요청·결과 검증을 제공한다. JSON 숫자가 파싱 중0이나 정수로 반올림되는 입력을 거부한다.
- `learning-thermal-evaluator.ts`와 `learning-thermal.worker.ts`: 올바른 요청에 시작 알림을 보내고 해당 가족을 계산한다. 잘못된 요청·입력·계산 실패를 구별한다.
- `learning-thermal-controller.ts`: 새 요청이 이전 Worker를 종료한다. 취소·해제·오류·결과 수신 시 자원을 정리하며 이전 콜백·다른 요청 결과를 표시하지 않는다. Worker 생성 실패 시 메인 스레드 계산으로 전환하지 않는다.
- `learning-thermal-worker-factory.ts`: 향후 실제 읽기 화면에서 사용할 번들 가능한 Worker 진입점이다.

결과 검증은 관측값의 구간·부호·로그 표현과 정확 유리수 구조를 검사한다. 두 가족의 고정 배열 크기, 단계·원자 순서, 도달 불가능한 끝점의 null, 정준 초기화/JE 표시, 시간 기록 수, 상태·채널 경로와 입력 귀속도 확인한다. 이는 전송 계약 검사이며 임의 송신자의 수학적 정확성을 증명하지 않는다.

## 검증과 한계

`npx tsx --test scripts/learning-thermal-worker.test.ts scripts/learning-driven-thermal-*.test.ts scripts/learning-reservoir-thermal-*.test.ts`에서229검사를 통과했다. 기존 수치·원고121검사를 모두 포함한다. 독립 참조49+50준비의 결과가 structured clone 뒤에도 전송 계약을 만족한다. 작은 양의 열·binary64 미만 확률의 표현도 기존 검산을 유지한다.

수치 엔진 없는 브라우저 대상 controller 번들을 검사했다. 실제 Worker 진입 코드를 브라우저 대상으로 번들한 뒤 **Node worker_threads의 self/postMessage 어댑터**로 실행해 두 가족의 시작/완료 메시지를 확인했다. 추가로 `npx tsx scripts/check-thermal-worker.ts`에서 실제 Chromium Worker로99준비의 시작/완료 전송과2가족의 시작 직후 종료를 확인했다. 검증용 JS 응답은 Playwright 경로로 제공했으므로 Next가 생성한 정적 Worker 자산의 검증은 아직이다. 교체·지연 콜백·취소·해제·오류·재진입 처리는 가짜 포트로 검사했다. 읽기 화면 검증도 별도로 남아 있다. 타입 검사와 lint도 통과했다.

기존 등록 소스4470파일과 정적 출력4321파일의 해시 및 출력 파일 집합이 이전 검증본과 일치한다. 두 번째 독립 참조의 재생성 결과도 저장본과 같으며 기존48준비를 유지했다. 이번에는 새 빌드나 브라우저 교재 검증을 수행하지 않았다. 근거: `.runtime/learning/thermal-worker-integration/verification.json`.

## 다음 구현

1. 두 순간 준위 변화의 에너지 장부·일 분포·열 이완과 두 저장고의 채널 전류·열·상태 관측 대비 그림을 구현한다.
2. 고정 개념·실습·안내·퀴즈, 입력 저장·복원, 계산 대기/실패/재시도 및 모션 정지를 연결한다.
3. 기존 홈페이지 색·서체를 유지해 **도서 → Interactive Book**에서 등록한다. Online Book에 진입 링크를 추가하지 않는다.
4. 실제 한영·명암·모바일·키보드/터치·접근성·Worker 수명주기·정적 출력을 검증한 뒤 입문 주제 완료 여부를 판단한다.
