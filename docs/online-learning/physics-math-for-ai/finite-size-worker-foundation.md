# 유한계 세 가족: 취소 가능한 worker 기반

2026-11-07. 세 가족의 계산을 worker로 분리하고 독립 Node·브라우저 환경에서 요청 교체·취소·복구를 검증했다. 실제 Next.js 읽기 화면과 정적 자산의 worker URL 검증, 개념·실습·평가 UI와 등록은 아직 미완료다. 이 기록을 실제 교재 통합 검증으로 해석하지 않는다.

## 코드 경계와 입력

`src/lib/learning-finite-size-{protocol,evaluator,controller,worker-factory}.ts`, `learning-finite-size.worker.ts`와 `src/components/learning/use-finite-size-calculation.ts`를 추가했다. protocol에는 모델의 type-only import만 있고, 실제 세 수치 모델은 evaluator에서 가져온다. hook의 브라우저 번들 의존성 검사는 세 모델·BigInt 산술·초월함수·evaluator가 주 스레드에 포함되지 않음을 확인했다.

요청에는 version·양의 안전 정수 requestId·가족별 입력이 필요하다. 입력은 정확한 키 집합·범위·분모·서로 연결된 제약을 검사한다. snapshot은 입력의 속성 순서와 signed zero를 정규화하고 소유한 객체를 고정한다. 응답은 같은 요청·가족·입력 키와 모든 필수 결과 필드, 기록 순서·끝점, 분포 행·사건 목록, 미제공 값과 복소 근의 지수/회전수를 확인한다.

유리수 점 구간과 로그의 비영 폭 구간을 구별하고,0분모·잘못된 부호/지수·무한 Number·미해결 정밀도·과도한 정수 문자열을 거절한다. 검증기는 구조와 준비 동일성을 확인하며, 전송자가 수학적으로 참이라는 인증은 아니다. 수치 정확성은 각 모델의 독립 기준 검사로 별도로 확인한다.

## 취소와 화면 상태

활성 요청마다 worker 하나를 생성한다. 새 요청은 이전 worker를 종료해 오래 걸리는 계산 뒤에 줄 서지 않는다. 시작 ACK를 받으면 starting에서 calculating으로 이동하고, 정상 결과·오류·취소·교체·해제에서는 이벤트 핸들러를 제거하고 worker를 종료한다. 이전 요청의 저장된 콜백과 늦게 도착한 메시지는 새 상태를 발행할 수 없다.

hook은 렌더 입력 키와 결과 입력 키가 다르면 응답을 노출하지 않는다. 비활성화와 null 입력, unmount에서도 결과를 숨기고 계산을 해제한다. 생성·전송·실행·역직렬화 실패를 별도 상태로 제공하고 재시도를 지원한다. 주 스레드에서 동기 계산으로 대체하지 않는다. SSR은 Worker를 읽지 않는다.

## 검사 근거

`scripts/learning-finite-size-worker.test.ts`의14검사와 관련440검사, 합계454개 및 typecheck·lint가 통과했다. 세 독립 기준의27+32+34=93준비를 실제 evaluator·구조 검증·structuredClone·JSON 경로로 대조했다. 필수 필드 삭제, 응답 신원 변경, 부정확한 사건·근·미제공 값과 오류 봉투도 검사했다.

실제 Node worker에서는 세 가족 각각 시작 ACK 뒤 취소와 새 스레드 복구를 확인해 소유한6개 스레드를 종료했다. 빠른 경계 계산의 CPU를 어느 시점에 중단했다고 주장하지 않는다.

`scripts/check-finite-size-worker.ts`는 임시 포트의 독립 페이지에서 실제 브라우저 Worker와 React StrictMode hook을 실행한다. 정상3·최대 결과3·시작 후 취소3·복구3, 입력 교체·비활성화·생성 실패/재시도·계산 중 해제를 확인했다. 큰 정수 자리 수, 유한하지만 표시 불가능한 꼬리 비와 극소 양의 로그 구간도 전달됐다. worker17개 생성/17개 종료, 남은0개, 이전 입력 결과 노출0개, 페이지 오류0개다. 이 서버와 브라우저는 닫았고 기존4175미리보기는 유지한다.

## 보존과 다음 단계

이전 소스4,269개와 정적 출력4,287개의 바이트·경로 집합을 보존하고 새 소스8개를 추가했다. 증거는 `.runtime/learning/finite-size-worker/verification.json`, `tests.log`, `related-tests.log`, `node-verification.json`, `browser-verification.json`, `hook-metafile.json`이다. 전체 프로젝트 테스트·정적 빌드·실제 Next 교재 검증은 이번 단계에서 실행하지 않았다.

등록530단원·1,590안내와 원래289/362입문 검증·73미완료를 유지한다. 다음은 실제 Next worker 번들과 기존 스타일의 개념6단계·안내3개·최종 문제3개·저장 실습·표지의 연결이다. 실제 읽기 통합 후 열역학 종합 점검과 나머지 영역·연구 심화를 이어간다. 진입은 도서 → Interactive Book이다.
