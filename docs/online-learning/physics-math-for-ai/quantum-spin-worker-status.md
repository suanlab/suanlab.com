# Spin·angular momentum: Worker 연결 검증

2026-11-08. [두 계산 가족과 상세 한영 본문](quantum-spin-foundation-status.md)에 입력/응답 계약·evaluator·실제 Worker entry·controller·React hook을 추가했다. 교수 화면과 교재 등록은 다음 단계다.556단원·1668안내·303/362입문 검증·59미완료는 유지한다.

## 계산과 화면의 경계

`learning-spin-protocol.ts`는 스핀1/2의17정수와 유한 ladder의21정수를 구분하고, 입력 소유권·범위·비영 상태·비활성 좌표0을 확인한다. 입력은 복사·동결하며 canonical key로 식별한다. JSON 초안·저장 복원은 숫자가 반올림되기 전의 정수 조건·underflow·범위·저장 버전을 검사한다.

모든 응답 필드·고유 속성·배열 길이/성분·유리수 끝점·그림용 수치·단위와 입력 일치를 검사한다. 보고 구간의 폭은 max(1,|끝점|)×10⁻¹²⁵ 이하다. 반각 가족은 표본 index·분수·부호 있는 turn·정확 초기 방향과 측정 계정을, ladder 가족은 차원·binomial metric·Casimir·Jz 분포·평균/분산 계정·ladder norm·영 image의 null·chain 적용 횟수와 action^(2k) 단위를 확인한다. 입력 일치와 유한 scalar 계정 검사이며 임의 송신자의 회전 동역학을 증명하거나 계산 엔진을 다시 실행하는 검사는 아니다.

`learning-spin-evaluator.ts`와 `learning-spin.worker.ts`만 두 수치 엔진을 실행한다. reader hook과 controller의 브라우저 의존성에서 spin 엔진·삼각함수·구간 진화·행렬 대수는 제외된다. 동기 fallback은 없다. SSR은 Worker를 생성하지 않으며 mounting 이전에는 idle이다.

## 수명주기

요청마다 Worker 하나를 만들고 새 요청은 이전 작업을 종료한다. 시작/계산 중/완료/오류 상태를 구분한다. 오래된 callback과 다른 요청의 응답은 현재 결과를 바꾸지 못한다. 성공·실패·취소·unmount 모두 handler를 해제하고 종료하며, 실패 후 재시도도 새 요청으로 시작한다. 구독자의 재진입과 factory 안의 취소에서도 Worker가 남지 않는지 검사했다.

hook은 render 직후의 입력 key와 완료 응답을 비교한다. 입력 교체 effect가 실행되기 전에도 이전 결과를 새 입력 아래 노출하지 않는다. 실제 교수 화면에서의 입력·저장·모션 상호작용은 아직 검증하지 않았다.

## 확인한 근거

- 새 Worker97검사 + 이전 집필·수치309검사, 합계406검사 통과. 타입·전체 lint 통과.
- 독립 참조에 사용한32개 spin-half와50개 ladder 준비 모두 transport 검사를 통과했다. 필드 삭제·성분 누락·symbol 장식·변조된 norm/metric/분포/단위·넓은 잔차·잘못된 요청·숫자 초안·stale 결과·재진입·오류 복구를 포함한다.
- 번들한 실제 Worker entry를 Node worker thread에서 실행해 두 가족의 started/완료 응답을 확인했다. React SSR에서 Worker 접근0회를 확인했다.
- 브라우저의 native module Worker82입력에서 Node의 같은 엔진과166049개 수치 관측의 끝점·값·부호·해결 상태가 일치했다. log는 플랫폼 오차 허용 범위로 비교했다. 독립 기호/고정밀 참조의 수학 검증은 이전 수치 검사에서 유지하며 이 native 비교와 구별한다.
- 시작 직후 종료2회와 실제 controller의 취소2회·요청 교체·실패 후 재시도·dispose를 확인했다. controller가 생성한7개는 모두 종료됐고 live0·페이지 오류0이었다. 직접 검증한82개와 시작 직후 종료2개도 각 경로에서 종료한다.

브라우저 확인은 기존 로컬 서버에서 별도로 번들한 검증 asset을 제공해 수행했다. Next 생산 asset·새 reader·접근성·새 static export를 검증했다는 뜻은 아니다. 이전 수치 엔진·독립 참조·4개 본문·화면·정적 출력4354파일은 보존했다.

## 다음 순서

고정 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 단원 등록·검색 → Next 생산 Worker·실제 읽기·한영·명암·모바일·키보드/터치·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 교재 카드를 넣지 않는다. 실제 읽기 통합 이후 원래 spin 주제의 입문 완료 수를 갱신한다.

전체51영역509주제·분량/주차 제한 없는 연구 깊이·궤도/정의역/결합·공용 그림 후속·양자 이후의 모든 영역은 계속 진행한다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-spin-worker/verification.json`, `combined-tests.log`, `native-worker.json`, `browser-dependency-graph.json`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-spin-worker.ts
npm run typecheck
npm run lint
```
