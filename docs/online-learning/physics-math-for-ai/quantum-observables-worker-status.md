# 관측·연산자: Worker 전송과 수명주기

2026-11-08. [두 수치 모형과 한영 원고](quantum-observables-foundation-status.md)에 요청·응답 계약, Worker 실행, controller와 React 훅을 연결했다. **교수 화면과 실제 교재 읽기는 아직 미등록**이다. 등록552단원·1656안내, 원래301/362입문 검증·61미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 입력과 응답

- 두 가족의 전체 소유 필드·정수 범위·양의 분모·비영 원시 벡터를 확인한다. canonical 입력을 복사·동결하고 요청 ID·가족·입력 키를 응답과 대조한다.
- JSON 입력·저장값은 숫자의 원래 문자열도 검사한다. `1.00000000000000001`처럼 binary64에서 정수로 반올림되는 소수나 `1e-9999`의 비영 underflow를 정수 준비로 받지 않는다.
- 결과의 전체 구조·배열 길이·소유 원소·행렬 차원·정확/구간 관측 형식을 확인한다. 구간 순서·최대4096자리 계수·분모·반올림 표시와 magnitude-log도 검사한다.
- C² 고유값과 rank의 짝, 축퇴 병합, 평균·이차 모멘트·분산, 수동 기저 norm과 평균을 대조한다. C³는 coarse/세분화 분기·표지, 결합 확률의 행/열 주변합·조건부 법칙·영 분기, coherence 차이와 commutator 진단을 확인한다.

이 검사는 전송 형식·입력 귀속·스칼라 회계의 일관성을 확인한다. 임의 송신자가 만든 모든 사영 행렬의 수학적 진실을 증명하는 것은 아니다. 실제 수치 엔진은 별도의 독립 참조로 검산하며, 응답 검사에서 행렬 사영이나 수치 엔진을 다시 실행하지 않는다.

## 취소·교체·오류 복구

계산은 요청별 소유 Worker에서 실행한다. 새 준비·취소·완료·오류·unmount에서 Worker를 종료하고 handler를 해제한다. 늦은 응답과 이미 종료한 port의 callback이 새 상태를 덮어쓰지 못하게 한다. Worker 사용 불가·구조화 복사 실패·오류·손상된 응답은 명시적 오류 상태로 처리하고 재시도할 수 있게 한다. 주 스레드 동기 계산으로 대체하지 않는다.

React 훅은 canonical 입력 키로 계산을 구동한다. 입력 변경 직후, 교체 effect가 실행되기 전에도 이전 결과를 새 준비의 결과로 노출하지 않는다. SSR에서는 Worker를 읽거나 만들지 않으며 응답은 null이다. 훅·controller의 브라우저 의존성 그래프에는 두 수치 엔진·복소 행렬 기반·Hilbert 정규화 core·초월함수 계산이 들어오지 않는다.

## 확인한 근거

- 기존 수치·집필123검사와 새 전송·수명주기·SSR64검사, 합계187검사 통과. 타입·전체 lint 통과.
- 독립 참조의 C²24준비·C³26준비 모두 왕복 검증. 새 필드 누락·장식/희소 배열·잘못된 분기·음수/동일 고유값·거짓 주변합·손상 구간·표시값 등을 거절하는 검사를 포함한다.
- 실제 Chromium Worker에서50준비 실행. 시작 메시지와 최종 응답을 확인하고 수치 관측30260개를 Node에서 검산한 엔진 결과와 대조했다. 정확 경계·부호·표시값·해결 상태를 비교하고 magnitude-log는 플랫폼 수치 차이 허용 범위로 대조했다.
- 시작 직후 종료2회, controller 취소2회·가족 교체·의도한 native Worker 오류 후 재시도·dispose 확인. controller 생성7개·종료7개·잔존0개. 페이지 오류 없음.
- Node Worker 진입점과 SSR, 호출자 입력 변경·reentrant 구독자·완료 후 재진입·늦은 callback·오류 프레임의 처리도 확인했다.

브라우저 검증은 기존 로컬 서버와 같은 출처에서 현재 소스로 bundle한 Worker/controller를 실행했다. **Next.js가 내보낸 새 Worker 자산이나 실제 교재 읽기 화면을 검증한 것은 아니다.** 원래 수치 모형·참조·한영 원고와 기존 등록 교재·정적 출력은 보존했다.

## 다음 순서

두 가족의 고정 개념 비교·예측·해설과 자유 실험·안내·퀴즈·저장 복원·모션을 구현한다. 이어 도서 → Interactive Book에 본문·표지·목차·검색을 등록하고 한영·명암·모바일·키보드/터치·접근성·정적 출력을 검증한다. 그 뒤에만 관측·연산자 입문 완료 수를 갱신한다. Online Book 목록에는 새 교재 카드를 추가하지 않는다.

일반 POVM·측정 장치·혼합 상태 입력·연속 스펙트럼·연산자 정의역·연구 심화는 계속 남는다. 다음 원래 주제는 unitary evolution·Schrödinger이며 나머지 영역의 원래 순서도 유지한다.

새 빌드·공개 배포·commit·push는 수행하지 않았다. 근거는 `.runtime/learning/quantum-observables-worker/verification.json`, `native-worker.json`, `browser-dependency-graph.json`, `tests.log`와 보존 기록이다.

```bash
npx tsx --test scripts/learning-observable-worker.test.ts scripts/learning-quantum-observable*.test.ts scripts/learning-quantum-hilbert.test.ts
npx tsx scripts/check-observable-worker.ts
npm run typecheck
npm run lint
```
