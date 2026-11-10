# Density matrix: Worker 계약과 실제 브라우저 검증

2026-11-08. [두 계산 가족과 상세 한영 원고](quantum-density-foundation-status.md)에 Worker 전송 계층을 연결했다. 새 계산이나 원고 등록 없이 입력·응답 검증, 취소·교체·종료, React hook과 실제 브라우저 Worker를 준비했다.

## 입력과 응답 계약

두 kind를 구별하여 ensemble의49개 정수와 spectrum의11개 정수 입력을 그대로 사용한다. 스냅샷·정규화된 입력 키·요청 번호로 응답을 요청에 귀속한다. 입력 JSON과 저장 자료의 숫자는 반올림 전 원문을 검사한다. 초기·완료·오류 프레임은 소유 필드만 허용하고 누락·추가·잘못된 배열·다른 가족을 거부한다.

모든 관측량은 유한한 표시값, 정확 유리수 또는 방향성 경계, 부호·로그와 정밀도 상태를 검증한다. 구간 폭 max(1,|경계|)×10⁻¹²⁵와 표시값의 경계 중점 귀속을 확인한다. 잔차는0을 포함해야 하고 정확 계정의 잔차는 구조적0이어야 한다. 넓은 구간을 통과시키거나 null을0으로 바꾸지 않는다.

Ensemble 응답은 raw 입력으로 밀도 성분·가중치·성분 norm·probe·관측량을 귀속하고 순도·분산·Gram minor/특성다항식·tomography·조건부 null·퇴화 결과를 확인한다. Spectrum 응답은 요청한 eta 표본과 정확 영점 삽입, Bloch/density·기저 측정·고유값 경계·퇴화와 log0·Choi 관례/스펙트럼·Kraus 개수·거리 계정을 구별한다.

이 계정은 제한된 유리수 검증이다. Reader에서 고유값 계산·로그·방향성 계산 엔진을 다시 실행하지 않으며 임의의 발신자의 모든 수학적 주장을 증명한다고 표현하지 않는다. 기존 독립 참조와 엔진 검사가 계산의 근거다.

## 수명 관리와 hook

한 활성 요청마다 Worker 하나를 사용한다. 교체·취소·완료·오류·dispose에서 이벤트 핸들러를 해제하고 종료한다. 이전 응답과 이전 callback은 새 입력에 결과를 노출하지 않는다. Worker 생성 실패·전송 실패·messageerror는 명시적 오류가 되며 동기 계산 fallback은 없다.

Hook은 정규 입력 키로 실행하며 렌더 중 새 입력으로 바뀐 순간에도 이전 결과를 보여주지 않는다. SSR에서는 Worker에 접근하지 않는다. 저장 초안의 해석은 추가했지만 실제 저장 UI·수업 화면은 다음 단계다.

## 확인한 근거

- 새 Worker 검사97개 + 이전543개, 총640개·타입 검사·전체 lint 통과.
- 두 가족의 독립 참조81사례를 응답 검증에 통과시켰다. 구조 변조·잘못된 관측 경계/표시값·밀도/측정 계정 바꿔치기·null/퇴화 의미·eta 순서·Choi 정규화·취소와 재진입을 확인했다.
- 실제 browser-native module Worker81개 입력·수치 관측78175개가 Node 기준 결과와 일치했다. 로그 표시의 플랫폼 차이는 상대10⁻¹² 허용이며 유리수 경계·부호·표시값·해결 상태는 동일하다.
- 계산 시작 직후 종료2회. Controller의 cancel2회·교체·실제 오류/재시도·dispose에서 생성7개/종료7개·잔여0개. 브라우저 pageerror0.
- Reader hook과 controller bundle의 의존성에서 수치 엔진·행렬 계산·방향성/초월함수 구현이 제외됨을 확인했다. 실제 Worker entry의 시작·완료 프레임도 별도 thread에서 검사했다.

초기 변조 검사 하나는 같은 확률 둘을 뒤집어 실제로는 응답이 바뀌지 않았다. 서로 다른 유효 관측값으로 바꿔 실패 조건을 검증했다. 수치 엔진·참조·원고는 수정하지 않았다.

## 다음 순서와 전체 범위

이번 브라우저 검사는 source bundle Worker를 격리 URL에 제공한 것이다. 새 Next 생산 자산·실제 reader·접근성·정적 출력 검증은 아직 아니다. 두 단원은 미등록이며 기존558단원·1674안내·304/362입문 검증·58미완료와4360개 정적 파일을 유지한다.

다음은 고정 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book reader 등록 → 생산 Worker 자산·실제 읽기·접근성·정적 출력이다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 넣지 않는다. 전체51영역509주제·일반 차원/복합계 심화·연구 깊이·공용 그림 후속 범위는 유지한다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-density-worker/verification.json`, `combined-tests.log`, `typecheck.log`, `lint.log`, `native-worker.json`, `browser-dependency-graph.json`, `preservation.json`.

```bash
npx tsx --test scripts/learning-density-worker.test.ts scripts/learning-quantum-density-authoring.test.ts scripts/learning-density-spectrum.test.ts scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-density-worker.ts
npm run typecheck
npm run lint
```
