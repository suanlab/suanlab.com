# Density matrix: 개념·실험·시각화 화면

2026-11-08. [두 수치 가족과 원고](quantum-density-foundation-status.md), [Worker 검증](quantum-density-worker-status.md)에 한영 학습 화면을 연결했다. 아직 reader에는 등록하지 않은 source-component 단계다. 기존 홈페이지의 CSS 변수·버튼·표·명암 테마와 모션 정책을 사용한다.

## 두 학습 화면

유한 ensemble 화면은1…4차원 예제16개와 전체49개 정수 입력을 제공한다. 밀도 rho·Gram·가중 열 B를 전환하고 준비/기준 확률, 순도·분산 분해, X/Y 쌍 tomography와 outside 확률, Lüders 조건부 밀도와 순도를 비교한다. 불가능한 결과는 조건부 null로 표시하며 퇴화한 혼합 결과와 구별한다. 영 가중치·반대 대표·거의 나란한 rank2도 포함한다.

혼합 qubit 화면은 예제16개와 전체11개 정수 입력을 제공한다. 매개변수 슬라이더로 eta1부터 끝점까지 이동하며 entropy bit/nat·기저 대각 entropy·coherence·squared Uhlmann fidelity·trace distance를 읽는다. 고유값과 중복도, log0/null, 퇴화 사영, I/Z 확률, Choi 스펙트럼/rank·Kraus를 제공한다. 곡선 가로축은 실제 u값이며 영점 삽입 표본을 등간격으로 잘못 그리지 않는다.

Bloch ball은 실제 혼합 벡터 길이를 유지하며3D 카메라 조작·2D 투영·WebGL 복구를 제공한다. d≠2에는 qubit 그림을 적용하지 않는다. 행렬·곡선·확률 막대에는 숫자·label·선 모양과 키보드/터치 스크롤이 있고, 정확 유리수/방향성 경계와 모든 잔차는 별도로 펼쳐 볼 수 있다. 작은 행렬 폭은 실제 열 수에 맞췄다.

## 교수 흐름과 저장·모션

각 단원에 고정 개념3단계·예측 문제3개·안내 실험3개를 준비했다. 같은 rho의 여러 준비, Gram/순도/rank, tomography/조건부 혼합, Bloch ball/log0, 기저 coherence, 부호 있는 채널/거리로 이어진다. 고정 비교는 저장된 자유 실습과 독립적으로 Worker에서 계산한다.

유효한 준비만 가족별로 저장한다. 입력 오류·원문 숫자 underflow·손상된 저장 자료·저장 불가·Worker 오류와 재시도를 구별한다. 설정 변경 시 이전 결과를 즉시 숨기고 실행 중단을 제공한다. 표본 재생은 읽기 안내이며 물리 시간이나 상태 붕괴가 아니다. 동작 줄이기·사이트 정지·화면 밖·숨은 탭에서는 자동 진행을 멈추고 수동 선택은 유지한다. SSR은 Worker와 WebGL을 시작하지 않는다.

## 확인한 근거

새 교수 화면 검사9개와 이전640개를 합친649개·타입 검사·전체 lint를 통과했다.32개 예제의 독립 참조 입력 일치와 유일성·전체 입력 label·숫자 검증·개념/해설·손계산 비교·SSR·Worker 전용 수치 의존성을 확인했다. 중복 입력 예제 하나는 복소 qutrit으로 교체했다.

한영×명암×320/1600px×두 가족의16조합에서 실제 source-component 화면과 module Worker를 검사했다.16개 접근성 검사에서 위반0, 영어의 한글 부재와 페이지 가로 넘침 부재를 확인했다. 예제64회·고정 단계/예측 각12회·안내12회·잘못된 초안20회·저장 복원16회·Worker 오류/재시도4회·손상 저장4회·쓰기 실패4회·키보드 스크롤8회·실제 터치4회를 포함한다. 모션12회·화면 밖 정지4회·실제3D8회·WebGL 손실/복구4회와 Worker350개 생성/350개 종료·잔여0개도 확인했다. 상세 실측 개수는 runtime 보고서를 근거로 한다.

행렬의 모바일 여백을 직접 검토하여 고정 폭을 수정한 뒤 최종16조합 검증을 반복했다. 스크롤 뒤의 화면 캡처는 처음 위치로 복원하여 읽기 상태도 확인했다. 위 검증은 독립 수치 참조나 최종 Next reader 검증을 대신하지 않는다.

## 등록과 다음 단계

고정 개념6단계·예측6개·안내 실험6개는 준비 상태이며 전역 목록에는 아직 추가하지 않았다.558단원·1674안내·304/362원래 입문 주제 검증·58미완료와 기존4360개 정적 출력은 유지한다. 두 수치 엔진·참조·원고·Worker 계층과 기존 교수 화면을 보존했다.

다음은 교재 본문/목록·안내·퀴즈·검색·표지 등록 → 실제 Next 생산 Worker·reader → 접근성·키보드/터치·정적 출력 검증이다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다. 전체51영역509주제·일반 차원/복합계 심화·연구 깊이·공용 그림 후속 범위는 유지한다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-density-teaching/verification.json`, `combined-tests.log`, `typecheck.log`, `lint.log`, `browser-verification.json`, `browser-dependency-graph.json`, `app-bundle.js`, `worker-bundle.js`, 화면 캡처와 `preservation.json`.

```bash
npx tsx --test scripts/learning-density-interface.test.ts scripts/learning-density-worker.test.ts scripts/learning-quantum-density-authoring.test.ts scripts/learning-density-spectrum.test.ts scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-density-interface.ts
npm run typecheck
npm run lint
```
