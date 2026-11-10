# Density matrix: Interactive Book 실제 교재 등록·검증

2026-11-08. 두 density matrix 단원을 실제 교재에 등록했다. **도서 → Interactive Book**에서 목록·표지·본문·실험으로 열 수 있고 검색에도 연결된다. Online Book 목록에는 교재 카드를 넣지 않는다. 기존 URL·북마크와558개 이전 단원/안내의 내용·상대 순서,51개 영역의 분류를 보존했다.

## 등록한 두 단원

- `finite-ensemble-density-gram-tomography`: 일반 density 조건·convexity·극점·ensemble 비유일성·Gram 관계를 유도한다. 유한1…4차원 혼합의 순도/rank·tomography·outside 확률·Lüders 조건부 혼합을 비교한다.
- `qubit-density-spectrum-entropy-dephasing`: 일반 엔트로피·기저 coherence·부분추적·purification을 설명하고 qubit 스펙트럼·부호 있는 dephasing·Choi·거리/fidelity를 실험한다. 복합계와 reset 손계산은 현재 계산기의 추가 입력 기능으로 표시하지 않는다.

각 한영 본문은14절·12개 표시 수식·2개 표·8개 해설 문제다. 합계4편·56절·48개 표시 수식·8개 표·32개 해설 문제이며 번역을 별도 원래 주제로 세지 않는다. 각 단원에 개념3단계·예측3개·안내 실험3개·본문 퀴즈3개를 연결했다.32개 자유 실험 예제와49/11개 전체 정수 입력, 저장·오류·재시도·모션 정책을 제공한다.

목록 표지의 미정의 색상 변수를 발견해 기존 사이트의 muted 테마 토큰으로 수정했다. 한영 목록에서 실제 계산된 배경색이 테마 토큰과 일치하는지 검사했다. 행렬·그림은 좁은 화면에서 내부 스크롤을 제공하며 페이지 자체는 가로로 넘치지 않는다.

## 검증한 근거

- 전체10728검사 통과. 표지의 색상 수정 후 관련14검사를 재실행했다. 타입·lint·콘텐츠·최종 빌드와1798 HTML의 내부 링크/자산/수식 앵커·sitemap·canonical을 확인했다.
- 실제 정적 reader에서 한영×명암×1600/390px×두 단원16조합·접근성16회를 통과했다. 예제32회·개념/예측 각24회·고정 비교124회·안내48회·본문 퀴즈48회·잘못된/유효 입력 각16회·저장 재방문16회·슬라이더32회·메뉴16회를 확인했다.
- 실제3D16회·WebGL 손실/복구4회·행렬 전환24회·조건부 결과 선택16회·키보드/실제 터치 가로 이동 각8회. Reader Worker350개 생성/350개 종료·브라우저 pageerror0. 별도 오류/재시도2회도 통과했다.
- 생산 Worker 파일은 실제 생성된 classic factory와 같은 방식으로 실행했다. 독립81사례의78175수치 관측이 Node 기준 결과와 일치했다. 유리수 경계·부호·표시값·해결 상태는 동일하며 로그 표시만 플랫폼 차이의 상대10⁻¹² 허용을 사용했다. 이전 고정밀 참조의 폭 기준10⁻¹²⁵는 유지한다.
- 실제 클라이언트 전환3회에서 Worker22개를 모두 종료하고 canvas를 해제했다. 한영 목록 카드4개·검색4개·Online Book 제외2회와 이전 spin/unitary reader8회도 확인했다.

초기 모바일 검사는 화면에 맞는 첫 표를 무조건 스크롤하려 해 멈췄다. 실제 scrollWidth가 clientWidth보다 큰 그림/표를 명시적으로 선택하도록 검증 코드를 고쳐 전체16조합을 통과시켰다. UI에 불필요한 넘침을 만들거나 검사를 생략하지 않았다.

## 현재 범위와 다음 순서

현재560개 한영 단원·1680개 안내 실험이다. 원래 입문 주제305/362검증·57미완료이며 quantum 영역은5/11입문 주제를 검증했다. 다음은 **uncertainty·measurement**다.

이는 density matrix의 입문 처리 검증이다. 전체51영역509주제, 일반 차원/복합계 심화·증명·연구 깊이·공용 그림 후속은 계속 남는다. 전체 교육과정이나 원래 전체 목표를 완료로 표시하지 않는다. 로컬 사이트에 반영했으며 공개 배포·commit·push는 수행하지 않았다.

로컬 목록: <http://localhost:4175/book/interactive/>. 두 단원은 기존 `/book/online/physics-math-for-ai/` 읽기 주소를 유지한다.

근거: `.runtime/learning/quantum-density-integration/verification.json`, `full-tests.log`, `final-focused-tests.log`, `build.log`, `typecheck.log`, `lint.log`, `validate-content.log`, `export.log`, `native-reader.json`, `production-worker.json`, `client-navigation.json`, `discovery.json`, `preservation.json`, `export-hashes.json`과 화면 캡처.

```bash
npm test
npm run build
npm run typecheck
npm run lint
npm run validate:content
npm run check:export
npx tsx scripts/check-density-reader.ts
npx tsx scripts/check-density-assets.ts
npx tsx scripts/check-density-client.ts
npx tsx scripts/check-density-discovery.ts
```
