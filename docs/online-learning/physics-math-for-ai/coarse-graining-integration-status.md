# Coarse graining: 두 단원의 실제 교재 통합

2026-11-08. 두 수치 가족과 한영 원고를 **도서 → Interactive Book**의 실제 교재 읽기에 등록했다. 등록550단원·1650안내, 원래300/362입문 주제 검증·62미완료다. 비평형 원래10주제의 입문 종합 점검도 마쳤다. 전체51영역509주제와 연구 깊이·공용 그림 후속 범위는 계속 진행한다.

## 등록된 내용

| 단원 | 학습 내용 | 읽기 |
| --- | --- | --- |
| 집계한 상태의 닫힘과 기억 | 모든 초기 법칙의 닫힘, lifting·숨은 좌표, 정확 기억식, 잘린 근사의 음수와 half-L1 차이 | `/book/online/physics-math-for-ai/markov-partition-lumpability-memory/` |
| 숨은 변수의 기억과 Markov 근사 | 연속 소거·초기 층·기억 적분·시간척도, 준정상 준비와 정확한 닫힘의 차이 | `/book/online/physics-math-for-ai/linear-hidden-memory-markov-limit/` |

각 한영 본문은14절·표시 수식10개·비교표2개·풀이6개다. 두 가족의 준비20개, 개념 단계6개와 자유 실험·예측·해설·단원 퀴즈·기록 재생·저장 복원을 연결했다. 목차 카드·표지·선수 단원·앞뒤 이동·한영 검색과 sitemap에 포함된다. 기존 URL 규칙을 유지하며 Online Book 목록에는 교재 진입 링크를 추가하지 않는다.

이산 횡축은 갱신 번호, 연속 횡축은 초 단위 시각이다. 마지막 갱신의 다음 상태 진단을 추가 시각으로 그리지 않는다. 음수 근사를 자르지 않으며, 비영의 정밀도 미해결과 구조적0·미제공을 구별한다. 정확 유리수와 방향성 구간의 근거는 결과 표에서 확인할 수 있다.

## 수치·원고·기존 교재 보존

Markov 가족은 독립 스칼라 숨은 좌표의30준비·381기록·14,919정확 유리수 비교, 연속 가족은 독립220자리 확장 행렬 지수의39준비·407기록·9,786구간 비교로 검산했다.47개 극단적인 작은 참조는 부호 있는 작은 구간으로 대조했다. 이 수치·원고·전송·화면 근거는 이전 검증 기록을 유지한다.

새 등록 관련181검사를 포함해 전체9752검사, 타입·lint·콘텐츠·빌드를 통과했다. 기존548단원과 안내의 의미·상대 순서, 기존 교육과정의 내용은 등록 전 해시와 대조했다. 교육과정 변경은 비평형 목록에 두 ID를 덧붙이는 것이다. 이전 수치 모형과 독립 참조·원고는 변경하지 않았다.

## 실제 읽기 검증

- 한영×명암×데스크톱1600px/모바일390px×두 단원의16읽기·접근성16조합.
- 준비20회 선택, 개념 단계24회·고정 비교72회·예측24회·안내48회·단원 퀴즈48회.
- 잘못된 입력16회·유효 편집16회·저장 복원16회, 슬라이더32회, 키보드·실제 터치 가로 이동 각8회.
- 실제 정적 Worker 자산으로 계산하고 의도한 오류 후 재시도를 두 가족에서 각각 확인. 기본 읽기 추적 Worker 생성210개·종료210개.
- 클라이언트 단원 이동3회와 목차 복귀에서 Worker 생성16개·종료16개·잔존0개.
- 한영 목차 카드·표지·검색4결과·메뉴 활성 분류, Online Book 목록 제외 확인.
- 정적 출력1788 HTML의 내부 링크·자산·본문 fragment·sitemap·canonical 검사 통과. 실제 명암 캡처를 검토했다.

추가로 [비평형 입문 종합 점검](nonequilibrium-introductory-audit.md)에서 원래10주제와17단원의 대응, 모든 단원의 한영34읽기·접근성34조합을 확인했다. 이 범위를 모든 입력·모든 화면 상태의 전수 증명이나 연구 수준의 전체 완성으로 확대하지 않는다.

## 다음 순서

다음 영역은 **양자역학과 양자정보**, 첫 주제는 **상태·Hilbert 공간**이다. 이후 관측·연산자, unitary evolution·Schrödinger, spin·angular momentum, density matrix, uncertainty·measurement, tensor product·entanglement 등을 순서대로 진행한다. 일반 비선형 사영·조건부 기대·재규격화와 보편성 등 심화 범위도 계속 남는다.

로컬 미리보기는 <http://localhost:4175/book/interactive/>다. 공개 배포·commit·push는 수행하지 않았다. 근거는 `.runtime/learning/coarse-integration/verification.json`, `native-reader.json`, `nonequilibrium-audit.json`, `discovery.json`, `client-navigation.json`과 각 검사 로그다.

```bash
npm test
npm run lint
npm run validate:content
npm run build
npx tsc --noEmit
npm run check:export
npx tsx scripts/check-coarse-reader.ts
npx tsx scripts/check-coarse-discovery.ts
npx tsx scripts/check-coarse-client.ts
npx tsx scripts/check-nonequilibrium-audit.ts
```
