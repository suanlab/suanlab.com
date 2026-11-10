# 양자 회로·채널 실제 reader 통합

2026-11-09. **도서 → Interactive Book**의 실제 reader에 네 단원과 검증한 정적 출력을 반영했다. 작업본에서도74통합 검사·타입·lint·출력·교육과정 재생성 및4개 실제 reader를 다시 확인했다. commit/push·공개 배포는 포함하지 않는다.

| 순서 | 단원 | 안내·해설 문제 |
|---|---|---:|
|572|`finite-qubit-circuit-order-interference`|각3|
|573|`qubit-channel-kraus-choi-verification`|각3|
|574|`system-environment-stinespring-reference`|각3|
|575|`quantum-instruments-records-feedforward`|각4|

한영 본문8개, 표지4개, 개념13단계,33입력 예제와 안내13개·해설 문제13개를 실제 reader의 본문·선수학습·목차·이전/다음·퀴즈와 연결했다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 교재 링크를 추가하지 않았다. 기존 reader URL과 저장 키를 유지한다.

별도 후보 작업본에서 등록·빌드·전체 검사를 먼저 수행했다. 기존571단원·안내의 내용/순서와51영역의 범위를 개별 SHA-256으로 대조했다. 반영 결과는 **575단원·1726안내 실험**이다. 기존 정적 출력4406파일과 변경 전 소스는 별도 보관한다.

## 실제 생산 자산과 화면

- 전체 **14805/14805검사**, 통합 관련74검사, lint·타입·콘텐츠·정적 빌드·출력 검사 통과. 생성된1813HTML의 링크·자산·fragment·sitemap·canonical을 확인했다.
- Next가 생성한 계산/독립 검증 Worker를 실제 HTTP 자산으로 실행했다.595입력의 정확 결과를 독립 참조와 비교하고 표시 관측값의 Node/browser 일치 및 독립 receipt를 확인했다. 다섯 종류의 수치 변조와 좌표 변조를 각각 거부했다. 빌드가 실제로 전달한 실행 모드는 classic이며 소스 factory의 module 옵션과 구별한다.
- 한영×명암×320/1440px의 **32 reader 조합·72접근성 상태**를 확인했다.104개 개념 단계와104개 문제의 오답/정답 해설, 입력·답·완료 표시의 재로딩, 계산 중 페이지 이동 후 Worker 종료를 확인했다.
- 실제 Three.js 장면·시점 버튼·WebGL 손실/재시도와55개 실제 JavaScript 자산의 응답/파일 바이트를 확인했다. main reader 의존성 그래프에는 새 qubit 계산기와 수학 검증기가 포함되지 않는다.
- 여섯 준비에서 입력 반응17–83ms, pending heartbeat 최대 간격22–44ms, stale-ready0건이었다. 검산 중 취소는64ms였고 재시도도 통과했다. 실제 reader에서 한영16개 저장소 손상/차단 흐름을 확인했다. 이는 로컬 Chromium의 관측값이며 실제 숨김 탭 자동화로 확대하지 않는다.

자산 검사 스크립트의 `[unit]` URL 인코딩 처리를 고친 뒤32조합을 다시 실행했다. 테스트 경계나 수학 기준을 완화하지 않았다. 이전 단계의 [416개 실습 접근성 상태·88복구 시나리오](quantum-qubit-labs-status.md)와 실제 reader 검증의 범위는 구별한다.

## 완료 범위와 다음 순서

`quantum circuit·channel`의 **입문** 처리를 완료했다. 양자 영역은10/11입문 주제, 원래 범위는 **310/362검증·52미완료**다. 다음은 **quantum information**이다. 전체51영역509주제의 나머지 집필·영역 감사·증명·연구 깊이는 완료되지 않았다.

근거는 `.runtime/learning/quantum-qubit-reader-integration/`의 후보/작업본 검사·생산 자산·reader·발견 경로·반응성 결과, 보존한 원본과 해시 목록이다.

작업본의 브라우저 재검증은 `QUBIT_PREVIEW_BASE=http://localhost:4175 npx tsx scripts/check-qubit-reader.ts`처럼 실제 미리보기 주소를 지정한다. 후보 검증용4176서버와 작업본4175서버는 구별한다.
