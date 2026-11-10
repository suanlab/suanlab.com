# Tensor product·entanglement: Interactive Book 등록과 실제 읽기 검증

2026-11-08. 순수 합성계와 혼합 두-qubit의 두 단원을 **도서 → Interactive Book**에 등록했다. 목록·표지·목차·상세 원고·개념·실험·퀴즈·검색을 연결했으며 Online Book 목록은 별도로 유지한다. 기존 읽기 URL과562개 이전 단원/가이드의 내용·순서,51개 영역의 기존 분류를 보존했다.

## 등록한 두 단원

- `bipartite-pure-schmidt-partial-trace`: 텐서 좌표·부분추적·복소 Gram·Schmidt 분해의 유도와 축퇴·rank1 분리·조건부 상태·무신호성·국소 반사·dephasing을 설명한다. 부분계1…4차원, 전체 최대16차원의 복소 순수 상태를 실험한다.
- `two-qubit-ppt-correlations-chsh`: 일반 두-qubit 밀도·국소 혼합도와 얽힘의 차이·PPT와 negativity·국소 Pauli 상관·선택한 CHSH와 최적 크기·국소 조건부 상태를 설명한다. Bell·고전적 혼합·독립 혼합·Werner 상태를 비교한다.

한영 원고4편은 합계56절·42표시 수식·8표·24해설 문제다. 각 가족에 개념3단계와 예측·안내 실험3개·본문 퀴즈3개, 자유 실험 예제16개를 연결했다. 50/32개 정수 필드와 전체 JSON 편집, 저장·재방문, 오류·재시도를 제공한다. 진폭·확률·Schmidt 가중치·부분전치 부호·조건부 상태·A/B Bloch·상관 격자를 기존 홈페이지 색으로 표시한다.

수치 Schmidt 기저·최적 CHSH 측정축을 계산한다고 표시하지 않는다. 축퇴 가중치는 중복도를 유지하고 확률0의 조건부 상태는 null이다. PPT의 분리 충분 조건은2×2에 한정한다. 일반 다부분계·무한 차원·고차원 bound entanglement·자원 이론과 연구 심화는 계속 남는다. 모션은 읽기 보조이며 물리 시간이나 상태 변화를 나타내지 않는다.

## 확인한 근거

- 전체11369검사·타입·lint·콘텐츠·빌드 통과. 1802 HTML의 내부 링크·자산·수식 앵커·sitemap·canonical을 확인했다. 등록 전 해시와 이전562개 단원/가이드·51개 영역을 비교하는4개 통합 검사도 통과했다.
- 실제 reader에서 한영×명암×1600/390px×두 단원16조합과 접근성16회를 통과했다. 예제32회·개념/예측 각각24회·고정 비교88회·안내48회·본문 퀴즈48회·잘못된/유효 입력 각각16회·저장 재방문16회와 퀴즈 답 복원48회를 확인했다.
- 키보드/실제 터치 가로 이동 각각8회·메뉴16회·실제 Three.js3D4회·Worker 오류/재시도2회를 확인했다. reader Worker271개가 모두 종료되고 페이지 오류는0이다.
- 배포용 static Worker를 실제 컴파일 factory와 같은 classic 방식으로 실행했다. 120개 입력의99340수학 관측이 검증한 Node 계산과 일치했다. 유리수 구간·부호·값·해결 상태는 동일하며 로그 표시의 플랫폼 차이에만 상대10⁻¹²를 허용했다. 앞선 독립120사례/119유일 준비의 정확92917·방향성2143대조와10⁻¹²⁵구간 폭 기준은 유지한다.
- 실제 클라이언트 단원 이동4회에는 계산 중 이동이 포함된다. Worker21개가 모두 종료되고 canvas가 해제됐다. 한영 목록 카드4개·검색4개·Online Book 제외2회와 이전 uncertainty/density/spin/unitary reader16회도 통과했다.

[이전 학습 화면 검증](quantum-tensor-teaching-status.md)과 이번 실제 교재 검증을 구별한다. 이번 등록에서 수치 엔진·입력·정밀도·Worker 계약·원본 원고를 바꾸지 않았다. 기준6170개 파일 중 통합에 필요한8개 변경과 이후 진행 문서 갱신을 기록하고 나머지 파일을 해시로 확인한다. 생성된 출력4376개 파일은 새 빌드의 해시를 별도로 보존한다.

## 현재 범위와 다음 순서

현재564개 한영 단원·1692개 안내 실험, 원래 입문 주제307/362검증·55미완료이며 quantum 영역은7/11입문 주제를 검증했다. 다음은 **variational principle**다. [집필 설계](quantum-variational-authoring-plan.md)·[직접 유도](quantum-variational-derivation-notes.md)와 독립 손계산 검산을 준비했으며 아직 구현 단원으로 세지 않는다.

전체51영역509주제·영역별 감사·증명과 연구 깊이·공용 그림 후속은 미완료다. 전체 목표를 완료로 표시하지 않는다. 공개 배포·commit·push는 하지 않았다.

로컬 목록: <http://localhost:4175/book/interactive/>. 읽기 주소는 `/book/online/physics-math-for-ai/`를 유지한다.

근거: `.runtime/learning/quantum-tensor-integration/`의 `verification.json`, `full-tests.log`, `content-tests-final.log`, `build.log`, `typecheck-final.log`, `lint-final.log`, `validate-content.log`, `export-check.log`, `native-reader.json`, `production-worker.json`, `client-navigation.json`, `discovery.json`, `preservation.json`, `export-hashes.json`, `next-topic-hand-check.json`과 화면 캡처.
