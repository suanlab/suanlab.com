# Uncertainty·measurement: Interactive Book 등록과 실제 읽기 검증

2026-11-08. 준비 공분산과 POVM/instrument의 두 단원을 실제 교재에 등록했다. **도서 → Interactive Book**에서 목록·표지·목차·본문·실험으로 열고 검색할 수 있다. Online Book 목록에는 교재 카드를 넣지 않는다. 기존 URL과560개 이전 단원/가이드의 내용·순서,51개 영역의 분류를 보존했다.

## 등록한 두 단원

- `finite-state-covariance-uncertainty`: Hilbert–Schmidt Gram에서 Schrödinger/Robertson 하한을 유도한다. 공분산/교환자 포화·순수 고차원 반례·복소 사영·실수 공분산 타원·ensemble 분해·Gaussian 위치/운동량과 표본 오차를 설명한다. 유한1…4차원 혼합과 임의 Hermitian 관측량을 실험한다.
- `qubit-povm-instrument-backaction`: 효과와 상태 갱신을 구별하고 같은 POVM의 양의 제곱근·피드백·재준비를 비교한다. 조건부/비선택 상태·순차 법칙·읽기값 잡음/bias·raw/보정 RMS 오차·관측량 교란·상태 거리·Ozawa 보정 관계·Choi를 설명한다.

한영 원고4개는 전체48절·34표시 수식·8표·24해설 문제다. 각 단원에 고정 개념3단계·예측3개·안내 실험3개·본문 퀴즈3개를 연결했다.32개 자유 실험 예제, 유계 정수 입력과 전체 JSON 편집, 저장·오류/재시도·모션 정책을 제공한다.

공분산 타원과 방향 재생은 이차형식의 읽기 보조이며 비가환 공동 분포나 물리 시계가 아니다. Var(A)=0의 비유일 계수/null과 정의된 최소 잔차를 구별한다. Bloch ball은 실제 상태 길이를 유지하고, 확률0의 조건부 상태와 κ=0의 보정은 null로 남긴다. 일반/무한 차원·일반 POVM·복합계 입력을 이 유한 실험의 기능으로 표시하지 않는다.

## 검증한 근거

- 전체11015검사·타입·전체 lint·콘텐츠·빌드 통과.1800 HTML의 내부 링크·자산·수식 앵커·sitemap·canonical을 확인했다. 기존560개 단원/가이드와51개 영역은 등록 전 해시와 대조했다.
- 실제 정적 reader에서 한영×명암×1600/390px×두 단원16조합·접근성16회를 통과했다. 예제32회·개념/예측 각각24회·고정 비교120회·안내48회·본문 퀴즈48회·잘못된/유효 입력 각각16회·저장 재방문16회·슬라이더32회·메뉴16회를 확인했다.
- 공분산 표현 전환16회·instrument 전환24회·그릴 상태 전환24회·조건부 결과16회·meter 보정 전환16회·키보드/실제 터치 가로 이동 각각8회를 확인했다. 실제3D8회·WebGL 손실/복구4회, reader Worker359개 생성/359개 종료·페이지 오류0이다. 별도 Worker 오류/재시도2회도 통과했다.
- 생산 Worker는 실제 컴파일된 classic factory와 같은 방식으로 실행했다.91입력의102584수학 관측이 Node 기준과 일치했다. 유리수 경계·부호·그림 값·해결 상태는 동일하고, 로그 표시의 플랫폼 차이만 상대10⁻¹²를 허용했다. 앞선 독립91사례/88유일 입력·정확78128/방향성23176대조의 폭 기준10⁻¹²⁵는 유지한다.
- 실제 클라이언트 단원 전환3회에서 Worker24개가 모두 종료되고 canvas가 해제됐다. 한영 목록 카드4개·검색4개·Online Book 제외2회와 이전 density/spin/unitary reader12회도 확인했다.

이전 [교수 화면 단계](quantum-uncertainty-teaching-status.md)의32예제·저장/모션·null·제어 검증과 실제 교재 검증을 구별해 기록한다. 기존 CSS와 색을 사용하며 모바일/명암 화면을 직접 확인했다. 이번 등록에서 새 수치 엔진이나 정밀도 기준을 바꾸지 않았다.

## 현재 범위와 다음 순서

현재562개 한영 단원·1686개 안내 실험이다. 원래 입문 주제306/362검증·56미완료이며 quantum 영역은6/11입문 주제를 검증했다. 다음은 **tensor product·entanglement**다.

이는 uncertainty·measurement의 입문 처리 검증이다. 전체51영역509주제, 일반/무한 차원·복합계·증명·연구 깊이·공용 그림 후속은 계속 남는다. 전체 교육과정이나 원래 목표를 완료로 표시하지 않는다. 로컬 사이트에 반영했으며 공개 배포·commit·push는 수행하지 않았다.

로컬 목록: <http://localhost:4175/book/interactive/>. 두 단원은 기존 `/book/online/physics-math-for-ai/` 읽기 주소를 유지한다.

근거: `.runtime/learning/quantum-uncertainty-integration/`의 `verification.json`, `full-tests.log`, `content-tests.log`, `build.log`, `typecheck.log`, `lint.log`, `validate-content.log`, `export.log`, `native-reader.json`, `production-worker.json`, `client-navigation.json`, `discovery.json`, `preservation.json`, `export-hashes.json`과 화면 캡처.
