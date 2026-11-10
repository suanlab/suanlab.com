# Quantum variational principle: 상세 한영 원고 준비

2026-11-09. Rayleigh 에너지와 비직교 Ritz의 두 계산 가족에 대응하는 한영 원고4편을 작성·검산했다. 실제 교재 등록 전이므로564개 한영 단원·1692개 안내, 원래 입문 주제307/362검증·55미완료를 유지한다.

## 작성한 원고

- `finite-state-rayleigh-energy-variance-chapter.{ko,en}.md`: 상태 ray·정규화·하한·복소 방향1/2차 미분·분산/잔차·gap 누출·두 준위·작은 양수·Gaussian 조화 진동자·제한 ansatz·VQE/Monte Carlo 연구를 설명한다.
- `nonorthogonal-ritz-subspace-certificates-chapter.{ko,en}.md`: Gram metric·일반화 문제·물리 사영·min–max·포함 공간·잔차 분해·손계산 평면·복소 좌표 변경·조건 문제·연속 가정·움직이는 기저 미분을 설명한다.

언어별 각 장은14절·표시 수식12개·비교표2개·해설 문제6개다. 전체56절·수식48개·표8개·해설24개이며 영어3039/2893단어, 한국어11352/10560문자다. 가정·직관·실패 사례·연구 연결을 포함하고, 진단의 의미와 계산기의 실제 기능을 구별한다.

## 검증과 근거

21개 새 원고 검사를 통과했다. 실제 Markdown/KaTeX 렌더링, 표의 행/열 수, 중복 없는 절 앵커, 한영 표시 수식·문제 순서, 영어의 한글 혼입과 제어 문자 부재를 확인했다. 타입과 전체 lint도 통과했다.

대표 예제를 검증한 수치 엔진과 대조했다. Gaussian 정규화·모멘트·소멸 연산자 인수분해, 경로 미분, Ritz 평면과 움직이는 metric의 미분은 독립 SymPy 계산으로 확인했고 기준 파일 재생성의 byte 동일성을 확인했다. 의미·가정·한계의 직접 검토는 [원고 감사](quantum-variational-manuscript-audit.md)에 기록했다.

근거: `.runtime/learning/quantum-variational-manuscripts/`의 `verification.json`, `authoring-checks.json`, `tests-final.log`, `reference-generation.log`, `reference-reproduction.json`, `lint.log`, `typecheck.log`, `preservation.json` 및 원고별 렌더링 HTML.

## 다음 순서와 전체 범위

다음은 두 수치 가족의 Worker 전달·응답 검증·취소·재시도·수명주기다. 이후 개념 그림·자유 실험·예측·안내·퀴즈·저장을 만들고 실제 reader/접근성과 정적 출력을 검증한다. 진입은 **도서 → Interactive Book**으로 유지하며 Online Book 목록에는 새 교재 링크를 넣지 않는다.

두 단원 ID는 아직 registry·검색·사이트에 등록하지 않았다. 전체51영역509주제·영역 감사·증명과 연구 깊이·공용 그림 후속은 미완료이며 양자 변분 원리도 원래 주제 완료로 세지 않는다. 공개 배포·commit·push는 하지 않았다.
