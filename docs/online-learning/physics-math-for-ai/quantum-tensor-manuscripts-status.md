# Tensor product·entanglement 상세 한영 본문 준비

2026-11-08. [두 수치 가족](quantum-mixed-entanglement-foundation-status.md)에 대응하는 원고4편을 작성했다. 일반 대학원생이 정의·직관·유도·예제·반례·해설 문제를 따라갈 수 있도록 각각14절로 구성했다. 아직 교재에 등록하지 않았으며562단원·1686안내·306/362입문 검증·56미완료를 유지한다.

## 작성한 원고

- 순수 합성계: [한국어](bipartite-pure-schmidt-partial-trace-chapter.ko.md), [English](bipartite-pure-schmidt-partial-trace-chapter.en.md). 텐서곱·부분추적·Schmidt·복소 Gram 전치·rank/minor·엔트로피·조건부 측정·국소 unitary/dephasing·순수 부분전치·AI tensor 근사를 다룬다.
- 혼합 두-qubit: [한국어](two-qubit-ppt-correlations-chsh-chapter.ko.md), [English](two-qubit-ppt-correlations-chsh-chapter.en.md). 일반 상태 양성·Pauli 상관·분리/PPT·negativity·정보량·무신호성·CHSH 가정/최적화·Werner·정확 경계를 다룬다.

각 언어마다28절·12해설 문제·4표·21표시 수식이다. 영어 원고는 약3000단어씩이며, 한국어는 약1.2만자씩이다. 도식적 개념 흐름과 비교표를 포함했다. 실제 조작 가능한 시각화·모션·실험 화면은 다음 구현 단계에서 연결한다.

## 확인한 근거

새19집필 검사와 이전164관련 검사를 합친183검사, 타입·전체 lint가 통과했다. 한영 수식·문제 순서 일치, Strict KaTeX 렌더링, 표 행/열, heading ID와 영어의 한글 혼입을 확인했다. 손계산 값은 기존 두 수치 모델과 대조했고, GHZ/W 확장은 직접 부분추적으로 검산했다. [원고 검산 문서](quantum-tensor-manuscript-audit.md)에 관례와 가정을 기록했다.

수치 기반의120참조 사례/119유일 입력·92917정확 대조·2143방향성 대조와 이전1190검사·참조 재생성 일치 근거는 보존했다. 원고 단계의183검사가 이전 전체 수치 회귀1190검사를 새로 모두 실행했다는 뜻은 아니다. 수치 모델을 수정하지 않았으며 두 모델과 근 분리의147검사는 이번에도 통과했다.

기존6044개 소스·검증 파일 중 진행 문서9개만 갱신했다. 수치 엔진·입력·프로토콜·참조·이전 원고/화면·4371개 정적 출력 파일은 보존했다. 새 build·Worker·신규 reader 브라우저/접근성 검증·배포·commit·push는 진행하지 않았다.

## 다음 순서

두 가족의 입력 키·결과 schema·Worker 전달과 수명주기 → 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·검색·실제 읽기·접근성·정적 출력 순서다. 기존 홈페이지 스타일과 색을 사용하며 진입은 **도서 → Interactive Book**, Online Book 목록에는 넣지 않는다.

이 단계는 원래 주제 완료나 연구 깊이 완료가 아니다. 전체51영역509주제·분량/주차 무제한 범위, 다부분계·LOCC·bound entanglement·일반 positive-map 이론과 공용 그림 후속 작업은 남는다.

근거: `.runtime/learning/quantum-tensor-manuscripts/`의 `authoring-tests.log`, `combined-tests.log`, `rendering.json`, 네 HTML, `typecheck.log`, `lint.log`, `verification.json`, `preservation.json`.
