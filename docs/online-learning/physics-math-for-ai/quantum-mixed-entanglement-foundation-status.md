# Tensor product·entanglement: 두 수치 가족 준비

2026-11-08. [첫 순수 합성계 가족](quantum-tensor-pure-foundation-status.md)에 이어 `two-qubit-ppt-correlations-chsh`의 일반 혼합 상태 수치 모델을 구현·검산했다. 두 모델은 준비되었으며 상세 한영 본문·Worker·교수 화면·등록은 남는다. 교재 등록562단원·1686안내·306/362입문 검증·56미완료를 유지한다.

## 추가한 계산과 구분

32개 소유 유계 정수로 일반 복소 Hermitian4×4 분자와 네 국소 spinor를 받는다. 양의 trace와 모든15 principal minor로 상태를 검증하고, 정확한 부분추적·Bloch vector·Pauli 상관 tensor·전체 Pauli 재구성·선택 측정의 공동/조건부 통계와 무신호성을 계산한다. 임의 혼합 상태를 Bell·Werner 형태로 제한하지 않는다.

전체·축약·부분전치의 일반 스펙트럼과 엔트로피·mutual information·조건부 엔트로피·negativity를 제공한다. PPT⇔분리는 이 모델의2×2에만 적용한다. 선택 CHSH |S|와 연속 전체 projective 방향의 최댓값2√M을 구분하며, 실제 최적 측정축을 구했다고 표시하지 않는다. G=TᵀT, q=TrG−1에서 G−qI의 정확 principal minor/행렬식으로 M<1,=1,>1을 판정한다. 부동소수점 threshold로 정확 경계를 분류하지 않는다.

Bell·고전적 상관·독립 I/4의 국소 상태가 모두 I/2임에도 공동 정보·얽힘이 다른 사례, 얽혔지만 CHSH 위반이 없는 Werner 상태, 좋은/나쁜 측정 설정, 음의 선택 CHSH, 정확 경계와 양쪽 유효 상태, 복소 rank1…4·분리 혼합·영 사건을 포함했다. 유계 분자의 약10⁻²⁴ 양의 고유값도0으로 합치지 않는다.

## 검증 근거

- 새65검사 통과. 관련999+191=1190검사에 이전1021검사의 모든 이름이 포함됨을 대조했다. 타입과 전체 lint도 통과했다.
- 독립53사례/53유일 입력. SymPy 직접 tensor·부분추적·모든 minor와 mpmath250dps Hermitian eigensolver를 사용했다. 독립 CHSH 경계는 shifted-minor 대신 정확 다항식 근 개수로 판정한다.
- 정확 유리수32867대조·방향성 관측1337대조.210자리 참조에 대해 구간 폭 max(1,|참조|)×10⁻¹²⁵를 확인했고, 재생성 결과가 바이트 단위로 일치한다.
- 기존 순수 모델의67사례/66유일 입력·16차원 조합과 함께 두 수치 가족이 준비되었다. 첫 모델의 Schmidt 기저와 둘째 모델의 최적 측정축·분리 혼합 분해는 계산하지 않는다.

초기 참조 생성에서는 정확 복소식의 정규형 변환과 CHSH 경계 상태의 유효한 양방향 perturbation을 수정했다. 처음의 rank3 경계에 대한 한쪽 perturbation은 비물리 상태였으므로, 양쪽 perturbation을 허용하는 full-rank 경계로 구성했다. 사례·정밀도를 줄이거나 음의 고유값을 보정하지 않았다. 정확0을 포함하는 entropy 차이의 관측은 숫자 projection이 null일 수 있어, 검사는 유리수 구간의0 포함과 폭을 확인한다.

## 다음 순서와 범위

두 단원의 상세 한국어/영어 본문 → 입력·Worker 계약과 수명주기 → 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·검색·실제 읽기·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다.

기존6012개 소스·검증 파일 중 진행 문서9개만 갱신했다. 기존 수치 엔진·프로토콜·참조·본문·교수 화면과4371개 정적 출력 파일은 보존했다. 새 build·신규 Worker/reader 브라우저 검증·배포·commit·push는 진행하지 않았다. 전체51영역509주제, 일반 다부분계·증명·연구 깊이와 공용 그림 후속 작업은 완료 처리하지 않는다.

수치 계약: [source audit](quantum-mixed-entanglement-source-audit.md). 근거: `.runtime/learning/quantum-mixed-entanglement/`의 `new-tests-final.log`, `combined-tests.log`, `additional-regression-tests.log`, `regression-coverage.json`, `reference-checks.json`, `reference-reproduction.json`, `typecheck-final.log`, `lint-final.log`, `verification.json`, `preservation.json`.
