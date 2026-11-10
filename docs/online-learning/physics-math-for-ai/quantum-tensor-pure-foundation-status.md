# Tensor product·entanglement: 순수 합성계 수치 기반

2026-11-08. [집필·구현 계약](quantum-tensor-entanglement-authoring-plan.md)의 첫 가족 `bipartite-pure-schmidt-partial-trace`를 구현했다. 각 부분계1…4차원, 전체 최대16차원의 유계 복소 정수 계수 상태를 다룬다. 두 번째 혼합/PPT/CHSH 가족·상세 한영 본문·Worker·교수 화면·등록은 남는다. 현재562단원·1686안내·306/362입문 검증·56미완료를 유지한다.

## 계산 범위

50개 소유 정수로 차원·4×4 raw 계수 슬롯·두 국소 probe를 정한다. A-major 텐서 인덱스, 전체/축약 밀도·복소 Gram 전치·국소 순도·Schmidt rank·특성 다항식·전체 principal minor를 정확 복소 유리수로 계산한다. Rank1에는 명시적 곱 인수와 정규화 상태를, rank>1에는 비영2×2 minor 증거를 제공한다.

일반4차 축약 스펙트럼을 정확 square-free/Sturm 방법으로 분리한다. 중근의 중복도, 정확한 영 인자와 작은 양수를 구별한다. 기본 근 분리1024bit, 구간 작업768bit·보고512bit를 사용한다. Schmidt 가중치/계수·엔트로피·순수 상태 부분전치의 양/음 pair·negativity·log-negativity를 계산한다. Schmidt 기저 벡터를 수치적으로 구했다고 표시하지 않는다.

국소 projector/여공간 사건의 공동 확률·조건부 축약 상태·확률0/null·비선택 무신호성을 비교한다. 국소 Householder 반사의 텐서 작용과 축약 스펙트럼 불변량을 검산한다. 같은 기준 기저 확률의 대각 혼합을 분리 가능한 비교 상태로 제공해, Bell 상태와 같은 국소 I/2가 얽힘을 결정하지 못함을 보여준다.

## 확인한 근거

- 새82검사(순수 합성계78·일반 근 분리4)와 이전939관련 검사를 합친1021검사, 타입·전체 lint 통과.
- 독립67사례/66유일 입력·16개 모든 차원 조합. SymPy의 직접 tensor/partial trace·rank·특성 다항식·조건부 행렬과 mpmath250dps 일반 Hermitian 고유값을 사용했다. 전체 최대16×16 부분전치 고유값에서도 negativity를 대조했다.
- 정확 유리수60050대조·방향성 관측806대조.210자리 참조에 대해 구간 폭 max(1,|참조|)×10⁻¹²⁵를 유지한다. 참조 재생성의 바이트 일치와 보존 결과는 runtime 기록을 확인한다.
- 복소 C의 B축약 밀도와 Gram의 허수 부호, Bell±/허수 위상과 고전적 혼합, rank3·4·직사각형의 서로 다른 영 중복도, 작은 비영 minor·네 번째 가중치, 반복 가중치, 영 사건과 혼합 여공간 조건부 상태를 확인했다.
- 모든16차원 조합의 복소 곱 상태에서 명시한 인수의 곱이 모든 raw 계수를 A-major 순서로 복원하고 정규화 진폭도 일치한다. 전역 대표 위상·국소 반사와 정확 rank 증거도 확인했다.
- 별도 일반 근 helper 검사에서는 사용자 입력 범위와 구별해10⁻¹⁸⁰ 간격의 서로 다른 근을 구성했다. 이를 중근으로 합치지 않고, 비실근·지원 차수 밖·부적절한 정밀도/분모를 거부한다.

초기 독립 생성기의 비정규형 복소 식을 정확 유리수 행렬로 정리했고, 영 인자가 다른 단순근과 같은 square-free 인자에 포함되는 경우의 독립 중복도 검사를 수정했다. 사례나 정밀도를 줄이지 않았다. 자세한 수치 관례는 [수치 검산 계약](quantum-tensor-pure-source-audit.md)에 기록했다.

## 다음 순서와 전체 범위

혼합 두-qubit PPT·상관·CHSH 가족 → 두 단원의 상세 한영 본문 → Worker → 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·실제 읽기·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다.

이 준비는 아직 교재 단원으로 등록하지 않았다. 새 build·Worker·브라우저 검증이나 공개 배포·commit·push는 없다. 기존 수치·원고·Worker·화면과4371개 정적 파일은 보존한다. 이전11015전체 검사와 reader 검증을 새 합성계 단원의 완료 근거로 세지 않는다. 전체51영역509주제·일반 다부분계·증명·연구 깊이·공용 그림 후속 범위를 유지한다.

근거: `.runtime/learning/quantum-tensor-pure/`의 `new-tests.log`, `combined-tests.log`, `reference-checks.json`, `reference-reproduction.json`, `typecheck.log`, `lint.log`, `verification.json`, `preservation.json`.
