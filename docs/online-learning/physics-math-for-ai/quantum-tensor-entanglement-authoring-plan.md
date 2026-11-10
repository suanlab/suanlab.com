# Tensor product·entanglement 집필·구현 계약

2026-11-08. uncertainty·measurement 다음 원래 주제다. 현재562단원·1686안내·306/362입문 검증·56미완료다. 전체51영역509주제와 분량·주차 제한 없는 연구 깊이를 유지하며 준비 작업을 교재 등록이나 검증 완료로 세지 않는다.

## 두 계산 가족과 전체 이론

1. `bipartite-pure-schmidt-partial-trace`: 각 부분계1…4차원·전체 최대16차원의 복소 유계 정수 계수 순수 상태를 다룬다. 텐서 인덱스·부분추적·복소 Gram 전치·Schmidt rank/가중치·엔트로피·곱상태 분해/비영 minor·국소 조건부 상태/무신호성·국소 반사 불변량과 기준 기저 dephasing을 계산한다. Qutrit/ququart의 rank3·4를 qubit 예제로 대신하지 않는다. 수치 Schmidt 기저 벡터는 제공하지 않으며 일반 Schmidt 분해의 증명·기저 모호성은 상세 본문에서 다룬다.
2. `two-qubit-ppt-correlations-chsh`: 일반 두-qubit Hermitian 밀도의 양성과 정규화, 국소/공동 통계·PPT·negativity·선택한 CHSH 설정과 최적 CHSH 크기를 비교하는 수치 가족을 구현했다. 32개 소유 유계 정수로 일반 Hermitian 분자와 네 복소 국소 spinor를 받는다. 모든15 principal minor로 상태를 검증하며, shifted-Gram의 정확 PSD/행렬식으로 CHSH 경계도 구별한다. [수치 계약](quantum-mixed-entanglement-source-audit.md)을 따른다. 상세 한영 본문·Worker·학습 화면을 준비했고 실제 교재 등록은 아직 남는다.

본문은 텐서곱과 직합의 차이, 선형 확장과 곱벡터의 비유일한 대표, A-major 좌표와 swap, 부분추적의 정의/기저 독립성/국소 기대값 보존을 유도한다. C=UΣV†에서 B의 Schmidt 벡터에 켤레가 들어가는 이유, 비영 축약 스펙트럼의 일치, 중복 가중치의 기저 모호성, 순수 상태의 곱 분리와 rank1·순도1의 동치를 설명한다.

혼합 상태에서는 국소 혼합도가 얽힘을 판정하지 못한다. Bell 상태·기준 기저의 고전적 혼합·독립 I/4, 국소 측정의 조건부 변화와 비선택 무신호성을 비교한다. PPT가 분리에 충분한 차원과 고차원의 한계, 얽힘과 Bell 비국소성의 차이, 유한 표본·측정 설정·숨은 변수 가정을 명시한다. 일반 다부분계·GHZ/W·얽힘 조작/monotone·bound entanglement·자원/연구 깊이는 두 유한 계산기로 완료 처리하지 않는다.

## 수치 원칙과 순서

첫 가족은50개 소유 정수(절댓값10⁶ 이하)와 정확한 비영/비활성 좌표 검사를 쓴다. 부분추적·밀도·rank·principal minor·특성 다항식·국소 조건부 상태·국소 반사·곱상태 인수분해는 정확 복소 유리수다. 일반4차 스펙트럼은 정확 square-free/Sturm 분리로 중복도를 구별하고, 작은 양수나 작은 gap을0으로 합치지 않는다.

첫 수치 가족 → 혼합/PPT/CHSH 수치 가족 → 상세 한영 본문 → Worker 계약/수명주기 → 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·실제 읽기·접근성·정적 출력 순서다. 두 수치 가족·[상세 한영 원고4편](quantum-tensor-manuscripts-status.md)·[Worker 기반](quantum-tensor-worker-status.md)·[학습 화면](quantum-tensor-teaching-status.md)을 준비했으며, 다음은 실제 reader 등록/검증이다. 원고는 언어별14절·해설6문제와 한영 동일 수식으로 검산했다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다. 기존 홈페이지 스타일과 색을 따른다.

기초 관례는 [Preskill §2.3–2.4](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), PPT 배경은 [Peres 원 논문](https://arxiv.org/abs/quant-ph/9604005)과 [Horodecki 등의 분리 조건](https://arxiv.org/abs/quant-ph/9605038)에서 확인했다. CHSH 최적화는 [저자들의 Quantum entanglement §IV.A–B](https://arxiv.org/pdf/quant-ph/0702225)에서 두-qubit·traceless±1 국소 projective 관측의 가정과 대조했다. 선택한 유계 spinor 방향과 연속 전체 방향 최적화를 구분한다. 이 문서는 집필 범위이며 원래 주제 완료 근거가 아니다.
