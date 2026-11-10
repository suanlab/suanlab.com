# Uncertainty·measurement: 두 수치 가족 준비

2026-11-08. 준비 불확정성에 이어 `qubit-povm-instrument-backaction`을 구현했다. 두 수치 가족이 준비되었으며 상세 한영 본문·Worker·교수 화면·Interactive Book 등록은 남는다. 등록560단원·1680안내·305/362입문 검증·57미완료는 유지한다.

## 측정 확률과 상태 갱신

22개 소유 정수로 Bloch 상태·측정/후속 Pauli 방향·측정 강도 κ·결과별 SU(2) 피드백을 정한다. 크기는10⁶ 이하이며 Bloch 공·양의 분모·비영 방향과 quaternion을 정확히 검사한다. 효과는 Eₛ=(I+sκA)/2다.

같은 효과에 대해 양의 제곱근 갱신, 결과별 unitary 피드백, 결과 방향으로 재준비하는 세 instrument를 계산한다. 확률·비정규화/조건부 상태·비선택 channel·후속 순차 확률·affine 표현·dual 관측량·Choi와 최소 Kraus 개수를 대조한다. 확률0의 조건부 값은 null이며, 순차 결과를 초기 비가환 관측량의 공동 분포로 표시하지 않는다.

읽기값 s와 보정값 s/κ의 분산·내재 잡음·bias·isometry RMS 오차를 구별한다. κ=0이면 보정 장치는 null이다. 명시한 Kraus/isometry의 RMS 교란, 상태 trace distance, 후속 분포의 total variation은 각각 계산한다. Ozawa의 보정된 오차-교란 관계와 단순 곱 관계를 구별하고, 관측 구간이 해결하지 못한 부호를 임의로 확정하지 않는다.

## 검증 결과

- 새52검사와 이전716검사를 합친768검사를 통과했다. 타입 검사와 lint 결과는 별도 실행 기록에 보관한다.
- 독립40사례/39유일 입력으로120개 instrument 결과를 대조했다. 정확 유리수 계수64772회·방향성 관측12875회, 최대 계수161자리다. 관측 구간 폭은 max(1,|참조|)×10⁻¹²⁵ 이하를 유지한다.
- 생산 코드는 Q(√(1−κ²))에서 부호·rank·제곱 잔차를 정확히 계산하고, 표시는768bit 작업·512bit 보고 구간을 쓴다. 작은 양수의 상쇄는 수체에서 먼저 유리화한다.
- 독립 SymPy 참조는 효과의 행렬 제곱근과 KρK†·K†BK·commutator norm을 직접 계산한다. 피드백은 SU(2) 행렬로, 재준비는 별도의 계산기저 Kraus 분해로 계산한다. 생산 코드를 import하지 않는다.
- κ=0의 같은 POVM이 상태를 그대로 둘 수도, 완전히 혼합할 수도 있다. I/2가 sharp Z 갱신 뒤 그대로여도 η(X)=√2다. Sharp 효과 뒤 피드백은 반복성과 unital 성질을 바꿀 수 있다.
- κ=3/5, 입력 +Z에서는 읽기값 분산16/25·raw RMS 오차 제곱4/5·보정 RMS 오차 제곱16/9를 구별한다. Sharp 측정의 ε=0은 단순 εη 하한을 보장하지 않으며, 준비 Robertson 관계의 위반을 뜻하지 않는다.

첫 가족까지 합친 독립 참조는91사례/88유일 입력, 정확 계수78128회·방향성 관측23176회다. [측정 수치 계약과 출처](quantum-povm-instrument-source-audit.md), [첫 공분산 가족](quantum-uncertainty-covariance-foundation-status.md)에 정의·범위·독립 검산 방식을 기록했다.

## 다음 작업

두 단원의 상세 한영 원고4개 → 입력/Worker 수명주기 → 고정 개념·자유 실험·그림·예측·안내·퀴즈·저장·모션 → 등록·실제 reader·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 카드를 넣지 않는다.

현재는 수치 준비 단계다. 일반 다차원 POVM·무한 차원 정준 관측량·복합계·일반 장치의 운용적 오차 정의·증명과 연구 깊이를 유한 qubit 예제로 완료 처리하지 않는다. 전체51영역509주제와 공용 그림 후속 작업을 유지한다. 새 build·브라우저·Worker 검증·공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-povm-instrument/`의 `new-tests.log`, `combined-tests.log`, `reference-checks.json`, `typecheck.log`, `lint.log`. 재생성·보존 결과와 최종 단계 상태는 `verification.json`을 확인한다.
