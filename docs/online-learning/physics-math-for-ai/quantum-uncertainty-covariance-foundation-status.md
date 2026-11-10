# Uncertainty·measurement: 상태 공분산 수치 기반

2026-11-08. [집필·구현 계약](quantum-uncertainty-authoring-plan.md)의 첫 가족 `finite-state-covariance-uncertainty`를 구현했다. 임의의1…4차원 Hermitian 관측량과 유한 혼합 상태를 다룬다. 두 번째 POVM/instrument 가족·상세 한영 본문·Worker·교수 화면·실제 읽기는 남는다. 등록560단원·1680안내·305/362입문 검증·57미완료를 유지한다.

## 첫 계산 가족

75개 소유 정수로 차원·네 ensemble 가중치/복소 raw 대표·두 Hermitian 행렬의 전체 윗삼각 성분과 분모·실수 선형 결합 u,v를 정한다. 정수 크기는10⁶ 이하, 분모는 양수이고 차원 밖 좌표는0이다. 가중치0인 영벡터는 허용하지만 점유한 벡터는 비영이어야 한다.

상태·관측량·중심화·분산·복소 공분산·교환자 기대를 직접 계산한다. Gram determinant는 Schrödinger slack, 여기에 실수 공분산 제곱을 더한 값은 Robertson slack이다. 준비 내부/준비 평균 사이 분산·교차항, 최적 복소 투영과 실수 결합 관측량의 분산을 정확 유리수로 검산한다. 단순 Pauli 전용 모형으로 제한하지 않는다.

표준편차·표준화 성분·가중 변동 열·2×2 Gram/실수 공분산 스펙트럼에는768bit 작업·512bit 보고 구간을 쓴다. 작은 고유값은 det/high, 사영 방향은 제곱의 비를 먼저 정규화하는 방식으로 계산한다. 정확한0과 작은 양수, 영행렬의 rank0과 영 고유값의 중복도2, 퇴화한 고유공간을 구별한다.

이 가족은 같은 준비에서의 분산을 계산한다. 비가환 관측량의 공동 분포·측정 장치·기기의 오차/교란·정준 x,p 또는 임의 d의 관측량 고유값 계산을 제공하지 않는다. 실수 공분산 타원은 Var(uA+vB)를 설명한다.

## 확인한 근거

- 새62검사와 이전654검사를 합친716검사, 타입 검사·전체 lint 통과. 초기 lint의 미사용 import3개를 제거했으며 수치 계산이나 정밀도 기준은 바꾸지 않았다.
- 독립51사례/49유일 입력·차원1…4. SymPy 직접 복소 행렬 trace·Gaussian rank와 mpmath250dps 일반 Hermitian 고유값/고유벡터 사영을 사용했다. 생산 코드를 import하지 않는다.
- 정확 유리수 인증13356회·방향성 관측10301회. 표준편차·사영·가중 열은210자리 참조와 비교하고 구간 폭 max(1,|참조|)×10⁻¹²⁵를 유지한다. 참조 재생성의 바이트 일치도 확인했다.
- 순수 Bloch(3/5,4/5,0)와 X/Y에서는 공분산 항이 하한144/625를 만들고 약한 교환자 하한은0이다. I/2와 순수 qutrit의 엄격한 slack1도 확인해 순수 qubit의 포화를 일반화하지 않는다.
- 같은 rho의 Z/X/Y ensemble에서 전체 Gram은 같고 내부/사이 분산은 달라진다. shift 불변·부호 있는 재척도·음의 공분산·상수 관측량·확률0인 성분·거의 종속인 양의 rank2와1차원 영 공분산을 포함한다.
- Var(A)=0일 때 최적 계수는 비유일/null이고 최소 잔차 Var(B)는 정의된다. 별도 helper 검사에서는 사용자 입력 범위와 구별하여10⁻³⁰⁰ gap을 구성하고, 작은 간격을0으로 합치거나 방향을 잃지 않음을 확인했다.

자세한 유도·부호 관례·정밀도·참조 방법은 [수치 검산 계약](quantum-uncertainty-covariance-source-audit.md)에 기록했다. 실제 기기의 오차-교란 부등식으로 상태의 준비 불확정성을 대체하지 않는다.

## 다음 순서와 전체 범위

다음은 POVM/instrument 수치 가족 → 두 상세 한영 본문 → Worker → 개념·실험·시각화·예측·안내·퀴즈·저장·모션 → Interactive Book 등록·실제 읽기·접근성·정적 출력이다. 아직 새 교재 단원으로 등록하지 않았다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다.

기존 수치·원고·Worker·교수 화면과4366개 정적 파일은 보존했다. 이번 단계에는 새 build·브라우저·reader 검증이 없다. 이전10728전체 검사와 실제 읽기 결과를 새 uncertainty 단원의 완료 근거로 세지 않는다. 전체51영역509주제·일반 차원/무한 차원·복합계·증명·연구 깊이·공용 그림 후속은 계속 남는다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-uncertainty-covariance/verification.json`, `combined-tests.log`, `new-tests.log`, `reference-checks.json`, `reference-reproduction.json`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-uncertainty-covariance.test.ts scripts/learning-density-content.test.ts scripts/learning-density-interface.test.ts scripts/learning-density-worker.test.ts scripts/learning-quantum-density-authoring.test.ts scripts/learning-density-spectrum.test.ts scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-uncertainty-covariance-references.py --output .runtime/learning/quantum-uncertainty-covariance/reference-regenerated.json
npm run typecheck
npm run lint
```
