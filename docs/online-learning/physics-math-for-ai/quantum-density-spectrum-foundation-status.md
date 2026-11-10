# Density matrix: 스펙트럼·엔트로피 수치 기반 검증

2026-11-08. 첫 [유한 ensemble 가족](quantum-density-ensemble-foundation-status.md)에 이어 `qubit-density-spectrum-entropy-dephasing`을 구현했다. 두 가족의 상세 한영 본문·Worker·교수 화면·실제 읽기는 남는다.558단원·1674안내·304/362입문 검증·58미완료를 유지한다.

## 두 번째 계산 가족

11개 소유 정수로 혼합 qubit의 Bloch 좌표·probe·끝점 eta·표본 수를 정한다. 정확한 ball 조건과 −1≤eta≤1을 검사하며 invalid state를 잘라 넣지 않는다. rho·순도·determinant·Pauli 통계·기저 변환·channel/Choi 계정은 정확 유리수다. 고유값·사영·entropy·coherence·거리·Kraus 그림은 방향성 구간이다.

작은 고유값에는(1−|r|²)/(2(1+|r|))를 사용한다. 순수 상태의 log0은 null·entropy 항은 극한0으로 구별하고, 완전 혼합에서는 rank2 고유공간을 합친다. nats/bits, 기저 의존 coherence, squared Uhlmann fidelity와 trace distance를 따로 제공한다.

eta는 I/Z의 확률 혼합을 정한다. 음수도 유효하며 eta−1 끝점은 unitary Z다. eta0을 지나는 표본을 정확히 삽입하고 경로를 물리 시간이나 Markovian law로 부르지 않는다. |+X⟩에서 eta−1로 가면 entropy는0으로 돌아오지만 초기 상태와 직교한다. Choi는 output⊗input의 unnormalized trace2 행렬이며 상태 trace1과 구별한다.

## 확인한 근거

- 새52검사 + 이전475검사, 합계527검사·타입·전체 lint 통과.
- 독립38사례(37개 유일 입력)·221표본. SymPy operator sum·symbolic eigenvalues·직접 측정 trace와 mpmath250dps 일반 Hermitian eigensolver·사영·행렬 제곱근·singular-value fidelity·로그를 대조했다. 생산 안정화 식이나 로그를 재사용하지 않았다.
- 정확 유리수 인증31007회·방향성 관측28641회. 구간 폭은 max(1,|참조|)×10⁻¹²⁵ 이하다. 참조 재생성이 바이트 단위로 일치했다.
- exact pure/maximally mixed·작은 양의 determinant/radius·음의 eta·격자 밖0 교차·기저/위상·실제 측정 행렬·coherence/entropy 계정·Choi 양성/TP·Kraus completeness·한 선택 상태의 불변과 channel 항등의 차이를 확인했다.
- 첫 가족의 혼합 상태와 같은 qubit 입력의 rho·purity·평균/분산·완전 dephasing 결과도 정확히 일치했다. 첫 가족의43사례/39유일 입력·정확15073/방향성3558 대조를 유지한다.

768bit 연산·512bit 보고 구간, 로그800bit 작업/768bit 경계를 사용한다. 구조적 영과 수치 잔차 구간을 구별하고 norm을 보정하지 않는다. 자세한 정의와 유도는 [스펙트럼 검산 계약](quantum-density-spectrum-source-audit.md)에 기록했다.

이번 단계의 새 build·브라우저·Worker·reader 검증은 없다. 기존 수치·참조·한영 원고·교수 화면과4360개 정적 출력은 보존했다. 이전10492전체 검사·1796 HTML을 새 density 읽기의 근거로 세지 않는다.

## 다음 순서와 전체 범위

다음은 두 상세 한영 본문4개 → Worker → 개념·실험·시각화·예측·안내·퀴즈·저장·모션 → 등록·실제 읽기·접근성·정적 출력이다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 넣지 않는다.

일반 d의 density/entropy 이론·부분추적/복합계·연구 깊이와 전체51영역509주제·공용 그림 후속은 계속 남는다. 공개 배포·commit·push는 수행하지 않았다. 전체 목표는 진행 중이다.

근거: `.runtime/learning/quantum-density-spectrum/verification.json`, `reference-checks.json`, `combined-tests.log`, `reference-reproduction.log`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-density-spectrum.test.ts scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-density-spectrum-references.py --output .runtime/learning/quantum-density-spectrum/reference-regenerated.json
npm run typecheck
npm run lint
```
