# Density matrix: 유한 ensemble 수치 기반 검증

2026-11-08. spin 다음의 density matrix 주제에서 첫 계산 가족 `finite-ensemble-density-gram-tomography`를 구현했다. 스펙트럼/엔트로피 가족·상세 한영 본문·Worker·교수 화면·실제 읽기는 남는다. 기존558단원·1674안내·304/362입문 검증·58미완료는 유지한다.

## 구현한 계산

49개 소유 정수로 d=1…4, 최대 네 순수 상태·비음 weight·probe·유리수 관측량을 지정한다. 점유 상태는 비영이며 비활성 좌표를 잘라내지 않는다. 확률적 혼합과 진폭 중첩을 구별하고 rho·순도·rank·모든 principal minor·Gram 특성다항식 불변량을 정확히 계산한다.

기준 기저와 실수/허수 pair 측정으로 모든 행렬 성분을 복원한다. d>2에서는 pair 부분공간 밖의 확률도 포함한다. 관측량의 같은 고유값은 먼저 합치고 d=1에 rank0 결과를 만들지 않는다. 확률0의 조건부 density는 null이며 퇴화한 조건부 상태는 혼합일 수 있다.

전체 분산을 준비 내부/준비별 평균 차이의 두 항으로 분해하고, 각 항은 같은 rho에서도 원래 ensemble에 따라 달라질 수 있음을 보여준다. B·Gram과 정규화 진폭만 방향성 제곱근 구간을 사용한다. 나머지는 정확 복소 유리수이며 출력 norm을 보정하지 않는다.

## 확인한 근거

- 새55검사와 기존420검사, 합계475검사·타입·전체 lint 통과.
- 독립43참조 사례(39개 유일 입력)·모든4차원값. SymPy Gaussian rank·Berkowitz·독립 고유공간·실제 pair projectors와 mpmath250dps/210자리 좌표로 대조했다. 생산 모듈을 재사용하지 않았다.
- 정확 유리수 인증15073회·방향성 관측3558회. 구간 폭은 max(1,|참조|)×10⁻¹²⁵ 이하다. 참조 재생성의 바이트 일치를 확인했다.
- 같은 rho의 Z/X/Y ensemble, 같은 Z histogram의 다른 순수/혼합 상태, 허수 coherence의 Y 부호, 작은 점유 방향의 정확 rank, 퇴화 측정·영 확률·raw 위상/재척도·입력 소유권을 확인했다.

손계산에서 rho=[[7/10,3/10],[3/10,3/10]]의 순도는19/25=0.76이고, 대각 성분만 남긴 순도는29/50=0.58이다. 차이9/50=0.18은 비대각 coherence 제곱이다. 초기 손계산 기대값에 대각 성분만 사용했던 문제를 독립 참조와 대조해 수정했으며 계산 모형의 정의나 검사 정밀도를 바꾸지 않았다. 전체 Z 분산21/25도 두 분산 항으로 검산한다.

이번 단계에서 새 빌드·브라우저·Worker·reader를 검증한 것은 아니다. 이전 수치·참조·한영 본문·등록 화면과4360개 정적 출력은 보존했다. 전체10492검사·1796 HTML은 앞선 spin 통합의 기록이며 새 density 읽기의 근거로 세지 않는다.

## 다음 순서와 전체 범위

다음은 `qubit-density-spectrum-entropy-dephasing`의 계산 기반이다. 이후 두 상세 한영 본문 → Worker → 개념·실험·시각화·예측·안내·퀴즈·저장·모션 → 등록·실제 읽기·접근성·정적 출력 순서로 진행한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 넣지 않는다.

일반 density 이론·부분추적/복합계·연구 깊이와 전체51영역509주제·공용 그림 후속은 계속 남는다. 공개 배포·commit·push는 수행하지 않았다. 전체 목표는 진행 중이다.

근거: `.runtime/learning/quantum-density-ensemble/verification.json`, `reference-checks.json`, `combined-tests.log`, `reference-reproduction.log`, `typecheck.log`, `lint.log`, `preservation.json`. [집필 계약](quantum-density-authoring-plan.md)과 [유도/검산](quantum-density-ensemble-source-audit.md)에 범위를 기록했다.

```bash
npx tsx --test scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-density-ensemble-references.py --output .runtime/learning/quantum-density-ensemble/reference-regenerated.json
npm run typecheck
npm run lint
```
