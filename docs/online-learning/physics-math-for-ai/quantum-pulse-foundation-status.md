# Unitary evolution: 두 수치 가족의 기반 검증

2026-11-08. 첫 [스펙트럼·Schrödinger 가족](quantum-unitary-spectral-foundation-status.md)에 이어 `piecewise-hamiltonian-pulse-time-ordering`의 계산 기반을 구현했다. 두 가족의 상세 한영 원고·Worker·화면·읽기 연결은 아직 남는다. 새 교재 등록은0개이며 기존554단원·1662안내·원래302/362입문 검증·60미완료를 유지한다.

## 새 계산 가족

두 Hermitian Hamiltonian의 실제 시간 전파 A→B, 양의 시간 역순 B→A, 지속 시간으로 가중한 상수 평균 Hamiltonian을 같은 시간 격자에서 비교한다. 양쪽 전환 시각을 정확히 포함하며 비직교 축의 평균도 전체 Hermitian 지수로 계산한다. 영 지속 시간과 상쇄된 traceless 작용을 별도 처리한다.

상태 진폭·여러 축의 Born 확률·직접 norm·각 Hamiltonian 기대값·전파 행렬과 상태/process 차이를 반환한다. 가환성의 끝점 일치와 중간 스케줄의 같음을 구별하며, 양의 pulse 역순과 진짜 역전파를 구별한다. 작은 작용의 교환자 leading 항과 나머지를 분리한다. 증명과 제약은 [수학·독립 검산 계약](quantum-pulse-source-audit.md)에 정리했다.

## 확인한 근거

- 새 pulse58검사와 기존 스펙트럼/구간71검사·관측량62검사, 합계191검사 통과. 타입·전체 lint 통과.
- 독립35준비·187시간 표본, SymPy 정확 대수와 mpmath250자리 직접 행렬 exponential 대조. 참조 재생성이 바이트 단위로 일치했다.
- 정확 유리수6999회·방향성 점 관측23827회. 점 구간 폭은 max(1,|참조|)×1e−125이하로 확인했다.
- 별도 구간별 DOP85310사례·648실수 성분을3e−11이내로 대조했다. 이 허용 오차를 엄밀한 ODE 오차 상한으로 부르지 않는다. 미해결 영 구간2개는 대조 내부 중점만 사용하고 공개값은 null을 유지한다.
- 비음 prefix 시간·경계·영 시간·scalar/상쇄 평균·가환 prefix 반례·여러 측정 축·재척도·역전파·교환자 부호와 국소 차수·소유 필드 검증을 포함했다.

앞선 스펙트럼31준비·152시간·독립 ODE9사례와 별도 삼각함수 근거는 그대로 유지한다. 이번 단계에서 새 빌드·브라우저·Worker 검증을 했다고 세지 않는다. 기존 정적 출력과 기존 수치·교재·화면은 보존한다.

## 다음 순서와 전체 범위

두 가족의 상세 한영 원고4개 → 입력/Worker와 실제 Worker 수명주기 → 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 등록·검색·Interactive Book 실제 읽기와 접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이고 Online Book 목록에는 교재 카드를 넣지 않는다.

원래 주제의 실제 읽기까지 확인한 뒤 입문 검증 수를 갱신한다. 전체51영역509주제·연구 깊이·공용 그림 후속 작업과 이후 spin·density matrix·entanglement 등의 주제는 계속 남는다. 전체 목표를 완료로 바꾸지 않았다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-pulse-foundation/verification.json`, `reference-checks.json`, `combined-tests.log`, `reference-reproduction.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-pulse-evolution.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-pulse-evolution-references.py --output .runtime/learning/quantum-pulse-foundation/reference-regenerated.json
npm run typecheck
npm run lint
```
