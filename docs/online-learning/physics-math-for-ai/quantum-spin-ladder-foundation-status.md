# Spin·angular momentum: 두 수치 가족의 기반 검증

2026-11-08. 첫 [스핀1/2 측정·회전](quantum-spin-half-foundation-status.md)에 이어 `finite-spin-ladders-casimir-metric`을 구현했다. 두 가족의 상세 한영 원고·Worker·교수 화면·실제 읽기는 남는다. 기존556단원·1668안내·원래303/362입문 검증·59미완료를 유지한다.

## 새 계산 가족

j=0,1/2,…,4의 단일 유한 irrep를 선택한다.21개 소유 정수로 j·양의 ℏ·최대9개 복소 raw 좌표를 지정하며 선택 차원을 넘는 성분은0이어야 한다. 유효한 비영 활성 상태만 계산하며 숨겨진 성분을 잘라내지 않는다.

정규직교화 전의 다항식 기저와 가중 내적에서 Jz·J±·Jx/Jy·Casimir·확률·평균/분산·연속 적용을 정확 유리수로 계산한다. 표준 정규직교 행렬·진폭을 방향성 구간으로 복원하고 두 좌표의 norm·Hermiticity·작용·평균을 대조한다.

J±의 norm을 확률로 부르거나 chain을 매번 재정규화하지 않는다.0 action에는 정규화된 image가 없다. k번 적용의 norm 제곱은 작용^(2k) 단위다. 작은 ℏ에서 정규화 좌표의 구간이 넓어지는 문제는 양의 정확 공통 척도로 조건화했으며, 원시 image와 norm·검사 기준은 유지했다. 자세한 유도는 [가중 기저·검산 계약](quantum-spin-ladder-source-audit.md)에 기록했다.

## 확인한 근거

- 새 ladder62검사·앞선 spin1/2 42검사·기존 수치191검사, 합계295검사 통과. 타입·전체 lint 통과.
- 독립50준비·모든9개 j값·630개 chain 기록. 정규직교 제곱근 ladder에서 시작하는 SymPy similarity transform·직접 모멘트·물리 좌표로 대조했다. 독립 참조 재생성이 바이트 단위로 일치했다.
- 정확 유리수 인증92785회와 방향성 점 관측48672회. 점 구간 폭은 max(1,|참조|)×1e−125이하다. 작은/큰 ℏ의 정규화 image가 같은 정밀도를 유지하면서 원시 norm은 달라지는지 검사했다.
- 가중/일반 adjoint·기저 변환·물리 확률/naive 좌표 histogram·J²와 성분 제곱·교환관계·사다리 끝·nilpotence·0 image·coherent 상태·스핀1/2 Pauli 대응·복소 재척도·입력 소유권을 확인했다.

앞선 첫 가족의32준비·172각도 표본·정확2744/방향성18808 대조와 π/회전 검증은 유지한다. 두 가족의 수치 기반이 준비되었다는 뜻이며 본문·화면·새 단원 등록의 완료가 아니다. 기존10261전체 검사·1794 HTML을 새 spin 읽기의 근거로 세지 않는다. 이번 단계의 새 빌드·브라우저 검증은 없고 기존 수치·원고·화면·정적 출력은 보존한다.

## 다음 순서와 전체 범위

두 가족의 상세 한영 본문4개 → 입력/Worker·실제 Worker 수명주기 → 고정 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 교재 등록·검색·실제 읽기·한영·명암·모바일·키보드/터치·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 해당 교재 카드를 넣지 않는다.

원래 주제의 실제 읽기까지 검증한 뒤 입문 완료 수를 갱신한다. 전체51영역509주제·일반 j의 이론·궤도/정의역·연구 깊이·공용 그림 후속과 모든 이후 영역은 계속 남는다. 전체 목표는 진행 중이다. 공개 배포·commit·push는 수행하지 않았다.

근거는 `.runtime/learning/quantum-spin-ladder-foundation/verification.json`, `reference-checks.json`, `combined-tests.log`, `reference-reproduction.log`, `typecheck.log`, `lint.log`, `preservation.json`이다.

```bash
npx tsx --test scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-spin-ladder-references.py --output .runtime/learning/quantum-spin-ladder-foundation/reference-regenerated.json
npm run typecheck
npm run lint
```
