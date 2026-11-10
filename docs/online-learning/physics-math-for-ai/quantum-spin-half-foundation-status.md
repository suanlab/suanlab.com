# Spin·angular momentum: 첫 스핀1/2 수치 기반

2026-11-08. 다음 원래 주제인 spin·angular momentum의 첫 계산 가족을 구현했다. 두 가족 중 유한 ladder와 한영 원고·Worker·교수 화면·실제 읽기는 남는다. 기존556단원·1668안내·원래303/362입문 검증·59미완료를 유지한다.

## 구현한 내용

`spin-half-measurement-su2-rotation`은 초기 복소 상태·측정 방향·회전 방향과 정확 유리수 turn을 받는다. S=ℏσ/2의 측정과 능동 U=exp(−iφn·σ/2), φ=2πν를 계산한다.17개 소유 정수·양의 ℏ/분모·부호 있는 각도·1…16표본을 검증한다.

π의 교대급수 구간·정확 modulo2와 기준점 처리로 반각·2π 부호·4π 주기를 구별한다. 복소 진폭·spin 평균/분산·Born 확률·Bloch와 Rodrigues 대응·ray fidelity·대표 차이를 반환한다. spin 대수·Casimir는 정확 유리수이고 일반 회전은768bit 작업/512bit 공개 구간이다. 각도 매개변수와 물리 시간을 구별하며 결과 norm을 재정규화하지 않는다.

계약과 이후 범위는 [집필·구현 계획](quantum-spin-authoring-plan.md), 직접 유도와 원문·π/구간 논증은 [수치 근거](quantum-spin-source-audit.md)에 기록했다.

## 검증

- 새 스핀42검사와 기존 스펙트럼/구간·pulse·관측량191검사, 합계233검사 통과. 타입·전체 lint 통과.
- 독립32준비·172각도 표본. SymPy의 기호 spin 대수/고유공간과 mpmath250자리의 일반 행렬 exponential을 대조했다. 생산 π·turn 축소·삼각함수를 재사용하지 않는다.
- 정확 유리수 인증2744회·방향성 점 관측18808회. 점 구간 폭은 max(1,|참조|)×1e−125이하다. 독립 참조 재생성이 바이트 단위로 일치했다.
- unitarity·determinant1·반각 합성·역전파·norm·확률/평균 계정·SU2/SO3·2π 부호·4π 주기·Casimir·교환자·active 회전 방향·ℏ 척도·raw 복소 재척도·입력 소유권을 확인했다.
- π 구간의 정확 나머지·폭·불변 캐시, 음수/큰/작은 각도와 기준점 근처의 비영 차이를 구별했다.2π 뒤 같은 ray와 local 통계가 된다는 결과를 제어 분기의 상대 위상까지 무의미하다고 확대하지 않는다.

이번 단계에서 본문·등록·Worker·화면·새 빌드·브라우저 검증은 추가하지 않았다. 앞선 unitary 전체10261검사·1794 HTML을 새 스핀 읽기의 근거로 세지 않는다. 기존 수치·전송·교재·화면과 정적 출력은 보존한다.

## 다음 순서

가중 다항식 좌표와 정규직교 좌표를 대조하는 유한 j의 ladder·교환관계·Casimir 가족 → 두 가족의 상세 한영 원고 → 입력/Worker → 교수 화면·예측·안내·퀴즈·저장·모션 → 실제 Interactive Book 등록·읽기와 접근성·정적 출력 순서다. Online Book 목록에는 해당 교재 카드를 추가하지 않는다.

원래 주제의 읽기까지 검증한 뒤 입문 완료 수를 갱신한다. 전체51영역509주제·연구 깊이·공용 그림 후속과 이후 모든 주제는 계속 남는다. 전체 목표는 진행 중이며 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-spin-half-foundation/verification.json`, `reference-checks.json`, `combined-tests.log`, `reference-reproduction.log`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-spin-half-references.py --output .runtime/learning/quantum-spin-half-foundation/reference-regenerated.json
npm run typecheck
npm run lint
```
