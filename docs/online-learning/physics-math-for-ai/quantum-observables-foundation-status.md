# 관측·연산자: 두 수치 가족과 한영 원고

2026-11-08. [집필 계약](quantum-observables-authoring-plan.md)에 따라 두 수치 모형과 한영 원고4개를 구현했다. **Worker·교수 화면·실제 읽기는 아직 연결하지 않았다.** 등록552단원·1656안내, 원래301/362입문 검증·61미완료를 유지한다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위도 계속 진행한다.

## 준비한 내용

| 계획 ID | 계산과 설명 |
| --- | --- |
| `hermitian-spectrum-born-expectation` | adjoint·실수 고유값·스펙트럼 사영·Born 확률·평균/분산·수동 기저 변환·비Hermitian 반례 |
| `degenerate-projectors-sequential-observation` | C³ 축퇴 사영·조건부/비선택 Lüders 출력·rank1 세분화·coherence·purity·순차 결합 법칙·commutator |

첫 가족은 정수 복소 상태·축·수동 기저와 유리수 중심·간격16개 필드다. 둘째는 상태·두 축과 네 고유값26개 필드다. 모든 값은 소유 정수·절댓값≤1e6, 양의 분모·비영 벡터를 요구한다. 매개변수는 검증 후 복사·동결한다.

행렬·확률·모멘트·사영 항등식·순차 법칙·출력 purity는 정확 복소 유리수다. 정규화 좌표만 방향성512bit 제곱근 구간을 사용한다. 동일 고유값은 전체 고유공간 사영으로 합치고, 확률0 분기의 조건부 상태·둘째 법칙은 null이다. rank1 세분화는 coarse 관측량과 다른 도구로 표시한다.

각 한영 원고는12절과 풀이6개를 갖춘다. C² 원고는 같은 평균·다른 분포, 유리수 중심/간격의 수치 예제와 기저 불변성을 다룬다. C³ 원고는 중첩 보존/소실·같은 A 수치와 다른 후속 B 확률, 순서 효과의 정확 확률 표와 한 상태로 가환성을 인증할 수 없는 반례를 설명한다. C³ 좌표를 물리 위치라고 부르거나 시간 진화를 흉내 내지 않는다.

## 검증 근거

- 새 수치62검사·집필7검사와 기존 Hilbert 수치54검사, 합계123검사 통과. 타입·전체 lint 통과.
- 독립 SymPy C²24준비·C³26준비, 정확 유리수12028대조·방향성 좌표369대조. 정규화 구간 폭은 max(1,|참조|)에 대한1e−145이하로 확인했다.
- 독립 참조의 재생성이 바이트 단위로 일치했다. 생산 사영 분해와 별개로 고유공간을 구하고, 순차 확률은 밀도행렬 trace로 계산한다.
- 사영의 Hermitian·멱등·직교·완전성, 확률 합, 직접/스펙트럼 모멘트, 수동 기저 재구성, 순차 주변합, 영 분기, 위상·배율 불변성과 음수/동일 고유값을 확인했다.
- 원고4개의 한영 수식 일치, KaTeX 엄격 렌더링,12절·풀이6개와 계산 예제를 확인했다. 준비50개는 서로 다른 입력이다.

[원문 확인과 독립성](quantum-observables-source-audit.md)을 함께 따른다. 근거는 `.runtime/learning/quantum-observables-foundation/verification.json`, `reference-checks.json`, `tests.log`, `reference-reproduction.log`다. 이전 전체 회귀9879검사·정적1790 HTML 검증은 Hilbert 통합 때의 근거이며 이번 새 단원의 전체 읽기 검증으로 세지 않는다.

## 다음 순서와 남은 범위

두 가족의 입력/Worker 계약과 취소·교체·오류 복구를 연결한다. 이어 개념 고정 비교·자유 실험·안내·예측·퀴즈·저장·모션 → 도서 → Interactive Book 실제 읽기를 만든다. 한영·명암·모바일·접근성·정적 출력까지 확인한 뒤에만 입문 완료 수를 갱신한다. Online Book 목록에는 새 교재 카드를 추가하지 않는다.

이 유한 실험은 일반 POVM·측정 장치·혼합 상태 입력·불확정성·연속 스펙트럼·연산자 정의역·연구 깊이의 완성을 뜻하지 않는다. 다음 원래 주제는 unitary evolution·Schrödinger이며 뒤의 양자·장/다중척도·상대론·물질/계산·역문제·연구 방법론 범위도 유지한다.

이번에는 새 빌드·공개 배포·commit·push를 수행하지 않았다.

```bash
npx tsx --test scripts/learning-quantum-observable*.test.ts scripts/learning-quantum-hilbert.test.ts
python3 scripts/learning/generate-quantum-observable-references.py --output .runtime/learning/quantum-observables-foundation/reference-regenerated.json
npm run typecheck
npm run lint
```
