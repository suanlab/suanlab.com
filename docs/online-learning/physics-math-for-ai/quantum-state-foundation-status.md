# 양자 상태·Hilbert 공간: 두 수치 가족과 한영 원고

2026-11-08. [집필 계약](quantum-state-authoring-plan.md)의 두 수치 가족과 한영 원고4개를 준비했다. **Worker·교수 화면·실제 읽기는 아직 미등록**이다. 등록550단원·1650안내, 원래300/362입문 검증·62미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 두 가족의 역할

| 계획 ID | 수치 준비 | 핵심 구별 |
| --- | --- | --- |
| `complex-state-rays-basis-amplitudes` | C² 복소 유리수와 방향성512bit 정규화 | 상태와 좌표·확률, 전역/상대 위상, 정규직교·Gram 기하 |
| `wavefunction-l2-projection-completeness` | 구간별 affine 복소 함수의 정확 적분·Legendre 사영 | 점 값과 L² 동치류, 밀도와 bin 확률, 사영·재정규화·완비성 |

첫 모형은 상태·비교 상태·선택 기저의 비영 복소 정수 쌍과 전역 phasor·실수 배율을 사용한다. 입력16개는 소유 정수이며 크기10⁶ 이하, 배율 분모는 양수다. 영 상태를 작은 수로 보정하지 않는다. 내적의 첫 인자는 켤레 선형이다. 선택 기저는 정규직교이고 재구성·Gram·확률 합을 정확 유리수로 확인한다. 순수 C² 상태의 실험을 임의 차원·혼합 상태·무한 차원 실험으로 세지 않는다.

두 번째 모형의 입력15개는 복소 구간별 affine g(u), u=x/L을 지정한다. 0<L≤10, 분할점0…1, 차수0…12, bin1…32, 샘플 간격1…64, 정수 크기10⁶ 이하이며 L² norm은 양수여야 한다. ψ=g/√(LZ)로 길이 측도를 유지한다. bin 확률·계수·모멘트·직접 오차는 샘플 적분이 아니라 정확 다항식 적분이다.

유한 사영은 포착 norm을 그대로 유지하고 재정규화 결과를 별도로 제공한다. 영 사영에서는 재정규화와 fidelity가 미제공이다. 점 하나의 대표값이 달라도 L² 상태와 구간 확률이 같을 수 있다. 유한 절단으로 완비성이나 모든 점 수렴을 증명하지 않으며 Hamiltonian·벽 경계조건·연산자 정의역·물리 시간은 지정하지 않았다.

## 독립 검산

- 유한 가족: 생산 복소 유리수 계산과 별도 SymPy 정규화 벡터·켤레 전치·행렬 곱을24개 서로 다른 준비에서 비교.
- 연속 가족: 생산 Legendre 점화식·Horner 적분을 독립 이항 계수 전개·단항식 적분과22준비·247샘플에서 비교.
- 합계 정확 유리수7,103개·방향성 진폭 구간856개 대조. 비유리 진폭 참조는170자리, 작업 정밀도는220자리이며 전체 구간 폭은 max(1,참조 절댓값)×10⁻¹⁰⁰ 이하로 검사.
- 두 독립 생성기를 다시 실행해 저장 참조 JSON과 바이트 단위 일치를 확인.
- 수치·집필61검사, 타입·전체 lint 통과.

같은 계산 기저 확률을 가진 직교 상태, ±i의 상대 위상·켤레 부호, 전역 위상과 배율, 정규직교가 아닌 기저의 단순 제곱 합2와 Gram norm1을 확인했다. 연속 사례는 정확 affine 재구성, 영 사영, 좁은 지지의 질량, 물리 길이 변환, 샘플 수 불변성, 끝점 대표 변경과 정확 비음수 오차·Pythagoras 잔차를 포함한다. 제한된 사례 검산을 전 입력의 형식 증명으로 확대하지 않는다.

## 원고와 다음 순서

각 원고는14절·표시 수식10개·비교표2개·풀이6개다. 한영 수식 일치, 엄격한 Markdown/KaTeX, 손계산·경계 조건을 확인했다.

- 복소 상태·ray: [한국어](complex-state-rays-basis-amplitudes-chapter.ko.md) · [영어](complex-state-rays-basis-amplitudes-chapter.en.md)
- L² 파동함수·완비성: [한국어](wavefunction-l2-projection-completeness-chapter.ko.md) · [영어](wavefunction-l2-projection-completeness-chapter.en.md)
- [공식 원문과 계산 근거](quantum-state-source-audit.md)

다음은 두 가족의 입력·Worker 요청/응답 계약과 취소·교체·오류 복구, 개념 비교·자유 실험·안내·퀴즈·모션·저장 복원이다. 이어 **도서 → Interactive Book**에서 실제 한영·명암·모바일·키보드/터치·접근성·정적 출력을 확인한 뒤에만 입문 완료 수를 갱신한다. 다음 원래 주제는 관측·연산자이며 전체 양자·나머지 영역과 연구 깊이도 계속 진행한다.

이번 준비는 기존 교재의 새 빌드·브라우저 실습·공개 배포·commit·push를 포함하지 않는다. 기존550개 등록 단원과 검증된 정적 출력은 유지한다. 근거는 `.runtime/learning/quantum-state-foundation/verification.json`이다.

```bash
npx tsx --test scripts/learning-quantum-hilbert*.test.ts
python3 scripts/learning/generate-quantum-state-references.py
python3 scripts/learning/generate-wavefunction-l2-references.py
npx tsc --noEmit
npm run lint
```
