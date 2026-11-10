# Density matrix: 두 계산 가족과 상세 한영 원고

2026-11-08. 두 수치 가족에 대응하는 상세 원고4편을 완성했다. 각14절·12개 표시 수식·2개 표·8개 해설 문제, 합계56절·48개 표시 수식·8개 표·32개 해설 문제다. 같은 문제의 번역은 중복하여 새 학습 주제로 세지 않는다.

## 원고와 학습 흐름

- 유한 ensemble: [한국어](finite-ensemble-density-gram-tomography-chapter.ko.md)·[English](finite-ensemble-density-gram-tomography-chapter.en.md). 일반 d의 density 조건·양성·convexity·순수 상태의 극점·ensemble 비유일성·Gram 관계에서 측정 복원과 Lüders 상태로 이어간다.
- 스펙트럼과 엔트로피: [한국어](qubit-density-spectrum-entropy-dephasing-chapter.ko.md)·[English](qubit-density-spectrum-entropy-dephasing-chapter.en.md). Bloch ball·안정적인 고유값·일반 entropy 정의와 상한·기저 coherence·부분추적·purification·부호 있는 dephasing·Choi·거리/fidelity를 연결한다.

첫 원고는 2/5의 e₀와3/5의 X 양의 상태를 섞어 순도19/25, 대각화 후29/50, 비대각 기여9/50을 계산한다. 동일 I/2를 만드는 Z/X/Y ensemble의 전체 분산은 같아도 준비 내부/준비 사이 분해는 다름을 설명한다. 퇴화한 측정 결과가 혼합 상태로 남는 qutrit 예와 양의 Y 상태 tomography의 허수 부호도 확인한다.

둘째 원고는 entropy와 준비 label/측정 결과 entropy를 구별하고, 오목성과 doubly stochastic 행렬로 대각 pinching 부등식을 유도한다. Bell 순수 상태의 축약 I/2, ensemble purification과 reset을 손계산한다. 이 복합계와 reset 예는 해석적 확장이며 현재 단일 qubit 엔진의 입력 기능으로 표시하지 않는다.

## 이번 검증

새 원고 검사16개와 이전527개를 합친543개, 타입 검사와 전체 lint를 통과했다. 네 원고의 strict KaTeX·한영 표시 수식 일치·연습 순서·표 열 보존을 확인했다. 정상적인 수식 명령을 누락으로 판정한 초기 테스트 정규식과 특성다항식의 객체 접근을 수정했으며 수치 엔진이나 정밀도 기준은 바꾸지 않았다. 영문 원고에서 발견한 수식 명령의 역슬래시 누락2곳도 수정했다.

손계산 검사는 unequal mixture·분산 분해·Gram 영 고유값 인자·near-parallel rank2·tomography·조건부 null·혼합 조건부 상태·부호 있는 eta 표·log0·퇴화·기저 coherence를 확인한다. 독립된 작은 행렬 계산으로 Bell/고전 공동 혼합의 부분추적과 전체 순도 차이, purification의 축약 상태, reset의 trace 보존과 비unital 성질도 확인한다.

앞 단계의 독립81사례/76유일 입력·정확 유리수46080/방향성32199대조를 회귀 검사에서 유지했다. 원고 테스트의 일반 부동소수점 손계산은 이 고정밀 수치 인증을 대신하지 않는다.

## 등록과 다음 단계

이번에는 새 Worker·교수 화면·reader·정적 출력을 만들지 않았다. 두 단원은 아직 미등록이며558단원·1674안내·304/362원래 입문 주제 검증·58미완료를 유지한다. 기존4360개 정적 파일과 기존 수치 엔진·참조·교수 화면은 보존한다. 웹 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 넣지 않는다.

다음은 입력/Worker 계약·수명 관리·실제 Worker → 고정 개념·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 실제 reader 등록·접근성·정적 출력이다. 전체51영역509주제·일반 차원/복합계 심화·연구 깊이·공용 그림 후속 범위는 계속 남는다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-density-manuscripts/verification.json`, `authoring-tests.log`, `combined-tests.log`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-quantum-density-authoring.test.ts scripts/learning-density-spectrum.test.ts scripts/learning-density-ensemble.test.ts scripts/learning-spin-content.test.ts scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npm run typecheck
npm run lint
```
