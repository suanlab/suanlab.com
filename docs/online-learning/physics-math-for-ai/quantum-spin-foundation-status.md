# Spin·angular momentum: 상세 한영 본문 검증

2026-11-08. [두 수치 가족](quantum-spin-ladder-foundation-status.md)에 이어 두 단원의 한국어·영어 본문4개를 작성했다. 일반 대학원생이 정의·유도·손계산·반례·해설을 따라갈 수 있도록 전개했다. 아직 준비 원고이며 Worker·교수 화면·Interactive Book 읽기에는 등록하지 않았다. 기존556단원·1668안내·303/362입문 검증·59미완료는 유지한다.

## 본문과 학습 흐름

- [스핀1/2 한국어](spin-half-measurement-su2-rotation-chapter.ko.md) / [English](spin-half-measurement-su2-rotation-chapter.en.md): Pauli 행렬과 단위 → 방향 측정의 확률·평균·분산 → 성분 제곱과 Casimir → 능동 회전의 반각 → π/2·π·2π·4π 손계산 → ray와 대표 벡터 → 제어 경로의 상대 위상 사고 실험 → Rodrigues 비교·공변성 → 구간 해석·해설 문제다.
- [유한 스핀 ladder 한국어](finite-spin-ladders-casimir-metric-chapter.ko.md) / [English](finite-spin-ladders-casimir-metric-chapter.en.md): 교환관계 → norm 양성과 사다리 종료 → 일반 j의 계수 유도 → j=3/2 표 → 다항식 기저·가중 내적·adjoint → spin1의 다른 무게 손계산 → 연속 작용의 norm과 단위 → coherent 상태·같은 평균의 다른 순수 상태 → 조건화·해설 문제다.

각 본문12절·해설 문제6개다. 언어별 합계48절·display 수식34개·표6개·해설 문제24개이며 서로 다른 학습 단원 수는2개다. 두 언어의 display 수식과 문제 순서가 일치한다. 스핀의 내부 자유도와 고전 자전, 기하학적 각도와 물리 시간, 연산자 image norm과 확률을 구별한다. 유한 j≤4 실험과 일반 j 유도를 분리하고 궤도·정의역·각운동량 결합·혼합 상태·연구 심화는 이후 범위로 유지한다.

## 검사와 실제 발견 사항

새 집필14검사와 이전 수치295검사, 합계309검사를 통과했다. strict KaTeX·표의 행/열 구조·한영 display 수식·영어의 한글 부재를 확인했다. 타입 검사와 전체 lint도 통과했다. 손계산은 수치 기반의 독립 참조 준비와 대조했다.

검사 예시는 +X 회전의 −Y 부호, +Z 회전의 +Y 부호, 2π의 fidelity1·대표 차이4, spin1의 N=5/2·평균4/5·분산 합34/25·raising norm6/5·chain1,6/5,8/5,0, j=3/2의 ladder 계수 제곱0,3,4,3과 Casimir15/4, 같은 평균을 가진 서로 다른 Jz 분포다. 양의 ℏ 재척도에서 기하 각도의 진폭이 유지되는지도 확인했다.

+Z의 π/2 회전에서 이론상0인 횡성분은 일반 삼각함수의512bit 구간에서 부호를 확정할 수 없어 단일 값이 null이다. 검사에0을 강제하는 대신 정확 유리수 끝점이0을 포함하고 폭이10⁻¹²⁵보다 작은지 확인했다. 엔진의 미해결 값 계약과 정밀도 기준을 바꾸지 않았다. ladder 표의 설명은 정규화된 기저 상태 사이의 연산자 계수라는 뜻으로 명확히 했다.

## 검증 범위와 다음 단계

이번 단계는 본문·수학 렌더링·손계산 검증이다. 새 브라우저 실험·Worker·빌드·등록 완료를 뜻하지 않는다. 기존 수치 모델·독립 참조·교수 화면과 정적 출력4354파일을 보존했다. 원래 spin 주제의 입문 완료 수는 실제 읽기 통합 후 갱신한다.

다음은 입력/Worker 계약과 수명주기 → 고정 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 등록·검색·실제 Worker/읽기·한영·명암·모바일·키보드/터치·접근성·정적 출력이다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 이 교재 카드를 추가하지 않는다.

전체51영역509주제와 분량·주차 제한 없는 연구 깊이, 공용 그림 후속 및 나머지 양자·후속 영역은 계속 진행한다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-spin-manuscripts/verification.json`, `combined-tests.log`, `typecheck.log`, `lint.log`, `preservation.json`. 수치 독립 참조는 앞선 두 가족의 검증 기록을 유지하며 이번 단계에서 새로 생성했다고 세지 않는다.

```bash
npx tsx --test scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npm run typecheck
npm run lint
```
