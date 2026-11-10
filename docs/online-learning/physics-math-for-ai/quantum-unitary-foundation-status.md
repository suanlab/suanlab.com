# Unitary evolution: 두 수치 가족과 한영 원고

2026-11-08. 스펙트럼 시간 진화와 구간별 Hamiltonian pulse의 두 수치 기반에 상세 한영 원고4개를 추가했다. 현재는 집필·계산 준비이며 Worker·교수 화면·실제 읽기 연결은 남아 있다. 기존554단원·1662안내·원래302/362입문 검증·60미완료를 유지한다.

## 준비한 본문

- 스펙트럼·Schrödinger: [한국어](two-level-spectral-schrodinger-evolution-chapter.ko.md), [English](two-level-spectral-schrodinger-evolution-chapter.en.md). 실제 시간·선형 초기값 문제·unitarity·group 성질·에너지 사영·단위 복원·전역/상대 위상·평균과 분산·Euler norm 증가를 유도한다.
- Pulse·시간 순서: [한국어](piecewise-hamiltonian-pulse-time-ordering-chapter.ko.md), [English](piecewise-hamiltonian-pulse-time-ordering-chapter.en.md). 같은 시각의 prefix·비직교 평균 생성자·순서와 역전파·가환 끝점과 다른 중간 경로·교환자 국소 전개·보존량과 비교 지표를 구별한다.

각 원고는12절과 해설 있는6문제다. 스펙트럼 원고에는 비교표1개·표시 수식7개, pulse 원고에는 표2개·수식8개를 넣었다. 두 언어의 표시 수식은 동일하며 영어 원고에 한글이 섞이지 않는다. 영어 본문은 각각1881·1995단어 수준이고, 한국어는 각각6915·7608자 수준이다. 수치는 최종 본문 통계에 기록한다.

일반 대학원생이 따라갈 수 있도록 회전의 직관, 초기 +X와 Z의 손계산, 에너지 고유상태·scalar 이동, Euler1·2·4걸음, pulse 노출2/3·2/15와 가환 prefix 반례를 설명했다. 단순 histogram 같음·ray 같음·연산자 같음을 혼동하지 않도록 한다. 작은 작용 전개를 유한 pulse 항등식이나 전역 오차 상한으로 표현하지 않는다.

## 확인한 근거

- 원고 렌더·한영 수식 일치·예제 계산13검사와 이전 수치191검사, 합계204검사 통과. 타입·전체 lint 통과.
- Markdown·GFM·수식 파이프라인으로4원고를 렌더했다. KaTeX 오류 없이 총30표시 수식·6표·48개 절·24해설 문제를 확인했다. 표의 열 수와 각 예제의 수치도 별도로 검사했다.
- 손계산 예제를 검증된 생산 모형과 대조했다. X/Z pulse의 +X 확률은 약0.913411과0.5, fidelity는1−sin²2 sin²1≈0.414549다. 같은 Z histogram만으로 같다고 판단하지 않는다.
- 앞선 스펙트럼31준비·152시간·ODE9사례와 pulse35준비·187시간·ODE10사례의 독립 근거를 유지한다. 원문은 Preskill §2.1과 NIST DLMF §4.19를 다시 열어 확인했다. 예제·유도는 이 교재를 위해 직접 전개했다.

이번 원고는 `docs/online-learning/physics-math-for-ai/*-chapter.{ko,en}.md`에 준비했다. 아직 `content/learning/`·단원 목록·읽기 화면에 등록하지 않았다. 이전 수치 모형·참조·교재·화면과 정적 출력은 보존한다. 기존 전체10026검사·1792 HTML을 새 읽기의 검증으로 재사용하지 않는다.

## 다음 순서와 전체 범위

입력/Worker 전송·취소·교체·오류 복구 → 고정 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 교재 등록·목차·검색·표지 → 실제 한영 읽기·명암·모바일·키보드·접근성·정적 출력 순서다. 진입은 **도서 → Interactive Book**으로 유지하며 Online Book 목록에는 교재 카드를 넣지 않는다.

원래 주제의 실제 읽기까지 확인한 뒤 입문 검증 수를 갱신한다. 전체51영역509주제·연구 심화·공용 그림 후속과 이후 양자·장·상대론 등 모든 남은 범위는 그대로다. 전체 목표는 진행 중이며 새 빌드·공개 배포·commit·push는 수행하지 않았다.

근거는 `.runtime/learning/quantum-unitary-manuscripts/verification.json`, `manuscript-checks.json`, `tests.log`, `typecheck.log`, `lint.log`, `preservation.json`이다.

```bash
npx tsx --test scripts/learning-quantum-unitary-authoring.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-quantum-observable.test.ts
npm run typecheck
npm run lint
```
