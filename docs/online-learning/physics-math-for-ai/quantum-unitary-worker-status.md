# Unitary evolution 입력·Worker 연결 검증

2026-11-08. 스펙트럼·Schrödinger와 구간별 Hamiltonian pulse의 두 계산 가족에 입력 전송·응답 검증·Worker 수명주기·React 계산 훅을 추가했다. [두 가족과 한영 원고4개](quantum-unitary-foundation-status.md)를 유지하며, 다음은 교수 화면과 실제 읽기다. 새 단원은 아직 미등록이다.

## 연결한 동작

- `learning-unitary-protocol.ts`: 두 가족의 소유 입력·순서에 독립적인 canonical key·불변 복사·JSON 정수의 원래 표기·저장 버전을 확인한다. 반올림으로 정수가 된 분수·underflow·초과 범위·추가 키·다른 가족 매개변수를 거절한다.
- 응답은 요청 ID·가족·입력 key·입력 귀속을 확인한다. 전체 필드·고정 행렬 크기·배열 소유 슬롯·수치 구간과 표시값, 에너지 스펙트럼 계정·물리 시간·Euler 격자·pulse 노출/단계·평균 사용 가능 여부·교환자 계정·분류를 검사한다.
- 정확 유리수와 방향성 구간을 구별한다. unitary·norm·보존량·경계·역전파 잔차에는0을 포함하는 구간을 허용하며 정확한 영이라고 바꾸지 않는다. 공개값 null도 유지한다. 확률·fidelity는[0,1] 구간이어야 한다.
- 한 활성 요청에 Worker 하나를 사용한다. 시작 알림·준비 완료·오류를 구별하며 교체·취소·완료·unmount·오류에서 이전 Worker와 핸들러를 종료한다. 이전 요청의 늦은 결과를 새 입력에 표시하지 않는다.
- `useUnitaryCalculation`과 `useUnitarySeries`는 단일 자유 실험과 고정 비교 표본을 Worker에서 실행한다. 새 입력이 렌더되고 effect가 바뀌기 전에도 이전 결과를 노출하지 않는다. SSR에서는 Worker를 시작하지 않으며 메인 스레드 계산 fallback을 제공하지 않는다.

응답 검증은 임의 발신자의 동역학을 증명하는 절차가 아니다. 생산 모형의 수학은 앞선 독립 행렬 exponential·기호 대수·ODE 근거로 확인한다. 메인 스레드는 제한된 scalar 계정만 검사하고 행렬 exponential이나 삼각함수 엔진을 계산하지 않는다.

## 확인한 근거

- 새 전송·수명주기·실제 Worker entry·SSR79검사와 기존 수치·집필204검사, 합계283검사 통과. 최종 타입·전체 lint 통과.
- 실제 Chromium module Worker에서 독립 참조 입력66개를 모두 실행했다. 결과의 수치 관측41049개를 Node의 검증된 생산 계산과 대조했다. 엄밀 구간·표시값·해결 상태·필드 구조를 비교했고 log 표시값에는 명시적 허용 오차를 사용했다.
- 각 가족에서 시작 알림 직후 종료2회, 실제 controller에서 취소2회·교체·고의 native 오류 후 재시도·dispose를 확인했다. controller가 생성한7개가 모두 종료되어 live0이며 page error는0이다.
- sparse/장식 배열·누락/추가 필드·다른 입력 귀속·잘못된 시간·확률 계정·구간·단계·분류·Worker 오류·clone 실패·재진입 subscriber·늦은 callback을 검사했다.
- reader/controller 의존성 그래프에는 입력·관측 검사·scalar 유틸만 포함되고 두 계산 모형·구간 삼각함수·행렬 엔진은 포함되지 않는다. 실제 Worker entry를 별도로 bundle하여 브라우저에서 실행했다.

이번 native Worker 검증은 localhost의 검증용 module 경로를 사용했다. Next의 최종 Worker asset·실제 교수 화면·읽기 경로는 아직 검증하지 않았다. 새 정적 빌드나 홈페이지 단원 등록으로 세지 않는다. 기존 교재·모형·원고·정적 출력은 유지한다.

## 다음 순서와 전체 범위

두 가족의 준비 목록·입력 해설 → 고정 개념 비교·자유 실험·시각화·예측·안내·퀴즈·저장·모션 → 등록·목차·표지·검색 → 한영·명암·모바일·키보드/터치·접근성·실제 Worker asset·읽기·정적 출력 순서다. 진입은 **도서 → Interactive Book**으로 유지하며 Online Book 목록에는 해당 교재 링크를 추가하지 않는다.

현재554단원·1662안내·원래302/362입문 검증·60미완료를 유지한다. 원래 unitary 주제의 읽기 검증이 끝나기 전에는 완료 수를 올리지 않는다. 전체51영역509주제·연구 깊이·공용 그림 후속과 이후 모든 영역은 계속 남는다. 전체 목표는 진행 중이다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-unitary-worker/verification.json`, `tests.log`, `transport-tests.log`, `native-worker.json`, `browser-dependency-graph.json`, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-unitary-worker.test.ts scripts/learning-quantum-unitary-authoring.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-unitary-worker.ts
npm run typecheck
npm run lint
```
