# 유한 장의 열 자유에너지: 생산 계산과 독립 Worker 검산

2026-11-11. [첫 수학 기반](thermal-field-foundation.md) 다음 단계다. 한영 21절 초안은 등록 전 상태로 유지하고, 계산·입력·구간 관측·Worker를 구현했다. 이 보고서는 실제 교재 reader나 홈페이지 새 단원의 완료 보고가 아니다.

## 실제 구현

`learning-thermal-field-input.ts`는 descriptor로 필드를 읽고 분수를 기약형으로 정규화·동결한다. 1/2차원과 평균 영 제약, 온도 경로 jet, 기준 측도를 포함한 14개 명시적 예를 제공한다. 수학적 발산은 입력 오류와 별개인 결과다.

생산 evaluator는 기존 정확 LDL·Gaussian 모멘트 계산과 숫자 관측을 묶는다. **독립 검산기는 생산 엔진·산술·구간 구현을 import하지 않는다.** incidence 행렬, Gauss–Jordan, 최소 에너지와 log determinant의 미분으로 모든 exact 필드를 재구성한다. 분배함수에서 얻은 U·Cv를 생산 모멘트 결과와 비교한다. 요청 ID·전체 입력 key가 다르면 이전 결과를 재사용할 수 없다.

log Z·F·열역학 S·Gibbs SG는 방향 반올림과 나머지를 포함한 유리수 구간이다. 생산 1024bit와 독립 2048bit 산술은 서로 다른 π 항등식을 쓴다. 최종 경계는 512bit dyadic 격자이며 최대 폭은 6quantum이다. 작은 determinant의 로그를 취하기 전에 영으로 반올림하지 않는다. 자세한 규칙과 해석의 경계는 [모형 계약](thermal-field-model-contract.md)에 있다.

## 수학·입력·변조 검사

- 독립 Fraction/Taylor 52모형과 명시적 예 14개, 총 **66응답**이 전체 검산을 통과했다. 정규화 가능 61개, 불가능 5개다. 32점에서 10^-1152 정도인 determinant도 처리했다.
- 원래 52모형의 exact 결과를 Python과 다시 대조했다. 모든 61개 정상 응답의 네 구간은 별도 Decimal/Gauss–Legendre 진단 244개와도 일치했다. Decimal 진단을 포함 구간의 증명으로 사용하지 않는다.
- 기본·2차원·정규화 불가능 응답의 **311 exact leaf**를 하나씩 바꾸어 모두 거부했다. 잘못된·뒤집힌·넓은·비정규·과도하게 긴 구간과 관측 metadata 등 **36개** 변경도 거부했다. 유효한 다른 포함 구간을 무조건 변조로 부르지는 않는다.
- source와 배경의 미분까지 귀속에 포함한다. 동치 분수 입력은 같은 key로 정규화한다. accessor·빈 배열 칸·추가 key·symbol·비정상 ID·과대 배열을 거부하고 getter를 실행하지 않는다. 결과는 소유한 동결 복사본이다.

## Worker 수명과 실제 Chromium

제어기는 생산 계산 결과를 별도 검산 Worker에 보내고 수락 뒤에만 ready를 공개한다. Node 제어기 검사는 교체·각 복사 중 취소·검산 중 취소·재시도·두 단계의 시간 초과·factory/post/error/messageerror 실패를 확인했다. hook은 SSR에서 Worker를 만들지 않고 idle을 렌더링했다. browser bundle 의존성 검사에서 reader hook의 행렬 계산과 독립 검산 엔진 제외도 확인했다.

격리된 localhost 경로를 Playwright가 메모리에서 응답하도록 해 **실제 Chromium module Workers**를 실행했다. 기존 홈페이지 출력이나 서버를 수정하지 않았다. 66개 순차 요청은 Node 결과와 전체 hash가 같고, starting→calculating→validating→ready를 통과했다. 일괄 계산이 끝나기 전에 타이핑이 가능했고 animation frame도 진행했다.

별도 실제 Worker 시나리오는 입력 교체·취소·잘못된 수학 응답·무응답 검산기·두 단계의 EventTarget messageerror를 확인한다. messageerror는 실제 Worker EventTarget에 dispatchEvent로 전달해 오류 경로를 점검했다. 일반 요청의 132개와 수명 검사 11개, 총 **143개 Worker**는 각각 한 번만 종료됐고 message/error 속성과 별도로 등록한 messageerror listener도 제거됐다. 종료 후 활성 Worker는 0개다. 생산 Next chunk나 실제 reader의 화면·hydration·visibility를 검증한 것으로 확대하지 않는다.

## 재현과 다음 단계

```bash
node --import tsx --test scripts/learning-thermal-field.test.ts scripts/learning-thermal-field-manuscripts.test.ts scripts/learning-thermal-field-protocol.test.ts scripts/learning-thermal-field-controller.test.ts
node --import tsx scripts/check-thermal-field-workers-native.ts
npm run typecheck
npm run lint
```

Node20/24에서 관련 **29검사**가 통과했다. 근거는 `.runtime/learning/thermal-field-protocol/`의 응답·변조 기록과 `.runtime/learning/thermal-field-workers/`의 native 결과·의존성·실행 로그다. 새 수학 함수는 아직 등록된 홈페이지 경로에서 호출되지 않으므로 기존 601단원과 1202원고를 수정하지 않았다.

다음은 기존 홈페이지 스타일의 장/에너지/부피 개념 그림, 격자·공분산·열 관측 시각화, 예측과 비교를 포함한 안내 실험, 해설 문제, 실제 reader와 Next 생산 자산 검증이다. 통계장론의 나머지 세 처리와 원주제 감사도 남는다. 입문 원주제 **316/362검증·46미완료**, 등록 전 초안 1개를 유지한다. 전체 51영역 509주제와 연구 심화의 완료를 주장하지 않는다.
