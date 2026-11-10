# 상세평형: worker와 계산 수명주기 검증

2026-11-07. 세 가족의 protocol·evaluator·worker entry·factory·controller·React hook을 구현하고 독립 Node/native 환경에서 검증했다. **학습 UI와 실제 Next 읽기/worker 자산 연결은 미완료**다. 등록527단원·1,581안내, 원래288/362주제 입문 검증·미완료74주제는 유지한다. 전체 교육과정·연구 심화도 계속 남아 있다.

## 요청·응답 계약

`learning-detailed-balance-protocol.ts`는 version1·양의 safe-integer 요청 ID·kind·정규화한 입력을 식별한다. 입력·요청 봉투의 누락·추가 필드, 희소 배열, 범위 밖 정수, 영 분모와 분모보다 큰 분자를 거부한다. 고정 필드 순서의 key와 동결된 사본을 만들고 배열을 소유한다. 세 모델은 정확한 유리수 계산이므로 적응 정밀도 옵션을 추가하지 않는다.

응답은 종류별 전체 필드·행렬/시계열 차원·순서·입력 identity를 검사한다. 정확한 관측은 동일한 정수 유리수 경계만 허용하고 구조적0과 비영 표시 범위 밖을 구별한다. 모형 범위의 최대 약3,330자리 정수를 포괄하는4,096자리 한계를 사용하며 main thread에서 큰 정수 계산이나 공약수 검사를 하지 않는다. 이는 전송 구조 검사이지 임의 송신자의 수치적 진실을 인증하는 절차가 아니다. 수학적 정확성은 모델별 독립 기준과 검사가 담당한다.

없는 제안의 조건부 수락, 비유일한 정상 참조, 이동 없는 점프 법칙,0지지집합 역전이 행과 상수 지시변수의 상관은 해당null 구조를 유지한다. 두 상태의 기록 범위·현재/마지막 행·극한 제공 여부와 혼합 기준 상태도 검사한다.

## 계산 분리와 중지

수치 모델을 import하는 evaluator는 worker 내부에만 있다. controller·hook의 main 의존 그래프에는 세 모델과 정확한 산술 helper가 없으며 SSR에서 Worker 접근은0이다. 요청마다 새 worker를 만들고 교체 시 기존 worker를 종료한다. 같은 입력의 재시도도 새 요청 ID와 새 worker를 사용한다.

상태는 idle, pending(starting/calculating), ready, error다. 실제 worker가 계산 전에 started ACK를 보낸다. 취소는 idle로 돌아가며, UI는 필요하면 사용자가 중지한 상태를 별도로 표현한다. 이전 generation·잘못된 key·종류·입력·응답·ACK가 새 결과를 덮지 못한다. 성공·실패·취소·교체·dispose에서 포트를 분리하고 종료하며 동기 수치 계산 fallback은 없다. hook은 교체 effect 전의 render에서도 새 입력 아래 이전 결과를 노출하지 않는다.

## 현재 검증 범위

새 검사16개와 관련241개, 합계257개가 통과했다. **96개 기준 입력(24+42+30)**의 실제 dispatcher·응답 검사·structured clone·JSON 전송을 확인했다. 이는 각 가족의 독립 수학 기준을 다시 수치적으로 인증했다는 의미와 구별한다. 관련 검사는 기존 모델/본문·산술·등록 보존을 포함한다.

Node 실제 worker entry에서 세 가족의 started ACK에 연결해 controller를 취소하고 각각 새 thread의 정상 요청으로 복구했다. 짧은 Hastings 계산의 CPU가 정확히 어느 시점에 중단됐다는 주장은 하지 않는다. 가짜 무거운 작업이나 대기 장벽을 모델에 넣지 않았다.

격리된 native browser·실제 hook에서 다음을 확인했다.

- 순환·두 상태는 계산 중 상태를 관측한 뒤 취소하고, 짧은 Hastings는 실제 started ACK 다음 microtask에 취소했다.
- 세 가족 재시도, 이전 요청 교체, disabled/null 입력 취소, 파일404 복구, 언마운트와 StrictMode 재마운트가 통과했다.
- 큰 결과2건을 끝까지 전달하고, 없는 수락률·항등·교대·흡수의 의미4건을 확인했다.
- worker28개 생성·28개 종료, 잔존0, stale render·page error0이다.

응답 수신 listener부터 microtask까지의 처리 측정은 이 검사 환경에서 최대 약2.4ms였다. 수치 계산·전송·structured-clone 역직렬화·실제 학습 화면 렌더링 전체의 지연 측정이나 일반적인 성능 보장은 아니다.

Native main 그래프에는 수치 모듈6개가 없고 별도 worker 그래프에는 모두 들어간다. 이는 격리 esbuild 증거이며 **실제 Next production chunk 검증을 대체하지 않는다**. lint·타입 검사도 통과했다. 기존4,223소스와4,280정적 출력은 보존했다. 이번에는 전체 테스트·정적 빌드를 다시 실행하지 않았으며, 직전 등록의8,005개 전체 검사 기록과 범위를 구별한다.

근거는 `.runtime/learning/detailed-balance-worker/verification.json`, `tests.log`, `regressions.log`, `node-verification.json`, `browser-verification.json`, `bundle-metafiles.json`이다. 소유한 QA 브라우저·임시 서버와 Node threads를 종료했고 사용자4175 미리보기는 유지했다.

```bash
npx tsx --test scripts/learning-detailed-balance-worker.test.ts
npx tsx scripts/check-detailed-balance-worker.ts
```

## 다음 단계

세 가족의6단계 개념·3안내·최종3퀴즈, 결과 시각화·저장·모션·표지를 작성한다. [연결 설계](detailed-balance-interaction-plan.md)의 각 시점/준비/참조/null 의미를 유지한다. 별도 화면 검증 후 기존527개 등록을 보존하여 실제 읽기·검색·menu에 연결한다. Next worker 분리·실제 한영/명암/반응형·저장·취소/복구·접근성·전체 회귀를 검증한 뒤 입문 진도를 올린다. 진입은 도서 → Interactive Book이며 Online Book 본문에 추가하지 않는다.
