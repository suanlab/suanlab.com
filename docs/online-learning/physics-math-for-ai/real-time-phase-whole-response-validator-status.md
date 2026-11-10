# 실시간 응답 검산: 세 mode 전체 gate

2026-11-10. `verifyRealTimePhaseResponse`로 chain·packet·fold의 모든 수치 검사와 요청 귀속을 연결했다. **독립 전체 응답 검산 요건을 완료하여 14/23 요건을 검증했다.** 실제 Worker·화면·reader·단원/원주제 감사는 아직 미완료다. 새 단원을 등록하지 않는다.

## 전체 검산 경로

요청의 version·양의 safe-integer ID·소유 입력을 먼저 읽는다. 성공 envelope 전체를 기존 bounded descriptor-only snapshot으로 복사한다. 원래 요청 ID·inputKey·envelope input·result input이 모두 같아야 한다. raw fraction spelling을 보존하며 동치 분수라도 다른 요청에 귀속시키지 않는다.

- chain: 최상위 7필드와 wave 4필드/8192·512 metadata를 확인한다. 전체 packed 역행렬·복소 응답, exact 자료, 무조절 kernel/parity-delta, Abel branch·scaled/log/raw 관찰값을 각각 독립 검산한다. 부분 exact certificate만으로 통과할 수 없다.
- packet: 최상위 4필드를 확인하고 exact flow·전체 공분산·center와 모든 파동 관찰값, 65개 표본을 검산한다.
- fold: 원점 Gamma certificate·중앙/점근 기준으로 실제 Ai·적분·척도·log·실수/복소 정상점·근사·메타데이터 전체를 검산한다.

모든 검사가 끝난 뒤에만 `phase='validated'`, `fullResponseValidated=true`인 receipt를 만든다. response는 모든 깊이에서 frozen인 소유 복사본이다. 호출자가 원래 객체를 바꿔도 영향을 주지 않는다. producer가 보내는 단순한 완료 주장이나 직렬화된 receipt를 검산의 대체물로 받지 않는다.

reader용 protocol은 여전히 입력 파서 외의 계산 모듈을 import하지 않는다. unverified success의 result는 unknown을 유지한다. 검산 이후의 결과 타입만 깊은 readonly 형태다. 실제 Worker의 전용 port·수명·오류 전달은 다음 단계에서 연결한다.

## 수학적 의미와 허용 범위

생산 계산기·그 산술·interval/rotation/Airy 엔진을 런타임 의존성으로 사용하지 않는다. 별도 15모듈 closure를 검사했고 reader protocol의 closure는 2모듈이다. 검산기는 원래 N2…32, 12자리 분수와 세 mode의 전체 물리 입력을 유지한다.

검사는 생산자의 출력 바이트를 복제하는 방식이 아니다. 더 좁은 올바른 구간, 허용 공유 척도, 유효한 fold truncation witness를 허용한다. [Abel raw 격자 한도](real-time-phase-abel-validator-status.md)와 [fold의 문턱 불확실성·work-count 의미](real-time-phase-airy-validator-status.md)도 그대로 적용한다. 인증되지 않은 diagnostic cell을 인증 구간이라고 부르지 않는다.

## 전체 envelope의 비용

32개 전체 producer envelope를 Node20·24에서 계산하고 실제 gate를 통과시켰다. input/ID/귀속 문자열까지 포함한다. 최대1432002bytes·1380446문자·단일20979문자·5605nodes였다. 이 기계의 cold gate 최대는 Node20 약11.5초/Node24 약11.1초였다. 기존 wire 한도를 늘리지 않았다.

기존 유리수 높이·producer 구조 상계를 현재 source hash로 다시 계산했다. inputKey와 owned input, receipt label에 16384자의 여유를 더하면 전체 문자열 상계는 chain1635898·packet766384·fold40384자다. 단일 문자열22041자·node10512·깊이14·배열65 상계로 기존 2000000/65536/100000/32/256 한도 안에 있다.

```bash
python3 scripts/learning/verify-real-time-producer-bounds.py
python3 scripts/learning/verify-real-time-whole-response-bounds.py
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-phase-validation.test.ts
npm run typecheck
```

숫자 크기와 전송 상계는 Worker runtime deadline의 증명이 아니다. cold 검산 시간에는 여러 component가 exact certificate를 다시 확인하는 비용도 포함한다. 실제 계산·검산 Worker의 기한·취소·교체를 별도로 검증한다.

## 통과한 검사

Node.js20.20.2/24.13.0에서 새46검사가 모두 통과했다. 타입 검사와 변경 파일 lint도 통과했다.

- 32개 전체 예제: 최대·최소·근접/정확한 초점·signed curvature·zero pivot·두 packet 폭 족·fold 합류/감쇠/진동을 포함한다.
- 작은 chain·parity·전체65표본 packet·merged fold envelope의 모든 primitive leaf 변조1342개를 거부했다. packed 계수는 허용 오차 밖으로 바꾼다. fold의 terms는 유효하지 않은0으로 바꾼다. 올바른 더 좁은 구간이나 더 많은 유효 항을 오류로 세지 않는다.
- 각 예제의 ID/inputKey/input 귀속 변조96개, 최대 N32의 후반 scalar6개와 coordinated phase-sign 반전을 거부했다. 다른 물리 모형을 완전히 계산한 뒤 입력 label만 바꾼 세 mode의 변조도 거부했다.
- top-level 추가/누락 필드·mode·wave 정밀도, 잘못된 ID·이전 응답·getter·상속·symbol·cycle·revoked Proxy·세 종류의 copy budget 초과를 확인했다.
- caller mutation에서 receipt 불변성을 확인했고, metadata header·부분 certificate·receipt 주장이 전체 검산을 대신할 수 없음을 확인했다.

증거는 `.runtime/learning/real-time-whole-response/verification.json`, 양쪽 Node profile/변조 기록·로그·wire accounting·source hash다. 이전 component 검사와 합쳐 전체 항목을 연결한 gate의 범위를 확인한다.

## 남은 순서

실제 계산/검산 Worker와 controller → 복사·취소·교체·오류·기한 → 기존 홈페이지 스타일의 개념도·Three.js/행렬·packet/fold 실험 → 설명·문제·저장·접근성 → Interactive Book reader/생산 자산 → 기존 콘텐츠 보존과 단원/원주제 감사 순서다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362 검증·47미완료와 전체51영역509주제·영역 감사·연구 심화는 유지한다.
