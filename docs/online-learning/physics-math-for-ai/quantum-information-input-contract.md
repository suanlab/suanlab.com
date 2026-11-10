# 양자정보 입력 소유권과 계산 진입 계약

2026-11-09. 아홉 단원의 reader용 입력과 Worker/CLI용 수학 계산을 분리했다. 아직 실제 Worker나 응답 수학 검증기는 구현하지 않았다.

## 두 검증 단계

`learning-quantum-information-input.ts`는 plain record와 dense array의 **데이터 descriptor만** 읽어 새 객체를 만들고 깊이 동결한다. getter·toJSON을 실행하지 않으며, 알 수 없는 키·symbol·상속 prototype·배열 구멍·순환 입력을 거부한다. 모델 import와 BigInt 계산이 없으므로 reader에서 작은 입력 형식을 확인할 수 있다.

이 단계는 Bloch 성분 범위와 확률 범위·정수 예산을 확인하지만, Bloch 반지름·양자 norm·Schmidt 합·희석 진폭 정규화까지 인증하지 않는다. 형식상 올바르지만 비물리적인 입력은 `learning-quantum-information-calculation.ts`의 수학 단계에서 거부한다. UI는 형식 오류와 계산 오류를 구분해야 한다.

계산 진입점은 다시 입력을 소유한 뒤 정확 유리수/복소 스칼라로 변환한다. 기존 모델의 물리성 검증을 거쳐 `{schemaVersion:1,kind,input,result}`를 반환한다. 입력 snapshot을 포함한다고 응답의 수학적 타당성이나 최신 요청 일치가 자동으로 보증되지는 않는다. 다음 단계의 독립 응답 검증과 Worker 순번 확인이 필요하다.

## wire 형식

| kind | 필수 필드 |
|---|---|
|discrimination|bloch0, bloch1, prior|
|entropy|correlationWeight, coherence, probabilityOne, copies, retainedRank|
|ensemble|bloch0, bloch1, prior, measurement: helstrom 또는 z|
|relative-entropy|rho, sigma|
|bb84|input, comparison, attackFraction, eveZProbability, sampleSize|
|teleportation|input, a, b, order: x-then-z 또는 z-then-x|
|error-correction|input, selectedError, physicalXProbability, errors|
|majorization / compare|mode, source, target|
|majorization / bell-dilution|mode, a, b|
|capacity|probability, input|

Bloch 벡터는 분수 문자열3개다. 순수 입력은 복소 스칼라2개이며 각 스칼라는 `{real:[a,b],imaginary:[c,d]}`로 `(a+b√2)+i(c+d√2)`를 표현한다. fraction은 정수 또는 `n/d`, 각 절댓값 정수 최대32자리, 양의 분모다. 소수/지수 문자열을 조용히 Number로 변환하지 않는다. 입력 예제는 `1/10`, `3/5`, `4/5`처럼 제공한다.

복사본 수는1–64, 표본 수는0–256, 오류 가족은1–16개, Schmidt 목록은1–16개다. rank는 최대20자리 **문자열**이며0–2^copies를 정확히 비교한다. reader는 십진 자리별 doubling으로 경계를 계산해 Number 반올림을 피한다. 내부 계산에서 커진 유리수 계수에는 외부32자리 예산을 다시 적용하지 않는다.

## 초안·저장·요청 정체성

JSON 초안은32768문자·중첩16단계를 한도로 읽으며 중복된 **디코딩 후** 키를 거부한다. 숫자 token은 작은 비음수 정수만 허용하므로 `2e0`, `2.0`, `-0`, 반올림되는 큰 JSON 숫자는 거부한다. 저장은 `{version:1,value:input}` 형식을 사용하며 다른 버전은 거부한다.

`quantumInformationInputKey`는 소유한 입력을 정해진 필드 순서로 직렬화한다. `1/2`와 `2/4`는 수학적으로 같아도 원래 입력 key가 다르다. 계산 결과를 정규화 값만으로 다른 요청에 연결하지 않는다. 외부 원본 변경은 이미 소유한 입력과 결과에 영향을 주지 않는다.

## 교수 예제와 다음 단계

`learning-quantum-information-presets.ts`에9단원44개 한영 예제를 준비했다. 등록 registry와 분리하며, 모든 입력·label·예제 목록을 동결했다. `npx tsx scripts/learning/capture-quantum-information-inputs.ts`로 실제 Worker 검증용 corpus를 재생성한다. 현재 corpus의 상태는 **draft-inputs-not-worker-verified**다.

다음은 응답의 소유권·원래 요청 일치·정확 행렬/분류/수치 구간에 대한 독립 검증, 변조 거부, 실제 Worker 수명주기다. 입력 테스트의 통과를 그 후속 게이트의 증거로 사용하지 않는다.
