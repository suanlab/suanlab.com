# 실시간 응답 검산: Abel 조절 파동

2026-11-10. [무조절 파동 검산](real-time-phase-unregulated-validator-status.md)에 이어 `verifyRealTimeRegulatedWave`를 추가했다. 이 함수는 **조절 파동 subrecord 전체**를 검산한다. packet·Airy 전체 응답, 검산 receipt/Worker와 화면 연결은 아직 미완료다. 완료 요건 13/23과 등록 수치는 유지한다.

## 독립 branch와 정규화

소유 물리 입력의 exact certificate를 먼저 독립 검산한다. 검증된 complex continuant의 연속 비율에서 각 pivot을 별도 checker Rational로 복원한다. 양의 실수부를 확인하고 Cartesian principal square root를 만든 뒤 그 켤레를 정규화한다. pivot별 위상을 곱하므로 **전체 determinant의 principal square root로 바꾸지 않는다.** N4에서 생길 수 있는 전체 부호 차이를 보존한다.

독립 9216비트 directed 산술과 1536비트 halving/doubling 위상으로 다음을 확인한다.

- prefactor `sqrt(s^N/(2πℏ|det Cε|))`, 모든 unit-phase 및 scaled complex 성분
- exact completed-square exponent의 비양수 실수부와 전체 허수 위상
- N log s, log(2ℏ), log|det Cε|, logπ를 각각 합성한 log magnitude
- negative-exponential 관찰값, raw complex 구간과 표시 플래그

생산 계산기·산술·구간/위상·Airy 모듈을 import하지 않는다. exact certificate가 물리 입력과 맞아야 하며, witness를 새 매개변수로 사용하지 않는다. 반환값은 `scope='abel-chain-wave-only'`, `fullResponseValidated=false`다. 조절자가 0일 때는 정확히 null인 subrecord만 허용한다.

## 극소 raw 값의 허용 정밀도

512비트 mantissa의 일반 구간 폭은 최대 8 observation quanta다. raw 값은 생산자의 절대 8192비트 격자에서 exp와 마지막 곱셈의 양수 하계가 사라질 수 있으므로, 다음의 명시적 바닥 오차를 추가로 허용한다.

`raw width ≤ 8·2^(scale−512) + 32·(1+prefactorUpper)·2^−8192`.

이 경우에도 독립 고정밀 raw 구간의 포함 관계, canonical dyadic endpoint, bounded exponent와 mantissa, shared normalization을 검사한다. Gaussian 적분은 유한 양의 조절자에서 0이 아니므로 `[0,0]+i[0,0]`은 인증할 수 없다. 더 좁은 올바른 구간은 허용하며 생산 반올림 바이트 일치를 요구하지 않는다.

바닥 오차는 임의의 raw 폭을 허용하는 장치가 아니다. 생산 exp의 32 guard bits에서 Taylor 항 수는 2743이다. 환원 각도 r은 반복 제곱이 있으면 `[1/32,1/8)`에 놓인다. 양의 Taylor/reciprocal 초기 구간 폭을 작업 quanta 2^16 미만으로 감쌀 수 있다. 반복 제곱의 각 중간 상계는 q=127/128 미만이다. `p q^(p−1)≤1/(1−q)=128`을 써서 초기 폭과 각 rounding injection을 마지막 거듭제곱으로 전달하면, 반복 수 32768 이하에서 폭은 작업 quanta 2^24 이하이다. 32비트 축소와 exponent 입력의 한 quantum 폭을 합쳐 exp 출력 폭은 4개의 8192비트 quanta 이내다. 마지막 복소 곱셈의 격자 오차까지 32·(1+prefactor) 한도가 감싼다. 나머지 scaled/phase 불확실성과 512비트 출력 반올림은 observation-quanta 항으로 제한한다.

## 표시 기준 수정

`rawValueRequiresScaledRepresentation`은 **출력되는 두 raw 구간이 모두 0을 포함하는지**로 결정한다. 기존의 작업 구간 기준보다 뒤 단계인 mantissa 관찰 반올림까지 반영한다. 숫자 kernel이나 입력 범위는 바꾸지 않았다. 예를 들어 양수 작업 하계가 마지막 dyadic 반올림에서 0 이하로 표시되면 사용자는 scaled/log 표현을 보아야 한다. 검산기는 출력된 canonical endpoint에서 이 플래그를 독립적으로 결정한다.

## 검증과 다음 단계

기존 17개 사슬 예제의 조절자가 0인 경우에는 1/3을 넣어 모두 실제 Abel branch를 확인한다. 작은 prefactor 경계와 심한 감쇠 경계를 추가한다. resolved 응답의 모든 primitive field 변조, 최대 혼합 분수 모형의 후반 phase/log/exponent 변조 및 phase·scaled/raw의 coordinated sign 반전, 임의 raw 폭·정확한 0·플래그 변경을 검사한다. getter·상속·소유권·귀속·부분 scope도 유지한다.

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-regulated-wave-check.test.ts scripts/learning-real-time-scaled-boundary.test.ts
npm run typecheck
```

Node.js20.20.2/24.13.0에서 새25검사와 기존 작은-prefactor 경계 회귀1검사, 합계26검사가 모두 통과했다. primitive 변조24개를 거부했다. 타입 검사와 변경 파일 lint도 통과했다. 실행 결과·source hash·17개 사례의 시간과 raw 오차 상계 상수는 `.runtime/learning/real-time-regulated-wave-check/verification.json`에 기록했다. 로컬 실행 시간을 Worker 기한 검증으로 세지 않는다.

다음은 packet의 모든 65개 표본과 Airy 전체 응답을 독립 검산한 뒤 세 mode의 전체 응답 gate를 구성하는 것이다. 실제 Worker·시각화·학습 화면·접근성·reader·홈페이지 회귀 점검은 그 뒤다. 입문 315/362 검증·47개 미완료, 전체 51영역509주제·영역 감사·연구 심화는 남는다.
