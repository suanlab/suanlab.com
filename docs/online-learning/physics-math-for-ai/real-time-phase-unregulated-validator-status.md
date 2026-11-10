# 실시간 응답 검산: 독립 수치 연산과 무조절 파동

2026-11-10. [전체 행렬 관찰값 검산](real-time-phase-matrix-validator-status.md)에 이어, 생산 산술을 import하지 않는 구간 연산과 사슬의 **무조절 파동 subrecord 전체**를 검산한다. 조절 파동·packet·Airy와 전체 응답 receipt/Worker/화면은 아직 미완료다. 완료 요건 13/23과 등록 수치는 유지한다.

## 독립 계산 경로

`learning-real-time-checker-interval.ts`는 런타임 import가 없다. 정수 directed arithmetic, 양의 2·3·4제곱근, 로그, 음의 지수, π, 작은 각도 Taylor와 큰 위상 halving/doubling을 구현한다. 제곱근의 최종 정수는 `r^k≤n<(r+1)^k`로 확인한다.

- 로그: 양의 값을 `2^e r`, `1≤r<2`로 환원한다. `z=(r−1)/(r+1)`의 양의 atanh 급수를 사용하며 잔여항은 `3 z^(2j+1)/(2j+1)`로 감싼다. 반올림으로 조금 넓어진 z도 1/2 미만이다. ln2는 z=1/3에서 별도 계산한다.
- 감쇠: `exp(−x)`는 x를 1/8 이하로 줄여 교대 Taylor의 첫 생략항을 적용한 뒤 반복 제곱한다. `x≥workBits`이면 `e>2`로 `[0,2^−workBits]`를 보증한다. 원래 값의 정확한 0을 주장하지 않는다.
- π: `π=8 atan(1/3)+4 atan(1/7)`과 교대 잔여항을 사용한다. tangent 덧셈과 첫 사분면에서 이 항등식을 확인할 수 있다. 생산자의 Machin 1/5·1/239 상수나 cache를 공유하지 않는다.
- 큰 위상: π로 나머지를 구하지 않는다. 유리수 각도를 `2^k`로 나누어 작은 각도에서 계산하고 `sin(2x)=2sin(x)cos(x)`, `cos(2x)=1−2sin²(x)`로 복원한다. 단위 범위로 교차시키며, k회 복원에 `2k+128`의 추가 작업 비트를 둔다. 각 복원에서 구간 폭의 증가는 최대 약 4배이며, 기본 급수의 반올림·잔여항도 포함한다.

위상 분자·분모는 각각 65536비트까지 받는다. magnitude 한도는 8192비트로, [전체 물리 입력의 작용/ℏ 상계](real-time-phase-rational-size-bounds.md)인 4568비트를 포함한다. 임의의 65536비트 크기 각도까지 검산한다고 주장하지 않으며, 원래 N2…32/12자리 물리 입력 범위는 줄이지 않는다.

## 실제 파동 검산

`verifyRealTimeUnregulatedWave`는 소유 물리 입력의 exact certificate를 다시 검산한다. 진폭은 생산자의 dimensionless continuant 식 대신 물리 Hessian determinant를 사용한 `sqrt(s^N/(2πℏ|det H|))`로 계산한다. 로그는 N log s, log(2ℏ), log|det H|, logπ를 각각 합성한다. 작용/ℏ의 회전에는 독립 halving/doubling을 사용하고 signature의 Fresnel octant를 곱한다.

9216비트 작업 구간을 기준으로 amplitude·unit phase·complex value·log magnitude의 모든 관찰값을 비교한다. endpoint는 canonical 512비트 이하 dyadic이어야 한다. 포함 관계·최대 8 observation quanta 폭·공유 척도를 확인하며 생산자의 특정 반올림 바이트를 요구하지 않는다. 초점에는 parity·Maslov index·unit phase·`pointwiseValue=null`을 확인한다. 일반 함수 값과 delta 분포를 혼동하지 않는다.

반환 scope는 `unregulated-chain-wave-only`, `fullResponseValidated=false`다. 전체 wave나 regulated/packet/fold 자료를 이 부분 검산으로 인증할 수 없다.

## 검증과 남은 범위

수치 연산 33검사는 Node.js20.20.2/24.13.0에서 통과했다. 기존 큰 위상 diagnostic fixture 중 물리 magnitude 범위의 사례, 실제 4568비트 경계와 독립 생산 회전의 교차 확인, signed rational/root 격자, π·로그·감쇠를 확인했다. Python Decimal 700/900자리 및 극소 감쇠 4000/4300자리의 새 기준은 **진단용 cell이며 인증 구간이 아니다.** 수학적 포함 근거는 구현한 directed 연산과 해석적 잔여항이다. 9216비트에서 exp(−5630), exp(−6000)의 양수성과 512비트 초과 상대 해상도를 확인했다.

무조절 파동 24검사도 두 Node에서 모두 통과했다. 17개 예제와 네 Fresnel 사분면, 모든 primitive field 변조 53개, 귀속·dyadic/canonical 문법·accessor·상속·전체 응답 scope를 확인했다. 합계 새 57검사이며 타입 검사와 변경 파일 lint도 통과했다. source/evidence hash와 양쪽 Node의 실행 결과는 `.runtime/learning/real-time-unregulated-wave-check/verification.json`에 기록했다. 로컬 cold 검산 최대는 Node20 약3.1초/Node24 약2.9초이며 Worker 기한 증거로 세지 않는다.

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-checker-interval.test.ts scripts/learning-real-time-unregulated-wave-check.test.ts
npm run typecheck
```

다음은 Abel branch·scaled/log/raw 감쇠 표현, packet 65개 표본 및 Airy 전체 응답의 독립 검산이다. 이후 실제 계산/검산 Worker, 시각화·학습 화면, 접근성, reader 등록과 홈페이지 회귀 점검을 진행한다. 입문 315/362 검증·47개 미완료, 전체 51영역509주제와 영역 감사·연구 심화는 남는다.
