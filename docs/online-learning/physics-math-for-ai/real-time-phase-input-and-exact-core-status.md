# 실시간 위상·고전극한: 입력·큰 위상·정확한 유한 계산

이전 exact core 단계의 기록이다. 후속 [세 모드 수치 producer](real-time-phase-numerics-status.md)에서 실제 진폭/위상·Airy·전체 응답과 전송 상계를 검증했고, pivot 중복 전송은 ratio 인수로 바꾸었다. 아래 수치와 미완료 표시는 당시 단계의 증거다.

2026-11-10. [상세 한영 원고](real-time-phase-manuscript-status.md) 다음으로 reader용 소유 입력, 큰 유리수 위상, 유한 이차 사슬/파동묶음의 정확한 계산과 손실 없는 행렬 전송을 구현했다. **전체 생산 계산기 완료가 아니다.** Airy 인증 계산, 실제 kernel/packet 진폭·위상 관찰, 전체 응답 정밀도/비용과 독립 응답 검산·Worker·화면·reader가 남는다.

## 입력과 교육 예제

`learning-real-time-phase-input.ts`는 다른 모듈을 import하지 않는다. 세 모드 모두 자체 data descriptor만 복사해 새 frozen 객체를 만든다. 누락/추가/혼합 필드, symbol, accessor, 상속/다른 prototype, 잘못된 정수·유리수와 음의 0을 거부한다. 입력 spelling은 유지하며 수학적으로 동등한 분수를 조용히 같은 요청으로 바꾸지 않는다.

- chain: `intervals,time,mass,hbar,omegaSquared,start,end,damping,probe`
- packet: `intervals,time,mass,hbar,omegaSquared,position,momentum,chirp,width,widthMode`
- fold: `control,hbar` — 물리 u와 ℏ는 독립적이다.

N2…32, probe0…N, 양수 time/mass/hbar/width≤16, 나머지 물리 크기≤16, damping≥0, 분자/분모 각각12자리다. packet의 fixed-variance 폭은 σ², hbar-scaled-variance 폭은 r=σ²/ℏ다. 32예제는 자유/역진동자·거친 분할·음의 모드·영 pivot·호환/비호환 초점·양쪽 near-caustic·조절자·극소/서로소 입력·두 packet족·chirp 초점·fold 양/음/0을 포함한다.

## ±2⁶⁴를 넘는 위상의 구간 계산

`learning-real-time-phase-rotation.ts`는 `π=16atan(1/5)−4atan(1/239)`와 교대 급수의 첫 생략항으로 π를 감싼다. 각 reciprocal power와 항의 나눗셈을 정수로 바깥 반올림한다. 항별 누적 반올림도 구간 폭에 포함한다. tan의 유리수 덧셈으로 `tan(4atan(1/5)−atan(1/239))=1`을 확인하고 해당 각이 (0,π/2)에 있음을 보이면 이 identity를 얻는다.

위상 크기의 비트 수와 관찰 정밀도에 128guard bit를 더해 π 정밀도를 정한다. π 구간의 중점으로 정수 회전 수 k를 고르되, **k를 정확한 최근접 정수라고 증명할 필요는 없다**. 어떤 정수 k도 sin/cos의 주기를 보존한다. 정확한 유리수 x와 `2kπ`의 구간 차를 감싸고, 잔여각을 [−4,4]에 포함하는지 검사한다.

잔여각은 관찰 정밀도+64bit에서 Taylor 구간으로 계산한다. 8번째 항 이후의 절대 항비는 1/16보다 작고, 명시한 항 수는 첫 생략항을 한 작업 quantum보다 작게 만든다. 최종 구간은 바깥 반올림하고 알려진 [−1,1]과 교차한다. 작은 비영 위상을 정확한 0으로 바꾸지 않는다. cache는 최대8정밀도만 보존한다.

19개 독립 mpmath 진단은 ±2⁶⁴ 초과, 4096/8192bit 위상, 최대65536bit 분자/분모, 큰 정수 회전 근처와64/512/2048bit 관찰을 포함한다. mpmath의 두 정밀도 일치는 **인증된 기준 구간이 아니다**. 생산 구간의 근거는 위의 유리수 연산과 급수 나머지이며, 독립 소수 계산은 구현 대조 자료다.

## 정확한 유한 사슬과 파동묶음

`learning-real-time-phase-exact.ts`는 전체 Hessian 성분, forcing/상수, continuant/행렬식, Sturm signature, 정상 작용·경로와 고정 시간1/2/4/8배 map을 계산한다. 영 첫 pivot을 허용하고 정확한 특이성은 θ_d=0으로 판정한다. 초점에서는 inverse 대신 null vector·호환 여부·affine family 대표와 Maslov index를 반환한다.

양의 damping에서는 복소 continuant·모든 inverse 성분·양의 실수부 pivots·복소 mean·완전제곱 exponent를 정확히 계산한다. 이는 Born 공분산이나 유니터리 damping 전파가 아니다. packet은 두 폭 족의 전체2×2공분산·중심·중심 작용·Möbius 폭과 branch index를 유지한다. 전체 파동함수의 복소 prefactor를 아직 관찰값으로 계산하지 않는다.

별도 Python의 Fraction·밀집 소거/Gauss–Jordan·1×1/2×2 congruence와25사례를 대조한다. 전체 H/역행렬·조절한 복소 역행렬의 모든 exact cell은 canonical JSON hash로 대조하고, vector/map/refinement/공분산/complex width는 직접 대조한다. 저장 공간 때문에 행렬을 몇 개 표본 성분으로 줄인 hash가 아니다.

## Airy 원점 상수

[양의 Gamma 적분 경계](real-time-airy-origin-certificate.md)로 Ai(0)·Ai′(0)의1536bit reference를 별도로 인증했다. 반올림을 포함한4096bit Taylor 적분과 양의 무한 꼬리를 사용하며, 내부 구간 폭은2⁻¹⁸⁰⁰보다 작다. stdlib checker가 통과했고 mpmath2200/2400bit의10진단도 대조했다. 전체 Airy 생산 구현은 아직 없다.

## 손실 없는 행렬 전송과 남은 budget

`learning-real-time-phase-core-wire.ts`는 전체 inverse를 다음 인수로 지정한다. 내부 첨자는0…d−1이다.

- 실수 inverse: θ 배열과 s. 성분은 `θ_min(i,j) θ_(d−1−max(i,j))/(s θ_d)`다.
- 복소 inverse: 조절한 Θ 배열·`(−is)^k` 배열·Θ_d. 성분은 `Θ_min(i,j) Θ_(d−1−max(i,j)) (−is)^|i−j|/Θ_d`다.

독립 Rational 구현으로 전송 인수에서 **모든 성분을 복원**해 Python 밀집 기준과 비교한다. 고정 끝점·영 성분·부호·복소 부분을 생략하지 않는다. 전체 행렬의 시각적 관찰값과 exact factorization은 이후 함께 제공한다.

기존 wire의 단일문자열65536자·전체문자열2000000자·배열256·node100000·깊이32를 바꾸지 않고 core snapshot을 검사한다. **이 측정은 core만의 비용**이며 Airy/진폭·위상 관찰과 전체 응답을 합친 budget 증명이 아니다. 전체 생산 계약의 `workingPrecisionAndWireBudgetFinalized=false`를 유지한다. 시간 측정도 이 기계의 관찰이며 Worker deadline 보증이 아니다.

## 검증 결과

관련54검사가 Node.js20.20.2/24.13.0에서 통과했다(입력4·위상22·유한core26·원고2). 최종 타입 검사와 신규 파일 lint도 통과했다. 이전의 중복 약분이 많은 검사는 Node24에서 통과했고 Node20 실행은 검사 알고리즘 개선을 위해 중단했다. 기존 로그를 보존하고 동일한 전체 성분 범위의 개선판을 두 버전에서 다시 통과시켰다.

25core 사례의 실제 snapshot 최대는1156385bytes·전체문자열1151819자·단일문자열20979자이며 최대 node수는565다. core 계산/직렬화 관찰의 최대는 이 기계에서 Node24 약6.6초·Node20 약8.1초였다. 독립 전체 성분 복원/검산까지 포함한 최대 검사 시간은 각각 약99/102초로, 생산 core 계산 시간과 구별한다. 기존 wire 한도를 통과했지만 전체 응답을 합친 budget은 아직 아니다.

값의 크기에 대한 보수적 [정상 작용/ℏ 상계](real-time-phase-rational-size-bounds.md)도 유도했다. 이는 모든 유리수의 분자/분모 비트 수나 실행 비용을 같은 수로 감싼다는 주장이 아니다.

## 재현과 범위

```bash
node --import tsx --test scripts/learning-real-time-phase-input.test.ts scripts/learning-real-time-phase-rotation.test.ts scripts/learning-real-time-phase-exact.test.ts scripts/learning-real-time-phase-manuscripts.test.ts
npm run typecheck
```

체크인 fixture는 `scripts/fixtures/real-time-phase-{rotation,exact}-references.json`, 생성기는 `scripts/learning/generate-real-time-phase-{rotation,exact}-references.py`다. rotation 재생성에만 mpmath가 필요하며 Node 검사에는 Python 패키지를 설치하지 않는다. exact 생성기는 `{id,input}` 배열 JSON을 인자로 받고 생산 계산 결과를 읽지 않는다. 입력 배열은 `realTimePhaseExamples`를 JSON으로 저장해 재생성할 수 있다.

증거와 실제 측정은 `.runtime/learning/real-time-phase-core/verification.json`, `real-time-phase-{rotation,exact}/measurements-node{20,24}.json`에 기록한다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362검증·47미완료를 유지한다. 새 단원의 완료 요건은 여전히8/23이다. 원래 경로적분 주제와 전체51영역509주제·영역 감사·연구 심화도 미완료다.

다음은 인증된 chain/packet 진폭·위상 관찰과 Airy 상수·중앙/점근 경로·scaled/log 표현 및 전체 응답 budget이다. 이후 독립 전체 응답 검산 → Worker → 기존 스타일의 시각화/실험/저장 → 실제 Interactive Book → 단원/원주제 감사를 진행한다.
