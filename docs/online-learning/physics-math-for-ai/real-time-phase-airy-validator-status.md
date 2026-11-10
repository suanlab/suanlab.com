# 실시간 응답 검산: Airy fold 전체 관찰값

2026-11-10. [packet 전체 표본 검산](real-time-phase-packet-validator-status.md)에 이어 독립 Airy 기준과 `verifyRealTimeFold`를 추가했다. 물리 u·ℏ를 유지하며, 실제 Ai·적분·scaled/log 표현과 모든 정상점 자료를 검산한다. **세 mode를 합친 응답 gate·검산 receipt·Worker·화면은 아직 미완료다.** 완료 요건은 13/23으로 유지한다.

## 원점 상수: 별도의 양의 Gamma 급수

`checkerRealTimeAiryOrigin`은 [DLMF8.7.1](https://dlmf.nist.gov/8.7#E1)의 양의 불완전 Gamma 급수를 사용한다. 생산자의 e⁻ᵗ³ 교대 Taylor 적분을 호출하지 않는다. a=r/3, T=1728=12³에서 다음을 계산한다.

`γ(a,T)=Tᵃ e⁻ᵀ Σ Tᵏ/[a(a+1)…(a+k)]`.

Tᵃ는 정확히 12 또는 144다. 항비가 1/2 이하가 된 뒤 남은 양의 급수를 첫 생략항의 두 배로 감싼다. 0<a<1에서 T 이후의 Gamma 적분은 e⁻ᵀ 이하이다. 작업 정밀도 6144비트에서 두 Gamma 구간과 Ai(0), Ai′(0)의 폭이 2⁻²⁴⁰⁰ 미만임을 실행 중 확인한다. 필요한 급수 항 수는 각각 4768/4767이었다. 상수는 frozen cache에만 저장하며 decimal literal을 쓰지 않는다.

기존의 독립 인증된 1536비트 적분 기준과 네 상수 구간이 겹침을 확인한다. Γ(1/3)Γ(2/3)=2π/√3도 별도로 대조한다. 상수 검산을 전체 Airy 검산으로 확대 해석하지 않는다.

## 중앙 및 큰 인수의 별도 계산

중앙 영역에서는 z³=−u³/ℏ²의 정확한 유리수 거듭제곱과 두 coefficient denominator의 정수 곱을 만든다. 각 Taylor 항을 한 번씩 directed 구간으로 변환한다. 생산자의 반올림된 이전 항 점화식을 재사용하지 않는다. 감소하는 항비가 1/2 이하이고 첫 생략항이 2⁻²³⁰⁴보다 작을 때 두 꼬리를 각각 감싼다. |z|≤72의 기존 성장 상계 2⁶⁴⁰을 적용해도 원점 상수 오차·상쇄·양의 scaled 변환에 여유를 남긴다.

점근 계수는 [DLMF9.7.2](https://dlmf.nist.gov/9.7#E2)의 곱에서 유도한 `uⱼ=(6j)!/[864ʲ(3j)!(2j)!]`로 별도 계산한다. ζ²=4|u|³/(9ℏ²)의 짝수 거듭제곱은 유리수이며 홀수 항만 √ζ²로 나눈다. [실수 오차 경계](https://dlmf.nist.gov/9.7#iii)에 따라 각 급수의 signed first-omitted remainder를 2⁻⁷⁶⁸ 이하로 감싼다. 생산자의 640비트 생략항과 다른 계산 경로·정밀도다. 진동 위상은 독립 halving/doubling으로 ζ−π/4를 직접 회전한다.

중앙 |z|≤72, 점근 |z|≥64와 겹침 두 경로를 지원한다. 따라서 정상점 합류를 scaled-control-only 입력으로 바꾸지 않는다. |z|64·68·72 양쪽 부호에서 실제 중앙/점근 결과를 검산한다.

## 전체 응답과 메타데이터

입력 귀속, method/regime, 모든 argument·phase scale·Ai·적분·scaled·감쇠·log 구간을 확인한다. 모든 실수 정상점의 위치/작용/곡률과 복소 정상점 두 branch·실수 contour 기여 여부도 대조한다. 정상 위상 근사는 실제 Airy 적분과 구별한다. endpoint는 canonical dyadic이고 관찰 구간의 포함 관계·폭도 검사한다.

scaled sign과 relative-log 표시는 **반환된 scaled 구간**에서 결정한다. 0을 포함하면 log는 null이어야 한다. `rawValueBelowObservationQuantum=true`는 독립 적분 상계가 2⁻⁵¹² 미만이어야 한다. false는 생산 상계의 문턱 불확실성을 위해 폭 2⁻¹⁰²⁴의 경계 띠에서 보수적으로 허용한다. 그 띠에서 멀리 떨어진 깊은 감쇠의 잘못된 false는 거부한다. 실제 raw 구간을 정확한 0으로 바꾸거나 scaled/log 증거를 지울 수 없다.

terms는 실제 실행 trace나 첫 정지 iteration을 복제하지 않는다. 보고한 항 수가 유효한 **truncation witness**인지 정확한 계수로 확인한다. 중앙/점근의 범위와 항 수 한도, 감소하는 비, 생략항의 보수적 경계를 검사한다. 더 많은 항을 쓰는 올바른 witness는 허용한다. 이 때문에 primitive 변조 검사에서 terms는 유효하지 않은 0으로 바꿔 거부를 확인한다.

반환 scope는 `canonical-fold-response-only`, `fullResponseValidated=false`다. 다른 mode의 응답이나 세 mode 전체 protocol receipt를 이 부분 검사로 발급하지 않는다.

## 확인한 범위와 다음 단계

40개 독립 고정밀 diagnostic 입력은 최대/최소 u·ℏ, 합류·근접 영점·양쪽 signed control과 세 겹침 크기를 포함한다. 새 기준 구간과 diagnostic cell의 겹침, 생산 응답 전체 및 두 branch의 정상점 자료를 확인한다. diagnostic cell 자체를 인증 구간이라고 부르지 않는다. 추가로 문턱 앞/뒤 ℏ=1/520·1/530, raw 입력의 동치 분수 표기와 응답 귀속도 검사한다.

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-fold-check.test.ts
npm run typecheck
```

Node.js20.20.2/24.13.0에서 새50검사가 모두 통과했다. 40기준 사례와 별도 두 문턱 입력, primitive 변조191개를 확인했다. 타입 검사와 변경 파일 lint도 통과했다. 실행 결과·원점 certificate·변조 수·source hash는 `.runtime/learning/real-time-fold-check/verification.json`에 기록했다. 초기 실행의 증거 저장 경로 오류는 분수가 들어간 ID를 파일명으로 인코딩해 수정했다. 수치 오류와 구별하며 초기 로그도 보존한다.

다음은 chain/packet/fold의 top-level 필드와 요청 귀속까지 합친 전체 응답 gate다. 실제 계산/검산 Worker의 복사·취소·기한, 개념도·실험·문제·접근성, reader와 홈페이지 회귀 점검은 이어서 진행한다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362 검증·47미완료와 전체51영역509주제·영역 감사·연구 심화는 유지한다.
