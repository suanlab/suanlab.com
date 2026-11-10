# 실시간 응답 검산: exact 자료·요청 귀속 기반

2026-11-10. [세 모드 수치 producer](real-time-phase-numerics-status.md) 다음으로 독립 exact-core 검산과 요청/귀속 계약을 구현했다. **전체 응답 검산기는 아직 완료하지 않았다.** 새41검사(기존 묶음40+전체 구간 격자1)가 Node.js20.20.2/24.13.0에서 통과했다. 완료 요건13/23과 등록 수치는 유지한다.

## 독립적인 exact 검산

`verifyRealTimeExactCore`는 생산 계산기·그 산술·구간/위상/Airy 모듈을 import하지 않는다. 별도 cross-cancel Rational과 정수 Gaussian arithmetic을 사용한다. 런타임 의존성 그래프 검사도 통과했다.

- 입력의 half-kick/drift/half-kick을 공통 정수 분모로 합성하고 시간 단계마다 직접 반복한다. 생산자의 유리수 행렬 binary power를 호출하지 않는다. 기본/2·4·8배 정련의 모든 map 성분과 symplectic determinant를 대조한다.
- real/complex continuant는 소유 입력에서 만든 정수 격자에 놓여야 한다. 초기 minor·점화 certificate를 검사하고 복원한 **전체 Green 행렬**에 실제 tridiagonal 행렬을 곱해 모든 행/열의 residual을 확인한다. witness를 새 물리 매개변수로 쓰지 않는다.
- signature는 Sturm 부호열 대신1×1/2×2 congruence로 계산한다. 첫 pivot0을 특이성으로 오인하지 않는다. null vector·corank·호환/비호환 초점·parity·affine representative와 모든 정상 방정식, endpoint 상수/작용을 확인한다.
- Abel determinant·cofactor 거리 power·양의 실수부 pivot branch·전체 complex mean·완전제곱 exponent를 정수 격자에서 확인한다. 고정된 양끝의 두 forcing은 N2에서 더한다.
- packet은 별도 canonical flow와 전체 초기 공분산으로 최종 공분산/중심을 계산한다. complex width는 producer의 Möbius 나눗셈 대신 검증된 Σ_qp/Σ_qq와 ℏ/(2Σ_qq)로 대조한다. 두 폭 족과 초점 위상 index를 유지한다.

반환값은 `scope='finite-exact-core-only'`, `fullResponseValidated=false`인 내부 context다. 전체 응답을 exact 자료 대신 넣으면 거부한다. fold에는 이 finite-core gate를 적용하지 않는다. packed 관찰값·진폭/위상·Airy/scale/log는 아직 이 함수가 검사하지 않으며, 전체 검산 receipt나 UI 연결을 만들지 않았다.

## 요청·귀속 계약

reader용 protocol은 getter를 실행하지 않고 version/양의 safe-integer requestId/자체 필드를 확인해 새 frozen 입력을 만든다. source/inputKey/결과의 input 귀속을 확인하지만 header만으로 수학을 인증하지 않는다. `RealTimePhaseUnverifiedSuccess.result`는 `unknown`이다.

raw fraction spelling을 보존하므로 time `2/2`와 `1`의 요청 귀속은 다르다. 파생 exact 값에는 기약 정수/분수만 허용한다. 잘못된 ID·추가/누락 필드·상속·accessor·revoked Proxy와 다른 입력의 응답을 거부한다. 생산 계산을 reader에 import하지 않는다.

## 확인한 사례와 변조

25개 최대/경계 exact fixture에 대해 독립 Python 밀집 기준의 map/inertia와 맞음을 확인했다. 별도 격자는 N2…32 각각에서6종 signed curvature(186사례)와 두 packet 폭 족(62사례)을 확인했다. 생성 입력의 반 구간 시간은 소수 표기 대신 허용된 `N/2` 분수로 구성한다.

작은8모형의 primitive leaf **502개**를 각각 canonical 숫자 변화·metadata 변화로 바꿔 모두 거부했다. 최대 혼합 서로소 N32모형의7개 후반부 scalar 변경과, 다른 mass로 생성한 self-consistent 자료의 input만 바꾼 변조도 거부했다. canonical 문법·추가 필드·symbol·cycle·accessor·상속·NaN·음의0도 검사했다. 범위는 exact subrecord이며 아직 wave 전체 변조 검사라고 세지 않는다.

이 기계의 cold exact 검산 최대는 Node24 약2.0초, Node20 약2.1초였다. Worker 기한/수명 검증과 구별한다. 타입 검사와 신규 파일 lint도 통과했다.

## 함께 수정한 감쇠 표시

최종 복소 값의 두 구간이 모두0을 포함할 때 `rawValueRequiresScaledRepresentation`을 켠다. 이전의 decay 계수 하계만 확인하는 조건은 작은 prefactor를 곱한 뒤 생기는 구분 불능을 놓칠 수 있었다.

N32,time16,mass1/100000,ℏ1/999999999999,양끝15,ε16의 재현 검사를 추가했다. decay<5630이고 ln2>0.69이므로 decay 계수 자체는2⁻⁸¹⁹²보다 크다. 그래도 prefactor와 곱한 raw box는0을 포함한다. 수정 후 scaled amplitude와 log를 보존하며 올바른 표시를 준다. 수치 kernel/위상/행렬 자체나 입력 범위를 바꾸지 않았다.

## 재현과 다음 단계

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-exact-core-check.test.ts scripts/learning-real-time-phase-protocol.test.ts scripts/learning-real-time-scaled-boundary.test.ts
npm run typecheck
```

증거는 `.runtime/learning/real-time-exact-check/verification.json`, 양쪽 Node의 검사/격자/변조 기록·실행 시간·source hash다. 초기 로그와 수정한 격자 생성 입력의 성공 로그도 보존했다.

다음은 독립 구간 산술·큰 위상·정규화/branch·지수 척도와 Airy를 포함한 **전체 응답**의 semantic 검산과 변조 검사다. 그 뒤 실제 계산/검산 Worker의 복사·취소·교체·오류/기한을 검증하고 시각화/학습 화면·reader를 연결한다.

등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362검증·47미완료와 전체51영역509주제·영역 감사·연구 심화는 남는다. 기존 홈페이지 출력과 메뉴는 변경하지 않았다.
