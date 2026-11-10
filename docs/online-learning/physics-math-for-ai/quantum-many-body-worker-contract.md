# 다체계 Worker 요청·계산 dispatch 계약

2026-11-09. 다섯 원고의 계산을 분리 실행하는 첫 transport 단계다. 응답을 reader에 공개하는 최종 검증기는 아직 구현하지 않았다. 현재 모듈은 실제 학습 화면에 연결하지 않는다.

## 계산 종류와 입력

| Kind | 예정 단원 | 입력의 소유 정수 수 |
| --- | --- | --- |
| `spin-chain` | finite-spin-chain-local-hamiltonians | 102 |
| `fermion-fock` | fermion-fock-parity-operator-algebra | 74 |
| `boson-fock` | boson-fock-cutoff-boundary-algebra: 원시 Fock 상태 | 52 |
| `boson-coherent` | 같은 단원: coherent 투영 | 26 |
| `mps` | matrix-product-state-contraction-gauge | 269 |
| `schmidt` | schmidt-cuts-bond-rank-truncation | 35 |

입력은 정확히 `{kind,parameters}`다. 각 가족의 기존 input 모듈이 차원·비활성 항·분모·상태·gauge의 범위를 검사한다. `learning-many-body-input.ts`는 descriptor의 소유 값만 읽고 동결된 canonical snapshot을 만든다. Symbol·추가 키·상속·accessor·reflection 오류를 거부하며 caller getter/get trap을 실행하지 않는다.

Identity key는 canonical snapshot의 JSON이다. 필드 입력 순서나 −0 표시는 의미가 같은 입력의 identity를 바꾸지 않는다. Draft와 `{version:1,value:input}` 저장 형식은 최대32768자이며 JSON 파싱에서 사라지는 비정수·underflow·overflow의 어휘 증거를 검사한다. 잘못된 저장을 기본 상태인 것처럼 조용히 성공 처리하지 않는다.

## 요청과 프레임

요청은 정확히 `{version:1,requestId:positiveSafeInteger,input}`다. `readManyBodyRequest`는 원래 객체와 분리된 snapshot을 반환한다. Envelope 오류는 `invalid-request`, 가족 입력 오류는 `invalid-input`이며 started를 보내지 않는다. 소유 data descriptor의 유효한 ID가 있으면 실패 응답에 유지하고, 없으면 null이다.

유효 요청은 먼저 `{version:1,requestId,phase:'started',kind}`를 보낸다. 이후 `{version:1,requestId,ok:true,kind,inputKey,input,result}` 또는 `calculation-failed` 실패를 보낸다. Engine이 제공한 정확 계수·인증 구간·null·축퇴·영 상태 의미를 그대로 유지하며 표시용 Number로 축약하지 않는다. MPS 영 수축은 성공한 원시 대수 결과이지 다른 상태로 보정한 계산이 아니다.

무거운 여섯 engine은 evaluator와 worker에서만 import한다. Protocol의 result type import는 번들에서 제거되며 입력·factory 번들에 수치 엔진이 포함되지 않는지 검사한다. Factory는 호출할 때만 module Worker를 생성하며 모듈 import 자체는 SSR에서 Worker를 읽지 않는다.

## 현재 증거의 범위

기존 독립 기준207준비(spin42·fermion45·boson33·coherent32·MPS29·Schmidt26)의 입력·저장·identity를 검사한다. Bundled 실제 entry를 Node worker_threads와 Chromium module Worker에서 실행하고, started·ID·kind·inputKey·입력 snapshot과 전체 독립 기준의 수학 관측을 대조한다. Chromium 검사는 기존 미리보기 origin의 격리 harness와 Blob 번들을 사용한다. 새 Next export 자산이나 실제 reader를 검증한 것이 아니다.

## 다음 관문

응답 검증기는 소유 data/배열·정확한 field 집합·차원·공통 observation metadata·Q(√2,√3)의 계수와 부호를 검사해야 한다. 요청에 따른 Hamiltonian·state·mode/cut·환경·rank·확률/정규화·실제 잔차의 귀속도 확인해야 한다. TypeScript 타입이나 잘 구성된 임의의 응답만으로 성공을 허용하지 않는다. 정상 기준 외에 필드 삭제·추가·accessor·잘못된 관측 metadata·수학값 오염·다른 입력 결과를 거부하는 검사가 필요하다.

그다음 controller에서 이전 Worker 종료·늦은 프레임·재진입·취소·dispose·실패·재시도를 검증하고 React hooks·저장과 연결한다. 메인 thread에서 실패 계산을 동기 재실행하지 않는다. 이 관문 뒤에 개념·실험·예측·세 안내·quiz·Three.js/표와 실제 reader를 구현·검증한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록은 사용하지 않는다.
