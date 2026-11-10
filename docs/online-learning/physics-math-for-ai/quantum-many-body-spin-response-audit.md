# Spin 응답의 소유 구조·수학 귀속 감사

2026-11-09. 여섯 계산 종류 가운데 `spin-chain` 응답에 대한 첫 검증기다. 다른 다섯 종류의 응답 검증이나 다섯 학습 단원 전체의 완료를 뜻하지 않는다.

## 소유 데이터 경계

`learning-many-body-owned-wire.ts`가 caller getter/get trap을 호출하지 않고 data descriptor를 복사한다. Plain object·dense array·유한 primitive만 허용하고 symbol·accessor·상속·cycle·sparse/추가 배열 항을 거부한다. 깊이64·방문250000회·배열65536항·record512항·문자열8192자의 상한이 있다. 공유 subtree는 alias로 보존하지만 cycle은 허용하지 않는다. `__proto__`는 prototype을 바꾸지 않는 소유 값으로 복사한 뒤 정확 schema에서 거부한다.

검증기는 원래 packet이 아닌 동결된 소유 복사본을 반환한다. 복사한 전체 트리에 대해 요청에서 도출한 정확 field 집합·배열 길이·primitive metadata와 모든 관측을 대조한다. 검증한 뒤 caller가 원래 객체를 바꾸는 문제가 없다.

## 요청에서 다시 구하는 수학

`learning-many-body-spin-validation.ts`는 제품 spin engine을 import하지 않는다. 국소 Pauli의 Kronecker 행렬로 H를 만들고 전체 복소 입력 벡터로 Hψ·N·E·H² 모멘트·잔차·분산을 계산한다. 제품의 basis bit-flip 계산 경로와 다르다. 국소 density는 직접 partial trace하여 Bloch·purity를 대조하며 모든 pair의 아홉 Pauli 평균·연결 상관을 확인한다.

Total Z/parity의 대각값에서 commutator norm·평균·분산·보존과 선택 상태의 sharpness를 계산한다. 대각 H일 때만 모든 basis 에너지를 정확히 묶어 level·multiplicity·확률·ground/gap·leakage·상한을 대조한다. 비대각 H에서는 spectrum과 ground 주장이 null이어야 한다. 파라미터·graph·term source/word/support·비용·단위 convention·계약 문장도 요청과 일치해야 한다.

정확 관측은 유리수 endpoints와 수치 표시 metadata를 확인한다. 정규화 좌표는 부호·제곱 관계·N·폭을 인증한다. 유효한 관측 모양의 다른 유리수로 바꿔도 허용하지 않는다. 공급된 residual=0이나 TypeScript 타입을 결과의 수학적 증거로 신뢰하지 않는다.

## 정상·변조·브라우저 증거

기존 독립 spin42준비를 모두 수락하고 전체 기준의 정확16024·방향성775관측을 대조한다. Root 삭제·nested 추가/symbol/accessor·wrong metadata·잘못된 유리수·다른 입력 결과·dense/sparse 배열·대각 level partition·거짓 ground·잘못된 gap/확률 등을 거부한다. 비가환 두-site 및 대각 네-site 결과에서는 모든 관측 위치를 각각 잘못된 유리수로 교체한다.

배열의 추가 속성은 JSON 문자열에서 보이지 않으므로 변조 확인에는 객체 비교를 사용한다. Getter/get trap이 실행되지 않는 것과 freeze/alias·복잡도 상한도 검사한다. 검증 번들에는 계산 engine·root isolation·로그 interval engine이 없다.

Native Chromium module Worker42개를 생성·종료하고 **브라우저에서 번들된 검증기 자체**로 수락 여부를 확인한다. 각 준비의 잘못된 ID·에너지 표시 metadata도 거부한다. 반환된 결과는 Node에서 독립 기준과 재대조한다. 격리 harness/Blob 번들이며 실제 reader나 Next 생산 자산 검증은 아니다. 관측된 최대 검증 시간은 약119ms였으며 환경에 의존한다. 학습 화면의 반응성·전체 취소 동작을 검증했다고 하지 않는다.

## 남은 관문

Fermion Fock·boson raw·boson coherent·MPS·Schmidt의 구조/수학 검증을 이어서 구현한다. 모든 종류의 검증 뒤 controller의 취소·늦은 프레임·재진입·dispose·재시도와 hooks를 확인하고 학습 화면·reader로 연결한다. 현재 공통 응답 guard를 만들어 미검증 가족을 성공 처리하지 않는다.
