# Fermion 응답의 부호·parity·sector 귀속 감사

2026-11-09. `fermion-fock` 응답을 검증하는 두 번째 가족이다. 공통 descriptor-only 소유 복사·정확 의미 트리 matcher를 재사용하며, 다른 네 계산 종류의 응답 검증을 대신하지 않는다.

## 요청에서 도출하는 계산

`learning-many-body-fermion-validation.ts`는 제품의 occupation bit-flip engine을 import하지 않는다. 각 모드의 앞쪽 Z 문자열과 국소 소멸 행렬을 Kronecker 곱으로 구성한다. 생성자는 adjoint이며 전체 행렬 CAR와 소멸자의 열별 row/sign을 구한다. 부호 없는 독립 ladder도 따로 만들지만 fermion의 대체 물리 모형으로 허용하지 않는다.

Offset·onsite·복소 hopping/pairing·density 항을 입력에서 구성하고 pairing의 adjoint를 역순으로 더한다. 정확 Hψ·노름·모멘트·잔차·분산·항별 에너지와 부호 없는 H의 비교를 계산한다. 제품은 잔차 노름으로 분산을 얻고 검증기는 H² 모멘트 차로 얻는다. 공급된 residual=0을 계산의 oracle로 쓰지 않는다.

전체 density·γ·anomalous·홀수 평균·모드 점유·number 분포/상관·N/parity commutator를 직접 대조한다. N 보존과 입력의 sharpness를 구별한다. Even/odd 확률·순수 parity coherence·SSR 호환성은 입력에서 계산한다. Twirl density·purity·에너지·홀수 평균은 원래 상태와 별도로 검사한다. 홀수 평균이0이라는 조건만으로 coherence를 지우지 않는다.

대각 H에서 full Fock level·확률·ground/gap/leakage와 parity/고정-number 최소를 각각 도출한다. Sector의 확률은 전체 입력 노름으로 나누는 projector 무게이며 조건부 확률이 아니다. Formal 입력의 선택 parity ground는 null이다. 비대각 H에는 계산하지 않은 spectrum·ground를 허용하지 않는다.

## 소유 구조와 수학 검증

Request ID·종류·canonical input/key·parameter·기저 convention·정확 field 집합·배열 길이·단위/계약 문장을 검사한다. 모든 정확 관측은 유리수 endpoints·표시 metadata와 일치해야 한다. 정규화 state/probe/residual은 부호·제곱 관계·노름과 좁은 구간을 대조한다. 검증 후 원래 packet 대신 동결된 소유 복사본을 반환한다.

공통 복사기가 getter/get trap을 실행하지 않고 accessor·symbol·상속·sparse/추가 배열 항·cycle·복잡도 초과를 거부한다. 전체 결과의 일부 필드만 whitelist로 뽑아 성공 처리하지 않는다. 아직 공통 성공 guard로 미검증 가족을 허용하지 않는다.

## 정상·변조·브라우저 증거

기존 독립45준비의 정확31905·방향성695관측을 대조한다. 새54검사에서541변조를 거부한다. 복소 formal 두 모드와 선택 parity의 대각 결과에서는 모든 관측 위치를 각각 잘못된 유리수로 바꾼다. 생성 순서·소멸자 row/sign·복소 pairing·실제/naive CAR·다른 입력 결과·보존/sharpness·표시 metadata·배열과 소유 구조 변조도 포함한다.

Formal cat의 홀수 평균이0이어도 SSR 호환성을 true로 바꾸지 못하며, twirl을 원래 pure density로 교체하지 못한다. 선택 even sector의 바닥을 full Fock 바닥으로 주장하는 응답과 비대각 결과에 spectrum을 끼워 넣은 응답을 거부한다. 기존 spin51·Worker214의265회귀와 타입·lint도 확인한다.

Native Chromium module Worker45개를 생성·종료하고 **브라우저의 번들된 검증기**로45결과를 수락했다. 각 준비에서 잘못된 ID와 에너지 표시 metadata도 거부했다. 반환 결과는 독립 수학 기준과 재대조한다. 최종 검증기 번들이 native에서 실행한 bytes와 같음을 확인했다.

격리 preview-origin harness/Blob Worker이며 실제 reader·Next 생산 자산·controller 검증은 아니다. 최대 브라우저 검증 시간 관측값은 약178ms로 환경에 의존한다. 실제 학습 화면의 반응성은 별도 관문이다. 다음은 boson raw/coherent·MPS·Schmidt의 응답 검증, 이어 controller/hooks와 학습 화면·reader다.
