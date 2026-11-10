# Coherent 응답의 지수함수·세 상태·절단 귀속 감사

2026-11-09. `boson-coherent` 응답을 검증한다. Spin·fermion·raw boson과 함께 네 번째 종류이며 MPS·Schmidt와 실제 reader 완료를 뜻하지 않는다.

## 독립 지수함수 인증

`learning-many-body-coherent-wire.ts`는 x∈[0,32]의 exp(−x)를 **직접 교대 Taylor 합**으로 둘러싼다. 0…512차 항의 양의 크기를2048비트 고정 격자에서 방향성 계산한다. 짝수 부분합은 상한이며 첫 생략 항513의 상한을 빼면 하한이다. 전체 tail은 이 지점부터 감소하므로 이중 경계가 성립한다. 항의 floor/ceil 오차도 합에 포함한다.

생산 모형의 양항·range reduction·inverse/squaring 및 transcendental engine을 가져오지 않는다. 캐시는64개의 동결된 유리수 구간으로 제한한다. 독립245자리 vacuum 확률/진폭 기준을102회 대조하고, 구간 폭10⁻³⁰⁰ 이하를 확인한다. 이는 한 표시용 소수와의 equality 검사가 아니다.

각 확률·진폭의 독립 인증 구간 전체가 응답 구간 안에 들어가야 한다. Probability는[0,1], 진폭은 정확 field 계수×exp(−μ/2), overlap은√S, 정규화 거리 제곱은2(1−√S)로 대조한다. 제곱근 관계는 유리수 제곱 부등식으로 검사하며 root engine을 reader에 넣지 않는다. Joint retention과 mode retention 곱의 차는 정확한0을 포함하는 좁은 구간으로 검사한다.

## 정확 준비와 유한 관측

유리수 복소 α와 μ=|α|²에서 αⁿ/√n!의 정확 field 계수·Poisson 다항 가중치·series norm을 만든다. 모드의 곱으로 전체 raw 계수를 구성하며 raw boson의 검증된 유한 기대값 계산에 이 준비를 제공한다. 가짜 Gaussian-integer 입력 parameters는 결과에서 제거한다. Raw 입력 경로는 별도 회귀로 유지한다.

모든 유한 H·노름·density/γ·number·cutoff CCR·quadrature·probe·유한 spectrum을 이 실제 준비에 대해 계산한다. Coherent recurrence의 `(a_C−α)` 잔차는 상단 계수에서만 남고, 노름 제곱은 μp_top이다. 상태·잔차의 정규화에는 정확 series norm을 사용한다.

무한 상태의 Poisson mean/variance·quadrature·같은 normally ordered H 에너지는 별도로 계산한다. 무한 coherent 상태, 비정규화 projection과 재정규화 finite 상태를 교환하지 못한다. S는 fidelity **제곱**, tail은1−S, 정규화 거리는2(1−√S)다. 무한 바닥 상태나 finite displacement를 대신 구현했다고 주장하지 않는다.

## 소유 구조·변조·브라우저 검사

새42검사에서 독립32준비의 정확178986·방향성5221관측을 대조하고863변조를 거부했다. 한 비영 coherent 결과의 모든 관측을 잘못된 유리수로 교체한다. Root/nested 추가·삭제·symbol/accessor·다른 준비·fake parameters·certificate 변경·배열·잘못된 표시 metadata도 검사한다.

서로 일관되게 위조한 S=1/tail=0/fidelity=1/distance=0도 독립 exp 인증에서 거부한다. Tiny μ의 양의 tail을0으로 바꾸거나 overlap을 fidelity 제곱으로 쓰거나 tail을 정규화 거리로 쓰는 응답도 거부한다. 진공의 정확한 값과 두 모드 product retention, finite 경계 잔차도 포함한다.

기존413회귀(Worker214·spin51·fermion54·raw boson/field94), 타입·lint를 확인했다. 네이티브 Worker152개 생성/종료에서 coherent32·이전120을 새 브라우저 검증기로 수락하고 wrong ID·에너지 metadata를 각각152개 거부했다. 전체 정확428617·방향성8770관측을 독립 기준과 대조했다.

회귀가 생성한 최신 graph는 이번 runtime에 따로 보존하고, 이전 세 단계 graph는 baseline SHA-256에 맞는 원래 bytes로 복원했다. 기존 engine·원고·홈페이지 출력은 유지한다.

## 남은 범위

격리 preview-origin Blob Worker이며 실제 reader·Next 생산 자산·controller 검증은 아니다. 이번 환경에서 coherent 검증 최대 약0.99초, 네 종류 전체 최대 약1.19초가 관측됐다. 실제 화면 반응성은 미검증이며 별도 관문으로 해결해야 한다. 다음은 MPS·Schmidt 응답, 이어 controller/hooks·반응성·학습 화면·실제 reader다.
