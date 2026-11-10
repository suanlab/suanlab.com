# Tensor product·entanglement Worker 계약

2026-11-08. [두 수치 가족](quantum-mixed-entanglement-foundation-status.md)과 [상세 한영 원고](quantum-tensor-manuscripts-status.md)를 조작형 교재에 연결하기 위한 전달 계층이다. 기존 수치 모델·입력·참조·원고는 변경하지 않았다.

## 입력과 식별

`learning-tensor-input.ts`는 두 `kind`와 정규화한 매개변수 순서를 소유한다. 순수 가족은 기존 가벼운50필드 guard를 사용한다. 혼합 가족은 기존 engine guard의 runtime 의존성을 가져오지 않고, 같은32개 필드·양의 trace·비영 spinor·모든15 principal minor를 작은 정확 유리수 연산으로 검사한다. Engine key 목록과120개의 일반 상태 perturbation에서 두 guard의 판정 일치를 검산한다.

Snapshot은 입력 객체와 매개변수를 freeze하고−0을+0으로 통일한다. JSON key는 종류와 canonical 입력 전체를 포함한다. Draft와 version1 저장값은 소유 필드만 허용하고, JSON.parse 이전의 숫자 표기를 보존하는 기존 parser로 반올림된 정수·underflow·overflow를 거부한다.

요청은 `{version:1,requestId,input}`의 세 필드다. Request ID는 양의 safe integer다. 유효 요청은 같은 종류와 ID의 `started` frame 뒤에 성공 또는 소유 오류 frame을 보낸다. 잘못된 envelope와 잘못된 입력을 구별하며 계산 실패는 명시적 오류다.

## 결과 전체와 수치 계정

Success는 version·ID·kind·입력·입력 key·결과를 모두 요구한다. 두 validator는 결과의 모든 루트/중첩 필드·차원·배열 길이·소유성·상수 관례·null 허용 조건을 확인한다. Sparse 배열·외부 필드·symbol·잘못된 관측값·잘못된 분모·넓은 구간을 거부한다.

관측값은 유리수 상하한, 유한 machine projection, 부호·해상도·로그 메타데이터를 검증한다. 계수는 canonical 십진 정수 문자열4096자 이하, 분모는 양수, exponent는0이다. 구간 폭은 max(1,|상하한|)×10⁻¹²⁵ 이하다. Machine projection은 midpoint에서 재현하고 로그 projection의 일관성도 확인한다.

순수 계정은 입력 계수·노름·정규화 진폭·전체/축약 밀도·Gram 전치·모든 minor·특성 계수·곱 인수·국소 확률/조건부·Householder·기저 dephasing을 대조한다. 조건부 검증은 최대4×4 계수 표의 사영과 Gram을 사용하며16×16 전체 행렬 곱을 반복하지 않는다. 부분전치 행렬·고유값 질량/순도·Schmidt coefficient 제곱·순수 ±pair·negativity 합·entropy 항 합을 확인한다.

혼합 계정은 일반 H/trace·부분추적·Pauli tensor·모든 principal minor·특성 계수·선택 공동 법칙/조건부·정확 CHSH·G와 shifted K를 입력에 묶는다. PPT/얽힘과 최적 CHSH 분류는 정확 minor 부호로 다시 확인한다. 엔트로피 차이·음의 스펙트럼 합·상위 두 G근의 합·최대 CHSH 제곱 관계도 검산한다.

근은 정렬된 서로 다른1024bit 분리 구간과 정확 중복도를 유지한다. 같은 근을 여러 행으로 쪼갠 frame을 거부한다. 특성 다항식·질량·2차 모멘트와의 관계를 검사한다. Reader에서 Sturm 분리·고유값 계산·고정밀 로그를 재실행하지 않는다. 이 계층은 임의의 악의적 sender가 수학적으로 정직함을 증명하는 체계가 아니다. 로그 자체와 일반 스펙트럼의 정확성은 독립 SymPy/mpmath 참조로 검증한 Worker 엔진에 의존하며, 구간 계정은 그 독립 검증을 대신하지 않는다.

## 수명주기와 화면 경계

Controller는 활성 요청마다 Worker 하나를 만든다. 교체·취소·dispose·완료·오류는 handler를 해제하고 해당 Worker를 정확히 한 번 종료한다. 오래된 callback과 다른 양의 ID의 frame은 현재 결과를 바꾸지 못한다. 동기적으로 호출되는 subscriber나 factory가 재진입하여 취소·다음 요청을 시작하는 경우도 검산한다.

Worker 미지원·생성 실패·postMessage 실패·error/messageerror·잘못된 현재 결과는 명시적 오류다. 화면에서 수치 엔진을 대신 실행하는 fallback은 없다. React hook은 mount 후에만 Worker를 만들고, 새 입력을 그리는 effect 이전 render에서도 이전 입력의 ready 결과를 내보내지 않는다. SSR에서는 Worker를 읽지 않는다.

Reader dependency graph에서 수치 엔진·일반 근 분리·구간 초월함수 모듈이 제외됨을 확인한다. 실제 브라우저 검증은 Worker entry를 번들해 Blob URL의 module Worker로 실행한다. 이는 새 Next 정적 asset·최종 reader·접근성 검증과 구별하며, 그것들은 교재 통합 단계에서 수행한다.
