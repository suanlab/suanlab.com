# Euclidean 양자 커널: 정확한 유한 모형 계약

2026-11-10. `euclidean-quantum-kernels-and-time-slicing`의 생산 수치 모형과 독립 응답 검산의 계약이다. [한영 원고](euclidean-kernels-manuscript-status.md)의 사다리꼴 potential 규칙을 사용한다. [실제 Worker와 수명 관리](euclidean-kernels-worker-status.md)를 검증했으며 학습 화면·실제 reader는 이후 단계다.

## 입력과 소유권

소유 필드는 정확히8개다. `intervals`는 정수2…32,`probe`는 정수0…N이며 숫자−0을 거부한다. 나머지 여섯 필드는 유리수 문자열이다.

- `time`,`mass`,`hbar`: 양수이며16이하.
- `omegaSquared`:0이상16이하. 자유 경우의`0/7` 같은 표기도 허용한다.
- `start`,`end`: 절댓값16이하.

분자·분모는 각각12자리 이내다. 분모는 양수이며 소수·지수·공백·선행0·양수 부호·음의0을 거부한다. 분모가 생략되면1이다. 입력 범위 판정에서 쓰는12자리 정수와 분모의16배는 안전한 JavaScript 정수 범위 안에 있다. 산술 모형은 정규화된 정확분수를 사용하지만 반환한 입력은 원래 철자를 보존한다.

descriptor로 자체 필드만 복사하며 getter·누락·추가·symbol 필드는 거부한다. 호출자가 나중에 원본을 수정해도 소유 입력에 반영되지 않는다. 원래 표기를 포함한 입력 귀속과 Worker requestId/version/kind의 전송 검증을 모두 수행한다. 필드 삽입 순서는 정규화하지만`1`과`2/2` 같은 원래 분수 표기는 구별한다.

## 생산 수치 경로

δ=τ/N,s=m/(ℏδ),v=ω²δ²,Q=s(L+vI)다. 점화식D₀=1,D₁=2+v,D_k=(2+v)D_(k−1)−D_(k−2)와 닫힌 역행렬로 양끝 포함33×33까지의 공분산,31×31까지의 내부 정밀도·평균·LDLᵀ·행렬식을 계산한다. N2 forcing은 양끝 항을 더한다.

`kernel`은 앞 인자 제곱×π인P,대각/교차 지수 계수g,h,최소 작용과 오른쪽 끝점 규칙의 로그 보정을 반환한다. π·제곱근·지수 자체는 계산하지 않는다. `expectedAction`은 평균 경로의 최소 작용,요동 기여(N−1)/2,기대 운동/위치 작용을 구별한다.

`trace`는 구속 potential에서 유한 trace 제곱을 반환한다. 자유 경우는`divergent-unbounded-line`과`null`이며0이나 유한 partition function으로 대체하지 않는다. `endpointIntegral`은 끝점 적분의 제곱 앞 인자와 지수 계수다. 양의 커널을 질량1 전이 법칙으로 해석하지 않는다.

## 전체 행렬의 전송 표현

최대 분수의 대칭/반사 반복을 중복 저장하지 않도록 공분산은 `covariance.values`와`covariance.indices`로 표현한다. 첫 등장 순서로 서로 다른 정확분수를 등록하고128개씩 묶는다. 모든(N+1)²인덱스를 유지한다. ID의 분수는 `values[Math.floor(id/128)][id%128]`다. 이는 정확한 전체 행렬의 저장 형식이며 행렬 크기나 정밀도를 줄인 근사가 아니다.

기존`learning-qubit-wire.ts`의 배열256개·문자열65536자·전체 문자열2000000자 한도를 변경하지 않는다. chunk는128개,인덱스 행은 최대33개다. 입력 한도를 축소하거나 긴 분수를 소수로 바꾸지 않는다. 반환값의 나머지 배열도 기존 한도 안에서 검사한다.

## 정련 범위와 독립 검산

N,2N,4N,8N에서 같은 물리 시각의 평균/분산·커널·trace·오른쪽 보정을 비교한다. 최초 행만 전체 행렬 계산이고 나머지는 최대256구간 스칼라 해석 계산이다. 연속 Mehler 값·인증 구간·임의 연산자 수렴은 이 응답에 포함하지 않는다.

독립 검산은 다른 유리수 산술로 시간 간선 발생 행렬을 구성하고,LDL 소거·삼각 연립방정식으로 공분산 열과 평균을 구한다. 정련은 Gaussian 커널의 적분 합성을 사용하며 생산 점화식/닫힌 역행렬을 호출하지 않는다. 생산 산술이나 계산기는 runtime dependency에 없다. 후보 문자열은 산술 피연산자로 파싱하지 않는다.

모든 결과 필드와 저장 인덱스·원래 입력 표기를 재구성한 예상값과 대조한다. 검산 결과는 동결한 소유 snapshot이다. 캐시는 소유 입력4개로 제한한다. 별도 Python 기준은 발생 행렬·Gauss–Jordan·주기 사슬 행렬식·Gaussian 합성을 사용하며 TypeScript 수치 모듈을 실행하지 않는다.

수치 검증 결과·비용·변조 범위는 [수치 구현 상태](euclidean-kernels-numerics-status.md)에 기록한다. [Worker 상태](euclidean-kernels-worker-status.md)는 전체201검사와 실제 브라우저 모듈 Worker를 포함한다. UI·교재 등록·생산 Next 자산은 별도 통과 조건이다.

## Worker와 화면 연결

요청은`version:1,requestId,input`,계산 성공은`version:1,requestId,ok:true,kind:'euclidean-kernel',input,inputKey,result`다. requestId는 양의 안전 정수다. 검산 Worker는 원래 요청·성공 응답의 입력과 중첩 결과의 입력까지 대조하고 전체 결과를 재구성한다. 승인 receipt도 같은 requestId/kind/inputKey에 귀속한다.

Controller는 계산 Worker를 종료한 뒤 독립 검산 Worker를 만들며,두 번의 큰 소유 복사는 작업을 나누어 이벤트 루프에 양보한다. 취소·교체·dispose 시 활성 token이 바뀌면 복사도 중단한다. 기본 전체 기한은120초이고10…120000ms 범위로 설정한다. 입력 한도를 줄이지 않고 실제 최대 사례를 이 기한 안에 확인했다.

React hook은 새 입력과 일치하지 않는 이전 응답을 렌더 시점부터 숨긴다. 비활성·숨김·unmount에서 종료하고,복귀·retry 시 새 요청을 만든다. 잘못된 입력은 Worker 없이 오류를 보고한다. main-thread controller/hook에는 수치 계산기나 독립 산술을 포함하지 않는다. 실제 브라우저 검사에서는 visibility 상태/이벤트를 모의했으며 물리적 탭 숨김 자동화로 표현하지 않는다.
