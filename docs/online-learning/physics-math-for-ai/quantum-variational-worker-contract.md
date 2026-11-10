# Quantum variational Worker: 전달·검증·수명주기 계약

2026-11-09. Rayleigh와 비직교 Ritz를 기존 고전 변분법 코드와 구별하는 `learning-quantum-variational-*` 전달 계층이다. 실제 reader 등록·Next 생산 자산·연구 주제 전체의 완료를 뜻하지 않는다.

## 입력과 요청

종류는 `finite-state-rayleigh-energy-variance`, `nonorthogonal-ritz-subspace-certificates`다. 요청은 소유 data 필드만 있는 `{version:1,requestId,input}`이고 id는 양의 안전 정수다. input은 정확히 kind와 parameters를 가진다.36/63개의 유계 정수·양의 분모·비영 상태/계수·차원과 비활성 원소를 확인한다. Ritz는 작은 정확 Gram 행렬식으로 열독립성을 확인한다.

브라우저용 Ritz guard는 수치 엔진을 불러오지 않으며 원래 입력 계약과 모든41기준·기저 변형에서 대조한다. 파라미터를 순서가 고정된 동결 복사본으로 만들고 JSON 키로 요청을 식별한다. 필드 순서와-0 표현은 같은 준비로 정규화한다. 편집/저장 JSON은32768자 이하이며 정확한 정수 token을 확인해 underflow·반올림된 분수·안전 범위 초과를 거부한다.

## 응답의 의미와 모든 소유 필드

유효한 요청에는 `{version:1,requestId,phase:'started',kind}`를 먼저 보낸다. 완료 응답은 version·id·ok·kind·input·inputKey·result의 일곱 필드다. 실패는 version·id·ok·error 네 필드이며 invalid-request, invalid-input, calculation-failed를 구별한다. 계산기 내부 예외 문자열을 결과처럼 전달하지 않는다.

모든 결과 필드·중첩 레코드·행렬·배열·상태 플래그와 해석 계약을 명시적으로 검사한다. 누락·추가·symbol·상속·getter·sparse 배열을 거부한다. 관측은 유한 표시값 또는 null, 부호·해결 상태·로그 상태, 지수0의 양의 분모/정수 분자 경계로 구성한다. 계수 문자열은8192자 이하이다. 수치 표시와 경계 중점의 투영을 대조하고 로그 표시의 플랫폼 차이에만 상대10⁻¹²를 허용한다.

모든 허용 입력에 fixture의10⁻¹²⁵ 폭을 강제하지 않는다. 넓지만 정당한 unresolved 구간도 관측 schema에 허용한다. 실제 결과는 별도의 입력 귀속·정확 항등식·구간 대조를 만족해야 한다. 데이터 검사는 독립 수치 기준을 대체하는 범용 연산자 증명기가 아니다.

## 수학적 귀속과 root 인증

Rayleigh의 원시/정규화 상태·H·중심화·에너지/분산·잔차·방향1/2차 미분·경로/null을 입력에서 작은 정확 대수로 대조한다. Ritz는 Gram/역·A/C·공간 사영·물리 상태·잔차 분해·좌표 변경·분류와 에너지 통계를 대조한다. 사영의 Hermitian 성질·멱등성·rank/trace·완전성·고유공간/Galerkin 관계·점유 확률도 수신한 구간에서 확인한다.

특성 다항식의 정확 square-free 인수와 중복도를 작은 유리수 다항식 계산으로 대조한다. primary 고유값에는 정확 lower/upper·exact·multiplicity를 요구한다. 인수의 끝점 부호·연속성과 차수, 정렬·구간 분리·총 중복도로 전체 root 구성을 검사한다. Rayleigh1024/Ritz4096비트 분리 폭도 확인한다.

Ritz의 ambient/Gram/열 prefix 최소에도 같은 root 경계 메타데이터를 추가했다. 반올림된 숫자만으로 순서와 중복도를 확인하는 약한 방식으로 바꾸지 않았다. prefix 최소는 정확 Sturm 개수로 아래쪽 root 부재와 지정한 root의 유일성을 확인한다. 브라우저에서는 root를 새로 이분 탐색하거나 수치 고유값을 계산하지 않는다. 조건수·interlacing·포함 공간 개선과 양의 gap/null 의미도 연결한다. 추가한 것은 인증 필드이며 이전 수치 값·정밀도·입력 계약은 유지한다.

## 제어기와 React hook

활성 요청마다 실제 Worker 하나를 사용한다. 입력 교체·취소·disable·unmount는 기존 Worker를 종료한다. 긴 동기 계산 뒤에 cancel 메시지가 대기하는 구조로 만들지 않는다. id뿐 아니라 요청 token과 Worker 객체가 일치하는지 확인해 늦은 callback을 무시한다.

상태는 idle, pending(starting/calculating), ready, error다. factory 실패·postMessage 실패·실행 오류·messageerror·잘못된 응답은 정리 후 명시적인 오류 상태가 된다. retry는 새 Worker와 요청을 만든다. 재진입한 subscriber/factory와 ready 중 다음 요청도 검사했다. 완료·오류·종료 때 handler를 해제하고 port를 한 번 종료한다.

`useQuantumVariationalCalculation`은 새 입력의 render와 교체 effect 사이에도 이전 응답을 노출하지 않는다. 활성 여부와 canonical key를 확인하고, SSR/disable에서는 Worker를 시작하지 않는다. 계산 엔진·root 분리·작업 구간 엔진·동기 fallback은 reader hook의 의존 그래프에서 제외한다. 작은 정확 귀속 검사는 main thread에 남으므로 main thread 연산이 전혀 없다고 주장하지 않는다.

## 실행 근거와 후속

119개의 Worker/관측 검사, 기존 수치·원고74검사를 통과했다. 실제 브라우저 module Worker97개가 독립97준비에서 정확9857/방향성7893대조를 통과하고 모두 종료됐다. 별도 StrictMode hook은 종류5회·동일 종류 변경4회·실제 시작 후 취소·재시도·disable·factory/실행 오류·unmount/remount를 확인하고16개를 모두 종료했다.

실행 오류 주입은 오류를 내는 스크립트만 일시적으로 제공하고, 정상 계산은 실제 엔진으로 수행했다. Native 번들은 재생성해 byte 동일성을 확인했다. Blob Worker와 분리한 hook harness 검증이며 Next에서 생성한 생산 Worker 자산과 실제 교재 reader 검증은 아직 아니다. 다음은 개념·자유 실험·시각화·안내·퀴즈·저장 화면이다.
