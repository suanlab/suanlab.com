# 그래프 U(1) 독립 응답 검증과 실제 Worker

2026-11-09. `graph-u1-local-frames-wilson-loops`의 계산/검증 Worker와 React hook을 연결했다. [한영 원고·수치 기반](graph-gauge-foundation-status.md)에 이어 진행한 단계다. **학습 화면·실제 Interactive Book 등록·Next 생산 자산 검증은 아직 남아 있다.**

## 독립 검증 방식

- `learning-graph-gauge-validation-math.ts`는 별도 유리수·복소수 산술을 사용하며 계산기와 Hilbert 산술을 가져오지 않는다.
- `learning-graph-gauge-validation.ts`는 incidence 행렬을 조립하는 계산기와 달리 스칼라 이차형식을 여러 기저 입력에서 평가하여 Hermitian 행렬을 polarization으로 복원한다.
- 기울기는 사차 에너지의 중심 차분에서 정확한 삼차 오차를 제거하여 Wirtinger 미분으로 환산한다. 해당 노드에 무관한 항은 상수이므로 제외한다. 검증 범위를 줄이지 않고 큰 분모의 불필요한 상쇄를 피한다.
- 역방향 루프는 실제 경로의 순서와 방향을 모두 뒤집어 계산한다. 연결 성분은 union-find 대신 그래프 탐색으로 구한다.
- 에너지뿐 아니라 행렬·기울기·모든 간선과 루프·잔차·그래프 수치·완전성/물리 범위 선언을 검사한다. 입력 분수 표기·경로 방향·원래 요청 번호도 귀속시킨다.

## 실행 구조

프로토콜·계산 evaluator·검증 evaluator, 두 Worker 진입점, controller/factory와 `use-graph-gauge-calculation.ts`를 추가했다. reader에는 입력·메시지·수명 관리만 들어가며 BigInt 계산기는 들어가지 않는다. 계산 Worker를 종료한 후 별도 검증 Worker가 결과를 확인하고, 통과한 소유 데이터만 화면 상태로 채택한다.

입력 교체, 분할 복사 중 취소, 검증 중 취소, 늦은 메시지, 오류·재시도, deadline, 비활성화·visibilitychange·unmount를 확인했다. 실제 탭 숨김은 visibility 이벤트 시뮬레이션이며, 자동화로 물리적 숨김 탭을 시험했다고 주장하지 않는다.

## 검증 증거

- 기존 독립 Python49사례와13예제를 포함한 기반56검사, 새 Worker55/controller18검사를 합쳐 **129검사** 통과.
- 기본 결과의 **231개 말단 항목**을 각각 바꾸어 거부함을 확인. 상쇄되는 국소 오류, 루프 허수부, 틀린 기울기와 근거 없는 Gauss/완전성 주장도 거부.
- 실제 module Worker **65입력**: 참조 입력49개·예제13개·최대예산3개. 결과 전체 해시를 Node 계산 결과와 비교하고, 독립 수치 값은 앞의 Python 기반 검사로 대조한다.
- 실제 Worker에서 에너지 변조65건과 요청 귀속 변조65건 거부. 두 module Worker 모두 종료.
- 최대12노드·32간선·8경로·경로당32단계와18자리 분수, 서로 다른 큰 위상/물질 분모를 시험. 공용 전송 예산을 늘리지 않고 처리했다.
- 실제 React hook 생성14Worker·종료14·활성0, 브라우저 오류0. 타입·대상 lint·서버 렌더링·번들 의존성 분리를 확인.

명령: `npx tsx --test scripts/learning-graph-gauge-{foundation,workers,controller}.test.ts`, `npx tsx scripts/check-graph-gauge-workers-native.ts`, `npx tsx scripts/check-graph-gauge-hook-native.ts`.

증거는 `.runtime/learning/graph-gauge-worker/{tests.log,native-workers.json,native-hook.json,verification.json}`에 둔다. 기존 로컬 정적 출력은 그대로 보존한다. commit/push·외부 배포는 하지 않았다.

## 다음 순서와 전체 범위

한영 개념 단계·안내 실험·해설 문제, 그래프/복소평면/행렬의 시각화, 직접 입력·저장·접근성·모션 검증을 진행한다. 이후 실제 reader·검색·선수 단원·생산 자산을 검증한다. 다음 두 단원은 게이지 궤도·트리 고정·경계 기준, 그리고 정준 전기 변수·Gauss 제약이다.

미등록 초안1개는 완료에 포함하지 않는다.587등록 단원·1776안내·312/362입문 검증·50미완료를 유지한다. 전체51영역509주제·영역 감사·연구 심화와 장이론의 나머지 원주제도 계속 남아 있다.
