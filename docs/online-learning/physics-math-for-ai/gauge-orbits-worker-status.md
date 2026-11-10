# 게이지 궤도·트리·앵커의 독립 응답 검증과 실제 Worker

2026-11-10. `gauge-orbits-tree-fixing-boundary-anchors`의 계산/검증 Worker와 React hook을 연결했다. [한영 원고·정확 수치 기반](gauge-orbits-foundation-status.md)에 이어 진행했다. **학습 화면·실제 reader 등록·Next 생산 자산 검증은 아직 남아 있다.**

## 독립 검증

검증기는 계산기와 Hilbert 산술을 가져오지 않는다. 별도 검증용 유리수 산술을 사용하며 모든 수치 연산은 Worker 안에 둔다.

- 동치 여부는 선택한 트리의 정상형 비교 대신 원래 그래프의 **모든 간선 위상 제약**으로 판정한다. 전체/앵커 군의 연결·물질 동치를 각각 확인한다.
- 앵커 호환 숲은 탐욕 union-find를 복제하지 않고 최대 크기 부분집합을 사전식으로 열거한다. 고정 위상과 좌표는 유일한 단순 경로를 찾아 계산한다.
- 두 정상형, 경로 방향과 곱, 물질 Gram, 안정자와 궤도 차원, 앵커 조건, 실제 변환과 모든 잔차를 재구성한다. 공유 선언 문자열 외에 계산기 구현을 재사용하지 않는다.
- 입력 소유권·분수 표기·트리 순서·루트·앵커 순서와 요청 번호를 귀속시킨다. 실제 결과 문자열을 BigInt로 읽지 않고, 제한된 입력에서 구한 정확한 기대값과 비교한다.

## 실행 구조

프로토콜·두 evaluator·두 Worker 진입점, controller/factory와 `use-gauge-orbits-calculation.ts`를 추가했다. 계산 Worker를 정리한 후 독립 검증 Worker가 결과를 확인하며, 분할 복사 중 취소와 늦은 메시지를 처리한다. 취소·교체·deadline·오류·재시도·visibilitychange·unmount를 검사했다. reader 번들에는 정확한 산술 엔진이 들어가지 않는다.

## 검증 결과

- 기존 기반85검사, 새 Worker89/controller18검사를 합쳐 **192검사** 통과. Python77참조 입력의 판정과 정상형은 기반 검사에서 대조한다.
- 기본 결과의 **462개 말단 항목**을 각각 한 곳만 바꾸어 거부했다. 의도된 객체 공유를 제거한 사본을 사용하므로 다른 항목까지 함께 바뀌지 않는다.
- 전체군의 변환을 앵커 변환으로 바꾸는 위조, 열린 앵커 경로의 정보 삭제, 영 물질 안정자·Gram·트리 고정 가능성의 잘못된 주장도 거부했다.
- 실제 module Worker **99입력**: 독립 Python 입력77·예제17·최대 입력5. 전체 결과 해시를 Node 결과와 대조하고, 잘못된 앵커 상태 판정99건·요청 귀속99건을 거부했다. 두 Worker 모두 종료했다.
- 최대12노드·32간선·서로 다른18자리 분수와 0/1/3/12앵커, 서로 독립적인 큰 분수 상태 쌍을 기존 전송 예산으로 처리했다.
- 실제 React hook에서 최대 입력의 숨김/재개·검증 중 교체·재시도·비활성화·해제를 확인했다.14Worker 생성·14종료·활성0·브라우저 오류0이다. visibility 검사는 이벤트 시뮬레이션이며 물리적으로 숨긴 탭 자동화는 아니다.
- 타입·대상 lint·서버 렌더링과 번들 의존성 분리 검사를 통과했다.

명령은 `npx tsx --test scripts/learning-gauge-orbits-{foundation,workers,controller}.test.ts`, `npx tsx scripts/check-gauge-orbits-workers-native.ts`, `npx tsx scripts/check-gauge-orbits-hook-native.ts`다. 근거는 `.runtime/learning/gauge-orbits-worker/`에 보존한다.

## 다음 순서와 전체 범위

다음은 한영 개념 단계·안내 실험·해설 문제, 숲·앵커·루프/열린 경로·변환의 시각화, 정확 입력·저장·접근성 검증이다. 이후 실제 Interactive Book reader·생산 자산을 확인하고, 정준 변수·Gauss 제약과 대칭·게이지 원주제 감사를 진행한다.

미등록 초안1개는 완료 수치에서 제외한다. **588등록 단원·1782안내·312/362입문 검증·50미완료**와 전체51영역509주제·영역 감사·연구 심화를 유지한다. 기존 정적 출력과 이전 작업을 보존했다. commit/push·공개 배포는 하지 않았다.
