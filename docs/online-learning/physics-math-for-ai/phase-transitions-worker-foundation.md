# 상전이 계산 worker 기반과 검증 경계

> 현재 단계(2026-11-07): 세 단원의 실제 교재 등록과 검증을 완료했다. 아래는 해당 준비 단계의 기록이며, 최신 결과는 [실제 읽기 화면 연결 검증](phase-transitions-integration-verification.md)을 따른다.


2026-11-07. 세 계산 가족을 위한 소스를 준비했다. 교재 등록·실제 브라우저 읽기·배포 완료를 뜻하지 않는다.

## 구성

- `learning-phase-transitions-protocol.ts`: 세 가족의 입력·정밀도 범위, 동결된 복사본과 입력 키, 관측값·유리 구간·상태 및 근 목록의 응답 형태를 검사한다. 숫자로 표시할 수 없는 비영, 수학적 무한대, 구조적0, 미해결을 구별한다. 순환·누락·잘못된 분모·입력 불일치를 거부한다. 임의 발신자의 수학적 계산을 다시 증명하는 검증기는 아니다.
- `learning-phase-transitions-evaluator.ts`와 `.worker.ts`: 무거운 수치 구현의 실행 진입점이다. 유효한 요청에는 시작 메시지를 먼저 보낸다.
- `learning-phase-transitions-controller.ts`: 요청 하나당 worker 하나를 소유한다. 교체·취소·dispose 때 이전 포트를 분리하고 종료한다. 이전 요청의 결과는 새 상태를 덮어쓸 수 없다. 생성·복제·실행·응답 오류에는 동기 계산으로 우회하지 않는다.
- `use-phase-transitions-calculation.ts`: 실제 읽기에 연결할 SSR 안전 hook이다. 렌더 중에는 계산이나 Worker 접근을 하지 않으며, 입력이 바뀐 첫 렌더에서도 옛 결과를 새 입력의 결과로 보여주지 않는다.

## 검증 범위

세 독립 fixture의 총65준비를 실제 dispatcher와 응답 계약으로 대조한다. fake port 검사는 재진입, 교체·취소, 오류 후 정리, 폐기된 콜백, 잘못된 응답·안정성 필드·근 행을 확인한다. SSR에서 Worker를 읽지 않는 것도 검사한다.

esbuild로 실제 worker 소스를 묶어 Node 별도 스레드에서 시작 메시지를 받은 뒤 N=128의 무거운 계산을 중도 종료했다. 새 스레드에서는 다음 요청의 결과가 계약을 통과했다. browser용 hook 의존성 그래프에서도 세 수치 모델과 정확 산술 helper가 제외됨을 검사했다. 이것은 Next 정적 export에서 생성되는 worker 자산과 브라우저 네트워크 실패 복구까지 검증했다는 뜻은 아니다.

16검사를 모두 통과했다. 최종 기록은 `.runtime/learning/landau-sixth-order/worker-tests-final.log`와 `verification.json`에 둔다.

[독립 교수 화면](phase-transitions-teaching-components.md)에서 실제 브라우저의 계산·취소·자산 실패 복구·새로고침도 후속 확인했다. 실제 Next worker 자산과 등록된 읽기 검증은 남아 있다.

## 다음 통합 게이트

고정 개념·안내·퀴즈와 시각화·입력 패널을 연결한 뒤 실제 브라우저에서 계산·교체·중도 취소·unmount·실패 재시도·이전 결과 차단·worker 누수를 확인한다. 한영·명암·감소 모션·작은 화면, 저장 복원 및 정적 자산·링크 검사도 남아 있다. 도서 목록의 진입은 Interactive Book이며 Online Book 본문에는 추가하지 않는다.
