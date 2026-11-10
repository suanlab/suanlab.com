# 상관·요동: worker 계산·수명 기반 검증

2026-11-07 현재 상태: 세 단원을527개 등록 교재에 연결하고 실제 읽기 검증을 완료했다. [최신 통합 기록](correlations-integration-verification.md)을 우선한다. 아래의 준비·미등록 표기는 당시 단계의 기록이다.

> 현재 단계(2026-11-07): 세 가족의 [학습 화면 준비와 별도 native 검증](correlations-teaching-components.md)을 마쳤다. 아래는 해당 준비 단계의 기록이며 다음은 실제 Interactive Book 등록·Next 자산·읽기 검증이다.

2026-11-07. 연결 공분산·주기 Ising 고리·AR(1) 평균 오차의 공통 worker 계층과 React hook을 작성했다. **별도 브라우저 검증 환경에서 검증했으며 실제 Next 읽기 화면·정적 worker 자산은 아직 연결하지 않았다.** 등록524단원·1,572안내와 원래 미완료 입문75주제는 유지한다.

## 구현과 경계

- `learning-correlations-protocol.ts`: 세 가족의 입력 범위·정밀도·정규화 key·불변 snapshot, 요청 ID와 가족별 결과 구조를 확인한다. 구조적0·표시 범위 밖 비영·수학적 무한대·정밀도 미해결을 구별한다. 전송 검사는 임의 sender의 수학적 진실을 증명하는 검증기가 아니다.
- `learning-correlations-evaluator.ts`와 `.worker.ts`: dispatcher만 수치 모델을 import하며 계산 전에 시작 응답을 보낸다.
- controller·factory·`use-correlations-calculation.ts`: 새 요청은 이전 worker를 종료한다. 시작·계산·준비·오류 상태를 관리하고 취소·재시도·입력 교체·비활성화·null 입력·unmount에서 오래된 결과를 숨긴다. 자산 실패 시 주 스레드 계산 fallback은 없다.

## 실제 근거

새 worker 검사 **15개**, 모델·본문·이전 등록 회귀를 합친 **132개**가 통과했다. 세 독립 참조의62개 준비가 실제 dispatcher와 전송 검사를 통과했다. 잘못된 key·입력·관측 누락·유리 경계·순환 구조·가족 표시·정밀도 이력, 생성·clone·실행·역직렬화 오류도 검사했다.

실제 Node thread에서 세 수치 분기를 각각 시작한 뒤 계산 중 종료했고, 종료 전 성공 결과가 없음을 확인했다. 새 thread의 정상 복구도 확인했다. SSR은 Worker를 읽지 않았고 계산을 실행하지 않았다.

`check-correlations-worker.ts`는 실제 hook과 module worker를 별도 메모리 서버에서 실행한다. 세 가족의 시작 후 취소와 재시도, 교체, disabled/null 입력, 자산 실패 복구, unmount와 React StrictMode 재마운트를 확인했다. worker **22개 생성·22개 종료·잔존0**, 오래된 렌더와 페이지 오류0이다. 검증 서버와 소유 브라우저는 종료했다.

esbuild 의존 그래프에서 hook·브라우저 주 번들에 세 수치 모델과 고정밀 helper가 없음을 확인했다. lint·타입 검사를 통과했고 이전 소스4,171개와 정적 출력4,272개는 변경하지 않았다. 직전 전체7,936검사·빌드는 [AR(1) 단계](ar1-average-foundation.md)의 근거이며, 이번132검사와 구별한다.

실행 근거는 `.runtime/learning/correlations-worker/verification.json`, native 브라우저 기록과 번들 그래프를 따른다. worker 입력·취소·UI 분리는 검증했지만 Next의 실제 자산 분리와 경로·읽기 브라우저 검증을 완료한 것으로 세지 않는다.

## 다음 단계

[학습 화면 연결 계약](correlations-interaction-plan.md)에 따라 고정 개념18단계, 안내9개·프리셋·해설 퀴즈와 시각화를 만든다. 이후 세 단원을 Interactive Book에 등록하고 한영·명암·모바일/데스크톱, 저장 복원·실패 복구·접근성·실제 Next worker 자산을 검증한다. 전체 교육과정·연구 깊이는 계속 미완료다.
