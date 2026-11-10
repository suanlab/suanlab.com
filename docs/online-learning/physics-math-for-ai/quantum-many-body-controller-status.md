# 다체계 계산·독립 검증 controller와 React hook

2026-11-09. 계산과 수학 검증을 화면 밖의 Worker에서 실행하고, 큰 응답의 복사를 짧고 취소 가능한 작업으로 나눴다. 여섯 응답 검증기를 controller와 hook에 연결했으며 **실제 학습 화면 통합은 아직 남아 있다.**

## 구현한 동작

요청별 계산 Worker·독립 검증 Worker를 사용한다. 새 입력·취소·해제 시 이전 포트와 continuation을 중단하며, 검증에 성공한 현재 입력의 소유·동결 응답만 ready로 공개한다. Hook은 새 입력 render에서 이전 결과를 숨기고 재시도·취소를 제공한다. 수학 검증기·생산 엔진을 main 번들에 가져오지 않는다.

큰 결과는 최대2048노드·약4ms 단위로 descriptor-only 복사한다. 초기 양보 방식에서 발견한 약620ms timer 지연을 일반 timer 양보로 수정했다. [실행 경계·신뢰 모델·취소 감사](quantum-many-body-controller-audit.md)에 검증 Worker receipt와 main header 검사의 역할을 구별해 기록했다.

## 검증 결과

- 새30검사·기존255회귀·타입·lint 통과.
- 여섯 종류의 정상 응답 수락과 위조 응답의 별도 수학 검증 거부.
- Native React StrictMode harness에서 큰 여섯 입력, raw 위조·복사/검증 중 취소·재시도·입력 교체 확인.
- Worker21개 생성/종료 일치, 검증 Worker10개, 잘못된 이전 ready 결과0.
- 작업 중 heartbeat 최대 68.4ms, 검증 중 타이핑 13–19ms, raw 복사 취소 57ms·검증 취소 41ms. 로컬 harness150ms 예산 안에서 확인했다.

근거는 `.runtime/learning/many-body-controller/`의 tests·regression·native 로그, metrics·native-verification·dependency graph·verification JSON이다. 이전 지연도 before-fair-yield 기록에 보존했다. Main에 남은 약50–61ms task도 기록했으며 이 결과를 실제 reader 전체의 성능 통과로 세지 않는다. Native는 preview-origin Blob Worker다.

## 다음 순서와 전체 범위

**시각화·개념 단계·안내·퀴즈·입력 저장 → 실제 reader와 생산 Next Worker 자산·반응성 → 등록·정적 출력·한영 화면·접근성 검증**을 진행한다. 다섯 한영 원고 쌍·수치 가족·여섯 응답 검증과 실행 기반은 준비됐으나 새 단원을 등록하지 않았다.

기존566한영 단원·1698안내, 원래 입문308/362검증·54미완료와 전체51영역509주제·연구 깊이를 유지한다. 홈페이지 출력·미리보기 서버를 보존했고 진입은 **도서 → Interactive Book**이다. Commit·push·공개 배포는 수행하지 않았다.
