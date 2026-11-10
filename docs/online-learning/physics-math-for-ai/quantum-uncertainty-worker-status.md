# Uncertainty·measurement: 두 가족의 Worker 검증

2026-11-08. 준비 공분산과 POVM/instrument의 두 수치 가족·한영 원고4개에 입력 계약·전용 Worker·controller·React hook을 추가했다. 계산은 Worker에서 실행하며, 화면의 입력이 바뀌면 이전 요청을 종료한다. 실패 시 화면에서 계산을 대신 실행하지 않는다.

## 입력과 반환 계약

두 종류는 `finite-state-covariance-uncertainty`, `qubit-povm-instrument-backaction`이다. 각각75개·22개 소유 유계 정수 입력을 받는다. 정해진 필드·유효 차원·양의 분모·Bloch 공·비영 방향을 검사하고, 정규 순서의 불변 snapshot과 입력 키를 만든다. JSON 초안/저장은 반올림되어 정수처럼 보이는 소수, underflow, 범위 초과와 외부 필드를 거부한다.

반환은 version·요청 번호·종류·입력 키·입력 snapshot이 일치해야 한다. 전체 소유 필드와 배열·조건부 null을 검사한다. 유리수 증명값은 분모·정수 문자열·부호·그림 숫자·로그·구간 폭을 확인하며, 폭 기준 max(1,|값|)×10⁻¹²⁵를 유지한다. POVM의 Q(√(1−κ²)) 계수는 정확한 부호와 보고 구간 포함을 확인하고 유리수인 제곱근의 미접힘 표현을 거부한다.

준비 공분산은 요청한 상태·관측량·중심화·모멘트·Gram·하한·사영·실수 조합과 반환값을 연결한다. 측정 가족은 효과·결과 확률·피드백 회전·세 가지 상태·조건부/순차 확률·meter·RMS·affine channel·dual·Choi 규약과 최소 Kraus 개수를 연결한다. 한 입력 상태의 출력만 일치하는 다른 channel이 통과하지 않도록 전체 affine 표현을 확인한다.

이 검사는 유계 유리수/대수 계정이며 일반 고유값 계산·제곱근 엔진·고정밀 초월함수 엔진을 화면에 다시 넣지 않는다. 독립 수치 참조를 대체하거나 임의 발신자가 모든 수학을 참되게 계산했다는 증명으로 표시하지 않는다. 표시 Kraus·고유공간의 수학은 앞선 독립 참조와 수치 검사에 근거한다.

## 수명주기와 검증

- 새107검사와 이전785검사를 합친892검사 통과. 타입·전체 lint도 실행 기록으로 확인한다.
- 독립 참조의91개 입력을 실제 Worker entry와 controller 계약으로 검증했다. 오류 frame·누락/외부 필드·손상된 분수·거짓 그림 값·틀린 종류/입력·null/퇴화 처리·잘못된 상태/하한/채널/보정 관계를 거부한다.
- 교체·취소·실패/재시도·늦은 callback·재진입 구독/Worker factory·완료 알림에서 재요청·dispose를 검사했다. 종료 시 handler를 분리하고 한 포트를 중복 종료하지 않는다.
- 실제 Chrome module Worker에서91입력·수학 관측102584개를 Node 계산과 대조했다. 계산 시작 뒤 취소2회, controller의 생성7/종료7·live0, 실제 Worker 오류와 재시도를 확인했다.
- 실제 React StrictMode harness에서 종류 전환5회·동일 종류 입력 변경2회·비활성화·생성 실패/재시도·unmount/remount를 검증했다.80개 render에서 결과 입력 키가 현재 요청과 일치했고, 생성10/종료10·live0·페이지 오류0이다. SSR과 비활성 초기 화면은 Worker를 시작하지 않는다.
- Hook/controller 번들 그래프에 두 수치 엔진·공분산 스펙트럼 엔진·구간/행렬 엔진이 들어가지 않음을 확인했다.

브라우저 검증은 로컬에서 실제 소스를 번들링한 별도 harness다. Next 정적 자산·교수 화면·등록 reader 검증으로 세지 않는다. 앞선 독립91사례/88유일 입력·정확78128/방향성23176대조와 참조 재생성 기록은 유지한다.

## 다음 순서와 전체 범위

다음은 고정 개념·자유 실험·공분산 타원/변동 벡터·Bloch 갱신·확률/오차 비교 그림·예측·안내·퀴즈·저장·모션이다. 이후 Interactive Book 등록과 실제 읽기·한영·명암·모바일·접근성·정적 출력·배포 자산 Worker를 확인한다. 기존 홈페이지 색과 스타일을 따른다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 카드를 넣지 않는다.

등록560단원·1680안내·305/362입문 검증·57미완료를 유지한다. 이 단계로 원래 주제나 연구 깊이를 완료 처리하지 않는다. 전체51영역509주제·일반/무한 차원·복합계·연구 과제·공용 그림 후속 범위가 남는다. 기존 원고·수치·Worker·교수 화면과4366개 정적 파일은 보존하며, 새 build·공개 배포·commit·push는 없다.

근거: `.runtime/learning/quantum-uncertainty-worker/`의 `new-tests.log`, `combined-tests.log`, `native-worker.json`, `native-client.json`, `browser-dependency-graph.json`, `client-dependency-graph.json`, `typecheck.log`, `lint.log`, `verification.json`, `preservation.json`.

```bash
npx tsx --test scripts/learning-uncertainty-worker.test.ts
npx tsx scripts/check-uncertainty-worker.ts
npx tsx scripts/check-uncertainty-client.ts
npm run typecheck
npm run lint
```
