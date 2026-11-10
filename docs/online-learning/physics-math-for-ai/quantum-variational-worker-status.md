# Quantum variational principle: Worker 계산 연결 준비

2026-11-09. 두 수치 가족과 상세 한영 원고에 Worker 전달·응답 검증·취소·재시도·React hook을 준비했다. 실제 교재 등록 전이므로564개 한영 단원·1692안내, 원래 입문 주제307/362검증·55미완료를 유지한다.

## 구현과 필요한 보완

소유 입력 snapshot과 요청 키, version/id/kind 귀속, 전체 응답 schema와 수학 대조를 추가했다. 수치 엔진과 고정밀 root/구간 연산은 Worker에 두고 reader는 작은 정확 입력·귀속 검사를 사용한다. 오래된 포트를 종료하는 요청별 Worker 제어기와 render 시점의 입력 key 방어를 구현했다. 동기 계산 fallback은 없다.

Ritz의 보조 스펙트럼에 반올림된 숫자만 있던 부분은 정확 root 구간·중복도·exact 메타데이터를 추가했다. ambient/Gram/root 전체와 prefix 최소를 검사할 수 있게 했으며 기존 에너지·상태·잔차·정밀도는 바꾸지 않았다. 상세 규칙과 한계는 [Worker 계약](quantum-variational-worker-contract.md)에 기록했다.

## 검증한 근거

- Worker115개·관측/다항식4개, 합계119개 새 검사를 통과했다.97개 정상 기준 입력, 요청/결과 귀속, 잘못된·누락·추가·상속·getter·sparse 데이터, 수치/표시/정밀도 변조, root/축퇴/최소 인증, 취소·오류·재진입·SSR을 확인했다. 넓지만 정당한 구간을 fixture 정밀도 때문에 거부하지 않는 검사도 포함한다.
- Ritz 수치/구간과 기존 원고74개 회귀 검사, 타입·전체 lint를 통과했다. 입력 guard와 기존63개 키·정확 rank 결정을 대조했다.
- 실제 브라우저 module Worker97개에서97개 유일 준비를 계산했다. 시작 frame97개·검증 결과97개, 독립 유리수9857/방향성7893대조를 통과했다. 모두 종료했고 페이지 오류는0이다. Native Worker 번들64871바이트는 재생성해 byte 동일성을 확인했다.
- 실제 hook/factory를 StrictMode에서 확인했다. 종류 전환5회·동일 종류 파라미터 변경4회, 입력 교체·disable·factory 실패/재시도·실행 오류/재시도·unmount/remount와 계산 시작 후 취소/재시도를 통과했다. Worker16개 생성/16개 종료·live0, 모든 render에서 응답 kind/key가 현재 입력과 일치하며 페이지 오류0이다.
- Reader 의존 그래프에서 Rayleigh/Ritz·root 분리·작업 구간 수치 엔진을 제외했다. 이 그래프 검증은 main thread의 작은 귀속 대수까지 없다는 뜻은 아니다.

근거: `.runtime/learning/quantum-variational-worker/`의 `verification.json`, `worker-tests-final.log`, `observation-tests.log`, `numerical-manuscript-regression.log`, `native-verification.json`, `native-client.json`, `reader-dependency-graph.json`, `native-bundle-reproduction.json`, `lint-final.log`, `typecheck-final.log`, `preservation.json`.

## 다음 순서와 전체 범위

다음은 두 가족의 교수 화면이다. 개념 비교·직관적 그림·자유 실험·예측·안내·퀴즈·저장을 기존 홈페이지 스타일과 색으로 연결한다. 이후 **도서 → Interactive Book**에 등록하고 실제 reader·접근성·Next 생산 Worker 자산·정적 출력을 검증한다.

현재97개 native 검증은 Blob Worker, hook 검증은 분리한 harness다. Next 생산 자산·실제 reader 검증으로 표시하지 않는다. 전체51영역509주제·영역별 감사·증명과 연구 깊이·공용 그림 후속 및 양자 변분 원리 원래 주제 완료는 남아 있다. 공개 배포·commit·push는 하지 않았다.
