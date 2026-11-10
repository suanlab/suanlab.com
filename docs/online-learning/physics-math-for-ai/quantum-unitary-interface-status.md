# Unitary evolution 개념·자유 실험 화면 검증

2026-11-08. 스펙트럼·Schrödinger와 구간별 Hamiltonian pulse에 고정 개념 비교·자유 실험·물리 시간 선택·시각화를 연결했다. 기존 홈페이지의 학습 색 변수·버튼·표·명암 스타일을 사용한다. 현재는 미등록 교수 화면이며 다음 단계는 교재 목록·원고·실제 읽기 연결이다.

## 제공한 학습 흐름

- 독립 참조와 일치하는 준비31개: 스펙트럼15개·pulse16개. 모든 정수·유리수 입력에 한영 이름과 에너지 E0·시간 T0·ℏ E0T0 단위를 표시한다. 스펙트럼 Euler 격자와 pulse 지속 시간의 제약도 설명한다.
- 각 가족3단계, 총6개 개념 비교·예측·정답 해설과 별도 자유 실험 안내를 제공한다. 위상/에너지 확률·공통 이동/단위·Euler 오차, 순서/여러 측정 축·끝점/중간 경로·교환자/상쇄/역전파를 비교한다. 고정 비교는 저장한 자유 실험과 독립적이다.
- 시간 slider·이전/다음·재생은 실제 계산된 시각을 선택한다. pulse의 두 전환 시각을 포함하고 보간한 상태를 만들지 않는다. 재생은 표본 순회이며 일정한 물리 속도를 뜻하지 않는다. 동작 줄이기·사이트 모션 정지·화면 밖에서 자동 진행을 멈춘다.
- 복소 진폭 화살표·Bloch 3D/2D·확률/fidelity 시간 그래프·직접 norm/Euler 비교·동일 시각의 세 pulse 경로를 제공한다. 표에는 정확 분수·방향성 구간·표현 상태를 남기며, 큰 Euler 성장·미해결 좌표를0이나 확률로 바꾸지 않는다.
- 초기 에너지 통계·공통 위상 제거 진폭·전파·보존량·경계 연속성·평균·교환자 leading/나머지·역전파의 근거를 펼쳐 읽을 수 있다. 기존 Bloch 화면을 수정하지 않고 같은 geometry engine을 사용하되, 새 화면은 선택한 물리 시각의 벡터임을 설명한다.
- 유효한 설정만 저장·복원한다. 개별/전체 JSON 입력 오류는 이전 준비를 유지하며, 저장 실패·계산 취소·Worker 오류 후 재시도를 처리한다. 새 입력에는 이전 결과를 표시하지 않는다.

## 확인한 근거

새 준비·설명·좌표·SSR·의존성 검사7개와 이전 수치·집필·Worker283개, 합계290검사·최종 타입·전체 lint를 통과했다. 실제 UI bundle에는 새 수치 진화·행렬·방향성 삼각함수 엔진이 없고 계산은 Worker에서 수행한다. norm을 강제로 맞추지 않는 좌표, 미해결 값 생략, 매우 큰 유한 수의 그림 좌표도 검사했다.

기본 브라우저 검사는 두 가족×한영×명암×320/1600px의16조합이다.16개 접근성 검사와 영어 표시·페이지 가로 넘침·입력 오류·저장 복원을 확인했다. 준비62회·개념/예측12회, Worker 오류 복구4회·손상 저장 복구4회·저장 쓰기 실패4회·모션12회·3D16회·WebGL 손실 복구4회를 포함한다.

추가 영어/밝은/데스크톱2흐름에서는 안내 실행6회·실제 시간 선택4회·pulse 경로 전환2회·화면 밖 정지2회를 확인했다. +X 확률0.913411/0.5가 같은 끝점의 표에 나타나며 경로 선택이 그래프를 바꾸는지 검사했다. 준비31회·개념/예측6회·접근성2회·3D2회·WebGL 복구2회도 포함한다. 기본/추가 검사에서 생성한 Worker294+92=386개가 모두 종료됐고 page error는0이다.

이 근거는 source component를 실제 provider·Worker와 함께 browser에 mount한 검증이다. Next의 최종 Worker asset·교재 등록·실제 읽기 경로의 검증으로 세지 않는다. 새 정적 빌드는 아직 하지 않았으며 기존 교재·수치·원고·정적 출력은 유지한다.

## 다음 순서와 전체 범위

두 단원의 본문을 `content/learning/`에 등록하고 목차·표지·개념 walkthrough·안내·퀴즈·검색·읽기 라우트에 연결한다. 이후 실제 Next Worker asset·한영·명암·모바일·키보드/터치·접근성·읽기 이동·저장/Worker 복구·정적 출력과 전체 검사를 확인한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에는 해당 교재 카드를 넣지 않는다.

현재554단원·1662안내·원래302/362입문 검증·60미완료다. 원래 주제의 읽기까지 검증한 뒤 완료 수를 갱신한다. 전체51영역509주제·연구 깊이·공용 그림 후속과 이후 모든 영역은 계속 남는다. 전체 목표는 진행 중이다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-unitary-interface/verification.json`, `tests.log`, `browser-verification.json`, `focused-browser-verification.json`, `browser-dependency-graph.json`, 접근성 JSON·화면 PNG, `typecheck.log`, `lint.log`, `preservation.json`.

```bash
npx tsx --test scripts/learning-unitary-interface.test.ts scripts/learning-unitary-worker.test.ts scripts/learning-quantum-unitary-authoring.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-unitary-interface.ts
UNITARY_INTERFACE_FOCUS=1 npx tsx scripts/check-unitary-interface.ts
npm run typecheck
npm run lint
```
