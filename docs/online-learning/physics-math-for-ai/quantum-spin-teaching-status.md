# Spin·angular momentum: 시각화·실험·예측 화면 검증

2026-11-08. [수치·한영 본문](quantum-spin-foundation-status.md)과 [Worker 연결](quantum-spin-worker-status.md)에 이어 두 계산 가족의 교수 화면을 구현했다. 아직 Interactive Book에 새 단원을 등록하지 않았다. 기존556단원·1668안내·303/362입문 검증·59미완료는 유지한다.

## 학습 화면

스핀1/2은18예제로 반각·능동 회전 부호·0/π/2π/4π·측정 방향·공통 위상·ℏ 단위를 비교한다. 각도 선택과 표본 재생에 확률/fidelity 곡선·Born 막대·복소 진폭·Bloch3D/2D·모멘트·정확 구간을 연결했다. 한 바퀴 추가 비교는 같은 local 확률과 반대 진폭 부호를 함께 보여준다. 물리 시간이 아닌 기하 각도이며 제어 경로의 간섭 장치는 모사하지 않는다.

유한 스핀은21예제와 모든9개 j값을 제공한다. ladder 도식은 자기 표지·인접 연결 계수 제곱·선택한 정규화 image의 Jz 확률을 보여준다. raising/lowering과 적용 횟수를 선택하고 원시 chain의 norm·각 행의 action^(2k) 단위를 읽는다. 원시 norm을 확률 막대로 그리지 않으며 영 image에는 정규화 상태나 확률을 만들어 넣지 않는다.

물리 확률과 naive 계수 histogram, binomial metric, 원래 준비의 평균/분산·Casimir를 따로 비교한다. 같은 세 평균이지만 다른 Jz 분포를 가진 스핀1 반례도 포함했다. 고차원 상태를 qubit Bloch 구나 평균 화살표로 대체하지 않고 모든 진폭과 행렬 근거를 제공한다.

두 단원마다 고정 개념3단계·예측 문제3개·해설·안내 실험3개를 작성했다. 고정 비교는 저장된 자유 실습과 독립적이다.17/21개 한영 입력·전체 JSON·유효 입력 저장/복원·초기화·오류 후 재시도를 제공한다. j를 줄일 때 비활성 계수를 몰래 버리지 않고 입력을 거부하며 명시적인 수정 방법을 안내한다.

## 디자인·모션

기존 교재의 레이아웃·버튼·표와 사이트 색상 토큰을 사용한다. 새 bitmap 자산이나 별도 색상 체계를 추가하지 않았다. 모바일 캡처에서 진폭 글씨가 작아 각 복소 평면을 좁은 화면의 한 열로 배치하고 ladder 라벨을 키웠다. 선택한 image의 Jz 분포도 표로 제공한다.

재생은 계산한 표본만 순회한다. 물리적인 시간 경로로 부르지 않으며 동작 줄이기·사이트 모션 정지·화면 밖에서 멈춘다. 수동 선택은 유지한다.3D 카메라는 자동 회전하지 않고 기존 Three.js 렌더러의 시점 조절·2D fallback·WebGL 복구를 사용한다. 미해결 값은 임의 좌표나 보정 norm으로 바꾸지 않는다.

## 검증 근거

새 화면9검사와 이전406검사를 합쳐415검사·타입·전체 lint를 통과했다.39예제의 입력 일치·라벨·모든 j·개념/예측 순서·손계산·같은 평균의 반례·영 image·SSR의 Worker 접근0회와 reader 의존성에서 계산 엔진이 제외되는지 확인했다.

실제 provider·소스 component·native Worker를 별도 번들한 브라우저 환경에서 한영×명암×320/1600px×두 가족의16조합을 확인했다.16접근성 검사·78예제 변경·12개념 단계/예측·12안내 선택·16저장 복원·20잘못된 입력·오류 재시도4회·손상 저장 복구4회·저장 실패4회를 포함한다. 수동 표본·한 바퀴 비교·양방향 ladder·영 image·동작 줄이기·사이트 정지·화면 밖 정지도 확인했다.3D8회·WebGL context loss/복구2회와 페이지 오류0·모든 Worker 종료를 확인했다.

첫 검증 후 모바일 그림의 가독성을 개선하고 최종 소스로 같은16조합을 다시 검사했다. 과학적 구간과 수치 기반을 바꾸지 않았다. 이 검증은 최종 Next 생산 asset·교재 reader·검색·정적 출력 검증과 구별한다. 기존 수치·Worker·참조·한영 본문·등록된 화면과4354개 정적 출력은 보존했다.

## 다음 순서와 전체 범위

다음은 두 단원의 실제 등록·본문·안내/퀴즈 데이터·읽기/개념/실험 연결·검색 → Next 생산 Worker·실제 읽기·한영·명암·모바일·키보드/터치·접근성·정적 출력 검증이다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 교재 카드를 추가하지 않는다. 원래 spin 주제의 입문 검증 수는 이 통합 이후 갱신한다.

전체51영역509주제와 분량·주차 제한 없는 연구 깊이·궤도/정의역/결합·공용 그림 후속·나머지 양자 및 후속 영역은 계속 진행한다. 공개 배포·commit·push는 수행하지 않았다.

근거: `.runtime/learning/quantum-spin-teaching/verification.json`, `browser-verification.json`, `combined-tests.log`, `browser-dependency-graph.json`, `typecheck.log`, `lint.log`, `preservation.json`,16개 접근성 결과와 화면 캡처다.

```bash
npx tsx --test scripts/learning-spin-interface.test.ts scripts/learning-spin-worker.test.ts scripts/learning-quantum-spin-authoring.test.ts scripts/learning-spin-ladder.test.ts scripts/learning-spin-half.test.ts scripts/learning-spectral-evolution.test.ts scripts/learning-pulse-evolution.test.ts scripts/learning-quantum-observable.test.ts
npx tsx scripts/check-spin-interface.ts
npm run typecheck
npm run lint
```
