# Brownian bridge: 한영 교수 화면·시각화 검증

2026-11-10. `brownian-bridge-time-slicing-measure`의 기존 홈페이지 색·스타일을 사용한 학습 화면을 연결했다. 아직 미등록 초안1개이며597등록 단원·1838안내·등록1194한영 원고·입문315/362검증·47미완료를 유지한다. 실제 reader와 생산 Next 자산은 다음 단계다.

## 제공하는 학습 흐름

8개념 단계마다 한영 직관·수식·설명 그림을 제공한다.16예제,6예측 안내와8해설 퀴즈를 연결했다. 유한 적분/끝점 밀도/조건부 법칙, 시간 정밀도와 공분산, 점 원천, 주변분포 정련과 면적 오차, 증분 제약·거칠기와 양자 연결의 한계를 구분한다.

현재 입력에는 근사 평균/원천0 평균과 시각별 한 표준편차, 위치·증분·내부 정밀도의 열지도와Three.js, 정확한 분수/잔차/분해 표를 제공한다. 띠는 전체 경로의 동시 포함 구간이나 표본 경로가 아니다. 재생은 선택 시각만 이동하며 새 표본이나 재계산을 만들지 않는다. 정련 표의 큰 격자는 해석적 비교이며 전체 행렬은 입력N≤32에서만 계산한다. 커널의π·제곱근·지수는 해석식으로 남긴다.

입력·행렬 종류/척도/시간/성분, 개념 단계·안내/해설 선택·퀴즈 답을 소유한 version1 schema로 저장한다. 잘못된 저장은 첫 예제로 복원하며 저장이 막혀도 현재 실험은 계속한다. 동작 줄이기·숨김·키보드·터치와3D 컨텍스트 손실 복구를 지원한다.

## 실제 확인한 근거

- 신규 교수8검사와 관련194검사가 Node.js20/24에서 통과했다. 타입·ESLint도 확인했다. 입력·화면 bundle은 계산기와 독립 수학 검산기를 포함하지 않는다.
- 한영×명암×320/1440px8조합에서64개념·48안내·64퀴즈의 정오답 해설과37예제 적용, 행렬/척도·키보드·3D/컨텍스트 손실 복구를 확인했다. 접근성8회,106Worker생성/106종료·활성0이다. 한영 기준 그림16상태를 실제 브라우저로 기록했다.
- 최대32구간 모바일에서1089개 위치 성분·Three.js 터치 회전·표시 재생의 비재계산·취소/오류 후 재시도와 새로고침 복원을 확인했다. N=2의 단일 내부 정밀도도 실제3D로 확인했다. 새로고침 전14/후2Worker가 모두 종료됐다.
- 양끝/원천의 부호 있는12자리 분자·12자리 분모를 입력창에서 온전히 적용했다. 범위를 벗어난 초안은 이전 결과와 구별하고 새 계산을 시작하지 않는다.
- 한영의 잘못된 입력 저장·잘못된 표시 저장·저장 불가6상황을 추가 확인했다. 최대 흐름과 합쳐7상황·접근성7회다. 마지막 흐름은 입력·표시와 개념/안내/답의 복원을 확인했다.
- 선택 상자의 접근성 이름, 최대 열지도의 숨김 텍스트가 모바일 페이지 폭을 넓히던 문제와 부호 있는 분수의26문자 입력 한도를 보완했다. 최초 실패 로그는 보존하고 수정본의 모든 브라우저 실행이 종료0임을 확인했다. 영어 본문에 한글과 문서 가로 넘침이 없다.

## 재현과 다음 순서

```bash
npx tsx scripts/preview-brownian-bridge.ts
BROWNIAN_LANGUAGE=en BROWNIAN_THEME=dark npx tsx scripts/check-brownian-bridge-teaching-native.ts
npx tsx scripts/check-brownian-bridge-storage-native.ts
npx tsx --test --test-concurrency=2 scripts/learning-brownian-bridge-{numerics,manuscripts,validation,workers,controller,teaching}.test.ts
```

Playwright Chromium과 이전 Worker 검사가 만든 `.runtime/learning/brownian-bridge-worker/{calculate,validate}.js`가 필요하다. preview는 `.runtime/learning/brownian-bridge-teaching/review-server.json`에 주소를 기록하고 기존 `out/`을 바꾸지 않는다. 코드를 수정하면 같은 preview 명령의 `--build-only`로 자산을 갱신한다.

[21요건](brownian-bridge-acceptance-matrix.json) 중20개를 확인했다. 다음은 Interactive Book의 실제reader등록·생산 자산·메뉴/검색/이전 내용 보존 검증이다. 첫 처리만으로 경로적분 원주제를 완료 처리하지 않는다. Euclidean potential·안장/요동 근사·실시간 위상과 전체51영역509주제·연구 심화도 계속 남는다.
