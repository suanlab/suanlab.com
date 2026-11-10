# 실시간 응답 검산: 전체 행렬 관찰값

2026-11-10. 독립 exact-core 검산 다음으로 `verifyRealTimeMatrixObservations`를 추가했다. 이 함수의 범위는 사슬의 packed 역행렬·복소 connected response뿐이다. **전체 wave/Airy 응답, Worker, 화면 연결은 아직 완료하지 않았다.**

## 검산 방법

물리 입력과 exact certificate를 먼저 독립 검산한다. 검증된 Green 행렬에서 각 관찰값의 참값을 얻고, 정수 교차곱으로 모든 upper-triangle 성분의 포함 관계를 검사한다. 복소 행렬은 실수·허수 모두 확인하며, conjugate symmetry를 가정하지 않는다. 조절된 응답에는 입력 ℏ를 곱한다.

512비트 계수의 정수 문법, 분모와 척도, 구간 폭(최대 3 coefficient quanta), 행 길이와 전체 크기를 확인한다. 공유 척도에서 최대 성분의 절댓값은 1/4보다 크고 1보다 작아야 한다. 정확한 0은 `[0,0]`이어야 한다. 생산자의 특정 반올림 바이트와 같을 필요는 없으며, 더 좁은 올바른 구간도 허용한다.

특이 사슬에는 역행렬이 없어야 하고, 요청하지 않은 조절 응답도 없어야 한다. formal response는 확률 공분산이 아니라 복소 생성 응답이라는 해석 및 원래 ℏ 표기를 보존한다. getter를 실행하지 않는 bounded snapshot으로 관찰값을 소유한다. 생산 계산기·packing·구간 산술을 런타임 의존성으로 사용하지 않는다.

반환값은 `scope='finite-matrix-observations-only'`, `fullResponseValidated=false`다. wave를 끼운 전체 응답이나 packet/fold를 이 부분 검산에 통과시킬 수 없다.

## 재현

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-matrix-check.test.ts
npm run typecheck
```

Node.js 20.20.2와 24.13.0에서 새 23검사가 모두 통과했다. 17개 사슬 예제의 실수 2537개·복소 실수/허수 2988개 scalar 성분을 확인했다. 작은 조절 사슬의 각 성분 변조 30개, 최대 N32 모형의 마지막 복소 성분, ℏ 배율 누락, 잘못된 척도·폭·차원·구조·정수 문법·귀속·accessor를 거부했다. 정확한 dyadic point로 좁힌 구간도 허용함을 확인했다. 타입 검사와 변경 파일 lint도 통과했다. 검사 결과와 source hash는 `.runtime/learning/real-time-matrix-check/verification.json`에 기록했다. 실행 시간은 로컬 측정이며 Worker 기한 검증으로 세지 않는다.

## 남은 순서

독립 진폭·위상·정규화·scaled/log 및 Airy 검산 → 전체 응답/귀속 검산과 변조 검사 → 실제 계산/검산 Worker 수명 → 시각화·설명·문제·접근성 → reader 등록과 홈페이지 회귀 점검 순서다. 완료 요건은 13/23으로 유지한다. 등록 600단원·1857안내·1200원고, 미등록 초안 1단원/2원고, 입문 315/362 검증·47개 미완료를 변경하지 않는다.
