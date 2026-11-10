# 실시간 경로적분: 검산된 행렬의 열지도와 Three.js 보기

2026-11-11. [13단계 개념 그림](real-time-phase-concepts-status.md) 다음으로 실제 계산/독립 검산 Worker의 소유 응답을 행렬 화면에 연결했다. **전체 시각화 요건은 아직 진행 중이며 16/23요건을 유지한다.** 위상·모드, 파동묶음의 밀도/위상/고전극한 비교, fold/Airy 그림은 다음 순서다. 원래 N2…32·12자리 입력과 세 계산 모드를 유지하며 아직 단원을 reader에 등록하지 않았다.

## 행렬과 해석

- 사슬: 전체 내부 Hessian, 가역할 때 역곡률과 형식적 iℏH⁻¹, 조절자가 있을 때 ℏCε⁻¹, 고전 map을 비교한다. 특이점에서 역행렬을 만들지 않는다. 내부 첨자는 고정끝을 제외한1…N−1이다.
- 파동묶음: 고전 map과 초기/최종 대칭화 공분산을 q,p 순서로 비교한다. q,p 행렬의 성분마다 단위가 다를 수 있음을 명시한다. 복소 연결 응답을 확률 공분산이라고 부르지 않는다.
- canonical fold에는 이 사슬 행렬이 적용되지 않음을 설명한다. fold 화면을 행렬 그림으로 대체하지 않는다.

이진 계수 구간의 원래 값은 `multiplier × 2^scalePowerOfTwo × f`이고 `f∈[lower,upper]×2^(−coefficientBits)`다. 배율을 소수 하나로 곱해 overflow/underflow를 숨기지 않는다. 형식적 연결 응답의 ℏ도 원래 분수 multiplier로 남긴다. 선형/부호 로그는 표시용 계수 중점의 변환이며, 물리량에 대한 새 오차 인증이 아니다. 원래 구간·배율·복소 두 성분은 선택 성분과 전체 표에 보존한다.

전체 열지도와3D 막대는 양·음 부호, 정확한0, 구간의 부호 미해결, 표시 해상도보다 작은 성분을 구별한다. 정확한0은 점, 미해결/작은 값은 기준면의 별도 표식으로 표시하며 작은 값을 정확한0으로 바꾸지 않는다. 행·열 선택기로 모든 성분에 접근할 수 있고 전체 구간 표는 필요한 때 열어 읽는다. 최대31×31의961성분을 줄이지 않는다.

기존 HSL/learning 색상·Button·표 스타일을 사용한다. Three.js는 사용자가3D를 열 때 별도 chunk로 불러오고 입력/회전/resize/테마 변경에 반응해 그린다. 자동 회전이나 계속 도는 render loop는 없다. 키보드·터치 회전, 컨텍스트 손실 후2D 복귀/재시도, unmount 시 GPU 정리를 구현했다. 원래의 고정된 설명용 개념도와 현재 검산 결과를 구별한다.

## 실행한 검사와 범위

- Node24.13.0/20.20.2 각26검사: 사슬17·packet8, 총25입력의 전체 응답을 독립 gate로 검산한 뒤352개 행렬/성분/척도 조합과78,672개 셀의 차원·전체 성분 유지·대칭성·유한 투영 등을 확인했다. 음의 손계산,0 pivot 역행렬, 형식적 응답의 실수0/ℏ multiplier, 복소 대칭성과 SSR/영문 및 reader 의존성 분리를 확인했다. Node20에서는 실제 극소 폭 예제의 비영 성분을 tiny 표식으로 유지하는 검사도 추가 확인했다. 이는 새 독립 수학 gate의 증거가 아니라 기존 gate 통과 응답의 표시 검사다.
- 실제 native module Worker를 사용하는 Chromium KO/EN × light/dark ×320/1440px8조합: 음의 모드,0 pivot 가역 행렬,3D 키보드 회전,4개 모바일 viewport의 CDP touch 회전, 실제 context loss/재시도, 접근성8회를 확인했다. 영어 본문에 한글이 없고 문서 가로 넘침이 없었다. 8개 screenshot을 남겼다.
- 데스크톱 최대N32에서961개 열지도 성분·전체 구간 표와3D를 확인했다. 조절 복소 응답과 packet 공분산 전환도 확인했다. 최대N32의 모든 물리 모바일 기기를 검증했다는 뜻은 아니다.
- WebGL context를 강제로 막은 별도 실제 Worker 검사에서2D·원래 구간·척도 조절·재시도를 확인했다. 이는 context 상실 검사와 별도다.
- 첫 browser 검사에서는 선택기의 접근성 이름에 옵션 텍스트가 섞인 문제를 발견해 명시적인 한영 이름을 붙였다. 최대 행렬 검사는 새 입력에서 폐기된 이전3D를 가정하던 대기 조건을 수정했다. fallback 검사의 초기 스크립트 bootstrap 오류도 수정해 실제 WebGL 차단을 확인했다. 최초 로그를 보존한다.

대상 lint/typecheck가 통과했다. reader 그래프에는 producer·exact/검산 엔진이 없고 Three.js는 별도 chunk에 있다. root out과 기존 preview를 변경하지 않았으며 이번 review 서버만 검사 후 종료한다. 실제 reader의 Next 생산 자산 검사는 아직 수행할 단계가 아니다.

## 재현과 다음 순서

```bash
node --import tsx --test scripts/learning-real-time-phase-matrix.test.ts
node --import tsx scripts/preview-real-time-phase-matrix.ts
node --import tsx scripts/check-real-time-phase-matrix-native.ts
node --import tsx scripts/check-real-time-phase-matrix-fallback-native.ts
```

미리보기 CLI가 새로 출력한 주소를 사용한다. 증거·source/log hash는 `.runtime/learning/real-time-phase-matrix/verification.json`에 보관한다. [Three.js WebGLRenderer](https://threejs.org/docs/pages/WebGLRenderer.html)와 [OrbitControls](https://threejs.org/docs/pages/OrbitControls.html)의 현재 API를 확인했으며 기존 사이트의 이벤트 기반 행렬 renderer 패턴을 따른다.

다음은 위상/모드·packet·fold 결과 그림을 구현하고, 안내 실험·해설 문제·전체 입력/학습 저장·접근성·reader·생산 자산·보존/범위 감사를 진행한다. **600등록 단원·1857안내·1200등록 원고**, 미등록1단원/2원고, **입문315/362검증·47미완료**, 전체51영역509주제 및 연구 심화는 유지한다.
