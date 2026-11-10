# 다체계 응답 검증 다섯 번째 종류: MPS

2026-11-09. MPS의 소유 구조·전체 수치 귀속 검증을 구현했다. Spin·fermion·raw boson·coherent와 함께 **여섯 계산 종류 중 다섯**이 준비됐다. Schmidt 응답 검증과 학습 화면 통합은 남아 있다.

## 구현과 수정

`readMpsSuccessForInput`은 모든 가상 경로의 직접 합, prefix/suffix Gram, 조밀 partial trace·Pauli 곱으로 응답을 독립 계산한다. 작은 Gram의 양의 주 소행렬식으로 rank를 구해 생산 Gaussian elimination과 구별한다. Gauge 전후 텐서·환경 covariance·좌표 norm·정규형과 물리 상태를 각각 검사한다. 영벡터의 raw rank0과 물리 Schmidt rank null을 구별한다.

정규화 구간의 폭을 raw 좌표 대신 실제 정규화 값 기준으로 강화했다. Raw 진폭10²⁴에서 폭2×10⁻¹²⁰인 위조 응답을 이전 검증기가 수락하고 현재 검증기가 거부함을 별도로 재현했다. 작은 노름의 합법 입력도 통과한다. [수학·소유 구조 감사](quantum-many-body-mps-response-audit.md)에 범위와 한계를 기록했다.

## 검증 결과

- 새38검사·1289변조 거부, 기존455회귀, 타입·lint 통과.
- 독립 MPS29준비의 정확26073·방향성782관측 일치.
- Native Worker181개 생성/종료: MPS29·이전152개 수락, wrong ID와 잘못된 metadata 각각181개 거부.
- 다섯 종류 전체 정확454690·방향성9552관측을 독립 fixture와 대조.
- MPS 최대 브라우저 검증 142.0ms. 실제 reader 반응성은 미검증이다.

근거: `.runtime/learning/many-body-mps-response/`의 tests·regression·native 로그, metrics·native-verification·normalized-width-regression·verification JSON과 dependency graph. Native는 격리된 preview-origin Blob Worker이며 실제 Next Worker 자산이나 reader 검증이 아니다. 이전 그래프는 hash가 일치하는 원래 바이트로 보존하고 현 회귀 그래프는 별도 저장했다.

## 다음 순서와 전체 범위

**Schmidt 응답 → 취소 controller/hooks·반응성 → 시각화·안내·퀴즈·저장 → 실제 reader·정적 출력 검증**을 진행한다. 다섯 원고 쌍과 수치 가족은 준비 상태이고 등록 단원은 아직 추가하지 않았다.

기존566한영 단원·1698안내, 원래 입문308/362검증·54미완료와 전체51영역509주제·연구 심화 작업을 유지한다. 기존 홈페이지 출력과 미리보기 서버를 보존했다. 진입은 **도서 → Interactive Book**이며 commit·push·공개 배포는 하지 않았다.
