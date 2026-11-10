# 다체계 응답 검증 여섯 번째 종류: Schmidt

2026-11-09. **여섯 계산 종류의 전체 소유 구조·수학 귀속 검증을 모두 준비했다.** 새 `readSchmidtSuccessForInput`은 모든 연속 절단의 물리 밀도·정확 rank·spectrum·projector·절단 상태·오차·entropy를 입력에서 독립 계산한다. 취소 controller와 실제 교재 통합은 다음 단계다.

## 구현한 계산

조밀 joint density의 직접 partial trace와 주 소행렬식으로 물리 부분계·rank·characteristic polynomial을 구한다. 독립 Descartes 근 인증·직접 atanh 로그 인증을 사용하고, Lagrange polynomial의 행렬 Horner 계산으로 각 고유값 그룹 projector를 구한다. 생산 Schmidt·Hilbert·행렬·Ritz·root·로그 엔진을 검증기로 가져오지 않는다.

3072비트 dyadic 산술에서 부호 있는 나눗셈·제곱·정수 Newton sqrt와 행렬 연산을 바깥 방향으로 계산한다. 고정 분모를 유지해 반복 gcd 비용을 줄였다. 퇴화 경계의 chosen null, rank0의 normalized null, 오른쪽 C Pᵀ, 원시/정규화 진폭과 discarded error·fidelity·normalized distance를 각각 대조한다. [수학·소유 구조 감사](quantum-many-body-schmidt-response-audit.md)에 범위와 한계를 기록했다.

## 검증 결과

- 새41검사·359응답 변조 거부·624독립 구간 산술 대조, 기존526회귀, 타입·lint 통과.
- 독립 Schmidt26준비의 정확9041·방향성8613관측 일치.
- Native Worker207개 생성/종료: Schmidt26·이전181개 수락, wrong ID와 잘못된 metadata 각각207개 거부.
- 여섯 종류 전체 정확463731·방향성18165관측을 독립 fixture와 대조.
- Schmidt 최대 브라우저 검증 1275.6ms, 전체 최대 2435.5ms. 실제 reader 반응성은 아직 검증하지 않았다.

근거: `.runtime/learning/many-body-schmidt-response/`의 tests·regression·native 로그, metrics·native-verification·verification JSON, dependency graph와 이전 그래프 보존 기록. Native는 격리된 preview-origin Blob Worker이며 생산 Next 자산·controller·실제 reader의 증거가 아니다.

## 다음 순서와 전체 범위

**검증 실행 구조·취소 controller/hooks·반응성 → 시각화·안내·퀴즈·저장 → 실제 reader·생산 Worker 자산·정적 출력 검증**을 진행한다. 약1초 이상 걸릴 수 있는 검증을 실제 reader의 동기 성공 callback으로 연결하지 않는다. 새 단원을 등록하기 전 계산과 검증 중 입력·취소가 반응하는지 확인한다.

다섯 한영 원고 쌍·수치 가족·여섯 응답 검증은 준비 상태이고, 새 학습 단원은 아직 등록하지 않았다. 기존566한영 단원·1698안내, 원래 입문308/362검증·54미완료와 전체51영역509주제·연구 깊이를 유지한다. 홈페이지 출력·미리보기 서버를 보존했으며 진입은 **도서 → Interactive Book**이다. Commit·push·공개 배포는 하지 않았다.
