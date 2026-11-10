# Schmidt 응답 준비: 독립 스펙트럼·로그 인증

2026-11-09. 여섯 번째 다체계 응답 검증을 위한 **독립 수학 인증**을 구현했다. 응답 종류 검증은 여전히5/6이며, 전체 Schmidt schema·projector·절단 벡터·오차의 입력 귀속은 다음 단계다.

## 준비한 계산

정확한 square-free factorization으로 고유값 중복도를 구하고, Descartes 부호 변화와 유리수 이분으로 [0,1]의 모든 근을4096비트 폭까지 분리한다. 생산 Sturm/root engine을 가져오지 않는다. 가까운 두 근과 실제 퇴화, 작은 양의 근과0을 구별한다.

ln(x)는 dyadic reduction 후 atanh 양의 급수600항과 나머지 상계를1536비트 방향성 산술로 계산한다. 생산 logarithm 엔진과 상수를 가져오지 않는다. [수학·범위 감사](quantum-many-body-schmidt-spectral-audit.md)에 유도·예산·cache·한계를 기록했다.

## 검증 결과

- 새33검사·기존 Schmidt/MPS77회귀·타입·lint 통과.
- 독립26준비의65절단·129고유값·130엔트로피 대조.
- 별도 mpmath1800자리 계산의34개 로그·520유효 숫자 기준 대조와 byte-identical 재생성.
- 10⁻¹⁸⁰ 간격의 근·10⁻²²⁰ 양의 근·정확한 퇴화·비이진 유리근·quartic 무리근·0/1 끝점 검증.
- 실제 Chromium에서65절단·129근·34로그 재검증, page error0. 최대 스펙트럼 계산 655.8ms이며 실제 reader 반응성 통과로 세지 않는다.

근거는 `.runtime/learning/many-body-schmidt-spectral/`의 tests·regression·native 로그, metrics·native-verification·log-reference-reproduction·verification JSON과 dependency graph다. Native 검사는 격리된 preview-origin 페이지이며 전체 Worker 응답·취소·실제 reader·생산 자산의 증거가 아니다.

## 이어서 할 작업

독립 interval matrix 연산으로 spectral projector와 절단 상태를 계산하고, 전체 응답 schema·ID·입력 귀속·변조 거부를 연결한다. 이후 controller/hooks·반응성, 시각화·안내·퀴즈·저장과 실제 reader·정적 출력을 검증한다.

다섯 원고 쌍과 수치 가족은 준비 상태다. 등록566한영 단원·1698안내와 원래 입문308/362검증·54미완료, 전체51영역509주제·연구 심화 범위를 유지한다. 홈페이지 출력·미리보기 서버를 보존했으며 진입은 **도서 → Interactive Book**이다. Commit·push·공개 배포는 수행하지 않았다.
