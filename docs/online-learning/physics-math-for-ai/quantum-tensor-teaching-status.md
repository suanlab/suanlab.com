# Tensor product·entanglement 학습 화면 준비

2026-11-08. 두 단원의 개념·예측·자유 실험·시각화·안내·퀴즈·저장·모션을 구현하고 독립 화면에서 검증했다. 교재에는 아직 등록하지 않아562단원·1686안내·306/362입문 검증·56미완료를 유지한다.

## 준비한 화면

- 순수 합성계: 복소 진폭/확률 표·부분추적·Schmidt 가중치/중복도·조건부 상태/무신호성·부분전치 ±항·곱 인수/minor·국소 반사·dephasing.
- 혼합 두-qubit: A/B 국소 Bloch 공·Pauli 상관·PPT/negativity·공동/조건부 확률·선택/최적 CHSH·정확 경계 인증.
- 각16예제·3개념 단계·3안내·3퀴즈,정확한 전체 필드/JSON 편집과 browser 저장. 모든16차원 쌍 새 곱 준비와 명시적인 Werner 비교 상태 생성.

기존 홈페이지의 색 변수와 학습 컴포넌트/CSS를 사용했다. Three.js는 기존 Bloch renderer를 재사용하고2D 대안을 유지한다. 선택 상자의 다크 명암 대비와 A/B 접근 라벨을 확인했고, 두 공의 부분계 표제를 명시했다. 재생은 계산된 항목을 읽는 보조이며 물리 시간이나 상태 변화가 아니다.

## 검증

새42/관련243검사·타입·전체 lint 통과. 한영·모바일/데스크톱의8화면과12동작 사례를 확인했다.128예제 선택·24개념 예측·24안내·24퀴즈/복원,field/JSON 편집·저장 복원과 오류 보존을 확인했다.8axe 검사에서 위반이 없었고3D/2D 복구와 가로 키보드 스크롤을 확인했다.

모션/visibility/reduced-motion·저장소 차단·다른 가족 저장값·Worker 실패 뒤 재시도·계산 교체/취소·null 조건부와 수치 근거 페이지를 검산했다. 전체358개 Worker를 종료했고 stale 결과와 page error는 없었다.12동작 화면의 영어 body 전체에도 한글 혼입이 없었다. 상세 범위는 [교수 화면 검산](quantum-tensor-teaching-audit.md)을 확인한다.

## 다음 단계와 보존

다음은 원고4편을 콘텐츠로 연결하고 두 단원을 Interactive Book의 registry·curriculum·cover·검색에 등록하는 것이다. 그 뒤 실제 reader·production Worker asset·한영 읽기·접근성·정적 출력·이전 콘텐츠 보존을 검증한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 넣지 않는다.

기존6108개 소스·검증 파일 중 진행 문서9개만 갱신한다. 수치·입력·Worker 프로토콜·참조·상세 원고·이전 교수 화면과4371개 정적 출력 파일은 보존한다. 새 build·배포·commit·push는 없다. 전체51영역509주제·분량/주차 무제한 범위·일반 다부분계·LOCC·bound entanglement·연구 깊이·공용 그림 후속 작업은 남는다.

근거: `.runtime/learning/quantum-tensor-teaching/`의 `combined-tests-final.log`, `browser-final.log`, `browser-verification.json`, `controls-localization-final.log`, `controls-verification.json`,8axe 결과와8preview 이미지, `browser-dependency-graph.json`, `typecheck-verified.log`, `lint-verified.log`, `verification.json`, `preservation.json`.
