# 다체계 첫 학습 화면: 국소 Hamiltonian과 spin 사슬

2026-11-09. 다섯 예정 단원 중 첫 번째의 **staged 학습 화면**을 준비했다. 한영 본문·세 개념 단계·여섯 고정 예제·세 안내·세 해설 퀴즈·정수/JSON 입력·입력/답/완료 저장을 계산·독립 검증 Worker와 연결했다. Reader 등록과 생산 Next 자산 검증은 다음 통합 단계다.

## 학습 내용과 시각화

곱 상태/GHZ의 연결 상관, 횡장 유무의 잔차, 좌절된 삼각형의 고유상태/바닥 상태, 복소 위상의 Y 관측을 비교한다. 전체14절·13수식·3표·6해설의 상세 한영 원고를 실제 content loader에 연결했다.

2D 위치·결합 그래프와 Three.js 국소 Bloch 화살표를 기존 홈페이지 색·명암으로 구현했다. 전체 진폭·잔차·상관·대각 spectrum·항별 에너지와 수치 근거도 표로 확인한다. 강조 재생과 카메라는 실제 상태·물리 시간을 바꾸지 않는다. [집필·입력·그래픽 감사](quantum-many-body-spin-teaching-audit.md)에 범위와 실패 복구를 기록했다.

## 검증 결과

- 새9검사·기존101회귀·타입·lint 통과.
- 한영·명암·320/1440px·모션16기본 상태와17접근성 상태 통과, 영어 표시의 한글 잔류·문서 가로 넘침0.
- 여섯 고정 예제, 잘못된 정수/영 상태의 마지막 유효 입력 유지, 입력·퀴즈·완료 reload와 저장 차단/손상 복구 확인.
- 실제 Three.js·camera·context loss·WebGL 차단 대체, Worker 실패/재시도, 모션 설정·키보드 강조 확인.
- Native Worker66개 생성/종료 일치, page error0. Main의 무거운 계산/검증 엔진 제외 확인.

근거: `.runtime/learning/many-body-spin-teaching/`의 tests·regression·native 로그, metrics·native-verification·dependency graph·verification JSON과 한영·명암·3D 화면 캡처. Native는 실제 reader CSS를 사용한 별도 harness와 Blob Worker이며 실제 reader route·생산 Next Worker/3D 청크 검증은 아니다.

## 다음 순서와 전체 범위

**Fermion Fock/parity → boson cutoff·coherent → MPS 수축/gauge → Schmidt 절단**의 학습 화면을 이어서 준비한다. 이후 실제 reader·생산 자산·반응성·정적 출력·한영 화면·접근성을 통합 검증한다.

학습 화면 준비1/5, 등록0/5다. 기존566한영 단원·1698안내, 원래 입문308/362검증·54미완료와 전체51영역509주제·연구 깊이를 유지한다. 홈페이지 출력·미리보기 서버를 보존했고 진입은 **도서 → Interactive Book**이다. Commit·push·공개 배포는 수행하지 않았다.
