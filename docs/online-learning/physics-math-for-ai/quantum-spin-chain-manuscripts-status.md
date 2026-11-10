# 다체계 첫 상세 원고: 국소 spin 사슬

2026-11-09. 준비한 다섯 수치 가족 가운데 첫 단원 `finite-spin-chain-local-hamiltonians`의 **상세 한영 원고 두 개**를 작성·검산했다. 다른 네 단원 원고와 Worker·교수 화면·reader는 남아 있다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**, 전체51영역·509주제와 연구 깊이를 유지한다.

- [한국어 원고](finite-spin-chain-local-hamiltonians-chapter.ko.md)
- [English manuscript](finite-spin-chain-local-hamiltonians-chapter.en.md)
- [가정·예제·출처 감사](quantum-spin-chain-manuscript-audit.md)

## 작성한 내용

각 언어 **14절·13표시 수식·3비교표·6개 해설 문제**다. 곱 공간과 bit 순서에서 시작해 국소 Pauli 작용·물리 spin 계수·그래프·Ising/XY/Heisenberg·에너지와 잔차·횡장 GHZ·partial trace·연결 상관·보존량·좌절·AI 연구의 검증까지 설명한다. 한국어 Hangul5363자, 영어3172단어이며 제목만 바꾼 요약으로 대체하지 않았다.

전등 비유는 기저 개수에만 사용하고 측정 기저와 복소 위상의 한계를 함께 설명했다. 기저별 작용·에너지 공간·sharpness/보존 비교를 표로 나타냈다. GHZ의 평균에 드러나지 않는 횡장 작용은 잔차의 네 직교 방향으로 유도한다. 연결 ZZ가 같은 순수/혼합 준비, 영분산의 들뜬 상태, XX·YY 상쇄와 두 종류 parity의 차이를 반례로 다룬다. 혼합 상태와 전체 X 곱은 해석적 비교이며 현재 pure-state·two-body 입력의 추가 모드로 표시하지 않는다.

## 검증한 범위

제품 모듈 없는 SymPy 생성기로 Kronecker·H²·symbolic h·partial trace·commutator·삼각형 열거를 검산했다. 새 **10개 검사·기존 spin61개 회귀**, 타입 검사·lint를 통과했다. 독립 기준 JSON도 재생성하여 SHA-256 바이트 일치를 확인했다.

실제 strict KaTeX/GFM 렌더링에서 두 언어 모두 오류가 없고, 수식13개·문제 번호·참고 링크가 일치한다. ket의 `|`가 표 구분자로 분리되지 않는지와 수식에서 norm 기호로 바뀌지 않는지 확인했다. 영어 원고에는 한글이 없다. 구조·분량 검사는 편집 관문이며 교육 효과나 실제 reader 검증을 대신하지 않는다.

근거는 `.runtime/learning/spin-chain-manuscripts/verification.json`, HTML fragments·metrics·생성기·독립 JSON·로그와 소스/근거 hash manifest에 있다. 기존 소스/근거와 정적 출력은 비교하여 보존했다. 원고는 등록 전 `docs/`에 있으며 새 학습 단원을 등록하거나 홈페이지를 빌드하지 않았다.

## 이어갈 순서

다음은 **fermion Fock·parity** 한영 원고, 이어 boson cutoff/coherent·MPS·Schmidt 원고다. 다섯 원고 뒤 Worker·고정 개념·자유 실험·예측·세 안내·해설 퀴즈·저장·Three.js와 접근 가능한 표를 연결한다. 실제 **도서 → Interactive Book** reader와 생산 자산을 검증한 뒤 등록·원주제 입문 완료를 검토한다.

기존 홈페이지와 미리보기 서버를 유지하며 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [전체 남은 입문 순서](remaining-introductory-queue.md)의 범위를 유지한다.
