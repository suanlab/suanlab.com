# 다체계 두 번째 상세 원고: Fermion Fock·parity

2026-11-09. `fermion-fock-parity-operator-algebra`의 **상세 한영 원고 두 개**를 작성·검산했다. 다섯 단원 중 spin·fermion의 원고 두 쌍이 준비됐으며 boson·MPS·Schmidt와 Worker·교수 화면·reader는 남아 있다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**, 전체51영역·509주제와 연구 깊이를 유지한다.

- [한국어 원고](fermion-fock-parity-operator-algebra-chapter.ko.md)
- [English manuscript](fermion-fock-parity-operator-algebra-chapter.en.md)
- [가정·부호·출처 감사](quantum-fermion-fock-manuscript-audit.md)

## 작성한 내용

각 언어 **16절·17표시 수식·4비교표·8개 해설**이다. 입자와 모드의 차이·직교정규 가정·진공에서 시작해 CAR·점유 부호·Jordan–Wigner·복소 hopping/pairing·N/P 보존·SSR·twirl·gamma/F·sector 최소·AI 검증으로 이어진다. 한국어 Hangul5730자, 영어3517단어다.

첫 두 생성의 순서와 두 실제 CAR 경로, 중간 점유를 가로지르는 hopping의 에너지−1, 복소 pair의 역순 adjoint를 직접 계산한다. 표는 소멸자의 전체 작용, number 보존/분포/parity, 같은 gamma/F와 다른 N 통계, full/sector ground를 비교한다. pure parity 호환성과 even Hamiltonian의 보존을 구별하고, formal odd 평균을 선언한 물리 관측으로 제시하지 않는다. twirl은 다른 상태이며 자동 수정이 아니다.

## 검증 결과

제품 코드를 가져오지 않는 SymPy 생성기로 JW Kronecker·H²·실제 commutator·density/twirl·symbolic2p_ep_o와 one-mode 최소를 검산했다. 새 **11개 검사·기존67개 회귀**(fermion57·spin 원고10), 타입 검사와 lint를 통과했다. 독립 기준 JSON을 다시 생성해 SHA-256 바이트 일치를 확인했다.

strict KaTeX/GFM 렌더링에서 두 언어 모두 오류가 없고, 수식17개·여덟 문제 번호·참고 링크가 일치한다. ket의 `|`가 표의 열을 깨뜨리지 않으며 영어 본문에는 한글이 없다. 구조·분량 검사를 교육 효과나 실제 reader 검증으로 확대하지 않는다.

근거는 `.runtime/learning/fermion-fock-manuscripts/verification.json`, HTML fragments·metrics·독립 생성기/JSON·로그·소스/근거 manifest에 있다. 첫 spin 원고와 수치 기반을 포함한 기존 파일, 정적 홈페이지 출력은 보존했다. 새 원고는 등록 전 `docs/`에 있으며 실제 content loader·단원 registry에는 아직 연결하지 않았다.

## 다음 순서

다음은 **boson cutoff·coherent 투영**의 상세 한영 원고다. 이어 MPS 수축·gauge와 Schmidt 절단을 작성한다. 다섯 원고 뒤 Worker·고정 개념·자유 실험·예측·세 안내·해설 quiz·저장·Three.js와 접근 가능한 표, 실제 **도서 → Interactive Book** reader와 생산 자산을 검증한다. 등록 전에는 새 단원·원주제 입문 완료 수를 올리지 않는다.

홈페이지·미리보기 서버를 유지하며 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 전체 설계](quantum-many-body-authoring-plan.md)와 [전체 남은 입문 순서](remaining-introductory-queue.md)를 유지한다.
