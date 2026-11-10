# 다체계 다섯 번째 상세 원고: Schmidt cut·최적 절단

2026-11-09. `schmidt-cuts-bond-rank-truncation`의 **상세 한영 원고 두 개**를 작성·검산했다. Spin·fermion·boson·MPS·Schmidt의 다섯 원고 쌍과 다섯 수치 가족이 준비됐다. **다섯 단원의 Worker·학습 화면·실제 reader 연결은 아직 남아 있다.** 기존566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료를 유지한다. 전체51영역·509주제와 연구 심화의 완료를 뜻하지 않는다.

- [한국어 원고](schmidt-cuts-bond-rank-truncation-chapter.ko.md)
- [English manuscript](schmidt-cuts-bond-rank-truncation-chapter.en.md)
- [정규화·방향·축퇴 감사](quantum-schmidt-manuscript-audit.md)

## 작성한 내용

각 언어 **16절·16표시 수식·4비교표·8개 해설**이다. 한국어 Hangul5674자, 영어3185단어다. Cut와 ket 계수에서 시작해 양쪽 물리 density·SVD·rank·purity/entropy·최적 projection 증명·세 상태·오차/fidelity/거리·축퇴·budget0·오른쪽 transpose·tiny weight·관측 오차와 다중 cut 한계로 연결한다.

2|00〉+|11〉의 오류1/5와 정규화 거리2−4/√5, Bell의 여러 최적해, diag(3,2,2,1)의 다섯 budget을 손으로 비교한다. CPᵀ를 CP로 잘못 바꾸면 복소 예제의 오차가1/5→4/5가 됨을 실제 진폭으로 확인한다. 한 cut의 최적성으로 전체 MPS 압축을 해결했다고 하지 않는다.

## 검증 결과

제품 코드 없는 SymPy eigenvectors·Gram–Schmidt·직접 partial trace·실제 절단과250자리 평가로 13준비의 모든 cut을 계산했다. 새 **20검사·정확2561/방향성3589관측·4374잔차 인증**, 기존 **84회귀**(Schmidt39·이전 원고45), 타입·lint와 기준 JSON 재생성 SHA-256 바이트 일치를 확인한다.

Strict KaTeX/GFM 렌더링과 한영 수식16개·여덟 문제 번호·참고 링크가 일치하며 영어 본문에는 한글이 없다. 다섯 원고 합계는 각 언어80절·77표시 수식·19표·38개 해설이다. 등록 전 `docs/`의 원고이며 실제 content loader·Worker·reader·생산 Three.js/접근 가능한 실험 표의 검증으로 세지 않는다.

근거는 `.runtime/learning/schmidt-manuscripts/verification.json`, 렌더링 HTML fragments·metrics·독립 생성기/JSON·검사 로그와 소스/근거 manifest다. 기존 소스·증거와 정적 홈페이지 출력을 보존한다.

## 다음 순서

다섯 수치 가족을 Worker transport에 연결하고, 각 단원의 고정 개념·자유 실험·예측·세 안내·해설 quiz·저장·Three.js와 접근 가능한 표를 구현한다. 그다음 실제 content loader·reader 등록과 생산 자산·한영/반응형/접근성/상호작용을 검증한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 추가하지 않는다.

이 관문 전에는 새 단원·원래 입문 완료 수를 올리지 않는다. 홈페이지·미리보기 서버를 유지하며 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다.
