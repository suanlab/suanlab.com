# 다체계 네 번째 상세 원고: MPS 수축·gauge

2026-11-09. `matrix-product-state-contraction-gauge`의 **상세 한영 원고 두 개**를 작성·검산했다. 다섯 단원 중 spin·fermion·boson·MPS의 네 원고 쌍이 준비됐다. Schmidt 원고와 다섯 단원의 Worker·학습 화면·reader 연결은 남아 있다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**를 유지한다. 전체51영역·509주제와 연구 심화의 완료를 뜻하지 않는다.

- [한국어 원고](matrix-product-state-contraction-gauge-chapter.ko.md)
- [English manuscript](matrix-product-state-contraction-gauge-chapter.en.md)
- [색인·가정·출처 감사](quantum-mps-manuscript-audit.md)

## 작성한 내용

각 언어 **16절·14표시 수식·4비교표·8개 해설**이다. 한국어 Hangul5853자, 영어3224단어다. 텐서의 물리/가상 다리와 열린 경계에서 시작해 GHZ 경로·C=LB·환경 closure·복소 위상·관측 삽입·gauge·좌표 길이·covariance·isometry·canonicalization의 역할·상관·rank/비용으로 연결한다.

같은 상태의 텐서 길이가6→19/2로 바뀌는 GHZ gauge와, 잘못된 Tr(ER)가 영 수축을 노름4로 오판하는 복소 반례를 손으로 전개한다. QR/SVD 설명을 자동 구현과 구별하고, 같은 pair 관측을 가진 GHZ/혼합의 XXX 차이를 해석적으로 비교한다. 수치 모형의 작은 범위를 교육과정의 분량 제한으로 삼지 않는다.

## 검증 결과

제품 모듈과 별개인 SymPy 생성기는 일곱 준비에서 모든 가상 경로·직접 Gram·dense partial trace와 Pauli 비교를 계산한다. 새 **13검사·정확2296관측 대조**, 기존 **73회귀**(MPS41·이전 원고32), 타입·lint가 통과했다. 기준 JSON의 재생성 SHA-256 바이트 일치를 확인했다.

Strict KaTeX/GFM 렌더링, 한영 수식14개·여덟 문제 번호·참고 링크가 일치하며 영어 본문에는 한글이 없다. 원고는 등록 전 `docs/`에 있다. 실제 content loader·Worker·reader·Three.js/접근 가능한 실험 표를 검증했다고 세지 않는다.

근거는 `.runtime/learning/mps-manuscripts/verification.json`, 렌더링 HTML fragments·metrics·독립 생성기/JSON·검사 로그와 소스/근거 manifest에 기록한다. 기존 소스·증거와 정적 홈페이지 출력의 보존을 확인한다.

## 다음 순서

다음은 **Schmidt cut·최적 절단**의 상세 한영 원고다. 이후 다섯 단원의 Worker·고정 개념·자유 실험·예측·세 안내·해설 quiz·저장·Three.js와 접근 가능한 표, 실제 reader와 생산 자산을 검증한다. 진입은 **도서 → Interactive Book**이며 Online Book 목록에 추가하지 않는다.

등록 전 새 단원·원래 입문 완료 수를 올리지 않는다. 홈페이지·미리보기 서버를 유지하며 commit·push·공개 배포는 수행하지 않았다. [다섯 단원 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다.
