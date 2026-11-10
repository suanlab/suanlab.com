# 다체계 세 번째 상세 원고: Boson cutoff·coherent

2026-11-09. `boson-fock-cutoff-boundary-algebra`의 **상세 한영 원고 두 개**를 작성·검산했다. 다섯 단원 중 spin·fermion·boson의 세 원고 쌍이 준비됐다. MPS·Schmidt 원고와 Worker·학습 화면·reader 연결은 남아 있다. 기존 **566개 한영 단원·1,698개 안내, 원래 입문308/362개 검증·54개 미완료**를 유지하며 전체51영역·509주제와 연구 심화는 진행 중이다.

- [한국어 원고](boson-fock-cutoff-boundary-algebra-chapter.ko.md)
- [English manuscript](boson-fock-cutoff-boundary-algebra-chapter.en.md)
- [가정·정의·출처 감사](quantum-boson-fock-manuscript-audit.md)

## 작성한 내용

각 언어 **18절·17표시 수식·4비교표·8개 해설**이다. 한국어 Hangul5080자, 영어3178단어다. √n의 정규화 유도에서 시작해 경계 CCR·실제 불확정성 bound·곱과 투영의 순서·hopping 증강·coherent 급수·Poisson tail·fidelity·서로 다른 거리·nilpotency·cutoff 안정성·unbounded 관측 오차까지 연결한다.

α=1,C=3의 number 평균15/16과 분산207/256을 손으로 계산하고, 동일한 embedded 상태에도 유한 quadrature와 원래 quadrature가 서로 다른 분산을 준다는 비교를 제시한다. 여덟 문제에는 상단 상태·누락된 hopping 경로·C=0 oscillator 영점 에너지·두 모드 tail·작은 norm 오차와 큰 number 오차의 해설이 있다.

## 검증 결과

제품 코드와 별개인 SymPy/multiprecision 생성기로 C=0…3의 행렬, 확장 공간의 투영 곱, H², exact coherent 계수와 incomplete-gamma tail을 계산했다. 새 **11개 검사·기존161개 회귀**(boson140·이전 원고21), 타입 검사·lint가 통과했다. 독립 기준 JSON을 다시 생성해 SHA-256 바이트 일치를 확인했다.

두 언어의 strict KaTeX/GFM 렌더링과 수식17개·여덟 문제 번호·참고 링크가 일치한다. 표의 ket가 열을 깨뜨리지 않고 영어 본문에는 한글이 없다. 원고는 `docs/`에 있으며 실제 content loader·단원 registry에는 아직 연결하지 않았다. 해석적 확장 비교를 계산기의 구현 기능으로 세지 않는다.

근거는 `.runtime/learning/boson-fock-manuscripts/verification.json`, HTML fragments·metrics·생성기/JSON·검사 로그와 소스/근거 manifest에 기록한다. 기존 소스·증거와 정적 홈페이지 출력의 보존을 확인한다.

## 다음 순서

**MPS 수축·gauge → Schmidt cut·최적 절단**의 상세 한영 원고를 작성한다. 이후 다섯 단원의 Worker·개념 설명·자유 실험·예측·세 안내·해설 quiz·저장·Three.js/접근 가능한 표·실제 reader와 생산 자산을 검증한다. 진입은 **도서 → Interactive Book**을 유지하며 Online Book 목록에는 추가하지 않는다.

등록 전 새 단원·원래 입문 완료 수를 올리지 않는다. 미리보기 서버와 홈페이지 출력을 유지하며 commit·push·공개 배포는 수행하지 않았다. [전체 설계](quantum-many-body-authoring-plan.md)와 [남은 입문 순서](remaining-introductory-queue.md)를 유지한다.
