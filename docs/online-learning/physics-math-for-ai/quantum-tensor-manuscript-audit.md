# Tensor product·entanglement 한영 원고 검산

2026-11-08. 두 계산 가족의 상세 원고4편을 작성했다. [집필 계약](quantum-tensor-entanglement-authoring-plan.md), [순수 수치 계약](quantum-tensor-pure-source-audit.md), [혼합 수치 계약](quantum-mixed-entanglement-source-audit.md)과 대조했다. 기존 수치·참조·본문·Worker·교수 화면은 수정하지 않았다.

## 유도와 관례

순수 원고는 텐서곱과 직합·쌍선형 보편 성질·곱 기저 내적에서 시작한다. A-major 펼침과 국소 작용 C′=ACBᵀ, 부분추적의 국소 기대 보존/기저 독립성, Bob 밀도의 Gram 전치, SVD의 켤레 Schmidt 벡터를 유도한다. 비영 축약 스펙트럼 일치·서로 다른 영 중복도·축퇴 기저 자유·rank1/minor/국소 순수성의 동치를 전체 순수 가정 아래 설명한다.

엔트로피와 rank·순도는 같은 양이 아님을 qutrit/ququart로 비교한다. 거친 여공간 조건부는 혼합일 수 있으며0 사건은 null이다. 국소 unitary와 곱 기저 dephasing, 순수 부분전치의 ±pair 및 negativity의1/2 규약을 구별한다. Schmidt truncation·purification·GHZ/W·tensor network로 연결하되 일반 다부분계 자원 연구를 완료 처리하지 않는다.

혼합 원고는 모든 principal minor에 의한 PSD 판정을 det(tI+H)의 계수로 설명한다. 일반 Pauli 재구성·분리 볼록 혼합과 ensemble 비유일성·PPT 필요성을 유도한다. PPT 충분성은2×2/2×3 차원 정리를 사용한다고 명시하며 일반 고차원 증명을 제공했다고 주장하지 않는다. 엔트로피 차이·조건부 상태·무신호성, 국소 숨은 변수 CHSH 상한과 설정 독립성, 연속 projective 최적화를 설명한다. G의 상위 두 고유값 공식과 유리수 shifted-PSD 경계 판정을 별도로 유도한다.

## 예제의 독립적인 손계산 값

- C의 두 행 (1,i,2),(2,2i,4)는 u=(1,2),v=(1,i,2)의 곱이며 N=30이다.
- diag(1,2,3)의 가중치는1/14,4/14,9/14,순도1/2,엔트로피 약1.1981174211bit다. diag(1,2,3,4)의 순도는59/150,엔트로피 약1.5559130952bit다.
- 최대 얽힘 qutrit의 P=|0⟩⟨0| 여공간 사건은 확률2/3,조건부 순도1/2다. 부분전치에는 양6개·음3개가 있고 negativity1이다.
- Bell의 Bob spinor (5,±2)는 국소 방향 (±20/29,0,21/29)를 준다. A0=Z,A1=X에서 E=(21,21,20,−20)/29,S=82/29다.
- Werner p=1/2는 부분전치−1/8,3/8,3/8,3/8,negativity1/8이며 최적 CHSH는√2다. p=1/3과1/√2는 서로 다른 문턱이다.
- T=diag(3/5,−4/5,1/2)의 Bell 가중치는29/40,1/40,3/40,7/40,negativity9/40이다. K=diag(11/100,39/100,0)으로 최적 CHSH가 정확히2다. 밀도에 ±XX/100000을 더하면 Txx는 ±1/25000만큼 변하고 양성을 유지한다.

19개 새 집필 검사는 이 값들을 두 기존 수치 모델과 대조한다. GHZ/W 예시는 별도의 직접 배열 부분추적으로 비교한다. 기존 독립 SymPy/mpmath 참조와 엔진 검사는 보존해 함께 실행한다. 표시 수식의 한영 원문은 바이트 단위로 동일하다.

## 렌더링과 검증의 한계

언어별14절·해설6문제·표2개, 순수10/혼합11표시 수식이다. 전체4편은56절·24해설·8표·42표시 수식이다. 영어 두 편은3037/2978단어이며 한글이 없다. 한국어 원고는12300/12606자다. Strict KaTeX 오류·잘린 표 열·중복 heading ID·미치환 수식 placeholder를 검사하고 HTML을 runtime에 저장했다.

이는 문서 렌더링이며 실제 reader·접근성·Worker·정적 배포 검증이 아니다. 등록562단원·1686안내·306/362입문 검증·56미완료를 유지한다. 진입은 **도서 → Interactive Book**, Online Book 목록에는 카드를 넣지 않는다.

출처 관례: [Preskill §2.3–2.4](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), [Horodecki PPT 차원 정리](https://arxiv.org/abs/quant-ph/9605038), [저자 종설의 CHSH 측정 가정/최적화 §IV.A–B](https://arxiv.org/pdf/quant-ph/0702225). 이 출처들에 구현의 유리수 shifted-PSD 알고리즘이 실려 있다고 주장하지 않는다.
