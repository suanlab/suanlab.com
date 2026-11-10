# Mixed two-qubit PPT·상관·CHSH 수치 계약

2026-11-08. `two-qubit-ppt-correlations-chsh`의 수치 기반이다. [순수 합성계 가족](quantum-tensor-pure-source-audit.md)과 별도로 일반 혼합 상태를 받는다. 본문·Worker·교수 화면·등록의 완료 근거로 사용하지 않는다.

## 입력과 정확한 상태 검증

32개 소유 정수는 모두 절댓값10⁶ 이하다. `hRealij`(i≤j)10개와 `hImagij`(i<j)6개가 Hermitian4×4 분자를 정한다. 아래 삼각은 켤레이며 대각 허수 입력은 없다. 양의 trace와 **모든15 principal minor의 비음성**을 확인하고 ρ=H/TrH로 정규화한다. Leading minor만 검사하거나 음수를 자르거나 상태를 PSD로 보정하지 않는다. 이 유계 정수 표현으로 나타낼 수 있는 일반 유리수 밀도 행렬을 받으며 Werner·Bell-diagonal·X-state로 제한하지 않는다.

`a0`,`a1`,`b0`,`b1` 각각의 `Real0,Imag0,Real1,Imag1`은 비영 복소 spinor다. P=zz†/‖z‖²의 두 사건 P,I−P와 traceless±1 관측량 O=2P−I를 만든다. 정수 spinor는 임의의 연속 방향 전체를 정확히 매개화하는 계약이 아니다. 최적 CHSH 정리는 별도로 **모든 연속 국소 qubit projective 방향**에 대해 계산한다. 선택한 spinor들이 최적 방향이라고 표시하지 않는다.

## 행렬·부분전치·정보량

A-major 좌표2a+b, Pauli Y의 위쪽 원소−i를 사용한다. 부분추적·국소 Bloch vector·Tᵢⱼ=Trρ(σᵢ⊗σⱼ)·공동 확률·선택 CHSH·조건부 축약 상태는 정확 복소 유리수다. 전체 Pauli 전개로 ρ를 재구성해 부호와 인덱스를 검산한다. 확률0 사건의 조건부 상태/순도는 null이며 비선택 국소 측정은 원격 축약 밀도를 보존한다.

B 부분전치는 (ρᵀᴮ)₂ₐ₊ᵦ,₂𝑐₊𝑑=ρ₂ₐ₊𝑑,₂𝑐₊ᵦ이다. 모든 principal minor와 정확 스펙트럼 부호가 일치해야 한다. 이 모델의2×2에서는 PPT⇔분리를 적용한다. 높은 차원 일반 혼합 상태까지 충분하다고 주장하지 않으며 분리 상태의 곱 혼합 분해를 계산하지 않는다. Negativity는 음의 고유값 절댓값의 합으로 계산해 trace-norm−1의 상쇄를 피한다.

기존 일반4차 square-free/Sturm helper를 변경 없이 재사용한다. 상태·부분전치4×4와 G=TᵀT3×3의 근/중복도를1024bit에서 분리한다. 영/음/양 부호는 tolerance로 결정하지 않는다. 엔트로피 S=−Σλlnλ(0ln0=0), mutual information, 조건부 엔트로피, negativity와 log₂(1+2N)는768bit 작업과512bit 바깥 방향 관측을 사용한다. 작은 양의 고유값을 삭제하지 않으며 수치 고유벡터를 제공하지 않는다.

## CHSH 경계의 정확 판정

선택 설정은 S=E₀₀+E₀₁+E₁₀−E₁₁, 정확 |S|>2로 판정한다. 최적 크기는2√M, M은 G의 가장 큰 두 고유값 합이다. 보고 구간에서는 이 두 근을 직접 더한다.

분류는 구간 midpoint나 부동소수점 근을 쓰지 않는다. G의 최소 고유값을 μ₁, q=TrG−1이라 하면 M>1⇔μ₁<q⇔G−qI가 PSD가 아니다. G−qI가 PSD이고 determinant=0이면 M=1이며, PSD이고 determinant>0이면 M<1이다. 모든 principal minor를 정확 유리수로 검사하므로 중복 최소근과 정확 경계도 구별한다. 이는 구현에서 유도한 행렬 판정이며 출처에 이 알고리즘이 실렸다고 주장하지 않는다.

Bell 위반의2상한은 국소 숨은 변수 가정과 두 가지±1 국소 설정에 대한 것이다. 계산은 표본 오차·측정 독립성 검증·검출 loophole·필터링·postselection·실험 증명을 대신하지 않는다. CHSH 비위반을 일반적인 local hidden-variable 모델의 존재로 판정하지 않는다.

## 독립 참조와 출처

`scripts/learning/generate-mixed-entanglement-references.py`는 SymPy 직접 tensor/partial trace·모든 minor와 mpmath250dps 일반 Hermitian eigensolver를 사용한다. CHSH 경계는 생산 코드의 shifted principal-minor 판정과 달리, 정확 q근을 나눈 후 SymPy 다항식의 q보다 작은 근 개수를 센다. 중복도는 square-free 인자의 정확 중복도를 사용한다.210자리 실수 참조에 대해 max(1,|참조|)×10⁻¹²⁵ 구간 폭을 검사한다.

출처 대조: [Horodecki 등의 원 논문](https://arxiv.org/abs/quant-ph/9605038)은 PPT 충분성의 차원 제한을, [저자들의 Quantum entanglement §IV.A–B](https://arxiv.org/pdf/quant-ph/0702225)는 국소±1 Pauli 관측, TᵀT 상위 두 고유값의 CHSH 최적화와 M>1 조건을 뒷받침한다. Negativity는 위에서 정의한 trace-norm 규약(최대 Bell 상태 N=1/2)을 사용하며 다른 규약의 계수를 혼용하지 않는다.
