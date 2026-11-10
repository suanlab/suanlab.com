## 1. 국소 무작위성만으로 원인을 알 수 없다

두-qubit 광원에서 qubit 하나를 받았다고 하자. 국소 상태가 I2/2라면 모든 Pauli 측정에서 두 결과의 확률이 같다. 이 광원은 Bell 상태, 고전적으로 상관된 혼합, 또는 서로 독립인 최대 혼합 qubit 둘을 준비했을 수 있다. 자신의 qubit에만 하는 실험으로는 이 가능성들을 구별할 수 없다.

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},\qquad
\rho_{\rm Bell}=|\Phi^+\rangle\langle\Phi^+|,\quad
\rho_{\rm classical}=\frac{|00\rangle\langle00|+|11\rangle\langle11|}{2},\quad
\rho_{\rm independent}=\frac{I_4}{4}.
$$

세 광원의 두 국소 밀도 행렬은 모두 같다. Bell 광원과 고전적 광원은 계산 기저의 공동 확률까지 같아서00과11이 절반씩 나온다. 그러나 다른 공동 측정에서는 결과가 달라진다. 개념 그림은 **유효한 공동 상태 → 국소·공동 법칙 → 분리 판정 → 선택한 Bell 검사 → 최적화한 Bell 검사**다. 각 화살표는 서로 다른 질문에 답한다.

| 광원 | 전체 순도 | bit 단위 mutual information | Negativity | 최적 CHSH 크기 |
| --- | --- | --- | --- | --- |
| Bell 쌍 | 1 | 2 | 1/2 | 2√2 |
| 고전적으로 상관된 혼합 | 1/2 | 1 | 0 | 2 |
| 독립 I4/4 | 1/4 | 0 | 0 | 0 |

마지막 열은 고유값이+1,−1인 traceless qubit 관측량에 대한 최적화다. 상수 출력을 주는 항등 관측량을 허용하면 측정 클래스가 달라진다. 특히 I4/4의0은 생각할 수 있는 모든 출력 방식에서 CHSH가0이라는 뜻이 아니다.

## 2. 정규화 전에 Hermitian 상태 전체를 검증한다

밀도 행렬은 Hermitian이고 양의 준정부호이며 trace가1이어야 한다. 유한 가족은 일반 Hermitian 분자 H를 받아 양의 trace로 나눈다. 위 삼각에는 실수 성분10개와 독립적인 허수 성분6개가 있고, 아래 삼각은 위 삼각의 복소켤레로 결정된다.

$$
H=H^\dagger,\quad \operatorname{Tr}H>0,\quad
\rho=\frac{H}{\operatorname{Tr}H},\qquad
H\succeq0\ \Longleftrightarrow\ \det H[J,J]\ge0
\quad\text{for every nonempty }J\subseteq\{0,1,2,3\}.
$$

비어 있지 않은 principal 부분행렬은15개다. 처음부터 연속된 좌표로만 만드는 leading principal minor 네 개만 보면 준정부호 판정에 충분하지 않다. 예를 들어 diag(1,0,−1,1)의 leading determinant는1,0,0,0으로 모두 음수가 아니고 trace도 양수지만, 세 번째 대각 방향은 음수다. Trace로 나누어도 이 결함이 고쳐지지 않는다.

모든 principal minor가 충분한 이유도 보자. 필요성은 PSD 행렬의 모든 principal 제한이 PSD라는 데서 나온다. 역으로 그 minor들의 합은 det(tI+H)의 계수가 된다. 모든 minor가 음수가 아니면 이 monic 다항식은 양의 모든 t에서 엄밀히 양수다. 음의 Hermitian 고유값 λ가 있다면 t=−λ라는 양의 근을 주므로 모순이다. Hermitian 가정은 고유값이 실수임을 보장하기 때문에 중요하다.

유효하지 않은 입력은 거부한다. 음의 고유값을 잘라버리면 다른 상태를 만들어 원래 문제를 감추게 된다. 순수 상태나 정확한 PPT 경계처럼 특이한 유효 상태는 그대로 허용한다. 양의 준정부호는 모든 고유값이 엄밀히 양수여야 한다는 뜻이 아니다.

## 3. Pauli 전개로 국소 평균과 상관을 나눈다

Y의 오른쪽 위 원소가−i인 X,Y,Z 순서를 사용한다. 이들과 I는 trace 내적에서 서로 직교한다. 그 텐서곱으로 임의의 두-qubit 밀도를 전개하면 trace 정규화 뒤15개의 실수 상태 매개변수를 얻는다.

$$
r_i=\operatorname{Tr}(\rho\,\sigma_i\otimes I),\quad
s_j=\operatorname{Tr}(\rho\,I\otimes\sigma_j),\quad
T_{ij}=\operatorname{Tr}(\rho\,\sigma_i\otimes\sigma_j),\qquad
\rho=\frac14\left[I\otimes I+\sum_i r_i\sigma_i\otimes I+\sum_j s_jI\otimes\sigma_j+\sum_{ij}T_{ij}\sigma_i\otimes\sigma_j\right].
$$

Bloch 벡터 r,s가 두 축약 상태를 결정한다. 실수3×3 tensor T는 marginal만으로 결정되지 않는 공동 Pauli 상관을 담는다. 국소 방향 a,b에 대한 (a·σ)⊗(b·σ)의 기대는 aᵀTb다. 국소 평균의 곱을 빼면 연결 상관 aᵀ(T−rsᵀ)b를 얻는다.

곱 밀도 ρA⊗ρB에서는 T=rsᵀ이므로 모든 이런 연결 상관이0이다. 그러나 분리 가능한 혼합은 공유 준비 라벨 때문에 고전적 연결 상관을 가질 수 있다. 따라서 연결 상관이0이 아니라는 사실만으로 얽힘을 증명할 수 없다.

이 전개에서 전체 밀도를 복원하면 Y 부호와 부분계 순서를 포함한 모든 성분을 검산한다. 여기서는 공동 색인이2a+b인 A-major 관례를 쓴다. 국소 Bloch 벡터만 그리는 그림은9개의 공동 매개변수를 생략한다.

## 4. 분리 가능성은 공동 볼록 분해의 존재에 관한 성질이다

상태를 국소 곱 밀도들의 고전적 혼합으로 준비할 수 있으면 분리 가능하다고 한다. 같은 준비 라벨 k가 양쪽 인수를 함께 선택하므로 보통의 고전적 상관도 허용한다.

$$
\rho\ \text{separable}\quad\Longleftrightarrow\quad
\rho=\sum_k p_k\rho_A^{(k)}\otimes\rho_B^{(k)},\qquad
p_k\ge0,\quad\sum_kp_k=1.
$$

얽힘이란 이런 분해가 하나도 없다는 뜻이다. 얽힌 벡터가 들어간 어떤 분해를 찾아냈다는 것과는 다르다. 예를 들어 |Φ+⟩와 |Φ−⟩를 같은 비율로 섞으면 대각 밖 coherence가 상쇄되어1절의 고전적 상태가 된다. 얽힌 상태를 재료로 사용하는 ensemble 하나가 존재하더라도 같은 밀도에 분리 ensemble이 있을 수 있다.

이전 장의 순수 계수 표 rank1 검사는 임의 혼합 밀도의 원소 표에 그대로 적용되지 않는다. 혼합 밀도에는 bra 색인 두 개와 ket 색인 두 개가 있다. 이를 하나의 순수 진폭 표로 읽으면 수학적 대상이 달라진다. 국소 혼합도만으로 공동 광원의 얽힘을 결정할 수도 없다.

## 5. 부분전치로 분리의 필요조건을 얻는다

고정한 곱 기저에서 Bob의 행렬 색인만 전치한다. 결과는 여전히 Hermitian이고 trace가1이지만 양수가 아닐 수 있다. 하나의 독립된 밀도를 전치하면 양성은 유지되지만, identity tensor transpose는 모든 공동 상태에서 양성을 보존하지 않는다.

$$
(\rho^{T_B})_{ab,cd}=\rho_{ad,cb},\qquad
\left(\sum_kp_k\rho_A^{(k)}\otimes\rho_B^{(k)}\right)^{T_B}
=\sum_kp_k\rho_A^{(k)}\otimes(\rho_B^{(k)})^T\succeq0.
$$

마지막 부등식은 모든 분리 상태가 positive partial transpose, 줄여서 PPT를 가진다는 증명이다. 따라서 부분전치의 음의 고유값 하나만 찾아도 얽힘이 인증된다. 여기서 전치는 수학적 검사이며, Bob의 실험실에서 물리적으로 불가능한 보편 전치 channel을 구현하라는 지시가 아니다.

Bell 상태에서는00과11 사이의 coherence가 부분전치 후01,10 블록으로 옮겨간다. 고유값은1/2,1/2,1/2,−1/2다. 양쪽 국소 I/2에는 보이지 않던 음의 방향이 전체 검사에 나타난다.

부분전치에는 선택한 기저가 들어가지만, 국소 기저를 바꾸면 부분전치 행렬은 Bob의 unitary를 켤레로 바꾼 unitary 켤레변환을 받는다. 따라서 고유값과 양성은 기저 선택과 무관하다. A와 B를 바꿔도 두 부분전치가 서로 전체 전치 관계이므로 스펙트럼은 같다.

## 6. 두 qubit에서는 PPT가 충분조건이기도 하다

2×2, 그리고2×3 차원에서는 PPT이면 분리 가능하다. 이는 특정 차원에 한정된 정리다. 두-qubit 계산기는4×4 입력에 이 정리를 적용한다. 임의의 더 큰 혼합 상태에 같은 함의를 적용하지 않는다. 고차원에는 PPT이면서 얽힌 상태가 존재한다.

5절의 필요성 증명은 간단하지만 충분성은 추가 구조를 요구한다. 보통 분리 가능성의 positive-map 특성화와 낮은 차원의 positive map 구조를 통해 설명한다. Principal minor를 검사한다는 사실만으로 모든 차원에서 그 정리가 증명되는 것은 아니다. 계산 인증은 해석에 사용한 정리와 차원 조건을 명시해야 한다.

이 모델에서 PPT라는 결과는 분리 분해의 존재를 보장하지만 실제 곱 인수들을 구성해주지는 않는다. 음의 부분전치라는 결과는 얽힘을 인증하지만, 선택한 CHSH 설정의 Bell 위반까지 결정하지는 않는다. 출력에서 이 결론들을 구분해서 읽자.

## 7. 스펙트럼 negativity와 공동 정보를 계산한다

Negativity는 음의 부분전치 고유값 크기를 더한 것이다. 이 장의 trace-norm 정의에는1/2 인수가 들어간다. 다른 정규화를 사용하는 문헌도 있으므로 숫자를 비교하기 전에 Bell 값을 확인해야 한다.

$$
\mathcal N(\rho)=\frac{\|\rho^{T_B}\|_1-1}{2}
=-\sum_{\nu_k<0}\nu_k,\qquad
E_{\mathcal N}=\log_2(1+2\mathcal N),\qquad
I(A:B)=S(\rho_A)+S(\rho_B)-S(\rho),\quad
S(A|B)=S(\rho)-S(\rho_B).
$$

명시적으로 bit로 바꾸지 않는 한 엔트로피는 자연로그를 사용한다. Bell 쌍의 전체 엔트로피는0이고 양쪽 국소 엔트로피는 ln2다. 따라서 mutual information은2ln2이고 조건부 엔트로피 S(A|B)는−ln2다. 고전적 상관 혼합의 전체 엔트로피와 mutual information은 ln2, 조건부 엔트로피는0이다. 독립 I4/4는 전체 엔트로피2ln2와 mutual information0을 갖는다.

음의 양자 조건부 엔트로피가 가능한 이유는 밀도 연산자의 엔트로피 차이를 계산하기 때문이다. 고전적 조건부 확률분포의 Shannon entropy와 같은 대상이 아니다. Mutual information은 고전적·양자 상관을 모두 포함하므로 일반 혼합 상태의 순수 얽힘 척도가 아니다.

두-qubit PPT 상태의 negativity는 정확히0이다. 수치 구간이0에 가깝다는 것이 정확한 양성 인증을 대신하지는 못한다. 계산기는 음의 크기를 직접 합하고, 정확한 영 고유값을 보존하며, tolerance로 부호를 정하지 않는다.

## 8. 각 측정 설정 쌍의 확률 법칙을 유도한다

0이 아닌 복소 spinor z로 rank1 projector P=zz†/‖z‖²와 O=2P−I를 만든다. 결과는+1,−1이다. 네 spinor가 A0,A1,B0,B1을 정한다. 각 (x,y) 설정 쌍은 같은 광원의 독립 복사본들을 반복 준비하여 수행하는 별도의 실험이다.

$$
P_a^x=\frac{I+a A_x}{2},\quad Q_b^y=\frac{I+b B_y}{2},\qquad
p(a,b|x,y)=\operatorname{Tr}[\rho(P_a^x\otimes Q_b^y)],\quad
E_{xy}=\sum_{a,b=\pm1}ab\,p(a,b|x,y)=\operatorname{Tr}[\rho(A_x\otimes B_y)].
$$

Bob의 결과를 합하면 그의 부분계에 I가 들어가므로 Alice의 marginal은 y와 무관하다. Alice의 결과를 합하면 반대 방향의 같은 성질을 얻는다. 이 항등식은 상관 값만이 아니라 네 확률 표 전체를 검산한다.

Alice의 결과로 Bob을 조건화하려면 국소 P를 적용하여 정규화 전 공동 측정 후 밀도를 만든 뒤 Alice를 부분추적한다. 결과 확률이 양수일 때만 그 확률로 나눈다. 확률0의 조건부 밀도와 순도는 null이다. 각 Alice 설정에서 정규화 전 조건부들을 합하면 원래 Bob 축약 밀도가 나온다.

조건화는 라벨을 붙인 부분 ensemble의 상태를 바꿀 수 있다. 결과를 무시하면 원격 법칙은 같다. 결과 라벨을 공유하려면 고전 통신이 필요하며 초광속 신호를 주지 않는다. 이 유한 모델은 이상적인 모집단 법칙을 계산한다. 유한 실험 표본이나 잡음 있는 tomography의 추론 절차를 제공하는 것은 아니다.

## 9. 국소 숨은 변수의 CHSH 상한을 유도한다

결정론적 국소 할당에서 Alice의 출력 a0,a1과 Bob의 출력 b0,b1은 각각−1 또는+1이다. b0+b1과 b0−b1 중 정확히 하나는0이고 다른 하나는±2다. 따라서 a0(b0+b1)+a1(b0−b1)의 절댓값은2다. 이런 할당을 확률적으로 섞어도 기대의 절댓값은2를 넘지 못한다.

$$
S=E_{00}+E_{01}+E_{10}-E_{11},\qquad
|S|\le2\quad\text{for local hidden-variable assignments},\qquad
\mathcal B=A_0\otimes(B_0+B_1)+A_1\otimes(B_0-B_1).
$$

확률적인 국소 반응도 그 무작위성을 숨은 변수에 포함하면 같은 논증에 들어간다. 비교에는 공유 숨은 변수의 분포가 설정 선택과 독립이고, 각 반응은 자신의 국소 설정과 숨은 변수에만 의존한다는 가정이 필요하다. 실제 실험에는 표본 선정과 검출에 대한 정당한 가정도 필요하다.

양자 이론은 네 개의 다른 설정 쌍 실험으로 S=Trρℬ를 계산한다. 비가환 A0,A1을 한 복사본에서 동시에 관측한 결과를 요구하지 않는다. Bell 위반은 관측 법칙을 그 국소 모형으로 설명할 수 없다는 뜻이며 확률론 자체가 틀렸다는 뜻이 아니다.

계산기는 선택한 설정의 정확한 유리수 |S|>2를 판정한다. 한 방향 집합에서 한 표현을 위반하지 않았다는 사실만으로 다른 방향, 결과 라벨 변경, 다른 Bell 시나리오에서도 비국소성이 없다고 결론지을 수 없다.

## 10. 모든 사영 방향에 대해 CHSH를 최적화한다

Ax=ax·σ,By=by·σ로 쓰고 a,b를 실수 단위벡터로 두자. b0+b1과 b0−b1은 직교하므로 각각2cosθ v,2sinθ w로 표현할 수 있다. v,w는 정규직교다. 고정한 v,w에 대해 Alice 축을 Tv,Tw에 맞추고 필요한 부호를 택한다. 이어 θ를 최적화하면 두 길이 제곱의 합에 대한 제곱근이 나온다.

$$
G=T^T T,\quad 0\le\mu_1\le\mu_2\le\mu_3,\qquad
\max_{A_0,A_1,B_0,B_1}|S|=2\sqrt{\mu_2+\mu_3}=2\sqrt M,
\qquad M>1\ \Longleftrightarrow\ \text{some CHSH directions violate }2.
$$

가장 큰 두 고유값이 나오는 이유를 확인하자. 정규직교 v,w에 대해 vᵀGv+wᵀGw를 최대로 만든다. G의 고유기저에서 두 벡터는 각 고유값에0과1 사이의 가중치를 부여하며 총 가중치는2다. 가장 큰 두 고유값에 각각1을 놓으면 최대다. 이 설명은 특정 고유축을 유일하게 고르지 않고 축퇴도 포함한다.

단위 방향 b에서 ‖Tb‖는 Alice 단위 방향을 바꿨을 때의 최대 상관이다. 대응하는 곱 Pauli 관측량의 고유값이±1이므로 상관의 크기는1을 넘을 수 없다. 따라서 μ2,μ3≤1이고 최적 크기는 Tsirelson 상한2√2 이하다.

이 최적화는 연속적인 모든 traceless projective 방향에 대한 것이고, 선택한 입력은 유계 정수 spinor 대표다. 보고한 최댓값을 현재 방향이나 정확히 표현 가능한 유계 spinor가 달성할 필요는 없다. 엔진은 최대 크기와 정확한 위반 분류를 주지만, 그 값을 달성하는 고유벡터나 측정축은 계산하지 않는다.

## 11. Bell과 Werner의 문턱을 직접 계산한다

|Φ+⟩에서는 T=diag(1,−1,1)이다. A0=Z,A1=X와 Bob spinor (5,2),(5,−2)를 고르자. Bob의 Bloch 방향은 (20/29,0,21/29),(−20/29,0,21/29)이며 S=82/29, 약2.827586을 준다.2보다 크지만2√2보다는 작다. 같은 밀도라도 방향을 나쁘게 고르면 위반을 놓칠 수 있다.

Werner 가족은 singlet와 독립 잡음을 섞는다. 상관 tensor는−pI3이고 부분전치 스펙트럼도 직접 구할 수 있다.

$$
\rho_W(p)=p|\Psi^-\rangle\langle\Psi^-|+(1-p)\frac{I_4}{4},\quad 0\le p\le1,\qquad
\operatorname{spec}(\rho_W^{T_B})=\left\{\frac{1+p}{4}\text{ (three times)},\frac{1-3p}{4}\right\},\quad
\mathcal N=\max\left(0,\frac{3p-1}{4}\right),\quad M=2p^2.
$$

따라서 얽힘은 p=1/3을 초과할 때 시작하지만 최적 CHSH 위반은 p=1/√2를 넘어야 시작한다. p=1/2에서 negativity는1/8인데 모든 projective CHSH 선택의 크기는 √2 이하다. 얽힘과 CHSH 위반은 같은 검사가 아니다.

| Werner 매개변수 | PPT와 얽힘 | 최적 CHSH 결론 |
| --- | --- | --- |
| 1/3 | PPT·분리 가능, 부분전치 영 고유값1개 | 위반 없음 |
| 1/2 | NPT·얽힘, negativity1/8 | 위반 없음 |
| 353553/500000과353554/500000 | 둘 다 얽힘 | 앞은 문턱 아래, 뒤는 문턱 위 |

이 유리수 이웃들을 쓰면1/√2를 정확한 유리수 입력인 것처럼 다루지 않을 수 있다. 모든 projective CHSH를 위반하지 않아도 모든 측정 시나리오, filtering, 여러 복사본 프로토콜에 대해 국소 숨은 변수 설명이 존재한다고 증명한 것은 아니다.

## 12. 반올림 없이 정확한 CHSH 경계를 인증한다

수학적 답이 정확한 등호여도 고유값 구간은 문턱 양쪽을 걸칠 수 있다. 두-qubit 모델은 midpoint로 결정하지 않는다. q=TrG−1로 두자. M=TrG−μ1이므로 G의 최소 고유값과 q를 비교하는 문제를 shifted 행렬의 양성 검사로 바꾼다.

$$
q=\operatorname{Tr}G-1,\quad K=G-qI,\qquad
M>1\ \Longleftrightarrow\ K\not\succeq0,\qquad
M=1\ \Longleftrightarrow\ K\succeq0\ \text{and}\ \det K=0,\qquad
M<1\ \Longleftrightarrow\ K\succ0.
$$

허용된 입력에서 K의 모든 원소는 유리수다. 일곱 principal minor와 determinant가 반복 최소근까지 포함한 정확 인증을 준다. M의 보고 구간은 여전히 가장 큰 두 분리 근을 더해 계산한다. 정확한 분류는 그 구간의 반올림에 의존하지 않는다.

Full-rank 예제로 국소 Bloch 벡터가0이고 T=diag(3/5,−4/5,1/2)인 상태를 보자. Bell 기저 가중치는29/40,1/40,3/40,7/40이다. Negativity가9/40이므로 얽혔다. 그러나 G=diag(9/25,16/25,1/4),q=1/4,K=diag(11/100,39/100,0)이다. 따라서 M=1이고 최적 CHSH 크기는 정확히2다.

첫 상관을 ±1/25000만큼 바꾸면 밀도의 양성을 유지하면서 M을1의 양쪽으로 이동시킨다. 이는 full-rank 상태의 물리적으로 유효한 perturbation이다. Rank가 부족한 경계를 양성 검사 없이 흔들면 유효하지 않은 “예제”를 만들 수 있다. Bell 결과를 해석하기 전에 상태부터 유효해야 한다.

## 13. 해설이 있는 연습문제

**1. 같은 국소 상태를 설명한다.** 1절의 세 광원을 Alice 전용 측정으로 구별할 수 없는 이유는 무엇인가? 어떤 하나의 공동 측정은 둘을 여전히 구별하지 못하는가?

**해설.** 모든 Alice marginal이 I2/2여서 모든 국소 측정 법칙이 같다. 계산 기저 공동 측정도 Bell과 고전적 광원에서00·11이 절반씩이므로 같다. 공동 X 측정은 둘을 구별한다. XX 상관이 각각1과0이기 때문이다.

**2. 유효하지 않은 상태를 거부한다.** diag(1,0,−1,1)을 양의 trace로 정규화하자. Leading principal minor만으로 밀도임을 보장할 수 있는가?

**해설.** 그럴 수 없다. 이미 trace가1이고 leading determinant는1,0,0,0이지만 세 번째 좌표의 singleton principal minor는−1이다. 그 기저 projector의 확률이 음수가 된다. Leading만이 아니라 모든 principal minor가 필요하다.

**3. 분해와 판정 조건을 구별한다.** |Φ+⟩,|Φ−⟩를 같은 비율로 혼합한다. 얽힌 재료를 사용했으므로 결과도 얽혔다고 할 수 있는가?

**해설.** 두 coherence가 상쇄되어 (|00⟩⟨00|+|11⟩⟨11|)/2가 된다. 곱 상태 혼합을 명시적으로 썼으므로 분리 가능하다. Mutual information은1bit,negativity는0이다.0이 아닌 고전적 상관과 분리 가능성은 양립한다.

**4. 선택한 Bell 값을 유도한다.** 11절의 spinor와 Alice 축에서 네 E와 S를 계산하라.

**해설.** E00=21/29,E01=21/29,E10=20/29,E11=−20/29이므로 S=82/29다. 이 밀도의 최적 최대는2√2다. 나쁜 설정으로 바꾸면 선택 결과는 바뀌지만 얽힘과 최적 최대는 그대로다.

**5. CHSH 위반 없는 얽힘을 찾는다.** Werner p=1/2에서 최소 부분전치 고유값·negativity·가능한 최대 CHSH 크기를 구하라.

**해설.** 최소 고유값은−1/8, 나머지 세 개는3/8이며 negativity는1/8이다. M=2p²=1/2이므로 최적 크기는 √2다. 얽혔지만 모든 projective 방향에서 CHSH를 위반하지 않는다. 이 명제만으로 일반적인 국소 모형을 인증하지는 않는다.

**6. 정확한 경계에서0이 아닌 negativity를 읽는다.** 12절 Bell-diagonal 상태로 최댓값이 정확히2여도 분리 가능하지 않을 수 있는 이유를 설명하라.

**해설.** 가장 큰 Bell 가중치는29/40이고 부분전치 고유값은−9/40,19/40,17/40,13/40이다. Negativity9/40이 얽힘을 인증한다. Shifted Gram은 PSD이며 determinant가0이므로 M=1,CHSH 최대는 정확히2다. 두 인증은 서로 다른 질문에 답한다.

## 14. 연구로 이어지는 한 단계로 모델을 읽는다

이 가족은 절댓값10⁶ 이하의 소유 정수32개를 받는다.16개는 Hermitian 분자,16개는 네 비영 국소 spinor 성분이다. 정확한 PSD 검증을 만족하면 일반적인 복소 coherence를 허용한다. 분자나 국소 spinor를 허용된0 아닌 대표 인수로 확대해도 정규화된 물리 법칙은 같다.

상태·부분전치·G의 스펙트럼은1024bit 정확 다항식 분리를 사용한다. 엔트로피와 비선형 크기는768bit 작업 구간과 바깥 방향512bit 관측을 사용한다. 정확한0과 작은 고유값은 다르다. Null 기계 숫자 projection과 영 사건의 조건부 상태는 각각 명시한 의미로 읽어야 하며, 추측한 물리 값을 넣으라는 뜻이 아니다.

다음 연구 질문에는 witness 측정의 유한 표본 신뢰구간, 양성 제약 아래 robust tomography, 다른 Bell 부등식, multi-copy activation, LOCC 변환, 열린 계의 tensor-network 모사가 있다. AI에서 얽힘 witness나 Bell 설정 최적화는 제약을 가진 모델링 문제다. 학습 loss가 자동으로 유효한 물리 인증이 되지는 않는다. Positive-map 이론, PPT 얽힘, 다부분계 자원은 두-qubit 모델을 넘어서는 전체 교육과정의 심화 범위다.

예정된 진입점은 **도서 → Interactive Book**이며 원고 준비와 Worker 전달·시각 실험·등록 검증을 구분한다. 국소 혼합과 부분추적 관례는 [Preskill Chapter 2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), PPT 충분성의 차원 제한은 [Horodecki 분리 조건 논문](https://arxiv.org/abs/quant-ph/9605038), projective CHSH 최적화와 측정 가정은 [저자들의 종설 IV.A–B절](https://arxiv.org/pdf/quant-ph/0702225)과 대조했다. 유리수 shifted 인증과 위 수치 예제는 이 장의 관례에서 직접 유도했다.
