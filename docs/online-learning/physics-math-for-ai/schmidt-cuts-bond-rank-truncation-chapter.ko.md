## 1. 얽힘을 말하기 전에 cut을 지정한다

MPS bond의 차원은 표현의 용량이지만 물리 상태가 그 용량을 얼마나 사용하는지는 알려주지 않는다. 이제 선택한 물리 cut에서 상태를 조사하자. 같은 벡터도 cut에 따라 rank와 스펙트럼이 다를 수 있으므로, 얽힘에 관한 주장은 어떤 양분할인지 밝혀야 한다. 숫자 하나를 전체 사슬의 고정된 얽힘처럼 제시하지 않는다.

비영 순수 n-qubit 벡터에서 앞의 k개 비트를 왼쪽 색인 i, 나머지를 오른쪽 색인 j로 묶는다. 원시 진폭의 순서를 바꾸지 않고 C로 reshape한다. Site0의 비트가 최상위 비트이며 전체 노름 제곱 N은 어느 cut에서도 같다. 정규화는 reshape와 별개다.

$$
|\psi\rangle=\frac1{\sqrt N}\sum_{i,j}C_{ij}|i\rangle_L|j\rangle_R,\quad C\in\mathbb C^{2^k\times2^{n-k}},\quad N=\sum_{i,j}|C_{ij}|^2>0.
$$

| 네 qubit의 cut k | 왼쪽 차원 | 오른쪽 차원 | 작은 스펙트럼을 계산하는 쪽 |
| --- | --- | --- | --- |
| 1 | 2 | 8 | 왼쪽 |
| 2 | 4 | 4 | 같은 차원이므로 왼쪽 |
| 3 | 8 | 2 | 오른쪽 |

배열의 reshape는 측정·정규화·상태 변화가 아니다. 어떤 물리 레이블을 행과 열에 놓을지 바꾸는 것이다. 계산기는 모든 연속 cut을 계산하며 선택 cut은 살펴볼 결과를 정한다. 임의의 비연속 부분계는 명시적으로 다시 묶어야 하고 현재 입력에 숨겨진 별도 모드로 가정하지 않는다.

## 2. Ket 계수에서 양쪽 물리 밀도를 유도한다

다른 쪽 부분계의 기저 레이블을 맞춰 trace하자. C는 ket 진폭이므로 bra의 계수에 복소 켤레가 붙는다. 행렬 공식을 외워 옮기기 전에 성분을 전개하면 오른쪽 부분계에서 자주 생기는 방향 오류를 피할 수 있다. 양쪽 밀도의 역할은 대칭이지만 배열의 행·열 convention은 그대로 유지된다.

$$
(\rho_L)_{ii'}=\frac1N\sum_jC_{ij}\overline{C_{i'j}}=\left(\frac{CC^\dagger}{N}\right)_{ii'},\qquad (\rho_R)_{jj'}=\frac1N\sum_iC_{ij}\overline{C_{ij'}}=\left(\frac{C^{\mathsf T}\overline C}{N}\right)_{jj'}.
$$

둘 다 Hermitian·양의 준정부호이며 trace1이다. 오른쪽 물리 밀도는 Cᵀconj(C)/N이다. C†C/N은 같은 고유값을 갖지만 그 물리 밀도의 복소 켤레이므로 원소나 위상에 민감한 관측량이 같을 필요는 없다. 같은 spectrum만으로 행렬 방향까지 옳다고 판단하면 안 된다.

곱 상태의 행 C=(1,i)를 생각하자. 오른쪽 density의 오른쪽 위 원소는−i/2다. 통상 Pauli convention에서 Y=−2Imρ₀₁=+1이다. C†C를 물리 density로 복사하면 부호가 반대가 된다. 따라서 뒤의 검산에서는 고유값뿐 아니라 density의 성분과 실제 절단 진폭을 함께 확인할 것이다.

## 3. Singular value가 Schmidt 계수가 된다

정규화된 계수 행렬에 singular value decomposition을 적용하자. 왼쪽·오른쪽 singular vector는 각 좌표 공간에서 직교정규다. 행렬 곱의 원소를 전개하면 두 부분계 ket에 어떤 계수가 들어가는지 알 수 있다. 이 단계에서는 λ와σ, 원시 진폭의 scale을 혼동하지 않는 것이 중요하다.

$$
\frac C{\sqrt N}=U\Sigma V^\dagger,\qquad |\psi\rangle=\sum_{a=1}^{q}\sigma_a|u_a\rangle_L|\overline{v_a}\rangle_R,\qquad\lambda_a=\sigma_a^2,\quad\sum_a\lambda_a=1.
$$

고정한 계수 convention에서 오른쪽 ket은 V의 열벡터를 복소 켤레한 것이다. λ는 정규화 density의 가중치이며 σ는 그 음이 아닌 제곱근이다. N=1이 아니라면 원시 C의 singular value와 같지 않다. 모든 원시 진폭에 공통 인자를 곱해도 정규화된 λ는 바뀌지 않는다.

양쪽의 비영 spectrum이 같은 이유는 각 Schmidt 성분이 두 partial trace에 같은 λ를 주기 때문이다. 더 큰 쪽에는 영 고유값이 추가될 수 있다. 서로 다른 가중치 group으로 spectrum을 표시하면 multiplicity를 보존해야 한다. 가중치1/2의 group 하나가 multiplicity2라면 전체 확률은1이지1/2가 아니다.

## 4. Rank는 bond를 제약하고 물리적 양분할에 의존한다

Schmidt rank는 multiplicity를 포함해 양의 가중치가 몇 개인지 센다. 비영 순수 상태에서는 계수 행렬의 rank와 같다. 앞 장에서 배웠듯 가상 bond를 통과하는 factorization은 그 rank의 상한을 준다. 이 관계를 이용하면 어떤 표현이 정확하게 가능한지를 먼저 판단할 수 있다.

$$
q=\operatorname{rank}C\le\min(2^k,2^{n-k}),\qquad C=LB\implies q\le D_k.
$$

GHZ₃는 두 연속 cut 모두 rank2다. 곱 상태는 더 큰 가상 차원에 저장해도 rank1이다. 반대로 어떤 cut에서 rank4인 상태를 그 bond가2인 MPS로 정확하게 표현할 수는 없다. Bond를 줄이려면 양의 성분을 버리고 그 오차를 명시적으로 받아들여야 한다. 용량의 축소를 단순 좌표 변경인 gauge와 구별하자.

| 네 qubit 순수 상태 | Cut1 rank | Cut2 rank | Cut3 rank |
| --- | --- | --- | --- |
| Bell pair01과 Bell pair23의 곱 | 2 | 1 | 2 |
| Bell pair02와 Bell pair13의 곱 | 2 | 4 | 2 |

가운데 cut에서 첫 상태의 각 pair는 한쪽 내부에 있지만 둘째 상태의 두 pair는 모두 양쪽을 가로지른다. 이 예제에서는 crossing pair 하나마다 rank가 두 배다. Site 순서를 바꾸면 연속 cut이 뜻하는 물리 부분계도 바뀐다. 고정한 양분할 내부의 국소 기저 변경에 대한 불변성과 모순되지 않는다. 무엇을 바꾼 실험인지 먼저 이름 붙이자.

## 5. Purity와 entropy는 가중치의 분포를 설명한다

Rank는 support의 개수이지 그 위에 확률이 어떻게 퍼져 있는지는 아니다. Purity와 entropy는 spectrum의 모양을 더 설명한다. 이번 장의 입력은 전체가 순수하므로 reduced-state entropy를 양분할 얽힘 entropy로 해석한다. 전체가 혼합인 경우까지 같은 해석을 자동 확장하지 않는다.

$$
\mathcal P=\operatorname{Tr}\rho_L^2=\sum_a\lambda_a^2,\qquad S=-\sum_a\lambda_a\ln\lambda_a,\qquad S_{\mathrm{bits}}=\frac S{\ln2},\qquad0\le S\le\ln q.
$$

곱 상태의 spectrum은(1,0,…)이고 entropy0이다. Bell pair는 두 가중치1/2이므로 purity1/2, entropy ln2이며 bits로는1이다. 0ln0=0은 영으로 가는 연속 극한이다. 가중치 하나의 multiplicity가 m이면 entropy 기여에도 m을 곱한다. 그림에서 막대 하나로 group을 표시하더라도 이 개수를 빠뜨려서는 안 된다.

상한은 q개 support 위의 정규화된 분포에서 균일 분포가 최대 entropy를 주기 때문이다. 작은 entropy가 rank1을 증명하지는 않는다. 아주 작은 둘째 가중치도 엄밀하게 양수이면 rank2다. 전체 상태가 혼합이면 reduced entropy에는 고전적인 mixing도 들어갈 수 있으므로 순수 상태의 얽힘 척도로 그대로 사용할 수 없다.

## 6. 한 cut에서 큰 가중치를 유지하는 것이 최적인 이유

정규화된 계수 행렬을 rank≤r인 행렬 B로 근사하고 Frobenius 거리를 최소화하자. 곱 기저가 직교정규이므로 이 거리는 대응하는 상태의 Hilbert 거리와 같다. 이 문제에서 근사 B의 노름은1일 필요가 없다. 정규화 조건이 있는 문제와 먼저 구별하자.

B의 열공간에 대한 직교 projector를 P라 하자. 그 공간 밖의 오차는 B로 상쇄할 수 없으므로 하한을 준다. ρL의 고유기저에서 P의 대각 원소는0…1이고 합은 최대 r이다. 따라서 가중합은 가장 큰 r개 λ에 용량을 배분할 때 최대다. 이 관찰이 단순한 큰 수 우선 규칙에 수학적 근거를 준다.

$$
\|C/\sqrt N-B\|_F^2\ge\|(I-P)C/\sqrt N\|_F^2=1-\operatorname{Tr}(P\rho_L)\ge1-\sum_{a=1}^{r}\lambda_a,\qquad\operatorname{rank}B\le r.
$$

P의 range 안과 밖의 두 오차 성분이 직교하므로 첫 부등식이 성립한다. 제한된 대각 질량을 가장 큰 λ에 배분하면 둘째 부등식을 얻는다. 해당 Schmidt 성분을 유지하면 두 하한에 도달한다. 따라서 이것은 하한만 제시한 추측이 아니라 실제 달성 가능한 최적 오차다.

여기서 최적화하는 것은 지정한 한 cut의 rank 제약과 Hilbert/Frobenius 노름이다. 선택한 에너지나 모든 관측량의 오차, 여러 bond 제약을 동시에 만족하는 MPS의 전역 목적함수를 최소화했다고 하지 않는다. 목적함수와 제약을 바꾸면 다른 최적화 문제가 된다. 뒤에서 연구 알고리즘으로 연결할 때도 이 구별을 유지하자.

## 7. 서로 다른 세 상태를 분리한다

정규화된 원래 상태, 남긴 Schmidt 합, 그 합을 다시 정규화한 상태를 각각 적자. 양의 λ를 내림차순으로 정렬한다. Budget이 support보다 커도 양의 성분만 모두 유지하며 임의의 영 가중치 벡터를 만들어 추가하지 않는다. 같은 이름의 상태처럼 화면에 교체해서 보여주면 서로 다른 노름을 놓치기 쉽다.

$$
|\widetilde\psi_r\rangle=\sum_{a=1}^{\min(r,q)}\sqrt{\lambda_a}|u_a\rangle|\overline{v_a}\rangle,\quad S_r=\sum_{a=1}^{\min(r,q)}\lambda_a,\quad |\widehat\psi_r\rangle=\frac{|\widetilde\psi_r\rangle}{\sqrt{S_r}}\quad(S_r>0).
$$

Tilde 상태는 원래 상태의 직교한 유지 성분이며 노름 제곱은 Sᵣ다. 버린 성분과 직교하므로 두 노름 제곱의 합은1이다. 재정규화는 살아남은 진폭에1/√Sᵣ를 곱한다. 없앤 성분을 복원하는 작업이 아니다. Retained probability가 정규화 후 사라졌다고 해석하지 않는다.

수치 출력에서도 원시 C_ret, 정규화 입력에서의 C_ret/√N, 재정규화 C_ret/√(NSᵣ)는 서로 다르다. 원시 행렬을 정규화된 원래 상태와 직접 비교하면 scale에 의존하는 다른 오차가 된다. 진폭을 표시할 때 분모를 함께 적으면 어떤 상태인지 명확해진다. 비교표의 각 열도 같은 정규화 convention을 써야 한다.

## 8. 오차·fidelity·정규화 거리의 식은 다르다

직교성으로 비정규화 오차를 얻는다. 선택한 위상에서 원래 상태와 재정규화 상태의 overlap은 양의√Sᵣ다. 두 단위 벡터의 거리 제곱을 전개하면 정규화 거리를 얻는다. 같은 retained weight에서 출발해도 서로 다른 물리적 질문의 답이라는 점을 확인하자.

$$
\epsilon_r=\|\psi-\widetilde\psi_r\|^2=1-S_r,\qquad|\langle\psi|\widehat\psi_r\rangle|^2=S_r,\qquad\|\psi-\widehat\psi_r\|^2=2(1-\sqrt{S_r}).
$$

| 정규화 원래 상태와 비교하는 양 | 공식 | 조건 |
| --- | --- | --- |
| 유지 성분의 노름 제곱 | Sᵣ | 모든 budget |
| 최적 비정규화 오차 제곱 | 1−Sᵣ | 모든 budget |
| 정규화 fidelity 제곱 | Sᵣ | Sᵣ>0 |
| 정규화 거리 제곱 | 2(1−√Sᵣ) | Sᵣ>0, 위상 정렬 |

Fidelity 제곱이라고 표시한 값은 overlap의 절댓값 제곱이다. Hilbert 거리는 전체 위상에 의존하므로 여기의 거리는 overlap이 양수가 되는 자연스러운 위상, 즉 전체 위상에 대해 최소화한 거리다. Fidelity를 제곱근과 구별하지 않거나 위상을 생략하면 맞는 숫자도 틀린 기하 해석으로 연결될 수 있다.

재정규화한 유지 상태는 정규화된 rank≤r 상태들 가운데 overlap도 최대화한다. 고정한 열공간에서 Cauchy–Schwarz로 overlap을 원래 상태의 projection 노름으로 제한할 수 있다. 가장 큰 projection 노름은√Sᵣ다. 이렇게 두 최적화 문제를 연결하면서도 그 오차 값이 서로 다르다는 사실을 보존한다.

## 9. 가중치가 다른 예제를 solver 없이 계산한다

두 qubit의 원시 벡터에서00의 진폭이2,11의 진폭이1이라고 하자. 원시 노름 제곱은5다. 정규화된 reduced density가 대각이므로 반복 알고리즘으로 근삿값을 찾기 전에 가중치를 바로 읽을 수 있다. 손계산 가능한 기준점을 확보하면 더 복잡한 화면도 검산할 수 있다.

$$
|\psi\rangle=\frac{2|00\rangle+|11\rangle}{\sqrt5},\quad(\lambda_1,\lambda_2)=(4/5,1/5),\quad\mathcal P=17/25,\quad\epsilon_1=1/5,\quad d_1^2=2-4/\sqrt5.
$$

Budget1이면 원시 유지 행렬에는00의2만 남는다. 정규화 입력에서의 성분은(2/√5)|00〉로 노름 제곱4/5이고, 재정규화하면 |00〉다. 비정규화 오차 제곱은1/5지만 정규화 거리 제곱은2−4/√5로 약0.2111이다. 두 식을 같다고 처리하면 유지 성분의 크기를 다시 조정한 효과를 잃는다.

Purity17/25와 entropy−(4/5)ln(4/5)−(1/5)ln(1/5)는 원래 spectrum의 양이다. 정규화한 유지 곱 상태는 purity1·entropy0이므로 같지 않다. 절단은 진폭뿐 아니라 얽힘도 바꾼다. 화면의 entropy가 원래 상태인지 근사 상태인지 명시해야 원래 입력의 성질과 절단의 결과를 구별할 수 있다.

## 10. 축퇴는 최적 상태의 비유일성을 만든다

Bell 상태는 같은 양의 가중치 두 개를 가진다. Budget1에서 두 차원 Schmidt 고유공간의 어느 방향을 유지할지 구별해주는 조건이 없다. 눈에 보기에 다른 두 벡터가 같은 최소 오차를 달성한다. 최적값이 하나라는 사실과 최적해가 하나라는 사실은 구별해야 한다.

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},\qquad |\widetilde\psi_1^{(0)}\rangle=\frac{|00\rangle}{\sqrt2},\qquad |\widetilde\psi_1^{(1)}\rangle=\frac{|11\rangle}{\sqrt2},\qquad\epsilon_1=1/2.
$$

재정규화한 두 선택은 |00〉와|11〉이다. 둘 다 원래 상태와 fidelity 제곱1/2, 위상 정렬 거리 제곱2−√2를 가진다. 최적값이 같다고 벡터까지 유일하지는 않다. 임의의 eigenvector를 고르면 수학이 정하지 않은 선택을 구현이 숨기게 된다. 독자는 그것을 추가 가정이나 선택 규칙으로 알아야 한다.

양쪽 Schmidt 기저를 함께 회전해도 같은 모호성이 있다. 계산기는 해당 가중치의 전체 고유공간 projector를 반환하지만 budget이 그 양의 group을 가르면 선택 상태를 반환하지 않는다. 최적 retained weight와 error는 계속 보고한다. Budget을 몰래 늘려 group 전체를 유지하거나 편리한 기저를 고르는 것은 요청한 rank 제약의 답이 아니다.

## 11. 유지 블록 내부의 축퇴는 문제가 아니다

네 qubit 가운데 cut에서 대각 성분이3,2,2,1인 C를 사용하자. 진폭 제곱에는 최대 가중치 하나, 같은 중간 group 두 개, 최소 하나가 있다. 이 예제로 spectrum에 축퇴가 있다는 사실과 절단 경계가 축퇴를 가른다는 사실을 분리할 수 있다.

$$
C=\operatorname{diag}(3,2,2,1),\quad N=18,\quad\lambda=(1/2,2/9,2/9,1/18),\quad\mathcal P=19/54.
$$

| Rank budget | 양의 유지 rank | 유지 Sᵣ | 버린 오차 | 비정규화 최적해 유일 | 정규화 최적해 존재 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | 예 | 아니오 |
| 1 | 1 | 1/2 | 1/2 | 예 | 예 |
| 2 | 2 | 13/18 | 5/18 | 아니오 | 예 |
| 3 | 3 | 17/18 | 1/18 | 예 | 예 |
| 4 | 4 | 1 | 0 | 예 | 예 |

Budget2는 두 차원2/9 고유공간에서 방향 하나를 골라야 하므로 상태가 비유일하다. Budget3은 group 전체를 유지하여 합친 projector가 유일하다. 그 내부 기저가 유일할 필요는 없다. Budget1에서 반복 group을 전부 버리는 것도 모호하지 않다. 따라서 축퇴 여부 하나로 모든 절단의 선택 가능성을 판단하면 안 된다.

Multiplicity는 실제 spectrum으로 판단해야 하며 가까운 값을 화면 tolerance로 합치지 않는다. 작아도 양의 gap이면 수학적으로 leading subspace를 구별한다. 수치 conditioning과 수학적 유일성은 다른 질문이다. 시각화에서 가까운 막대가 겹쳐 보이는 현상도 정확한 축퇴의 증거로 사용하지 않는다.

## 12. 영 budget과 support 이상의 budget도 정의된 경우다

허용 rank는 상한이다. Budget0에서 가능한 비정규화 벡터는 영벡터 하나이며 정규화 입력에 대한 오차는1이다. 정규화된 영벡터는 존재하지 않는다. 따라서 그 정규화 fidelity나 거리를 요청하면 정의된 값이 없다는 것이 답이다. 수치0을 반환하면 전혀 다른 의미가 된다.

$$
r=0:\ S_0=0,\quad\widetilde\psi_0=0,\quad\epsilon_0=1;\qquad r\ge q:\ S_r=1,\quad\epsilon_r=0,\quad\widehat\psi_r=\psi.
$$

계산기는 이런 정규화 출력을0 대신 null로 표현한다. Rank1 곱 상태에서 budget4를 지정해도 양의 support 하나만 유지한다. 버린 가중치는0이고 원래 상태가 복원된다. 물리적으로 점유한 Schmidt 모드가 넷인 것처럼 nullspace의 임의 벡터를 붙이지 않는다. 용량을 모두 채우는 것이 목적이 아니다.

전체 양의 support projector의 rank는 density의 주변 차원보다 작을 수 있다. 그래도 주어진 계수 행렬의 support 위에서는 identity로 작용한다. 더 큰 물리 쪽의 영 고유값은 구조적 padding이지 추가 양의 확률이나 유지 물리 벡터의 새로운 모호성이 아니다. 영 weight를 가진 방향을 어떤 기저로 쓰든 현재 상태에는 기여하지 않는다.

## 13. 오른쪽 물리 projector에는 전치가 필요하다

오른쪽 부분계에서 |i〉|j〉의 새 계수는 이전 오른쪽 색인 l에 대해 C_il P_jl을 합한 것이다. 따라서 행렬 곱은 CPᵀ다. 왼쪽에 작용할 때는 보통의 왼쪽 행렬 곱이다. 성분을 한 번 쓰면 어느 쪽에 transpose가 필요한지 암기하지 않고 확인할 수 있다.

$$
(I\otimes P)|\psi\rangle\quad\longleftrightarrow\quad C_{\mathrm{ret}}=CP^{\mathsf T},\qquad (P\otimes I)|\psi\rangle\quad\longleftrightarrow\quad C_{\mathrm{ret}}=PC.
$$

Hermitian 복소 projector는 P†=P지만 Pᵀ는 P와 다를 수 있다. 그러므로 전치를 adjoint로 바꾸는 것은 일반적으로 틀리다. 세 qubit에서 앞의 두 site 뒤를 자르면 오른쪽이 작아 spectrum을 계산한다. 다음 예제는 실제 진폭으로 이 차이를 드러낸다.

$$
C=\begin{pmatrix}2&2i\\1&-i\\0&0\\0&0\end{pmatrix},\quad\rho_R=\begin{pmatrix}1/2&-3i/10\\3i/10&1/2\end{pmatrix},\quad P_+=\frac12\begin{pmatrix}1&-i\\i&1\end{pmatrix},\quad CP_+^{\mathsf T}=\begin{pmatrix}2&2i\\0&0\\0&0\\0&0\end{pmatrix}.
$$

원시 노름은10이고 양의 가중치는4/5와1/5다. Leading 오른쪽 ket은(1,i)/√2다. 올바른 CP₊ᵀ는 행(2,2i)을 유지하고 행(1,−i)을 없애므로 버린 가중치1/5다. CP₊를 곱하면 오히려 작은 성분을 유지하여 버린 가중치4/5가 된다. 고유값만 일치하는지 검사하면 이 뒤집힘을 놓친다.

이 예제는 ket convention·물리 density 성분·projector·실제 진폭을 연결한다. 오른쪽 trace·purity·entropy만 맞추는 것보다 강한 검증이다. 위상을 나타내는 화살표도 올바른 ket 방향을 따라야 한다. 맞는 가중치 숫자를 복소 켤레 방향 옆에 붙였다고 물리 그림이 옳아지는 것은 아니다.

## 14. 작은 가중치·관측 오차·여러 cut의 문제

거의 rank가 부족한 행렬도 정확한 rank는 가득 찰 수 있다. 다음 정수 행렬에서 M=10⁶으로 두면 determinant는 여전히−1이다. 가장 작은 정규화 가중치는 약6.25×10⁻²⁶이지만 엄밀하게 양수다. 표시 자리수에 나타나지 않는다고 정확한 영으로 바꾸지 않는다.

$$
C_M=\begin{pmatrix}M&M-1\\M-1&M-2\end{pmatrix},\quad\det C_M=-1,\quad N=4M^2-8M+6,\quad\lambda_{\min}=\frac{1-\sqrt{1-4/N^2}}2>0.
$$

마지막 식에서 거의 같은 부동소수점 수를 빼면 가중치를 잃을 수 있다. 유한 모형은 화면의 Number 값을 정의로 삼지 않고 정확 rank와 인증 root 구간을 사용한다. 작은 weight는 통제한 근사의 이유가 될 수 있지만 정확한 상태를 곱 상태로 만들지는 않는다. 어떤 오차를 받아들였는지 기록해야 한다.

유지 노름이 양수인 정규화 최적 상태에서는 bounded Hermitian 관측량의 기대값 오차를 순수 상태 trace distance로 제한할 수 있다. 두 상태에 같은 연산자를 적용하여 비교한다는 가정도 필요하다. 입력 상태뿐 아니라 관측 연산자까지 바꾸면 이 비교가 아니다.

$$
|\langle A\rangle_\psi-\langle A\rangle_{\widehat\psi_r}|\le2\|A\|\sqrt{\epsilon_r}\qquad(A=A^\dagger\text{ bounded},\ S_r>0).
$$

Unbounded 연산자에는 추가 모멘트 제어 없이 이 bound를 그대로 적용하지 않는다. 각 cut의 독립 최적 절단이 동시에 최적인 MPS를 만든다는 결론도 아니다. 순차 절단은 다음 cut이 보는 상태를 바꾼다. 전역 오차·conditioning·variational refinement는 실제 절차를 대상으로 따로 분석해야 한다. 한 cut의 최적성 증명을 다른 알고리즘의 성능 보장으로 확대하지 않는다.

## 15. 여덟 문제와 해설

**1. 행렬의 크기.** 네 qubit의 cut3에서 C의 차원과 작은 물리 density를 정하라. Rank2이면 큰 쪽의 추가 영 고유값은 몇 개인가?

**해설.** C는8×2, 오른쪽 density는2×2, 왼쪽은8×8이다. Rank2이므로 오른쪽 영은 없고 왼쪽 영은6개다. 양의 spectrum이 같다는 사실이 전체 물리 행렬의 크기도 같다는 뜻은 아니다.

**2. 서로 다른 가중치.** 2|00〉+|11〉에서 rank·가중치·purity와 budget1 유지 상태를 재정규화 전후로 구하라.

**해설.** N=5,rank2,가중치4/5·1/5,purity17/25다. 정규화 입력에서 유지한 성분은(2/√5)|00〉이며 재정규화하면 |00〉다. 노름이 다르므로 오차 제곱도1/5와2−4/√5로 다르다. 각 상태의 분모를 표시하면 구별하기 쉽다.

**3. 위상 trace.** C=(1,i)의 오른쪽 Y가−1이 아니라+1인 이유를 설명하라.

**해설.** Ket 계수에서 ρR=Cᵀconj(C)/2이며 오른쪽 위가−i/2다. Pauli convention으로 Y=−2Imρ₀₁=+1이다. C†C는 그 성분을 켤레한 값이고 고유값은 같으므로 eigenvalue만 검사해서는 오류를 찾지 못한다.

**4. 반환하지 않는 상태.** Bell 입력의 budget1 오차는1/2다. 계산기가 이 오차를 표시하면서 선택 최적 상태는 반환하지 않을 수 있는 이유는 무엇인가?

**해설.** |00〉/√2와|11〉/√2 등 여러 선택이 같은 비정규화 오차를 최소화한다. 같은 양의 가중치가 유지 방향을 정하지 않는다. 올바른 최적값이 유일한 벡터를 제공하지 않으며 전체 group projector만으로 rank1 부분 projector가 결정되지 않는다.

**5. Group 전체 유지.** 대각3,2,2,1 예제에서 budget2와3을 비교하라.

**해설.** Budget2는13/18을 유지하고5/18을 버리지만2/9의 두 차원 고유공간을 갈라서 상태가 비유일하다. Budget3은 전체 group을 유지하여17/18을 남기고1/18을 버린다. 전체 고유공간 projector는 그 안의 기저 선택에 의존하지 않는다.

**6. 영 budget.** Budget0의 비정규화 최적해·오차·정규화 fidelity는 무엇인가? Rank1 곱 입력에서 budget4이면 어떻게 바뀌는가?

**해설.** Budget0의 최적해는 영벡터이고 오차1이며 정규화 fidelity는0이 아니라 정의되지 않는다. Budget4의 곱 상태는 양의 모드 하나를 유지하여 오차0·fidelity 제곱1이다. Rank 상한을 반드시 끝까지 사용할 필요는 없다.

**7. 오른쪽 projector.** 복소 세 qubit 예제에서 budget1의 CP₊ᵀ와 CP₊를 비교하라.

**해설.** 전치는 원시 행(2,2i)을 유지하여 유지 노름8/10을 준다. 전치를 생략하면 행(1,−i)이 남아2/10이다. 오차 제곱은 각각1/5와4/5다. Hermitian projector가 반드시 실수 대칭 행렬은 아니라는 사실이 중요하다.

**8. MPS 용량.** 인접 Bell pair01/23과 crossing pair02/13의 middle-cut rank를 비교하라. 그 cut의 최적 rank2 근사가 전체 MPS 압축도 해결하는가?

**해설.** 첫 상태는 가운데에서 곱으로 분리되어 rank1이다. 둘째는 두 Bell pair가 가로질러 rank4이므로 rank2에서 양의 weight를 버려야 한다. 한 cut의 최적 근사는 다른 모든 bond 제약을 동시에 강제하거나 최적화하지 않는다.

## 16. 다음 알고리즘을 고르기 전에 인증 범위를 읽는다

모형은 순서 있는2…4 qubit·비영 Gaussian-integer 원시 벡터와 rank 상한0…4를 입력받는다. 모든 연속 cut에서 물리 density·정확 rank/purity·multiplicity를 가진 가중치 group·entropy·최적 오차·정규화 비교를 반환한다. 양의 축퇴를 가르지 않으면 유지 projector와 실제 절단 진폭도 제공한다. 영 budget에서는 정규화 진폭을 만들지 않는다.

밀도 대수와 rank는 정확하다. Spectrum과 로그의 인증 구간이 표시 값을 둘러싸며 projector 항등식·재구성·실제 절단 오차를 따로 확인한다. 이 계산은 자동 MPS 압축·DMRG·최적화 sweep·열역학적 극한을 구현하지 않는다. 작은 예제는 선언한 준비의 증거이지 모든 큰 계의 보증이 아니다.

배경은 [Schollwöck의 SVD/Schmidt 설명](https://arxiv.org/html/1008.3477v2) §4.1.1을 참고하자. 위에서 계수 convention·projection 증명·예제를 직접 전개해 가정을 명시했다. 다음 구현은 다체계 다섯 단원을 제어·예측 질문·해설 quiz·접근 가능한 그림에 연결하고 실제 reader를 검증하는 단계다. Spectrum 그림은 용량·양의 support·유지 가중치·버린 오차·선택할 수 없는 축퇴 상태를 구별해서 보여주어야 한다.
