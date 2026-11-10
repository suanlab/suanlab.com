## 1. 무작위 준비와 진폭의 중첩은 서로 다른 동작이다

광원이 공정한 동전을 던져 e₀ 또는 e₁을 준비한다고 하자. 동전 기록을 버리고 양자계만 남기면 Z 측정에서 두 결과가 같은 빈도로 나온다. 순수 중첩(e₀+e₁)/√2도 같은 Z histogram을 만든다. 그렇다고 두 준비가 같은 것은 아니다. X를 측정하면 무작위 광원은 두 결과가 반반이고, 중첩 상태는 양의 결과가 확실하다.

한 번의 실행마다 하나의 준비를 고르는 것과, 측정 전에 진폭을 더하는 것을 구별해야 한다. 후자에서는 상대 위상이 중요하다. 벡터 평균은 무작위 준비의 규칙이 아니다. 극단적인 검산으로 ψ 또는 −ψ를 반반 고르면 매번 같은 순수 상태이지만 벡터의 평균은0이다. 그 평균을 다시 정규화해도 잘못된 연산을 고칠 수 없다.

**준비 표지 → 가중 순수 상태 사영 → 밀도 연산자 → 측정 통계**라는 흐름으로 생각하자. 준비 표지는 밀도 연산자보다 많은 정보를 가질 수 있다. 밀도 연산자는 측정할 수 있는 양자계의 상태를 기술하며, 확률 전이 행렬이나 매 실행에서 어떤 표지를 골랐는지 적은 목록이 아니다.

## 2. 측정 확률에서 밀도 연산자를 유도한다

정규화된 ψᵢ를 고전 확률 pᵢ로 선택하자. 정규화한 방향 v로 사영하는 검사의 확률은 Σᵢpᵢ|v†ψᵢ|²다. 내적을 trace 안으로 옮기면 모든 방향 검사에 공통으로 쓸 수 있는 연산자가 나온다.

$$
\rho=\sum_i p_i|\psi_i\rangle\langle\psi_i|
=\sum_{i:w_i>0}\frac{w_i}{\sum_jw_j}\frac{z_i z_i^\dagger}{z_i^\dagger z_i},\qquad
p(v)=v^\dagger\rho v=\operatorname{Tr}(\rho vv^\dagger).
$$

원시 벡터 zᵢ의 길이가1일 필요는 없다. 점유한 각 벡터의 고유 norm 제곱으로 따로 정규화한다. 하나의 raw 벡터에 영이 아닌 복소수를 곱해도 사영은 바뀌지 않는다. weight0은 기여하지 않지만, 영벡터 자체에는 정규화된 상태가 없으므로 양의 weight를 붙여 사용할 수 없다.

Hermitian 관측량 A의 모멘트도 같은 선형성으로 계산한다.2차 모멘트에서는 trace를 취하기 전에 A²를 만들어야 한다. 평균을 제곱하면 다른 양이 된다.

$$
\langle A\rangle_\rho=\operatorname{Tr}(\rho A),\qquad
\langle A^2\rangle_\rho=\operatorname{Tr}(\rho A^2),\qquad
\operatorname{Var}_\rho(A)=\operatorname{Tr}(\rho A^2)-[\operatorname{Tr}(\rho A)]^2.
$$

이 식들이 사영을 평균하는 이유다. 모든 결과의 확률과 모멘트를 일관되게 평균한다. 특정한 고전 준비 기록을 보존하는 연산은 아니다.

## 3. Hermiticity·양성·trace1이 모두 필요하다

유한 차원 밀도 연산자는 세 조건을 만족한다. 양성은 표시한 행렬 성분이 모두 비음이라는 뜻이 아니라 모든 벡터에 대한 이차형식이 비음이라는 뜻이다.

$$
\rho=\rho^\dagger,\qquad x^\dagger\rho x\ge0\quad\text{for every }x,\qquad
\operatorname{Tr}\rho=1
\quad\Longleftrightarrow\quad
\rho=\sum_{k=1}^d\lambda_k|k\rangle\langle k|,\quad
\lambda_k\ge0,\quad\sum_k\lambda_k=1.
$$

ensemble 구성에서 x†ρx는 pᵢ|ψᵢ†x|²의 합이므로 비음이고, 각 순수 사영의 trace는1이다. 반대로 유한 차원 스펙트럼 정리를 적용하면 정규직교 고유벡터를 얻는다. 고유값이 비음이고 합이1이면 이 고유벡터들의 ensemble로 표현할 수 있다. 그 표현이 존재한다고 실제 장치가 원래 고유벡터들을 준비했다는 뜻은 아니다.

대각 성분이 양수이고 trace가1이라는 조건만으로는 부족하다. 두 행이(1/2,3/4), (3/4,1/2)인 Hermitian 행렬은 Z histogram만 보면 타당해 보이지만 고유값이5/4와−1/4다. 음의 X 검사 확률이−1/4가 되어 양성이 실패한다. 이는 이론적인 invalid-state 반례이며 convex ensemble 계산기에 허용되는 준비가 아니다.

## 4. 순도·rank와 상태 집합의 기하

고유값은 확률 분포를 이룬다. Cauchy–Schwarz로 제곱합의 하한을 얻고, λ²≤λ로 상한을 얻는다.

$$
\frac1d\le\operatorname{Tr}(\rho^2)=\sum_k\lambda_k^2\le1,\qquad
L(\rho)=1-\operatorname{Tr}(\rho^2),\qquad
\rho\text{ is pure}\Longleftrightarrow\operatorname{rank}\rho=1\Longleftrightarrow\rho^2=\rho.
$$

하한의 등호는 ρ=I/d일 때, 상한의 등호는 고유값 하나가1이고 나머지가0일 때 성립한다. L은 정규화하지 않은 linear entropy로 최댓값이1−1/d다. von Neumann entropy와 같은 양이 아니다. 차원1의 유일한 상태는 순수이면서 최대 혼합이다. 이 경계 예제로 식의 의미를 확인할 수 있다.

밀도 연산자들은 convex 집합을 이룬다. rank1 상태는 극점이다. 그 support에 수직인 모든 벡터의 기대값이0이므로, 그 상태를 convex하게 분해하는 양의 상태들도 같은 support를 가져야 하기 때문이다. 반면 혼합 상태는 양의 고유값 둘 사이에서 충분히 작은 양을 옮겨 서로 다른 두 상태를 만들 수 있다. 두 상태의 평균이 원래 ρ이므로 극점이 아니다.

Hermitian positive semidefinite 행렬의 rank는 양수인 principal minor 중 가장 큰 크기와 같다. 모든 principal minor는 비음이다. semidefinite의 경우 leading minor만 검사하면 부족하다. diag(0,−1,2)는 leading determinant가 모두0이지만 음의 singleton minor를 가진다. 정확한 이 기준과 부동소수점 고유값이 작아 보인다는 판단을 구별하자.

## 5. 서로 다른 ensemble이 같은 상태를 나타낼 수 있다

두 Z 고유상태의 같은 혼합, 두 X 고유상태의 같은 혼합, 두 Y 고유상태의 같은 혼합은 모두 I/2다. 구성 상태는 다르지만 남겨진 qubit만 측정하면 모든 통계가 같다. 표는 버린 준비 표지의 측정이 아니라 양자계만의 법칙이다.

| 같은 weight의 준비 쌍 | 결과 density | p(Z 양의 결과) | p(X 양의 결과) |
| --- | --- | --- | --- |
| e₀와 e₁ | I/2 | 1/2 | 1/2 |
| (e₀+e₁)/√2와(e₀−e₁)/√2 | I/2 | 1/2 | 1/2 |
| (e₀+ie₁)/√2와(e₀−ie₁)/√2 | I/2 | 1/2 | 1/2 |

어떤 광원 설정을 골랐는지 외부 기록이 알려준다면 결합된 설명에는 추가 정보가 있다. 그 기록을 버린다는 것이 이 준비 시나리오의 일부다. 밀도 연산자는 표지가 처음부터 없었다고 주장하지 않는다. 표지가 없을 때 어떤 예측을 할 수 있는지 말한다.

이것이 무작위 준비와 coherent 중첩을 구별하는 이유이기도 하다. 순수 양의 X 상태는 I/2와 Z 법칙이 같지만 X 법칙과 순도는 다르다. 비대각 coherence는 기저에 상대적인 양인 반면 순수/혼합의 구별은 unitary 기저 변환에 불변이다.

## 6. 가중 열 행렬이 density와 Gram 기하를 연결한다

B의 i번째 열에 √pᵢψᵢ를 놓자. 상태 공간에서 곱하면 density, 표지 공간에서 곱하면 Gram 행렬을 얻는다. 두 공간의 차원은 다를 수 있다.

$$
B_i=\sqrt{p_i}\,\psi_i,\qquad \rho=BB^\dagger,\qquad G=B^\dagger B,\qquad
\rho v=\lambda v,\ \lambda>0\ \Longrightarrow\ G(B^\dagger v)=\lambda(B^\dagger v).
$$

옮겨진 벡터 B†v의 norm 제곱은 λ||v||²이므로 영이 아니다. 반대 방향의 구성도 적용하면 중복도를 포함한 영이 아닌 spectrum의 일치를 증명한다. 따라서 density와 Gram의 rank·순도는 같고, 큰 공간에는 영 고유값이 더 있다. 표지 슬롯4개·d≤4인 실험에서는 χG(λ)=λ^(4−d)χρ(λ)다.

혼합을 제곱하고 trace를 전개하면 순도의 다른 표현도 얻는다.

$$
\operatorname{Tr}(\rho^2)=\sum_{i,j}p_i p_j|\langle\psi_i|\psi_j\rangle|^2.
$$

서로 직교하는 구성 상태에서는 Σpᵢ²만 남는다. 같은 ray를 반복했다면 표지가 달라도 순도는1이다. raw overlap Sᵢⱼ=zᵢ†zⱼ와 tᵢ=pᵢ/Nᵢ를 쓰면 Gram principal determinant는 det(SJ)∏ᵢ∈Jtᵢ다. 그러므로 그림의 열 좌표에 제곱근이 들어가도 특성다항식 계수는 정확히 인증할 수 있다. 일반 행렬을 모두 실제로 수치 대각화했다고 주장하는 것은 아니다.

## 7. 직교하지 않는 다른 weight의 혼합을 손으로 계산하자

e₀를 확률2/5로, 양의 X 상태를 확률3/5로 선택하자. 진폭 대신 두 사영을 더한다.

$$
\rho=\frac25\begin{pmatrix}1&0\\0&0\end{pmatrix}
+\frac35\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}
=\begin{pmatrix}7/10&3/10\\3/10&3/10\end{pmatrix},\quad
\det\rho=\frac3{25},\quad
\operatorname{Tr}(\rho^2)=\frac{19}{25},\quad
\lambda_\pm=\frac{5\pm\sqrt{13}}{10}.
$$

Z 확률은7/10과3/10이다. 무차원 Pauli Z 관측량의 평균은2/5,2차 모멘트는1, 분산은21/25다. 순도0.76에는 대각 성분 제곱뿐 아니라 두 비대각 성분의 크기 제곱도 포함된다.

대각만 남기면 순도는29/50=0.58이다. 차이9/50=0.18은 제거한 비대각 부분의 Hilbert–Schmidt norm 제곱이다. 대각 제곱만 더해서 순도를 계산하면 이 기여를 놓친다. 한편 음의 측정 결과는 양의 density와 양립한다. 양성은 확률과 이차형식의 조건이며 관측량의 모든 고유값이 양수라는 뜻이 아니다.

## 8. 전체 분산은 ensemble에 의존하는 두 기여로 나뉜다

구성 상태 i의 평균과 분산을 μᵢ·Varᵢ(A), 전체 평균을 μ=Σpᵢμᵢ라 쓰자. 각2차 모멘트를 분산과 평균 제곱으로 전개한 뒤 μ²를 더하고 빼면 정확한 계정식을 얻는다.

$$
\operatorname{Var}_\rho(A)=\sum_i p_i\operatorname{Var}_i(A)
+\sum_i p_i(\mu_i-\mu)^2.
$$

앞의 다른 weight 혼합에서 두 Z 평균은1과0이다. 준비 내부 분산의 평균은3/5, 준비별 평균 차이의 분산은6/25이므로 합이21/25다. ensemble을 지정하지 않으면 두 항을 각각 ρ만으로 복원할 수 없다.

| I/2를 나타내는 ensemble | Z 평균 | 준비 내부 분산 | 준비별 평균 차이의 분산 |
| --- | --- | --- | --- |
| 같은 weight의 Z 고유상태 | 0 | 0 | 1 |
| 같은 weight의 X 고유상태 | 0 | 1 | 0 |
| 같은 weight의 Y 고유상태 | 0 | 1 | 0 |

세 행의 양자계만의 Z 분산은 모두1이다. 선택한 준비를 밝히지 않고 첫 항을 양자적, 둘째 항을 고전적이라고만 부르면 오해하기 쉽다. 외부 준비 설명에 따라 분해가 달라지지만 최종 예측 분포는 같다.

## 9. 추가 측정으로 빠진 행렬 성분을 복원한다

대각 histogram에는 독립 수가 d−1개다. 일반 trace-one Hermitian 행렬에는 실수 매개변수가 d²−1개 필요하다. coherence를 복원하려면 i<j의 각 쌍을 실수·허수 중첩 방향으로 측정한다. dij=ρii+ρjj라고 두자.

$$
p_{X\pm}=\frac{d_{ij}}2\pm\operatorname{Re}\rho_{ij},\qquad
p_{Y\pm}=\frac{d_{ij}}2\mp\operatorname{Im}\rho_{ij},\qquad
\operatorname{Re}\rho_{ij}=\frac{p_{X+}-p_{X-}}2,\quad
\operatorname{Im}\rho_{ij}=\frac{p_{Y-}-p_{Y+}}2,\quad
p_{\mathrm{outside}}=1-d_{ij}.
$$

Y 양의 방향은(eᵢ+ieⱼ)/√2로 정한다. bra에서 켤레를 취하기 때문에 확률 식에 마이너스 부호가 생긴다. 순수 양의 Y qubit의 ρ01=−i/2이므로 pY+=1이다. 허수 부호를 뒤집으면 두 Y 결과가 바뀐다.

d>2에서는 두 방향이 전체 공간이 아닌 부분공간만 이룬다. 그 직교 여공간을 outside 결과에 포함해야 한다. I/3에서는 어떤 쌍이든 X 양의 결과·X 음의 결과·outside가 각각1/3이며 Y 검사도 같다. 실험은 정확 확률을 사용한다. 유한 shot의 빈도·서로 일치하지 않는 관측 빈도·추정 불확실성은 별도 추론 모형이 필요하다.

## 10. 측정 결과를 조건으로 삼아도 항상 순수해지지는 않는다

직교 사영 Pa의 이상적인 spectral measurement에서 Lüders instrument는 결과 법칙과 상태 갱신을 함께 지정한다. 확률을 정하는 사영을 제시한 것과, 그 확률을 구현하는 모든 장치를 지정한 것은 다르다.

$$
p_a=\operatorname{Tr}(\rho P_a),\qquad
\rho_{\mid a}=\frac{P_a\rho P_a}{p_a}\quad(p_a>0),\qquad
\rho_{\mathrm{nonselective}}=\sum_aP_a\rho P_a.
$$

ρ=I/3, A=2e₀e₀†−I를 보자. 양의 결과는 확률1/3이며 조건부 상태가 순수하다. 음의 결과는 확률2/3이고 조건부 density는 diag(0,1/2,1/2)로 순도가1/2다. 퇴화한 고유공간에서는 혼합이 남을 수 있다. 비선택 상태는 I/3으로 유지된다.

pa=0이면 조건부 density가 없으므로 epsilon으로 나누거나 상태를 만들어 넣지 않고 null을 반환한다. A=aI+b(2P−I)에서 b=0이면 전체 고유공간을 먼저 합친다. d=1에서는 유일한 고유값이 a+b·rank1이므로 rank0인 여공간을 다른 물리 결과처럼 제시하지 않는다.

## 11. 비선택 측정은 특정 coherence를 제거한다

Spectral pinching Δ(ρ)=ΣPaρPa는 각 고유공간 내부의 block을 남기고 block 사이 성분을 지운다. Hilbert–Schmidt 내적에서 Δ는 직교 사영이다. 남기는 부분과 제거한 부분이 서로 직교하므로 피타고라스 계정식을 얻는다.

$$
\operatorname{Tr}(\rho^2)-\operatorname{Tr}[(\Delta\rho)^2]
=\|\rho-\Delta\rho\|_{\mathrm{HS}}^2\ge0,\qquad
\|M\|_{\mathrm{HS}}^2=\operatorname{Tr}(M^\dagger M).
$$

완전한 rank1 기저 측정에서는 Δ가 대각만 남긴다. 퇴화 측정에서는 block 내부 coherence가 살아남을 수 있다. 앞의0.18 차이는 완전 기저 측정의 항등식을 손계산한 예다.

Hilbert–Schmidt 차이 제곱은 trace distance나 검출기가 상태를 알아맞힐 확률이 아니다. 이 pinching 식이 모든 quantum operation에서 entropy가 늘거나 purity가 준다는 뜻도 아니다. 다음 단원에서 다른 거리와 unital dephasing·일반 변환의 차이를 배운다.

## 12. 정확 입력과 rank 경계 근처의 의미

유한 가족은 d=1…4, 비음 weight4개, 복소 raw 대표·probe·관측량 중심/간격 유리수를 위한 소유 정수49개를 사용한다. 점유 대표와 probe는 비영이어야 한다. 사용하지 않는 좌표를0으로 명시하며 차원을 줄일 때 잘라내지 않는다. weight0인 비영 벡터를 기술할 수는 있지만 rho에 weighted column으로 기여하지 않는다.

density 성분·principal minor·rank·Gram 불변량·tomography·측정 계정은 정확한 복소 유리수다. 정규화 진폭과 가중 열만 방향성 제곱근 구간을 사용하며 작업 연산768bit·보고512bit를 쓴다. 미해결 관측은 null일 수 있다. 출력 정규화나 반올림한 near-zero 분류로 그 근거를 대신하지 않는다.

예를 들어 M=1000000에서 원시 벡터(M,1), (M,−1)을 같은 확률로 섞자.

$$
\rho=\operatorname{diag}\!\left(\frac{M^2}{M^2+1},\frac1{M^2+1}\right),\qquad
\operatorname{rank}\rho=2\quad\text{for every finite nonzero }M.
$$

그림은 순수해 보일 수 있지만 작은 고유값은 양수다. 독립 기호 rank와 행렬 계산으로 확인한다. M→∞라는 극한과 제한된 입력에서의 정확한 등식은 다른 명제다.

## 13. 해설과 함께 풀어보기

**1. ψ와−ψ를 같은 확률로 골라도 순수한 이유는?**

**해설.** 두 사영이 같으므로 혼합은 ψψ†이고 순도는1이다. 벡터 평균은0이 되어 무작위 준비에 부적절한 연산임을 드러낸다.

**2. 대각1/2·비대각3/4의 행렬은 상태일까?**

**해설.** 아니다. 고유값이5/4와−1/4다. trace1과 유효한 Z histogram이 있어도 음의 X 확률 때문에 양성이 실패한다.

**3. 다른 weight 혼합의 순도를0.58이라고 하면 무엇을 빠뜨렸을까?**

**해설.** 두 비대각 크기 제곱의 합0.18이다. 전체 순도는0.76이며 작은 수는 완전 Z dephasing 뒤의 순도다.

**4. 분산의 두 기여는 ρ만의 양일까?**

**해설.** 아니다. I/2의 Z·X ensemble은 Z 측정의 내부/사이 기여가0/1과1/0이다. 전체 분산은 모두1이다.

**5. Gram이 rho보다 많은 고유값을 가질 수 있는 이유는?**

**해설.** 차원이 상태 좌표 수가 아니라 표지 슬롯 수이기 때문이다. 영이 아닌 spectrum은 같고 큰 행렬에 영이 더 있다. trace와 purity도 일치한다.

**6. ρ01=−i/2일 때 Y 양의 확률은?**

**해설.** 이 순수 qubit에서는 pY+=1/2−Imρ01=1이다. 더 높은 차원이라면 outside 결과까지 남겨 전체 확률 법칙을 해석해야 한다.

**7. 확률이 양수인 모든 측정 결과의 순도는1일까?**

**해설.** 아니다. I/3 예제의 rank2 음의 고유공간은 순도1/2를 남긴다. 확률0인 결과에는 조건부 상태 자체가 없다.

**8. leading determinant만으로 semidefinite 양성을 인증할 수 있을까?**

**해설.** 없다. diag(0,−1,2)의 leading determinant는 모두0이지만 음의 singleton minor가 있다. 필요한 principal minor를 모두 보는 이유다.

## 14. 학습·연구와 남은 범위

가중 열 구성은 밀도 연산자를 Gram 행렬·공분산 기하·양의 kernel·저랭크 분해와 연결한다. AI 모형에서 trace1만으로 양성이 증명되지는 않고, 양의 행렬이라는 사실만으로 데이터에 대한 확률 보정이 보장되지도 않는다. 매개변수화·추정 모형·측정의 의미를 각각 정해야 한다.

다음 단원에서는 일반 spectral entropy·축약 상태의 해석적 예제와 혼합 qubit 실험을 연결한다. 복합계 동역학·유한 shot tomography·일반 channel 학습·연구 수준의 연산자 문제는 이후 계속 다룬다. 교재는 **도서 → Interactive Book**에서 연다.

ensemble과 상태 규약의 배경은 [Preskill Chapter2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), [IBM convex-combination 설명](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/convex-combinations)에 연결했다. 본문의 유도·가중 열 인증·손계산 준비·수치 비교는 이 교재를 위해 전개했다.
