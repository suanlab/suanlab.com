## 1. 작은 화살표를 더하면 큰 수가 사라질 수 있다

같은 길이의 화살표 두 개를 반대 방향으로 놓으면 합은 0이다. 실시간 경로적분에서도 각 경로의 크기를 작게 만들어 버리는 것이 아니라, 작용에 따라 돌아가는 복소 화살표를 더한다. 대부분의 이웃 경로가 빠르게 방향을 바꾸면 서로 상쇄한다. 정상 경로 근처에서는 그 회전의 일차 변화가 사라져 비교적 일관된 기여가 남는다.

앞 단원의 Euclidean 적분은 낮은 작용 근처에 양의 무게가 모이는 문제였다. 이번에는 음의 곡률도 위상에 기여한다. 따라서 “가장 가능성 높은 경로 하나를 찾는다”는 설명만으로는 충분하지 않다. Gaussian 적분, 행렬, 복소수와 미분을 알고 있다고 가정하고 유한 격자부터 유도한다. 일반 비선형 경로적분의 존재 정리를 증명하는 단원은 아니다.

## 2. 위상·전파 커널·확률을 구별한다

위상의 크기는 항상 1이지만 그 합의 크기는 0일 수도 있다. 먼저 전파 커널을 초기 파동함수에 작용시킨 다음 절댓값 제곱을 취해야 실제 위치 확률을 얻는다.

| 대상 | 의미 | 주의 |
|---|---|---|
| exp(iS/ℏ) | 경로의 복소 위상 | 양의 확률이 아님 |
| K(b,a) | 초기 진폭을 최종 진폭으로 옮기는 커널 | 그 절댓값 제곱만으로 일반 전이 확률을 만들지 않음 |
| ψ(b) | 준비된 상태를 전파한 파동함수 | 서로 다른 초기 위치의 진폭이 간섭함 |
| abs(ψ(b))² | 정규화된 위치 확률밀도 | 실제 확률 공분산을 정의함 |
| iℏH⁻¹ | 이차 위상 적분의 연결 응답 | 위치 확률의 공분산이 아님 |

$$
\psi_\tau(b)=\int K_\tau(b,a)\psi_0(a)\,da,\qquad
\rho_\tau(b)=|\psi_\tau(b)|^2,\qquad \int\rho_\tau(b)\,db=1.
$$

## 3. 정상점이 없으면 부분적분으로 상쇄를 본다

먼저 경로가 아닌 한 좌표 적분을 생각하자. g는 매끄럽고 유한한 구간 밖에서 0이며, 그 지지집합에서 실수 위상 φ의 미분은 0이 아니라고 하자. 지수함수를 미분하면 φ′/ℏ가 나오므로 이를 나눠 적분을 부분적분한다. 경계 항이 사라져 아래 식을 얻는다.

오른쪽의 적분이 유한하면 적어도 ℏ에 비례하는 상계를 얻는다. 더 매끄럽고 기울기 하계가 유지되면 반복하여 임의의 고정 차수 ℏᵏ를 얻는다. 상쇄가 일어나는 것이지 integrand의 절댓값이 작아지는 것은 아니다. 경계가 있거나 φ′가 0에 가까워지면 이 논리를 조건 없이 옮길 수 없다.

$$
I(\hbar)=\int g(x)e^{i\phi(x)/\hbar}\,dx
=-\frac{\hbar}{i}\int\left(\frac{g}{\phi'}\right)'e^{i\phi/\hbar}\,dx,
\qquad |I(\hbar)|\leq\hbar\int\left|\left(\frac{g}{\phi'}\right)'\right|dx.
$$

## 4. 정상점 근처의 이차 위상

정상점 xⱼ에서는 일차 항이 사라진다. x=xⱼ+√ℏy로 바꾸면 이차 항의 위상이 ℏ와 무관해지고 부피 요소가 ℏᵈᐟ²를 만든다. 각 양의 곡률 방향은 +π/4, 음의 방향은 −π/4의 Fresnel 위상을 준다.

고정된 매끄러운 compact amplitude와 위상, 그 지지집합의 유한하고 분리된 비퇴화 정상점에 대해 아래의 선도식이 성립한다. σ는 양의 고유값 수에서 음의 고유값 수를 뺀 값이다. 나머지 상수는 위상의 미분과 곡률 하계 등에 의존한다. 두 정상점이 합쳐지며 determinant가 0으로 갈 때 같은 오차 상수를 보장하지 않는다. g(xⱼ)=0도 전체 적분이 정확히 0이라는 뜻은 아니다.

$$
\int_{\mathbb R^d}g(x)e^{i\phi(x)/\hbar}\,d^dx
=(2\pi\hbar)^{d/2}\sum_j\frac{g(x_j)}{\sqrt{|\det H_j|}}
 e^{i\phi(x_j)/\hbar+i\pi\sigma_j/4}
+O(\hbar^{d/2+1}),\qquad H_j=\nabla^2\phi(x_j).
$$

## 5. 양끝 규칙을 포함한 유한 실시간 작용

시간 τ를 N≥2개 구간으로 나누고 δ=τ/N, d=N−1이라 하자. 양끝 x₀=a, x_N=b를 고정한다. m,τ,ℏ는 양수지만 ω²는 부호를 허용한다. 음수이면 역진동자이며 허수 입력을 넣는다는 뜻이 아니다.

실시간 작용은 운동에너지에서 potential을 뺀다. Euclidean 단원의 더하기와 다르다. 양끝 potential의 절반 가중치는 내부 미분에서는 상수지만 최종 작용과 위상을 바꾼다. S와 ℏ는 작용의 단위, v=ω²δ²는 무차원이다.

$$
V(x)=\frac{m\omega^2x^2}{2},\qquad
S_N=\frac{m}{2\delta}\sum_{i=0}^{N-1}(x_{i+1}-x_i)^2
-\delta\left[\frac{V(a)}2+\sum_{i=1}^{N-1}V(x_i)+\frac{V(b)}2\right].
$$

## 6. 유한 분할의 유니터리성과 정확도의 차이

양의 시간 자유 단계의 제곱근을 exp(−iπ/4) branch로 고정한다. 적분은 양의 Lebesgue 확률 적분이 아닌 진동/Abel 의미다. 각 단계는 potential 반 단계, 자유 전파, potential 반 단계의 곱이다. 실수 potential 위상 곱과 자유 전파는 각각 L²에서 norm을 보존한다. 따라서 유한 곱도 유니터리다.

그러나 유니터리라는 사실만으로 원래 Hamiltonian의 정확한 시간 τ 전파와 같아지지 않는다. 시간 이산화 오차는 별도 문제다. 커널의 단위는 역길이이며 모든 b에서의 점별 유한값이 항상 존재하는 것도 아니다.

$$
K_N(b,a)=\left(\frac{m}{2\pi i\hbar\delta}\right)^{N/2}
\int_{\mathbb R^d}e^{iS_N/\hbar}\,d^dq,\qquad
U_\delta=e^{-i\delta V/(2\hbar)}e^{-i\delta\hat p^2/(2m\hbar)}e^{-i\delta V/(2\hbar)}.
$$

## 7. 두 이웃에서 Hessian을 유도한다

내부 좌표 qᵢ를 미분하면 왼쪽과 오른쪽 운동 항에서 각각 한 항이 나온다. s=m/δ, v=ω²δ²라 쓰면 대각은 s(2−v), 이웃은 −s다. 내부 벡터 q의 첨자는 여기서 1…d다.

N=2에서는 내부 좌표가 하나이므로 a와 b가 같은 forcing 성분에 더해진다. 끝점 값을 마지막에 그려 넣는 것과, 내부 Hessian에 끝점을 미지수로 포함하는 것은 다른 문제다.

$$
S_N(q)=\frac12q^THq-f^Tq+c,\qquad
H_{ii}=s(2-v),\quad H_{i,i+1}=-s,\quad f=s(ae_1+be_d),
\qquad c=\frac{s(a^2+b^2)}2-\frac{\delta m\omega^2(a^2+b^2)}4.
$$

## 8. 정상 경로는 반드시 최소 경로가 아니다

Hq=f를 풀면 이산 운동 방정식을 얻는다. H가 가역이면 정상 경로는 유일하지만 음의 고유값 방향으로 작용은 내려간다. 이때 정상 경로를 최소점이나 가장 높은 확률 경로라고 부르면 잘못이다.

q=q*+y를 대입하면 일차 항이 사라지고 정상 작용과 요동이 분리된다. 이차 모형이므로 Taylor 나머지가 없다. 이것이 아래 Gaussian 적분이 근사가 아니라 정확한 유한 결과인 이유다.

$$
q^*=H^{-1}f,\qquad S_*=c-\frac12f^TH^{-1}f,\qquad
S_N(q^*+y)=S_*+\frac12y^THy.
$$

## 9. 행렬식 점화식과 모든 역행렬 성분

삼중대각 행렬의 마지막 행을 전개하면 continuant 점화식이 나온다. θ₀=1은 크기 0 행렬의 determinant다. 이 수열은 단순히 determinant 하나만 구하는 도구가 아니다. 왼쪽과 오른쪽 소행렬을 연결하면 모든 Green 응답 성분을 얻는다.

아래 역행렬 식의 i,j는 0…d−1이다. θ_d=0에서는 사용하지 않는다. 반면 중간 θ가 0이어도 θ_d가 0이 아니면 전체 역행렬은 존재할 수 있다.

$$
\theta_0=1,\quad\theta_1=2-v,\quad
\theta_k=(2-v)\theta_{k-1}-\theta_{k-2},\qquad
\det H=s^d\theta_d,\qquad
(H^{-1})_{ij}=\frac{\theta_{\min(i,j)}\theta_{d-1-\max(i,j)}}{s\theta_d}.
$$

## 10. 음의 모드는 사라지지 않고 위상을 바꾼다

실수 대칭 H를 직교 대각화하면 각 모드에서 Fresnel 적분을 수행한다. 양의 고유값과 음의 고유값은 크기는 같지만 위상은 다르다. ν를 음의 고유값 수라 하면 σ=d−2ν다.

모드를 삭제하거나 절댓값 곡률만 남기면 간섭 위상이 바뀐다. 이 단원의 검산은 Sturm 부호 변화와 별도의 1×1/2×2 congruence를 비교한다. 첫 pivot이 0인 행렬을 단순한 무pivot LDL의 실패만으로 특이하다고 판정하지 않는다.

$$
\int_{\mathbb R^d}e^{iy^THy/(2\hbar)}\,d^dy
=\frac{(2\pi\hbar)^{d/2}}{\sqrt{|\det H|}}e^{i\pi\sigma/4},
\qquad \sigma=d-2\nu,\qquad \det H\ne0.
$$

## 11. 자유 단계 정규화와 Maslov 위상

N개의 자유 정규화는 −Nπ/4를 주고 내부 d=N−1개의 Fresnel 모드는 π(d−2ν)/4를 준다. 합은 −π/4−πν/2다. 음의 모드가 하나 추가될 때 −π/2가 생기는 구조가 보인다.

determinant의 크기는 진폭을 정하지만 ν를 복원하지 못한다. 아래 식은 비특이한 유한 이차 사슬의 정확한 커널이다. 작용을 실제로 계산한 동일한 끝점 규칙과 정규화를 사용해야 한다.

$$
K_N(b,a)=\sqrt{\frac{s}{2\pi\hbar|\theta_d|}}
\exp\left(\frac{iS_*}{\hbar}-\frac{i\pi}{4}-\frac{i\pi\nu}{2}\right).
$$

## 12. N=2에서 음의 곡률을 손으로 계산한다

δ=m=1, N=2, a=0, b=1, ω²=3을 선택하자. H=−1, f=1, c=−1/4이므로 q*=−1이고 S*=1/4이다. 내부 정상점은 최대점이다. 그래도 진동 적분은 Abel 의미로 정의할 수 있다.

ν=1, θ₁=−1이므로 전체 위상은 1/(4ℏ)−3π/4다. |detH|만 보고 자유 사슬과 같은 −π/4를 사용하면 π/2만큼 틀린다. 이 예에서 양의 ω²가 음의 내부 곡률을 만든 것은 시간 격자가 거칠기 때문이다. 물리적 역진동자와 구별하자.

## 13. 첫 pivot 0이지만 가역인 예

N=3, δ=m=1, v=2이면 H의 대각은 0, 이웃은 −1이다. determinant는 −1이고 역행렬은 H 자신이다. 고유값은 +1,−1이다. θ 수열은 1,0,−1로 끝나므로 0을 건너뛴 부호 변화가 하나다.

a=1,b=0을 선택하면 f=(1,0), q*=(0,−1), c=S*=0이다. 0으로 나누는 무pivot 소거법만 실패했을 뿐, 양끝 문제와 커널은 비특이하다. 계산 알고리즘의 분모와 물리 모형의 특이성을 구별하는 좋은 예다.

$$
H=\begin{pmatrix}0&-1\\-1&0\end{pmatrix},\qquad
H^{-1}=H,\qquad \det H=-1,\qquad \nu=1.
$$

## 14. 정확한 초점에서는 커널이 distribution이 된다

N=2, δ=m=1, v=2이면 H=0이다. a+b=0이면 모든 내부 q가 정상 경로이고 작용은 0이다. a+b≠0이면 정상 경로가 없다. 작은 양수를 대각에 더해 가짜 역행렬을 만드는 것은 두 경우를 모두 바꿔 버린다.

일반적인 정확한 초점에서는 이 삼중대각 Hessian의 nullity가 1이다. r=(θ₀,…,θ_{d−1})가 영벡터이며 호환 조건은 rᵀf=0이다. 아래 M은 고전 전파 행렬, k=ν+1이다. N2,v2에서는 M=−I,k=1이고 K=−iδ(b+a)다. 이 δ는 시간 간격 δ와 표기가 같지만 Dirac distribution을 뜻한다. 점별 “무한대 확률”을 반환하는 수식이 아니다.

$$
\theta_d=0:\qquad M=(-1)^kI,\quad k=\nu+1,\qquad
K_N(b,a)=e^{-i\pi k/2}\,\delta\bigl(b-(-1)^ka\bigr).
$$

## 15. Abel 조절자는 정의를 돕지만 물리를 바꾼다

내부 적분에 exp(−εqᵀq/(2ℏ))를 곱하면 ε>0에서 절대 적분 가능한 복소 Gaussian이 된다. Cε=εI−iH의 실수부는 양의 정부호다. 완전제곱으로 아래 식을 얻는다. Iε는 자유 단계 prefactor를 포함하지 않는 내부 적분이다.

복소 평균은 −iCε⁻¹f이고 연결 응답은 ℏCε⁻¹다. ε→0에서 각각 q*와 iℏH⁻¹로 간다. 이 값들은 복소 가중치의 대수적 응답이며 Born 확률의 평균/공분산이라고 해석하지 않는다. 유한 ε는 damping을 도입하므로 원래 유니터리 전파가 아니다.

$$
C_\epsilon=\epsilon I-iH,\qquad
I_\epsilon=(2\pi\hbar)^{d/2}\prod_{j=1}^d\alpha_j^{-1/2}
\exp\left(\frac{ic}{\hbar}-\frac{f^TC_\epsilon^{-1}f}{2\hbar}\right),\qquad
\alpha_1=\epsilon-iH_{11},\quad
\alpha_j=\epsilon-iH_{jj}+\frac{s^2}{\alpha_{j-1}}.
$$

## 16. determinant의 주 제곱근 하나로 대체하지 않는다

각 복소 pivot αⱼ의 실수부는 양수다. 첫 pivot에서 그렇고, 양의 실수부를 가진 수의 역수도 양의 실수부이므로 점화식으로 이어진다. 각 sqrt의 실수부를 양수로 선택하고 그 곱을 유지한다.

N=4 자유 사슬의 H 고유값은 모두 양수다. ε가 작으면 세 pivot은 각각 거의 −π/2의 argument를 갖는다. 제곱근 곱의 argument는 거의 −3π/4지만 determinant의 주 argument를 먼저 취하면 +π/4를 얻는다. 두 답은 부호가 다르다. 작용을 정확히 계산해도 branch를 잘못 고르면 최종 진폭이 틀린다.

## 17. 같은 분할에서 고전 symplectic map을 만든다

potential 반 kick은 p를 −mω²δq/2만큼 바꾸고, 자유 drift는 q를 δp/m만큼 바꾼다. 이를 순서대로 곱하면 아래 한 단계 행렬을 얻는다. determinant가 1이어서 위상공간 면적을 보존한다.

N번 곱한 행렬의 성분을 A₀,B₀,C₀,D₀라 쓰자. Hessian과 혼동하지 않게 아래 첨자를 붙인다. B₀=(δ/m)θ_d이므로 행렬식 초점과 고전 map의 초점이 일치한다. B₀≠0일 때 S*의 생성함수 식에서 a,b를 미분하면 초기/최종 운동량을 얻는다. 작용의 B₀에는 부호가 남아야 한다.

$$
M_\delta=\begin{pmatrix}1-v/2&\delta/m\\-m\omega^2\delta(1-v/4)&1-v/2\end{pmatrix},\quad
M=M_\delta^N=\begin{pmatrix}A_0&B_0\\C_0&D_0\end{pmatrix},\quad
B_0=\frac{\delta}{m}\theta_d,\quad
S_* =\frac{D_0b^2+A_0a^2-2ab}{2B_0}.
$$

## 18. 시간 격자 정련은 작은 ℏ와 다른 실험이다

τ,m,ω²를 고정하고 N,2N,4N,8N의 map을 비교하자. 자유 모형에서는 모든 분할이 [[1,τ/m],[0,1]]와 일치한다. 양의 ω²의 연속 map은 대각 cos(ωτ), 오른쪽 위 sin(ωτ)/(mω), 왼쪽 아래 −mωsin(ωτ)다. 음의 ω²에서는 κ=√(−ω²)를 두어 cosh와 sinh의 hyperbolic map을 얻는다.

v>4인 거친 양의 진동자 분할도 hyperbolic 고유값을 가질 수 있다. 이것은 물리적 ω²<0과 다르다. 작은 ℏ만으로 이산 map의 오차가 줄지는 않는다. 상대적인 커널 오차는 초점 근처에서 특히 민감하므로 map 오차와 signature/분포 분류도 함께 본다.

## 19. 정규화 Gaussian 파동묶음과 chirp

점별 커널이 특이해도 정규화된 상태의 전파는 잘 정의될 수 있다. 초기 위치 평균 q₀, 운동량 평균 p₀, 위치 분산 σ²>0과 실수 chirp χ를 사용하자. χ는 위치에 따라 평균 운동량이 얼마나 기울어지는지를 나타낸다. 위치 밀도는 χ에 의존하지 않지만 위치와 운동량의 상관은 달라진다.

아래 prefactor는 |ψ|²를 적분하면 1이 되게 한다. χ의 단위는 운동량/길이다. σ는 표준편차이고 σ²가 분산이다.

$$
\psi_0(x)=(2\pi\sigma^2)^{-1/4}
\exp\left[-\frac{(x-q_0)^2}{4\sigma^2}
+\frac{ip_0(x-q_0)}{\hbar}+\frac{i\chi(x-q_0)^2}{2\hbar}\right].
$$

## 20. 평균과 전체 공분산을 함께 운반한다

운동량 연산자 −iℏ∂ₓ를 초기 Gaussian에 적용하면 위치에서 q₀를 뺀 값에 χ를 곱한 실수 항과 폭에 따른 허수 항이 나온다. 이를 평균내면 아래 대칭 공분산을 얻는다. 교차 성분은 연산자 순서를 대칭화한 공분산이다.

이차 전파에서는 평균과 공분산이 같은 고전 map으로 변한다. determinant는 ℏ²/4로 유지되어 최소 불확정성 면적을 보존한다. 이것은 실제 정규화 상태의 공분산이며 앞의 복소 iℏH⁻¹와 다른 대상이다.

$$
\Sigma_0=\begin{pmatrix}\sigma^2&\chi\sigma^2\\
\chi\sigma^2&\chi^2\sigma^2+\hbar^2/(4\sigma^2)\end{pmatrix},\qquad
\binom{q_\tau}{p_\tau}=M\binom{q_0}{p_0},\qquad
\Sigma_\tau=M\Sigma_0M^T,\qquad\det\Sigma_\tau=\frac{\hbar^2}{4}.
$$

## 21. 복소 폭과 초점을 지나는 유한한 상태

Z₀의 실수부는 chirp, 허수부는 양의 역폭이다. Möbius 식으로 전파하면 Im Zτ=Im Z₀/|A₀+B₀Z₀|²>0이다. 따라서 정규화된 위치 분산은 양수다. B₀=0이어도 분모 A₀가 0이 아니므로 커널의 가짜 무한대가 나타나지 않는다.

평균 경로의 작용은 (qτpτ−q₀p₀)/2다. 이는 이차 운동에서 d(qp)/dt=2L을 적분하거나 분할의 kick/drift 작용을 합쳐 얻는다. B₀≠0일 때 복소 Gaussian 적분으로 얻는 prefactor의 제곱은 1/(A₀+B₀Z₀)이지만 제곱만으로 branch를 결정하지 않는다. α의 sqrt 곱과 Maslov 위상을 추적해야 한다. 정확한 조화 초점에서는 parity와 exp(−iπk/2)를 상태에 직접 적용한다.

$$
Z_0=\chi+\frac{i\hbar}{2\sigma^2},\qquad
Z_\tau=\frac{C_0+D_0Z_0}{A_0+B_0Z_0},\qquad
\sigma_{q,\tau}^2=\frac{\hbar}{2\operatorname{Im}Z_\tau},\quad
\chi_\tau=\operatorname{Re}Z_\tau,\quad
S_{\mathrm{center}}=\frac{q_\tau p_\tau-q_0p_0}{2}.
$$

## 22. 두 가지 ℏ→0 족은 다른 고전극한을 준다

첫째, σ²와 χ를 고정하고 ℏ만 줄이면 초기 공분산은 일반적으로 0이 아닌 rank 1 행렬로 간다. 위치마다 p−p₀=χ(q−q₀)인 고전 ensemble이다. 넓은 분포가 고전적으로 움직이는 것이며 단일 궤적으로 수축하지 않는다.

둘째, σ²=ℏr로 줄이고 r>0,χ와 유한한 M을 고정하면 모든 공분산 성분이 ℏ에 비례해 0으로 간다. 평균이 따르는 한 위상공간 점으로 집중한다. r의 단위는 σ²/ℏ이며 앞 족의 입력 폭과 다르다. 임의의 r을 진동자의 고유 coherent state 폭이라고 부르지 않는다. M도 ℏ와 함께 커지는 긴 시간 극한에는 같은 결론을 자동 적용할 수 없다.

$$
\left.\Sigma_0\right|_{\sigma^2\,\mathrm{fixed}}
\longrightarrow\sigma^2\begin{pmatrix}1&\chi\\\chi&\chi^2\end{pmatrix},
\qquad
\left.\Sigma_\tau\right|_{\sigma^2=\hbar r}
=\hbar M\begin{pmatrix}r&\chi r\\\chi r&\chi^2r+1/(4r)\end{pmatrix}M^T
\longrightarrow0.
$$

## 23. 위치가 모여도 단일 궤적이 아닌 반례

자유 전파에서 τ=m=1이면 qτ=q₀+p₀, pτ=p₀다. χ=−1인 초기 ensemble은 위치 편차와 운동량 편차가 반대여서 고전적으로 최종 위치 편차가 상쇄된다. 고정 분산 σ²를 유지하고 ℏ만 줄여도 아래처럼 최종 위치 분산이 0으로 간다.

그러나 최종 운동량 분산은 σ²에 남는다. 위치 그래프 하나만 보면 “고전 한 점”이라고 오해하기 쉽다. 전체 공분산과 determinant를 함께 보아야 한다. 이 자유 chirp 초점은 B₀=1이므로 조화 커널의 B₀=0 초점과도 구별한다.

$$
M=\begin{pmatrix}1&1\\0&1\end{pmatrix},\quad\chi=-1:
\qquad\sigma_{q,\tau}^2=\frac{\hbar^2}{4\sigma^2},\quad
\sigma_{p,\tau}^2=\sigma^2+\frac{\hbar^2}{4\sigma^2},\quad
\Sigma_{qp,\tau}=\frac{\hbar^2}{4\sigma^2}.
$$

## 24. 두 정상점이 만나는 별도의 fold 모형

이제 사슬과 별개로 한 좌표 위상 φ=x³/3−ux를 보자. u>0이면 ±√u에 두 정상점이 있고, u=0에서 합쳐지며, u<0에는 실수 정상점이 없다. x=ℏ¹ᐟ³t로 바꾸면 아래 Airy 표현을 얻는다. 여기서는 삼차 계수를 1로 정한 canonical 단위를 사용한다.

u와 ℏ는 독립적인 물리 control이다. 합류 영역의 크기는 u가 ℏ²ᐟ³ 정도일 때 드러난다. 이 모형은 일반 비선형 경로적분 전체를 계산하지 않으며 조화계의 정확한 delta 초점을 단순히 Airy 곡선으로 교체하는 것도 아니다.

$$
J(u,\hbar)=\int_{\mathbb R}e^{i(x^3/3-ux)/\hbar}\,dx
=2\pi\hbar^{1/3}\operatorname{Ai}\left(-\frac{u}{\hbar^{2/3}}\right).
$$

## 25. 두 정상점의 간섭과 Gaussian 근사의 실패

x=+√u에서 작용은 −2u³ᐟ²/3, 곡률은 +2√u다. 음의 정상점에서는 두 부호가 반대다. 각각의 Fresnel 위상을 더하면 아래 cos가 나온다. 두 기여의 크기를 그냥 더하면 간섭 영점을 잃는다.

고정한 u>0에서 ℏ→0이면 좋은 선도 근사지만, u→0에서 Gaussian 크기는 발산한다. 정확한 canonical 적분은 u=0에서 2πℏ¹ᐟ³Ai(0)으로 유한하다. u<0에서는 실수 정상점이 없지만 Ai의 양의 인수 값은 양수이며 지수적으로 작다. “실수 정상점이 없으므로 정확한 0”이라는 결론도 틀리다.

$$
J_{\mathrm{SP}}(u,\hbar)=2\sqrt{\frac{\pi\hbar}{\sqrt{u}}}
\cos\left(\frac{2u^{3/2}}{3\hbar}-\frac\pi4\right),\qquad u>0.
$$

## 26. Airy 계산에는 중앙 급수와 큰 인수 표현이 필요하다

Ai는 y″=zy와 두 초기값으로 정의할 수 있다. a₂=0이고 (n+2)(n+1)a_{n+2}=a_{n−1}을 얻으므로 지수 3의 간격을 가진 두 계열이 남는다. 각 계열의 절대 항비가 1보다 작아져 감소하면 기하급수 꼬리 상계를 만들 수 있다.

하지만 큰 양의 z에서는 큰 두 계열의 차가 작은 Ai를 만들므로 상쇄로 정밀도를 잃는다. 이때 exp(ζ)Ai(z), ζ=2z³ᐟ²/3의 scaled 값을 계산하고 지수 척도나 log 크기를 별도로 유지하는 편이 낫다. 큰 음의 인수에서는 sin/cos 점근 계열과 조건을 만족하는 생략항 경계를 사용한다. Γ 상수와 π 자체도 인증이 필요하다. 여기의 급수식은 생산 수치 구현이 완료되었다는 뜻이 아니다.

$$
\operatorname{Ai}(z)=\sum_{n=0}^{\infty}a_nz^n,\quad
a_0=\frac1{3^{2/3}\Gamma(2/3)},\quad a_1=-\frac1{3^{1/3}\Gamma(1/3)},\quad a_2=0,\quad
(n+2)(n+1)a_{n+2}=a_{n-1}\quad(n\geq1).
$$

## 27. 숫자가 수렴해 보이는 것과 오차 인증의 차이

직접 진동 적분의 구간을 잘라 격자를 촘촘히 하는 것만으로 무한 꼬리나 aliasing을 통제하지 못한다. contour를 바꾼 수치 적분도 해석적 경로 변형 조건과 꼬리/반올림 상계를 따로 확인해야 한다. 작은 ℏ에서는 phase를 큰 수의 차로 계산하는 손실도 생긴다.

이 단원의 기반 검산은 62유한 모형의 정확분수 41,191검사와 Decimal 13진단을 구분한다. 후자는 격자/branch의 일치 관찰이며 인증된 적분 오차가 아니다. 실제 계산 화면은 아직 연결하지 않았다. underflow를 정확한 0으로 표시하거나 구간이 0을 포함할 때 상대 오차와 log를 보장해서는 안 된다.

## 28. 어느 극한을 취하는지 먼저 선언한다

계산을 비교하기 전에 무엇을 고정했는지 적자. 특히 ε를 고정한 채 내부 좌표 수를 늘리면 누적 damping 자체가 달라진다. 이 단원에서는 고정된 유한 N에서 조절자를 제거하는 뜻을 먼저 정의한다.

| 조작 | 고정하는 것 | 확인할 것 |
|---|---|---|
| ℏ 감소 | N,작용,초기 상태의 족 | 위상 상쇄와 선택한 고전극한 |
| N 증가 | 총 시간 τ와 물리 계수 | map/정규화와 초점 분류 |
| ε→0 | 유한 N와 양끝 | Abel branch와 distribution 의미 |
| 초점 접근 | 물리 모형과 접근 경로 | 점별 커널과 상태 전파의 차이 |
| fold 합류 | u/ℏ²ᐟ³ 척도 | 두 Gaussian 대신 uniform Airy 표현 |

## 29. 먼저 예측하고 비교할 여섯 실험

아래는 손계산·본문 코드와 이후 계산 화면을 위한 비교 안내다. 아직 구현되지 않은 버튼이나 시각화가 있다고 가정하지 않는다.

1. **음의 모드:** N2,δ=m=1,a=0,b=1에서 v=1과3을 비교한다. 먼저 정상점의 종류와 ν를 예측한 뒤 H,f,c를 계산한다. determinant 크기만 비교했을 때 빠지는 위상을 설명한다.
2. **영 pivot:** N3,v=2와 N2,v=2를 비교한다. 둘 다 첫 pivot이 0인데 왜 전자만 역행렬이 있는지 determinant로 판정한다.
3. **초점과 상태:** N2,v=2에서 b=−a와 b≠−a를 비교한 뒤 정규화 Gaussian에 parity를 적용한다. 점별 커널과 상태의 norm을 구별한다.
4. **두 폭 족:** 같은 자유 map에서 고정 σ²와 σ²=ℏr를 비교한다. ℏ를 절반으로 줄일 때 위치/운동량 분산이 어떻게 변하는지 예측한다.
5. **chirp 초점:** τ=m=1,σ²=2/3에서 χ=0과−1을 비교한다. 최종 위치 분산만 보고 단일 궤적이라고 결론낼 수 있는지 전체 공분산으로 답한다.
6. **fold 합류:** u>0에서 u→0→음수로 옮긴다. 두 정상점 근사와 exact canonical 표현의 차이, 실수 정상점이 없는 쪽의 작은 비영 진폭을 설명한다.

## 30. 같은 Python 코드로 손계산을 확인한다

다음 코드는 표준 라이브러리 Fraction만 쓴다. 음의 곡률 정상점, 영 pivot이 있는 가역 행렬, 정확한 초점의 호환 조건과 두 packet족을 검사한다. ℏ=1/2,1/4,1/8에서 고정 폭의 chirp 초점은 위치 분산만 줄고 운동량 분산은 2/3으로 남는 모습을 출력한다.

Gaussian 전파의 전체 위상이나 Airy 값을 수치 적분하는 코드는 아니다. 유한 손계산과 공분산 반례를 정확분수로 재현하는 실습이다. 로컬 Python이나 Colab 셀에서 실행할 수 있다.

~~~python
from fractions import Fraction as F

# N=2, delta=m=1, v=3, a=0, b=1.
H, f, c = F(-1), F(1), F(-1, 4)
q = f / H
S = c - f * q / 2
assert (q, S) == (F(-1), F(1, 4))
print(f"negative-mode q={q} S={S} nu=1")

# N=3, v=2: H=[[0,-1],[-1,0]] is its own inverse.
H2 = [[F(0), F(-1)], [F(-1), F(0)]]
identity = [[sum(H2[i][k] * H2[k][j] for k in range(2))
             for j in range(2)] for i in range(2)]
assert identity == [[1, 0], [0, 1]]
print("zero-pivot det=-1 inverse=H nu=1")

# N=2, v=2: H=0, so compatibility is a+b=0.
a = F(2, 3)
assert a + (-a) == 0 and a + F(0) != 0
print("caustic compatible=True incompatible=False")

# Free propagation at tau=m=1. Covariances are symmetrized.
def propagate(variance, chirp, hbar):
    qq = variance
    qp = chirp * variance
    pp = chirp**2 * variance + hbar**2 / (4 * variance)
    return qq + 2 * qp + pp, qp + pp, pp

for h in [F(1, 2), F(1, 4), F(1, 8)]:
    v = F(2, 3)
    qq, qp, pp = propagate(v, F(-1), h)
    assert qq == h*h / (4*v)
    assert pp == v + h*h / (4*v)
    assert qq*pp - qp*qp == h*h/4
    free_qq, _, _ = propagate(v, F(0), h)
    assert free_qq == v + h*h/(4*v)
    r = F(2, 3)
    narrow_qq, _, narrow_pp = propagate(h*r, F(0), h)
    assert narrow_qq/h == r + 1/(4*r)
    assert narrow_pp/h == 1/(4*r)
    print(f"h={h} focus_qq={qq} focus_pp={pp} "
          f"unchirped_qq={free_qq} scaled_qq={narrow_qq}")
~~~

## 31. AI의 최적화·불확실성과 연결하되 동일시하지 않는다

Laplace 근사와 정상 위상은 모두 정상점과 Hessian을 쓰지만 가중치가 다르다. 양의 posterior의 국소 covariance에 음의 curvature가 나오면 그 지점은 Gaussian 최소점 근사의 조건을 만족하지 않는다. 실시간 적분은 음의 curvature를 위상으로 처리한다. iℏH⁻¹를 Bayesian uncertainty로 가져오면 안 된다.

역문제와 학습된 동역학에서는 symplectic map의 면적 보존과 유한 단계 안정성을 별도로 검사할 수 있다. norm 보존만 만족하는 모델도 원래 동역학을 부정확하게 근사할 수 있다. 또한 출력 위치의 좁은 분포 하나만으로 latent 위상공간 불확실성 전체가 작다고 판단하면 chirp 반례를 놓친다.

연구에서는 여러 정상점, 대칭, 퇴화, 상쇄와 정밀도 비용을 명시해야 한다. 실시간 sampling의 sign problem도 복소/부호 기여의 상쇄에서 생기는 통계적 어려움과 연결된다. 이 단원의 정확한 이차 해법이 일반 sign problem을 해결한 것은 아니다.

## 32. 해설이 있는 확인 문제

1. **각 경로 위상의 크기가 1이면 적분도 커야 하는가?** 아니다. 서로 다른 방향의 화살표가 상쇄한다. compact amplitude의 비정상 위상에서는 부분적분으로 작은 ℏ의 상계를 얻을 수 있다.
2. **유일한 정상 경로이면 최소점인가?** 아니다. N2,v3의 H=−1은 유일한 최대점을 준다. 진동 적분에서는 음의 모드의 Fresnel 위상이 남는다.
3. **determinant 절댓값이 같으면 같은 커널인가?** 아니다. 작용, forcing, 음의 모드 수와 branch가 필요하다. 음의 모드 하나는 −π/2를 바꾼다.
4. **첫 pivot이 0이면 역행렬이 없는가?** 아니다. N3,v2의 2×2행렬은 determinant −1이다. 전체 determinant와 적절한 pivot 처리를 확인한다.
5. **초점에서 커널에 큰 숫자를 반환하면 되는가?** 아니다. 정확한 조화 초점은 parity를 포함한 delta distribution이다. 정규화 상태에 작용시켜 해석해야 한다.
6. **ℏ→0이면 모든 Gaussian이 한 궤적으로 가는가?** 아니다. 고정 폭은 보통 고전 ensemble을 준다. σ²=ℏr 족과 고정 M에서는 전체 covariance가 0으로 간다.
7. **최종 위치 분산이 0으로 가면 충분한가?** 아니다. 자유 χ=−1 반례에서 운동량 분산은 σ²에 남는다. 전체 위상공간 covariance를 확인한다.
8. **실수 정상점이 없으면 fold 적분은 0인가?** 아니다. u<0에서는 Ai의 양의 인수 값이 양수이며 지수적으로 작다. underflow와 정확한 0을 구별한다.

## 33. 자료와 이번 원고의 경계

[Evans–Zworski의 Lemma3.13/Theorem3.15](https://math.mit.edu/~vwg/semiclassicalEvansZworski.pdf#page=38)는 비정상/정상 위상의 조건을, [Wipf§3.1](https://www.tpi.uni-jena.de/~wipf/lectures/pfad/pfad3.pdf)은 유한 이차 경로적분을 참고하는 자료다. 여기의 양끝 대칭 규칙은 직접 유도했다. [Horváthy§§2–3](https://arxiv.org/pdf/quant-ph/0702236#page=2)은 Maslov 위상과 조화 초점의 distribution 해석을 다룬다.

Airy의 방정식/초기값, 적분, 점근식과 나머지 조건은 각각 [DLMF9.2](https://dlmf.nist.gov/9.2), [9.5](https://dlmf.nist.gov/9.5), [9.7](https://dlmf.nist.gov/9.7)을 따른다. 양의 인수에서의 양성은 [9.6.2](https://dlmf.nist.gov/9.6#E2)의 modified Bessel 연결과 [10.32.9](https://dlmf.nist.gov/10.32#E9)의 양의 적분 표현으로 확인할 수 있다.

이번에는 이론·유한 손계산·실행 예제를 작성했다. 전체 입력의 인증된 생산 계산, 독립 검산, Worker, 시각화, 저장, 실제 reader와 단원/원주제 감사는 다음 단계다. 따라서 새 단원이나 원래 경로적분 주제 전체를 완료했다고 세지 않는다.
