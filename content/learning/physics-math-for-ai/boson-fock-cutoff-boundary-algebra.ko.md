## 1. 같은 모드에 여러 입자가 놓이면 공간이 무한해진다

앞 장의 fermion 모드는 점유수가0/1이었다. Boson에서는 같은 한 입자 모드를 여러 입자가 함께 점유할 수 있다. 광학 모드의 광자 수나 진동 모드의 excitation 수가 대표적인 예다. 한 모드의 가능한 점유수만으로 이미0,1,2,…가 모두 필요하다. 모드가 하나라는 사실과 상태 공간이2차원이라는 사실은 같은 말이 아니다.

선택한 한 입자 모드들은 직교정규라고 가정한다. 각 모드의 수 상태를 |n〉으로 쓰고, 여러 모드는 보통 tensor product로 결합한다. 서로 다른 모드의 boson 연산자는 교환한다. 앞 장의 fermion parity string을 그대로 가져오지 않는다. 같아 보이는 점유 그림이라도 통계와 연산자 대수가 다르다.

이 장은 **무한 ladder → 유한 projection → 경계 교환자 → coherent 급수 → 절단 오차와 관측** 순서로 진행한다. 식은 유한 Fock 중첩이나 필요한 모멘트가 존재하는 상태에서 사용한다. 무한 차원의 a와 a†는 unbounded 연산자이므로 모든 Hilbert 벡터에 모든 곱이 무제한 정의된다고 가정하지 않는다. 수치 예제는 한두 모드와 최대 점유3을 사용하지만 이론이나 교육과정의 상한은 아니다.

## 2. √n은 정규화에서 나온다

한 모드의 vacuum은 a|0〉=0과 노름1을 만족한다. Canonical commutation relation, CCR은 [a,a†]=I다. Fermion의 반교환자와 달리 여기서는 두 곱의 차를 사용한다. Number operator n̂=a†a의 고유상태를 정규화하면 다음 ladder 계수가 나온다.

$$
|n\rangle=\frac{(a^\dagger)^n}{\sqrt{n!}}|0\rangle,\qquad
a|n\rangle=\sqrt n\,|n-1\rangle,\qquad
a^\dagger|n\rangle=\sqrt{n+1}\,|n+1\rangle.
$$

a|n〉의 노름 제곱은〈n|a†a|n〉=n이다. 따라서 아래 상태가 단위 노름이면 계수는√n이어야 한다. 같은 방식으로 aa†=n̂+I를 사용하면 생성 계수√(n+1)을 얻는다. 두 번 생성한 결과는0이 아니며, a†a†|0〉=√2|2〉다. 단순히 점유 비트를 바꾸는 행렬로는 이 정규화와 관측량을 재현하지 못한다.

모든 n을 포함하면 연산자의 행렬은 무한하다. 컴퓨터에서 유한 배열을 쓰려면 어떤 부분공간을 남길지 정해야 한다. 배열의 마지막 원소를 없애는 구현 선택은 물리적인 근사 선택이기도 하다. 다음 절에서 경계에 생기는 항을 직접 계산할 것이다.

## 3. Cutoff는 최대 점유이며 차원은 하나 더 크다

최대 점유를 C라 하고0부터 C까지 남기는 projector를 정하자. C=3이면 차원4이며, C=0도 진공 하나의 차원1 공간이다. 여기서 cutoff와 차원을 혼동하면 ladder의 마지막 행과 열을 잘못 만든다.

$$
\Pi_C=\sum_{n=0}^{C}|n\rangle\langle n|,\qquad
a_C=\Pi_Ca\Pi_C,\qquad a_C^\dagger=\Pi_Ca^\dagger\Pi_C.
$$

마지막 상태 |C〉를 생성하면 원래 무한 공간에서는√(C+1)|C+1〉로 간다. 그러나 유한 projection은 그 방향을 버리므로 a_C†|C〉=0이다. 이는 boson이 실제로 점유 C를 넘을 수 없다는 새 법칙이 아니라 선택한 유한 표현의 경계다. 같은0이라도 fermion exclusion과는 이유가 다르다.

| C=3의 입력 | a_C의 출력 | a_C†의 출력 | [a_C,a_C†]의 고유값 |
| --- | --- | --- | --- |
| \|0〉 | 0 | \|1〉 | 1 |
| \|1〉 | \|0〉 | √2\|2〉 | 1 |
| \|2〉 | √2\|1〉 | √3\|3〉 | 1 |
| \|3〉 | √3\|2〉 | 0 | −3 |

표의 마지막 행을 위 행과 같은 규칙으로 추측하면 잘못된 교환자를 얻게 된다. 생성한 뒤 소멸하는 경로가 경계에서 사라지기 때문이다. 연산자 곱의 순서와 어느 경로가 표현 밖으로 나가는지 함께 그려 보자.

## 4. 유한 행렬은 무한 CCR을 그대로 만족할 수 없다

기저에 작용시키면 a_C†a_C는0,1,…,C의 대각 행렬이고, a_Ca_C†는1,2,…,C,0의 대각 행렬이다. 두 차를 취하면 경계 projector가 남는다.

$$
[a_C,a_C^\dagger]=I_C-(C+1)|C\rangle\langle C|,\qquad
\langle[a_C,a_C^\dagger]\rangle=1-(C+1)p_{\rm top},\qquad
p_{\rm top}=\langle C|\rho|C\rangle.
$$

이 차이는 부동소수점 오차가 아니다. 어떤 유한 행렬 A,B에 대해서도 Tr(AB−BA)=0인데 identity의 trace는 양의 차원이다. 따라서 유한 차원에서 [A,B]=I를 정확 만족할 수 없다. 경계를 더 높은 곳으로 옮길 수는 있어도 이 대수적 장애 자체가 없어지는 것은 아니다.

C=3의 top 상태에서는 기대 교환자가−3이다. (|0〉+|3〉)/sqrt(2)에서는 p_top=1/2이므로−1이다. C=0에서는 a_C=0·I_C=|0〉〈0|이어서 교환자는0이다. 표시 값을1에 맞추어 재보정하면 사용 중인 연산자와 계산을 다르게 보고한 셈이 된다.

## 5. 불확정성 부등식에는 실제 교환자를 넣는다

무차원 quadrature를 X=(a+a†)/sqrt(2), P=(a−a†)/(i sqrt(2))로 정한다. 원래 무한 모드에서는 [X,P]=iI다. 질량 m·진동수 Ω의 조화 진동자에 물리 단위를 복원하면 x=sqrt(ℏ/(mΩ))X, p=sqrt(mℏΩ)P다.

$$
X_C=\frac{a_C+a_C^\dagger}{\sqrt2},\qquad
P_C=\frac{a_C-a_C^\dagger}{i\sqrt2},\qquad
[X_C,P_C]=i\bigl(I_C-(C+1)|C\rangle\langle C|\bigr).
$$

분산은 현재 상태와 현재 연산자에 관한 양이다. Symmetrized covariance를 K라 하면 Robertson–Schrödinger 부등식은 다음과 같다. 오른쪽에는 유한 교환자의 **기대값**이 들어간다.

$$
K=\frac12\langle X_CP_C+P_CX_C\rangle-\langle X_C\rangle\langle P_C\rangle,\qquad
\operatorname{Var}(X_C)\operatorname{Var}(P_C)-K^2
\ge\frac14\left|1-(C+1)p_{\rm top}\right|^2.
$$

C=0에서는 두 연산자가0이므로 두 분산과 경계가 모두0이다. 이는 원래 위치·운동량의 불확정성 원리를 위반한 상태를 만들었다는 뜻이 아니다. 계산한 연산자들이 달라졌다. C=3의 top 상태에서는 두 분산이3/2이고 K=0, 경계가9/4여서 유한 부등식을 포화한다. 무한 모드의1/4를 모든 유한 입력에 강제로 적용하지 않는다.

## 6. 먼저 곱한 뒤 자르는 것과 먼저 자른 뒤 곱하는 것은 다르다

같은 유한 상태를 무한 공간에0으로 확장해 놓고 **원래 X**를 측정하는 경우를 생각하자. X의 평균은 X_C와 같지만 이차 모멘트는 다르다. X가 잠시 |C+1〉로 나갔다가 돌아오는 경로는 원래 X²에는 있고 X_C²에는 없다.

$$
\Pi_CX^2\Pi_C-X_C^2
=\Pi_CP^2\Pi_C-P_C^2
=\frac{C+1}{2}|C\rangle\langle C|.
$$

따라서 embedded 상태에서 원래 quadrature의 분산은 유한 분산에(C+1)p_top/2를 더한 값이다. C=0의 유한 진공에서0이던 분산도 원래 X,P로 측정하면 각각1/2다. C=3의 top 상태는 유한 분산3/2에서 원래 분산7/2로 바뀐다. 상태와 연산자 가운데 무엇을 비교하고 있는지 명시해야 한다.

같은 차이가 조화 진동자 Hamiltonian에도 나타난다. 먼저 원래 H를 project한 행렬과 잘라 만든 quadrature를 제곱한 행렬은 경계에서 같지 않다.

$$
\frac{H_C}{\hbar\Omega}=\hat n_C+\frac12I_C,\qquad
\frac{X_C^2+P_C^2}{2}
=\hat n_C+\frac12I_C-\frac{C+1}{2}|C\rangle\langle C|.
$$

C=3의 top 에너지는 첫 정의에서7/2, 두 번째에서는3/2다. 무한 CCR로 유도한 항등식을 유한 행렬에 그대로 옮긴 것이 원인이다. 이 장의 onsite·offset 입력으로 조화 진동자를 나타낼 때는 H_C=ℏΩ(n̂_C+I_C/2)라는 투영 모형을 명시한다. 그림의 부드러운 궤적보다 연산자의 정의가 먼저다.

## 7. 두 모드 hopping에는 bosonic enhancement가 있다

두 모드의 전체 차원은(C₀+1)(C₁+1)이고 색인은 n₀(C₁+1)+n₁로 잡는다. 순수 tensor product를 사용하며 모드를 바꾸어도 fermion의 점유 permutation 부호를 넣지 않는다. 생성·소멸의 √n 계수는 hopping 진폭에 함께 들어간다.

$$
a_0^\dagger a_1|n_0,n_1\rangle
=\sqrt{(n_0+1)n_1}\,|n_0+1,n_1-1\rangle
\quad\text{when the target lies within the cutoffs}.
$$

H=a₀†a₁+a₁†a₀를 |2,2〉에 적용하자. 두 cutoff가3이면 두 이동 경로가 모두 살아 있다. 둘째 모드 cutoff를2로 줄이면 |1,3〉 방향만 사라진다. 평균 에너지가0으로 같아도 분산은 이 차이를 드러낸다.

| 최대 점유(C₀,C₁) | H의 비영 출력 | 평균 E | Var(H) |
| --- | --- | --- | --- |
| (3,3) | √6\|3,1〉+√6\|1,3〉 | 0 | 12 |
| (3,2) | √6\|3,1〉 | 0 | 6 |

유한 실험은 onsite ε_i n_i, self-density U_i n_i(n_i−1)/2, cross-density Vn₀n₁, 복소 hopping과 f_i a_i†+f_i* a_i drive를 다룬다. 계수들은 에너지 단위다. drive가 없으면 이 모형의 전체 입자 수는 보존되고 drive는 이를 깨뜨릴 수 있다. 다만 진공이 한 순간 N=0이라는 사실과 H의 N 보존은 여전히 다른 질문이다.

## 8. Coherent 상태는 소멸 연산자의 고유상태다

무한 공간에서 a|α〉=α|α〉를 만족하는 정규화 상태를 구하자. α는 무차원 복소수다. 수 상태 계수를 d_n이라 하면 ladder 작용으로 계수 사이의 점화식이 생긴다. 전체 위상을 정해 d₀를 양수로 택한다.

$$
\sqrt{n+1}\,d_{n+1}=\alpha d_n,\qquad
|\alpha\rangle=e^{-\mu/2}\sum_{n=0}^\infty\frac{\alpha^n}{\sqrt{n!}}|n\rangle,\qquad \mu=|\alpha|^2.
$$

계수는 d_n=d₀αⁿ/sqrt(n!)다. 노름의 합이 |d₀|² exp(μ)이므로 d₀=exp(−μ/2)가 된다. α=0이면 진공이고, α≠0이면 무한히 많은 점유 방향이 일반적으로 필요하다. 복소 위상은 X와 P 평균의 방향을 정한다.

coherent 상태는 입자 수가 다른 성분의 중첩이다. Boson 모형에 앞 장의 fermion parity 가정을 그대로 부과하지 않는다. 실제 모드의 준비·관측과 phase reference는 해당 물리 문제에서 정해야 한다. 여기서는 위 무한 급수와 연산자 평균을 정의하고 비교한다.

## 9. Poisson 통계와 최소 불확정성을 유도한다

수 상태 측정 확률은 진폭의 절댓값 제곱이다. 급수의 색인을 하나 또는 둘 옮기면 n과 n(n−1)의 평균을 구할 수 있다. 두 번째 관계를 사용해 분산을 계산한다.

$$
p_n=e^{-\mu}\frac{\mu^n}{n!},\qquad
\langle\hat n\rangle=\mu,\quad
\langle\hat n(\hat n-1)\rangle=\mu^2,\quad
\operatorname{Var}(\hat n)=\mu.
$$

또한〈a〉=α,〈a²〉=α²,〈a†a〉=μ,〈aa†〉=μ+1을 X/P 식에 넣는다. 상태의 중심은 이동하지만 vacuum fluctuation은 변하지 않는다.

$$
\langle X\rangle=\sqrt2\operatorname{Re}\alpha,\qquad
\langle P\rangle=\sqrt2\operatorname{Im}\alpha,\qquad
\operatorname{Var}(X)=\operatorname{Var}(P)=\frac12,\qquad K=0.
$$

여기서 소멸 연산자의 고유상태라는 것이 에너지나 number의 고유상태라는 뜻은 아니다. H=ℏΩ(n̂+1/2)에서 α≠0이면 에너지 분산은(ℏΩ)²μ다. 비Hermitian 연산자 a의 고유상태 조건과 Hermitian H의 잔차 검증을 구별하자.

## 10. 무한 상태·비정규화 투영·재정규화를 분리한다

cutoff 안의 지수 인자 없는 계수를 z_C라 하고 그 노름을 Z_C라 쓰자. Z_C는 thermodynamic partition function으로 새로 가정한 양이 아니라 유한 급수의 노름이다. 같은 기호를 보았다고 물리적 의미를 섞지 않는다.

$$
z_C=\sum_{n=0}^{C}\frac{\alpha^n}{\sqrt{n!}}|n\rangle,\qquad
Z_C(\mu)=\sum_{n=0}^{C}\frac{\mu^n}{n!},\qquad
\Pi_C|\alpha\rangle=e^{-\mu/2}z_C,\qquad
|\psi_C\rangle=\frac{z_C}{\sqrt{Z_C}}.
$$

| 대상 | 노름 제곱 | 의미 |
| --- | --- | --- |
| 무한 \|α〉 | 1 | 원래 coherent 상태 |
| 비정규화 Π_C\|α〉 | S=e^(−μ) Z_C | cutoff 안에 남은 성분 |
| 재정규화 ψ_C | 1 | 조건부 유한 상태 |

재정규화는 버린 확률을0으로 만들지 않는다. 이상적인 projector 측정{Π_C,I−Π_C}에서 남은 결과를 선택한다면 S는 그 결과의 확률이고 ψ_C는 조건부 상태다. 그런 측정을 실제로 수행했다는 가정 없이도 근사 비교를 위해 이 두 벡터를 계산할 수 있다.

무한 공간에서는 displacement D(α)=exp(αa†−α*a)가 vacuum에서 coherent 상태를 만든다. 그러나 유한 D_C=exp(αa_C†−α*a_C)를 계산하는 것은 원래 급수의 투영과 다르다. C=1·α=1이면 D_C|0〉=cos(1)|0〉+sin(1)|1〉이지만 투영 후 재정규화는(|0〉+|1〉)/sqrt(2)다. 유한 CCR가 달라져 원래 normal-ordering 항등식을 그대로 쓸 수 없기 때문이다.

## 11. Tail과 거리에는 서로 다른 공식이 있다

남은 확률을 S, 버린 Poisson 확률을 ε라 쓰면 투영된 벡터는 원래 상태와 직교 분해된다. 재정규화한 상태의 overlap은 양의 √S다. 따라서 세 오차 지표를 구별할 수 있다.

$$
S=e^{-\mu}Z_C,\qquad \varepsilon=1-S,\qquad
\bigl\||\alpha\rangle-\Pi_C|\alpha\rangle\bigr\|^2=\varepsilon,\qquad
|\langle\alpha|\psi_C\rangle|^2=S,\qquad
\bigl\||\alpha\rangle-|\psi_C\rangle\bigr\|^2=2(1-\sqrt S).
$$

마지막 두 상태는 모두 정규화되어 있으며 fidelity **제곱**은 S다. 거리 제곱2(1−sqrt(S))를 버린 확률 ε와 같은 값이라고 표시하지 않는다. 작은 ε에서는 가까워도 정확한 정의가 다르다. cutoff 안의 조건부 확률은 μⁿ/(n!Z_C)이고 원래 확률 exp(−μ)μⁿ/n!와도 다르다.

Tail은 lower incomplete gamma로 표현할 수 있다. 아래 gamma는 one-body 행렬의 기호가 아니라 특수함수이며 upper와 방향을 바꾸지 않는다.

$$
\Pr(n>C)=\frac{\gamma(C+1,\mu)}{C!}
=1-e^{-\mu}Z_C(\mu),\qquad
\gamma(s,x)=\int_0^x t^{s-1}e^{-t}\,dt.
$$

두 독립 coherent 모드의 직사각형 cutoff에서는 S=S₀S₁이고 전체 tail은1−S₀S₁이다. 두 tail을 단순히 더하면 겹치는 사건을 두 번 센다. μ=10⁻¹²,C=3의 tail은 약 μ⁴/24=4.17×10⁻⁵⁰이며 양수다. 보통 소수 연산의1−CDF가0처럼 보이더라도 없는 확률이라고 단정하지 않는다. μ=16,C=3에서는 남은 확률이 약9.31×10⁻⁵에 불과해 매우 작은 유한 예제가 원래 상태를 잘 근사한다고 할 수 없다.

## 12. 유한 소멸 연산자는 같은 coherent 고유상태를 갖지 않는다

유한 a_C는 lowering을 C+1번 적용하면0이 된다. 따라서 nilpotent이고 고유값은 모두0이다. α≠0의 정확 고유상태를 찾으려 하면 a_C^(C+1)ψ=α^(C+1)ψ와 모순이다. 작은 잔차를 정확한 고유상태와 혼동하지 말자.

잘린 coherent 계수는 경계 아래에서 원래 점화식을 만족한다. 마지막 계수에서만 다음 방향이 없으므로 잔차가 남는다. p_top을 재정규화 상태의 top 확률로 사용하면 식이 간단하다.

$$
p_{\rm top}=\frac{\mu^C}{C!Z_C},\qquad
\|(a_C-\alpha)|\psi_C\rangle\|^2=\mu p_{\rm top},\qquad
\langle a_C\rangle=\alpha(1-p_{\rm top}),\qquad
\langle\hat n_C\rangle=\mu(1-p_{\rm top}).
$$

C=0에서는 ψ_C가 진공이고 p_top=1이다. 실제 a_C 평균과 number 평균은0인데 고유상태 잔차는μ다. α=0이면 모든 cutoff에서 진공이 원래 상태와 같고 tail과 잔차가0이다. 두 경우를 구별하면 재정규화가 물리적 정보를 자동 복원하는 연산이 아님을 볼 수 있다.

## 13. α=1·C=3에서 세 비교를 손으로 확인한다

이 예제의 정확 계수는[1,1,√2/2,√6/6]이고 Z₃=8/3이다. 조건부 top 확률은1/16, 평균 number와 a_C는15/16, 실제 교환자 기대는3/4다. 소멸 고유상태 잔차 제곱은1/16이다. 어느 것도 원래 무한 coherent 상태의 평균을 그대로 복사한 값이 아니다.

$$
\langle\hat n_C^2\rangle=\frac{27}{16},\qquad
\operatorname{Var}(\hat n_C)=\frac{207}{256},\qquad
\operatorname{Var}(X_C)=\frac{39}{128},\qquad
\operatorname{Var}(P_C)=\frac9{16},\quad K_C=0.
$$

이 상태를 무한 공간에 embed해서 원래 X,P를 관측하면6절의 경계 보정1/8이 두 분산에 더해진다. 원래 coherent 상태와도 다른 비교다. 표에서 **상태와 연산자**를 함께 읽어야 한다.

| 상태와 관측 연산자 | 평균 number | Var(X) | Var(P) | 〈[X,P]〉/i |
| --- | --- | --- | --- | --- |
| 원래 무한 coherent, 원래 X/P | 1 | 1/2 | 1/2 | 1 |
| 재정규화 ψ_C, 유한 X_C/P_C | 15/16 | 39/128 | 9/16 | 3/4 |
| embedded ψ_C, 원래 X/P | 15/16 | 55/128 | 11/16 | 1 |

두 번째 행의 분산 곱은351/2048이고 유한 경계9/64보다 크다. 세 번째 행의 곱은605/2048로 원래1/4보다 크다. 유한 곱이1/4 아래라는 이유만으로 불확정성 원리 위반을 주장하거나, 모든 낮은 X 분산을 같은 원인으로 해석해서는 안 된다.

## 14. 유한 바닥과 무한 공간의 안정성은 다르다

H=−n̂의 cutoff 행렬은 대각이고 최솟값−C를 갖는다. 더 큰 cutoff로 옮길 때 이 값은 계속 내려간다. 무한 모형은 아래로 유계가 아니므로 정규화 가능한 최저 에너지 상태가 존재하지 않는다. 한 유한 배열의 바닥을 정확히 찾았다는 사실은 무한 물리 모형이 안정적이라는 인증이 아니다.

반대로 적절한 정의역과 아래로 유계인 H의 variational 부분공간에서는 Rayleigh–Ritz 최소값을 상한으로 사용할 수 있다. 하지만 어떤 유한 행렬을 썼는지 중요하다. H_C=Π_C H Π_C의 형식인지, 잘린 X/P를 다시 조합한 다른 모형인지 구별해야 한다. cutoff 증가에 따른 수렴도 원래 연산자와 공간, 상태의 모멘트 가정 아래에서 확인해야 한다.

현재 실험은 대각 유한 H에서만 에너지를 열거하고 유한 바닥을 판정한다. 일반 비대각 바닥·무한 계·시간 진화·최적화를 계산한 것으로 해석하지 않는다. 원래 coherent 상태의 normally ordered 에너지 기대와 유한 상태 에너지를 비교할 수 있어도, 그것이 바닥 상태임을 뜻하지는 않는다.

## 15. 작은 tail이 모든 관측 오차를 보장하지는 않는다

정규화한 ψ_C와 원래 pure coherent 상태의 trace distance는√ε다. Bounded 관측 A에는 operator norm을 사용한 기대값 차이의 경계가 있다. 여기서 비교하는 것은 같은 A에 대한 두 상태이며, A를 잘라서 바꾼 효과는 별도로 다루어야 한다.

$$
\left|\langle A\rangle_\alpha-\langle A\rangle_{\psi_C}\right|
\le2\|A\|\sqrt\varepsilon\qquad\text{for bounded }A.
$$

Number처럼 unbounded한 관측에는 이 식을 유한 상수의 보장으로 그대로 쓸 수 없다. 원래 상태가 sqrt(1−ε)|0〉+sqrt(ε)|M〉이고 M>C라면 tail은ε로 같아도 평균 number 오차는εM로 임의로 커질 수 있다. 일반 상태의 작은 tail만 보고 energy·고차 모멘트까지 정확하다고 결론내리지 않는다.

Coherent 가족에서는 분포를 구체적으로 알기 때문에 mean number 차이를μp_top으로 계산할 수 있다. AI 학습에서 norm loss·number moments·energy·quadrature·operator projection의 오차를 나누어 기록하는 이유다. 원래 모델·state family·관측 집합과 cutoff 규칙이 다른 학습 문제의 점수를 바로 비교해서는 안 된다.

## 16. 경계와 상태를 구별하는 여덟 문제

**1. Ladder와 trace.** C=2에서 a_C†a_C·a_Ca_C†·교환자의 대각을 구하고 trace를 비교하라.

**해설.** 각각(0,1,2), (1,2,0), (1,1,−2)다. 교환자 trace는0이고 identity trace는3이다. 이는 반올림 오차가 아니라 유한 CCR가 원래 I와 다를 수밖에 없는 이유다.

**2. Top 불확정성.** C=3의 |3〉에서 유한 Var(X_C), Var(P_C), 수정된 경계를 구하라. 원래 X/P로 계산하면 어떻게 달라지는가?

**해설.** 두 유한 분산은3/2이고 공분산0이어서 곱9/4다. 기대 교환자는−3이라 경계도9/4다. 원래 분산에는 각각2를 더해7/2가 된다. 서로 다른 연산자를 같은 측정으로 부르지 않는다.

**3. 사라진 hopping 경로.** |2,2〉에서 cutoff(3,2)와(3,3)의 H 작용·평균·분산을 비교하라.

**해설.** 공통 경로는√6|3,1〉다. 둘째 cutoff가3일 때만√6|1,3〉도 남는다. 두 출력은 입력과 직교해 E=0이고, 출력 노름 제곱에 따라 분산은6과12다. 평균 하나만으로 경계 영향을 놓칠 수 있다.

**4. 무한 coherent와 유한 진공.** α=1·C=0에서 tail·재정규화 상태·소멸 고유상태 잔차를 구하라.

**해설.** Z₀=1, 남은 확률 exp(−1), tail은1−exp(−1)이다. 조건부 상태는 진공이지만 원래 coherent 상태는 아니다. a_C=0이므로 (a_C−1)|0〉의 제곱 노름은1이다. 진공을 얻었다고 버린 확률이 없어지지 않는다.

**5. α=1·C=3의 세 값.** p_top·평균 number·고유상태 잔차 제곱을 구하고, fidelity 제곱과 비정규화 오차도 구하라.

**해설.** Z₃=8/3이므로 p_top=1/16, 평균15/16, 잔차1/16이다. 남은 S=(8/3)exp(−1)가 fidelity 제곱이고 비정규화 오차는1−S다. 재정규화 거리는2(1−sqrt(S))로 따로 계산한다.

**6. 같은 진공, 다른 Hamiltonian.** C=0에서 투영된 조화 진동자 H_C/(ℏΩ)와 (X_C²+P_C²)/2의 에너지를 비교하라.

**해설.** 첫 번째는 n̂_C+1/2=1/2이지만 두 번째는0이다. 무한 공간에서 같던 식이 잘린 연산자의 곱에서는 달라졌다. 비교할 Hamiltonian을 먼저 정의해야 한다.

**7. 두 모드 tail.** 두 모드의 retained probability가 S₀,S₁이면 공동 tail을 구하라. 모드별 tail 합이 왜 다른가?

**해설.** 독립 product coherent 준비와 직사각형 cutoff에서 공동 retained는 S₀S₁, tail은1−S₀S₁이다. ε₀+ε₁−ε₀ε₁이며 단순 합은 둘 다 밖으로 나가는 사건을 두 번 센다. 상관된 일반 준비에는 독립 곱 공식을 자동 적용하지 않는다.

**8. 작은 norm 오차와 큰 number 오차.** sqrt(1−ε)|0〉+sqrt(ε)|M〉에서 M>C를 가정하라. cutoff 뒤 어떤 결론이 가능하며 어떤 결론은 불가능한가?

**해설.** tail과 비정규화 거리 제곱은ε다. 재정규화 상태는 진공이지만 원래 평균 number는εM이다. 고정한 작은ε에서도 M을 키우면 number 오차가 커진다. bounded 관측의 norm 경계를 unbounded number에 그대로 적용하지 않는다.

## 17. 정밀한 계산은 정의의 차이를 지우지 않는다

실험에서는 raw-state와 coherent 준비를 구별한다. Raw 모드는 비영 복소 원시 진폭을 직접 주고, coherent 모드는 유리수 복소 α에서 계수를 만든다. 최대 점유0…3의 ladder·hopping에 필요한 √2·√3·√6은 정확한 대수 계수로 유지할 수 있다. 이 작은 cutoff의 수 체계를 무한 factorial 급수 전체에 대한 단일 표현이라고 일반화하지 않는다.

무한 지수 인자·tail·정규화 좌표는 인증 구간으로 비교한다. 표시용 소수0과 정확한 영을 구별하고, 원래 확률·조건부 확률·top 점유·잔차의 정의를 확인하자. Coherent 입력의 모드별μ≤16은 이 계산기의 검증 범위이며 이론의 가능한α에 대한 제한은 아니다. cutoff를 바꿔 보는 조작과 실제 Hamiltonian 시간 진화도 서로 다르다.

먼저 진공·top·중간 점유로 경계 규칙을 예측하고, 다음에는α의 크기와 위상을 바꾸어 원래와 유한 평균을 비교하자. 마지막에는 상태를 고정하고 관측 연산자만 바꿔6절의 보정을 확인해 보자. 같은 그림을 보고도 무엇을 잘랐는지 설명할 수 있어야 한다.

## 18. 어떤 근사를 했는지 말할 수 있어야 한다

이제 boson의 무한 공간을 유한 배열로 다루는 데서 바뀌는 것이 무엇인지 구별할 수 있다. 생성 계수·경계 교환자·관측의 곱·조건부 상태·버린 확률은 서로 다른 항목이다. 낮은 에너지나 작은 loss 하나로 이 모든 조건을 대신하지 않는다. 이후 MPS는 점유 cutoff와 다른 종류의 표현 압축을 다루며, Schmidt 절단은 bipartition 가중치의 손실을 정량화한다.

추가 학습에는 [FAU/Max Planck의 oscillator·coherent·quadrature 연습](https://mpl.mpg.de/fileadmin/user_upload/Exercise_1.pdf), [Stuttgart의 coherent 급수 연습](https://www.itp3.uni-stuttgart.de/downloads/Adv._Quantum_Theory_Engl_ws11.12/ex9.pdf), [NIST DLMF의 incomplete gamma 항등식](https://dlmf.nist.gov/8.4)을 참고하자. 실제 물질·광학 모드·큰 계의 연구로 확장할 때는 연산자 정의역·수렴·관측과 자원의 조건을 다시 제시해야 한다.
