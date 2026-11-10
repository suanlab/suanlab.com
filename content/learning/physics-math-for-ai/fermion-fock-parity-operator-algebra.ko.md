## 1. 입자 번호 대신 모드의 점유를 묻는다

앞 장의 spin 위치는 구별되는 자유도였다. 동일한 fermion을 다룰 때는 입자1이 여기 있고 입자2가 저기 있다는 이름표를 물리적 정보로 사용할 수 없다. 대신 미리 고른 **한 입자 모드** 가운데 무엇이 점유되었는지 묻는다. 모드는 공간 궤도, 운동량, spin을 포함한 spin-orbital 등일 수 있다. 모드의 이름과 입자의 이름을 구별하는 것이 Fock 표현의 시작이다.

두 모드 함수 phi0와 phi1이 직교한다고 하자. 각 모드에 하나씩 놓인 동일 fermion의 두 입자 파동함수는 반대칭 조합이다. 아래 x는 위치뿐 아니라 필요한 spin 좌표까지 포함한 **전체 한 입자 좌표**다. 공간 부분만 언제나 반대칭이라고 주장하지 않는다.

$$
\Psi(x_1,x_2)=\frac{\phi_0(x_1)\phi_1(x_2)-\phi_1(x_1)\phi_0(x_2)}{\sqrt2},\qquad
\Psi(x_2,x_1)=-\Psi(x_1,x_2).
$$

좌표를 교환했을 때의 부호를 매번 긴 파동함수에 적는 대신, 생성·소멸 연산자의 대수에 담을 수 있다. 이것이 단순한 비트 압축보다 중요한 변화다. 전자의 spin이 두 값을 가진다는 사실과 한 fermion 모드의 점유가0/1이라는 사실은 서로 다른 두 자유도를 설명한다. 같은 공간 궤도의 up/down spin-orbital을 각각 모드로 잡으면 두 모드는 함께 점유될 수 있다.

이 장의 흐름은 **모드 → Fock 기저 → 연산 순서와 부호 → Hamiltonian → 보존량과 물리 가정 → 관측의 한계**다. 복소 내적과 Pauli 곱 공간을 알고 있다고 가정하되, 입자 수를 처음부터 고정하지는 않는다. 작은 계산은 최대 네 모드이며, 이론의 모드 수나 배울 분량의 제한은 아니다.

## 2. 진공과 영벡터는 다르다

선택한 한 입자 모드들은 서로 직교·정규화되어 있다고 가정한다. 정규화된 진공은 모든 모드가 비어 있는 상태다. 생성자 c_i†는 모드 i를 점유시키고 소멸자 c_i는 비운다. 모드 순서를0,1,…로 고정하여 다음 기저를 정의한다. 오른쪽 연산자가 먼저 작용한다는 규칙도 계속 유지한다.

$$
|n_0\cdots n_{m-1}\rangle=(c_0^\dagger)^{n_0}\cdots(c_{m-1}^\dagger)^{n_{m-1}}|\mathrm{vac}\rangle,\qquad
n_i\in\{0,1\},\qquad \dim\mathcal F=2^m.
$$

두 모드에서 |11〉=c₀†c₁†|vac〉이다. 이는 c₁†를 먼저 적용하고 c₀†를 나중에 적용하라는 뜻이다. 창조된 입자에 시간 순서의 물리적 이름표가 생겼다는 뜻은 아니다. 고정한 연산자 곱 convention이 좌표의 부호를 정한다.

입자 수 N을 고정한 부분공간의 차원은 m개 중 N개를 고르는 조합 수다. 전체 Fock 공간은 이 서로 다른 입자 수 공간을 모두 포함한다. 진공의 입자 수는0이지만 노름은1이다. 반면 c_i|vac〉=0의 오른쪽0은 영벡터로, 새로운 정규화 상태가 아니다. 비어 있는 모드를 소멸했을 때 결과를 진공으로 바꿔 버리면 연산자 대수가 달라진다.

## 3. 반교환관계가 점유 제한과 순서 부호를 만든다

Fermion 생성·소멸 연산자는 canonical anticommutation relations, 줄여서 CAR를 만족한다. 중괄호는 두 곱의 **합**이고 commutator의 차와 다르다. 같은 모드와 서로 다른 모드를 모두 포함한 행렬 항등식이다.

$$
\{c_i,c_j^\dagger\}=\delta_{ij}I,\qquad
\{c_i,c_j\}=\{c_i^\dagger,c_j^\dagger\}=0,\qquad
n_i=c_i^\dagger c_i,\qquad n_i^2=n_i.
$$

i=j인 두 번째 식은 c_i²=0과 생성자의 제곱0을 준다. 같은 모드를 두 번 점유시킬 수 없다는 Pauli 제한이다. n_i²를 계산할 때 c_i c_i†=I−n_i를 넣으면 n_i²=n_i가 되어 점유 고유값이0/1임을 확인한다. 서로 다른 n_i들은 교환하므로 공동 점유 기저를 사용할 수 있다.

두 모드 소멸자의 전체 작용을 보자. 기저 순서는00,01,10,11이며 n₀가 최상위 비트다. 표의0은 모두 영벡터다.

| 입력 | c₀의 출력 | c₁의 출력 |
| --- | --- | --- |
| \|00〉 | 0 | 0 |
| \|01〉 | 0 | \|00〉 |
| \|10〉 | \|00〉 | 0 |
| \|11〉 | \|01〉 | −\|10〉 |

마지막 칸의−가 두 독립 비트 ladder와 다른 점이다. c₁이 앞쪽 c₀†를 지나가려면 한 번 반교환해야 한다. 생성 순서를 바꾸어도 같은 차이가 생겨 c₁†c₀†|vac〉=−|11〉이다. 두 경로의 부호를 지워서 같게 만드는 것은 단순한 표시 정리가 아니라 모형을 바꾸는 일이다.

## 4. 점유 앞부분의 parity를 세면 부호를 계산할 수 있다

일반 모드 i의 소멸은 그 모드가 비어 있으면0이다. 점유되어 있다면 앞선 모드에 몇 개의 fermion이 있는지 센다. 생성자도 반대 점유 조건에서 같은 앞부분 부호를 갖는다.

$$
c_i|n_0\cdots n_i\cdots\rangle=
\begin{cases}
0,&n_i=0,\\
(-1)^{\sum_{j<i}n_j}|n_0\cdots0_i\cdots\rangle,&n_i=1.
\end{cases}
$$

예를 들어 c₂|011〉=−|010〉이다. 앞선 두 점유의 합이1이기 때문이다. c₀†c₂|011〉=−|110〉인 반면 c₂c₀†|011〉=+|110〉이므로 두 작용의 합은0이다. 부호는 선택한 모드 순서에 맞추어 계산하며, 입자가 실제 공간에서 어떤 궤적을 따라 지나갔다는 그림은 필요하지 않다.

이를 qubit 행렬로 쓰는 한 방법이 Jordan–Wigner 표현이다. a는 한 비트의 lowering 행렬이며, 앞선 Z들이 점유 parity를 전달한다.

$$
a=\frac{X+iY}{2}=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
c_i=\left(\prod_{j<i}Z_j\right)a_i,\qquad
c_0=a\otimes I,\quad c_1=Z\otimes a.
$$

Z string을 빼면 서로 다른 모드의 행렬들이 교환한다. 예를 들어 naive 소멸0·생성1의 반교환자를 |10〉에 적용하면2|01〉가 되지만, 올바른 두 경로는−|01〉와+|01〉로 상쇄된다. 행렬 크기가2^m으로 같다는 사실만으로 fermion 대수가 구현된 것은 아니다. 비트 인코딩과 물리 연산자 대응을 함께 확인해야 한다.

## 5. Hamiltonian의 adjoint는 곱의 순서를 뒤집는다

다음 even Hamiltonian을 사용하자. ε₀는 에너지 원점, μ_i는 이 장에서의 onsite **에너지 계수**, h는 hopping, Δ는 pairing, U는 밀도 결합이다. μ 기호를 썼다고 자동으로 열역학적 chemical potential과 같은 부호 convention을 뜻하지 않는다. h와 Δ는 복소수이고 나머지는 실수다.

$$
\begin{aligned}
H={}&\epsilon_0I+\sum_i\mu_i n_i+\sum_{i<j}U_{ij}n_in_j\\
&+\sum_{i<j}\left(h_{ij}c_i^\dagger c_j+h_{ij}^*c_j^\dagger c_i
+\Delta_{ij}c_i^\dagger c_j^\dagger+\Delta_{ij}^*c_jc_i\right).
\end{aligned}
$$

각 항은 생성·소멸 연산자를 짝수 개 포함한다. 한 번의 hopping은 하나를 비우고 다른 모드를 채우므로 입자 수를 유지한다. pairing은 둘을 함께 생성하거나 함께 소멸시킨다. 밀도 결합은 점유 확률에 따라 에너지를 부여한다. 결합을 지정하는 것과 준비 상태를 진화시키는 것은 다르다.

특히 (c_i†c_j†)†=c_jc_i다. c_ic_j로 쓰면 반교환 부호가 하나 더 생긴다. 복소 계수의 켤레만 취하고 연산자 순서를 그대로 두면 Hermitian Hamiltonian이 되지 않을 수 있다. 이 부호를 외우기보다 일반 규칙 (AB)†=B†A†를 먼저 적용하고, 필요할 때만 CAR로 순서를 정리하자.

일반적인 네 모드 산란과 홀수 source는 이 유한 실험의 입력 범위에 없다. pairing을 가진 유효 모형은 reservoir와의 pair 교환이나 평균장 기술 등에서 나타날 수 있지만, 이를 폐쇄된 전체 우주의 입자 수가 근본적으로 보존되지 않는다는 주장으로 확대하지 않는다. 무엇을 계로 포함했는지도 물리 모형의 일부다.

## 6. Hopping의 위상과 중간 모드의 부호를 계산한다

두 모드에서 H=c₀†c₁+c₁†c₀를 보자. 한 입자 기저 |01〉,|10〉 사이를 계수1로 교환하고 진공과11에는0을 준다. 따라서 다음 두 상태의 점유 확률은 같아도 에너지는 반대다.

$$
|\psi_\pm\rangle=\frac{|01\rangle\pm|10\rangle}{\sqrt2},\qquad
E_\pm=\pm1,\qquad
H=\frac{X_0X_1+Y_0Y_1}{2}.
$$

앞 장의 XX+YY와는1/2 계수가 다르다. 같은 교환 모양을 보고 에너지를 그대로 복사하면 안 된다. h=i로 바꾸면 한 입자 block은 Y 행렬이고 (|01〉+i|10〉)/sqrt(2)의 에너지는+1이다. 이 경우도 상대 위상이 에너지를 결정한다.

세 모드에서0과2 사이를 hop하면 중간 점유의 부호가 보인다. 모드 순서는0,1,2이고 H₀₂=c₀†c₂+c₂†c₀라 하자.

$$
c_0^\dagger c_2|011\rangle=-|110\rangle,\qquad
|\chi\rangle=\frac{|011\rangle+|110\rangle}{\sqrt2},\qquad
\langle\chi|H_{02}|\chi\rangle=-1.
$$

parity string을 버린 모형은 같은 좌표 배열에서+1을 준다. 두 모형은 동일한 상태 이름을 사용해도 연산자가 달라졌다. 모드 기저를 재정의하여 부호를 옮기는 것은 가능하지만 상태와 모든 Hamiltonian 계수도 함께 변환해야 한다. 잘못된 CAR를 상태 이름 변경만으로 정당화할 수는 없다.

## 7. Pairing은 입자 수가 다른 even 상태를 연결한다

두 모드의 T=c₀†c₁†는 진공을11로 바꾼다. adjoint T†=c₁c₀는11을 진공으로 되돌린다. 이 두 상태가 만드는 even block에서 실수 Δ=1의 pairing Hamiltonian은 X와 같다.

$$
H_{\rm pair}=T+T^\dagger,\qquad
|e_\pm\rangle=\frac{|00\rangle\pm|11\rangle}{\sqrt2},\qquad
E_\pm=\pm1.
$$

두 상태는 even parity를 가지지만 입자 수는0과2의 중첩이다. 평균 입자 수는1, 분산은1이다. 입자 수가 확정되지 않아도 parity는 확정될 수 있다. e_+의 에너지 분산은0이지만 e_-보다 에너지가 높으므로 영분산만으로 바닥 상태라 할 수 없다. 일반 비대각 입력의 solver를 제공하지 않는다는 점과 이 작은 block의 해석적 결론을 구별하자.

Δ=i에서는 진공+i|11〉의 정규화 상태가 에너지+1을 갖는다. 항별 에너지를 계산하면 복소 켤레와 순서가 실제 값으로 확인된다.

$$
|e_i\rangle=\frac{|00\rangle+i|11\rangle}{\sqrt2},\qquad
\langle c_0^\dagger c_1^\dagger\rangle=-\frac{i}{2},\qquad
2\operatorname{Re}\!\left(i\langle c_0^\dagger c_1^\dagger\rangle\right)=1.
$$

이 예제에서는 생성 pair의 평균과 소멸 pair의 평균을 같은 순서라고 착각해도 일부 실수 값이 우연히 맞을 수 있다. 일반 복소 Δ와 상태를 함께 검산해야 한다. T†의 순서를 뒤집는 규칙과 기대값의 켤레 규칙을 각각 확인하는 습관이 중요하다.

## 8. 입자 수 보존·parity 보존·sharpness는 세 질문이다

전체 입자 수와 parity를 정의하자. n_i가0/1이므로 parity는 even 점유에는+1, odd 점유에는−1이다. 양자역학에서 말하는 보존은 현재 상태의 분산이 아니라 전체 Hamiltonian과의 교환자에 관한 말이다.

$$
\widehat N=\sum_i n_i,\qquad P=(-1)^{\widehat N},\qquad
[\widehat N,c_i^\dagger c_j^\dagger]=2c_i^\dagger c_j^\dagger,\qquad
[H,P]=0\quad\text{for the even Hamiltonian above}.
$$

| 모형 / 준비 상태 | N 보존 | 현재 N 분산 | 현재 parity |
| --- | --- | --- | --- |
| pairing / 진공 | 아니오 | 0 | even |
| pairing / e_+ | 아니오 | 1 | even |
| onsite·hopping / e_+ | 예 | 1 | even |

진공이 한 순간 N=0이라고 해서 pairing Hamiltonian이 N을 보존하지는 않는다. 반대로 N을 보존하는 모형에서도 입자 수가 다른 even 성분을 처음부터 중첩해 준비할 수 있다. 보존된다는 것은 그 입자 수 분포가 Hamiltonian 진화 아래 변하지 않는다는 의미이지, 분포가 반드시 하나의 값이라는 의미는 아니다.

짝수 연산자만 포함한 현재 H는 parity를 보존한다. 그러나 이것만으로 모든 비영 입력 벡터가 물리적 순수 상태로 허용된다는 결론은 나오지 않는다. 동역학의 보존 법칙과 상태·관측 대수에 부과한 물리 가정을 다음 두 절에서 나누어 살펴보자.

## 9. Parity superselection 가정을 명시한다

이 장에서는 fermion parity superselection을 물리 가정으로 채택한다. 물리 density는 P와 교환하고, 허용되는 물리 관측은 parity-even 대수에 속한다고 정한다. 이 가정 아래 순수 상태는 definite even 또는 definite odd parity에 놓인다. 이 문장은 모든 상태의 입자 수를 하나로 고정한다는 뜻이 아니다.

$$
[\rho,P]=0,\qquad
\Pi_e=\frac{I+P}{2},\quad\Pi_o=\frac{I-P}{2},\qquad
\rho=|\psi\rangle\langle\psi|\ \Longrightarrow
\Pi_e|\psi\rangle=|\psi\rangle\ \text{or}\ \Pi_o|\psi\rangle=|\psi\rangle.
$$

even 내부의 진공+11 중첩은 이 parity 조건과 양립한다. 반면 한 모드의 (|0〉+|1〉)/sqrt(2)는 even과 odd를 잇는 순수 coherence라서 이 가정 아래의 물리적 순수 준비로 제시하지 않는다. 두 parity sector를 확률적으로 섞은 block-diagonal 혼합 density는 허용될 수 있다. 순수 상태에 대한 definite parity 조건을 혼합 상태 전체의 definite parity 조건으로 바꾸지 말자.

c_i나 c_i+c_i†는 parity-odd다. 형식적 계산에서 그 평균을 구할 수 있어도 이 물리 가정의 직접 허용 관측량으로 취급하지 않는다. 실제 계의 선택, 조작과 관측 대수를 정하지 않은 채 아무 벡터나 물리 상태라고 표시하는 것을 피하기 위한 구별이다. 입자 수 superselection이라는 추가 가정을 별도로 선택할 수는 있지만, 현재의 parity 가정과 동일한 말은 아니다.

## 10. Twirl은 비교 상태이며 입력의 자동 수정이 아니다

임의의 형식적 density에서 even–odd coherence를 제거한 비교 상태를 만들 수 있다. 이를 parity twirl이라 부르자. 원래 입력과 비교 상태를 함께 보고 무엇이 달라졌는지 확인한다.

$$
\bar\rho=\frac{\rho+P\rho P}{2}=\Pi_e\rho\Pi_e+\Pi_o\rho\Pi_o,\qquad
\|\rho-\bar\rho\|_F^2=2p_ep_o\quad\text{for a pure input},
\qquad p_e=\langle\Pi_e\rangle,\quad p_o=\langle\Pi_o\rangle.
$$

순수 벡터를 even·odd 성분으로 나누면 off-diagonal 두 block은 서로의 adjoint이고, 각각의 제곱 길이는 p_ep_o다. 그래서 제거한 전체 coherence가2p_ep_o가 된다. 두 성분 중 하나의 무게가0일 때만 순수 상태가 block-diagonal이다.

한 모드의 형식적 plus 상태에서는 p_e=p_o=1/2, 제거한 coherence의 Frobenius 제곱이1/2다. 원래 density의 purity는1이고 twirl 뒤에는1/2다. H=n의 평균 에너지는 둘 다1/2이며 odd 평균은 원래〈c〉=1/2에서 twirl 뒤0이 된다. 모든 even 관측 A는 Tr(ρA)=Tr(ρ̄A)를 만족한다. trace의 순환성과 PAP=A를 쓰면 이 등식이 바로 나온다. 이 사실이 두 density를 같은 순수 상태로 만드는 것은 아니다.

또한 모든〈c_i〉가0이어도 [ρ,P]=0이라고 결론낼 수 없다. 세 모드의 (|000〉+|111〉)/sqrt(2)는 입자 수 차이가3이어서 한 번의 소멸로 두 성분을 연결하지 못하므로 일차 odd 평균은 전부0이다. 그래도 even–odd coherence와2p_ep_o=1/2는 남는다. 몇 개의 평균값이0이라는 것과 물리 가정을 검사한 것은 다르다.

형식적 반례 모드는 이 차이를 배우기 위한 계산 허용이다. 선택한 even/odd 순수 sector를 벗어나는 진폭을 지우고 재정규화하거나, formal 입력을 twirl로 몰래 바꾸지 않는다. 준비 가정이 다른 상태는 다른 상태로 명시해야 한다.

## 11. One-body와 anomalous 평균이 알려주는 것과 빠뜨리는 것

계산할 때는 먼저 색인 convention을 적자. 여기서는 gamma_ij=〈c_i†c_j〉이며, 일부 문헌의 반대 색인 정의와 전치 관계일 수 있다. F는 두 소멸 연산자의 anomalous 평균이다.

$$
\gamma_{ij}=\langle c_i^\dagger c_j\rangle,\qquad F_{ij}=\langle c_ic_j\rangle,
\qquad \gamma^\dagger=\gamma,\quad F^T=-F,\quad
\langle c_i^\dagger c_j^\dagger\rangle=F_{ji}^*.
$$

Tr(gamma)는 평균 입자 수다. CAR는 gamma와 I−gamma의 비음성을 주어 각 one-body 점유가0…1 범위에 놓이게 한다. 이를 보려면 d=Σvᵢcᵢ를 만들자. v†gamma v는〈d†d〉이고, v†(I−gamma)v는 CAR를 이용하면〈dd†〉여서 둘 다 비음이다. 이것은 상관된 상태의 gamma가 항상 projector라는 뜻이 아니다. hopping의 i<j 항에 대한 에너지는 **2Re(h_ij gamma_ij)**다. 같은 색인의 h와 gamma를 그대로 Tr(h gamma)라고 쓰면 전치를 놓칠 수 있다. 복소 h=i의 한 입자 예제에서 gamma01=−i/2를 넣으면 에너지+1이다.

One-body 평균 전체와 F를 함께 알아도 일반적인 전체 상태나 입자 수 분포를 결정하지 못한다. 네 모드의 다음 두 pure even 상태를 비교하자.

$$
|u\rangle=\frac{|0011\rangle+|1100\rangle}{\sqrt2},\qquad
|v\rangle=\frac{|0000\rangle+|1111\rangle}{\sqrt2},\qquad
\gamma_u=\gamma_v=\frac I2,\quad F_u=F_v=0.
$$

| 상태 | 평균 N | Var(N) | 입자 수 분포 |
| --- | --- | --- | --- |
| u | 2 | 0 | N=2에 확률1 |
| v | 2 | 4 | N=0·4에 각각1/2 |

두 상태 모두 각 모드 점유가1/2이고 두 입자 소멸 평균은0이지만, 전체 입자 수의 이차 통계는 다르다. 낮은 차수의 관측을 정확히 맞추는 학습 모형이라도 그 정보만으로 전체 상태를 복원했다는 주장을 해서는 안 된다. 어떤 관측 집합이 목표 연구에 충분한지 별도로 정해야 한다.

## 12. 모드 재배열과 subsystem 해석을 함께 바꾼다

모드 순서는 물리 법칙이 아니지만 Fock 기저의 정의에는 들어간다. 이름을 재배열하면 점유한 생성자들의 순서를 원래 convention으로 다시 정리하는 데 필요한 반교환 횟수만큼 부호가 생긴다. 같은 두 점유 모드를 교환하면 c₀†c₁†가 c₁†c₀†로 바뀌어−가 붙는다. 단순히 비트 위치만 교환하는 것과 다르다.

진공+두 점유 중첩에서는 이 변화가 두 성분 사이의 상대 부호로 보일 수 있다. Hamiltonian의 pairing 계수와 상태를 함께 변환하면 물리 에너지는 유지된다. 한쪽만 바꾼 뒤 에너지가 달라졌다고 모드 순서가 물리적 힘을 만든다고 해석하지 않는다. 표현 좌표의 변환과 실제 상태 조작을 구별하자.

또한 qubit 좌표가 있다는 이유만으로 보통 qubit partial trace의 결과를 fermionic mode 얽힘으로 바로 부르면 안 된다. 모드 부분집합의 연산자 embedding과 parity-even 관측의 정의가 필요하다. 이 단원은 전체 density와 명시한 연산자의 평균을 다루며, 일반 fermion subsystem 얽힘 계산을 제공하지 않는다. 뒤 MPS·Schmidt의 보통 tensor-product 결과를 fermion에 옮길 때도 이 추가 구조를 유지해야 한다.

## 13. 전체 Fock 바닥과 선택 sector의 바닥을 구별한다

한 모드의 H=−n을 생각하자. 진공은 even이고 에너지0이다. 점유 상태는 odd이고 에너지−1이다. 어떤 공간에서 최소화하는지에 따라 정답이 달라진다.

$$
E_{\min}^{\mathcal F}=-1,\qquad
E_{\min}^{e}=0,\qquad E_{\min}^{o}=-1.
$$

| 비교 공간 | 최소 에너지 | 진공은 그 공간의 바닥인가 |
| --- | --- | --- |
| 전체 Fock 공간 | −1 | 아니오 |
| even sector | 0 | 예 |
| odd sector | −1 | 진공이 그 공간에 속하지 않음 |

parity 제약이나 고정 입자 수 제약을 두면 최적화 문제의 허용 집합이 바뀐다. 전체 최솟값을 계산한 뒤 선택 sector의 바닥이라고 이름만 바꾸어서는 안 된다. 각 projector의 무게도 전체 입력에 대한 확률인지 sector 안에서 조건부로 재정규화한 확률인지 명시해야 한다.

유한 실험은 H가 점유 기저에서 대각일 때만 전체 에너지와 parity·고정 N 부분공간의 최솟값을 열거한다. 일반 비대각 H의 바닥 판정은 계산하지 않은 값이다. 알려진 두 모드 pairing block의 해석적 바닥과 임의 입력의 solver를 구별하자. 상수 H에는 다음 서로 다른 준위가 없어 양의 gap이 정의되지 않는다.

## 14. 에너지 잔차를 물리 가정과 함께 점검한다

비영 원시 복소 벡터 z를 유지하면 모든 물리 기대는 노름으로 나누어 계산한다. 분산은 현재 H에서 고유상태인지 검증하지만, 물리적 parity 가정이나 바닥 공간의 선택을 대신하지 않는다.

$$
N_z=z^\dagger z>0,\qquad E=\frac{z^\dagger Hz}{N_z},\qquad
\operatorname{Var}(H)=\frac{\|(H-E)z\|^2}{N_z}.
$$

이 단원의 실험은1…4 ordered mode, onsite·밀도 결합·일반 복소 hopping/pairing과 순수 상태를 다룬다. 모든 CAR와 실제 H의 N/P 교환자, gamma/F와 항별 에너지 계정, formal parity coherence·twirl 비교를 확인한다. 영 입력은 상태가 아니므로 거부하지만 Pauli exclusion의 영 **출력**은 유효한 연산 결과로 남긴다. 연산 후 노름 비율을 측정 instrument 없이 관측 확률로 부르지도 않는다.

새 AI 연구에서는 같은 점유 표를 쓰는지보다 대수·보존량·sector·관측 집합이 일치하는지가 중요하다. 신경망이나 tensor 표현으로 상태를 근사할 때는 에너지·잔차뿐 아니라 복소 위상, 순서 부호, N 분포와 선택한 물리 가정을 검증하자. 학습 데이터의 one-body 통계를 맞춘 것과 fermionic 전체 상태를 맞춘 것은 다른 주장이다. 작은 정확 행렬은 이를 점검할 기준이지만 큰 계의 수렴·복잡도나 최적화 성공을 자동 보장하지 않는다.

## 15. 순서와 가정을 확인하는 여덟 문제

**1. 진공과 exclusion.** c₀†c₁†|vac〉, c₁†c₀†|vac〉, (c₀†)²|vac〉를 비교하라. 마지막 결과가 정규화된 진공인가?

**해설.** 고정한 기저에서 각각|11〉,−|11〉,0이다. 마지막0은 노름0의 영벡터이며 진공의 노름1과 다르다. 두 번째 생성자가 같은 모드를 다시 채울 수 없다는 연산 결과다.

**2. CAR의 실제 경로.** c₀c₁†와 c₁†c₀를 |10〉에 각각 적용하라. parity string을 뺀 naive 모형과 비교하라.

**해설.** 첫 경로는 c₁†가−|11〉를 만든 뒤 c₀가−|01〉를 준다. 두 번째는 c₀가 진공을 만든 뒤 c₁†가+|01〉를 준다. 합은0이다. naive 모형은 둘 다+|01〉여서 반교환자의 결과가2|01〉가 된다.

**3. 긴 hopping의 에너지.** H₀₂와 χ에 대해6절의−1을 직접 구하라. 현재 입자 수와 parity도 계산하라.

**해설.** 두 기저 성분 사이의 이동이 각각−부호를 주므로 H₀₂χ=−χ다. 평균 에너지는−1이고 에너지 분산은0이다. 두 성분 모두 입자가2개이므로 N=2, Var(N)=0, parity는 even이다. 계산 범위가 비대각 전체 spectrum을 제공한다는 뜻은 아니다.

**4. 복소 pairing adjoint.** Δ=i의 even 상태 e_i에서 생성 pair 평균과 pairing 에너지를 구하라. 왜 adjoint의 연산자 순서를 뒤집어야 하는가?

**해설.** T는 진공 진폭을11로 옮기므로〈T〉=conj(i)/2=−i/2다. 2Re(i〈T〉)=1이다. T†는 c₁c₀이며 c₀c₁=−c₁c₀다. 이−를 놓치면 Hermitian conjugate 항을 잘못 조립한다.

**5. 보존과 sharpness.** 진공을 H_pair 아래 평가했을 때 N 분산0인데도 N 비보존인 이유를 설명하라. e_+의 parity와 N 분산은?

**해설.** 진공 자체는 N=0 고유상태지만 H_pair가 이를 두 입자 성분으로 보낼 수 있어 [H_pair,N]이 비영이다. e_+는 even parity 고유상태이며 N 평균1·분산1이다. N 비보존, parity 보존, 현재 분포의 확정성은 서로 다른 질문이다.

**6. Formal coherence의 비교.** 한 모드의 plus 상태와 그 parity twirl에서 purity·H=n의 평균·〈c〉를 비교하라. 세 모드 cat의 모든〈c_i〉=0이면 parity 가정을 만족하는가?

**해설.** purity는1과1/2, 에너지는 둘 다1/2, odd 평균은1/2와0이다. twirl은 다른 혼합 상태다. 세 모드 cat은 일차 평균이0이어도 even–odd coherence 제곱1/2가 남아 가정을 만족하지 않는다. odd 평균을 물리 관측량으로 허용한다고 해석하지 않는다.

**7. 같은 gamma와 F.** u와 v의 N 이차 모멘트·분산을 구하라. 왜 one-body 복원이 전체 상태 복원이 아닌가?

**해설.** u는 N=2가 확정되어〈N²〉=4·분산0이다. v는 N=0과4가 반반이어서〈N²〉=8·분산4다. 둘 다 평균2·gamma=I/2·F=0이지만 고차 점유 통계가 다르다. 지정한 평균 집합만으로 모든 관측이나 상태가 정해지지는 않는다.

**8. 어떤 바닥을 찾았는가?** H=−n의 진공은 영분산이다. 전체 Fock 공간과 even sector에서 바닥 여부를 각각 판정하라.

**해설.** 진공의 E=0·분산0은 고유상태임을 말한다. 전체 공간에는 E=−1인 odd 점유 상태가 있어 전체 바닥이 아니다. even 공간은 진공 하나뿐이어서 그 sector의 바닥이다. 선택한 허용 공간을 보고 결론을 붙여야 한다.

## 16. 부호 대수에서 다음 표현으로 넘어간다

결과를 설명할 때 모드가 무엇인지, 기저의 생성 순서가 무엇인지, 상태가 어느 sector에 있는지를 먼저 말해 보자. 다음에는 실제 연산 경로의 부호·복소 adjoint·H와 N/P의 교환자·입력의 parity 가정·관측의 범위를 구별하자. 동일한 비트 배열, 같은 평균 에너지 또는 같은 gamma가 같은 물리 문제를 보장하지 않는다는 점이 이 장의 핵심이다.

다음 boson 단원은 점유수가0/1로 제한되지 않을 때 ladder의 √n 계수와 무한 공간의 cutoff를 다룬다. 그 뒤 MPS와 Schmidt는 표현의 수축·gauge·절단을 설명한다. Fermion 통계를 tensor network에 옮기려면 지금의 순서·대수·물리 관측 구조를 함께 유지해야 한다. 표현이 빨라졌다는 이유로 이 조건들을 생략할 수는 없다.

추가 학습에는 [공식 Jordan–Wigner 설명](https://quantumai.google/openfermion/tutorials/jordan_wigner_and_bravyi_kitaev_transforms)과 [Szalay 등의 fermion 정보 리뷰](https://arxiv.org/html/2006.03087v4)를 참고하자. 일반 subsystem embedding·fermionic partial trace·허용 연산과 얽힘의 연구는 이 입문 대수에서 더 나아가 별도의 정의와 검증을 요구한다.
