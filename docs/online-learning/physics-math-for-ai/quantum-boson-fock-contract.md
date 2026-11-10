# Boson cutoff의 정확 대수·입출력 계약

2026-11-09. 예정 단원 `boson-fock-cutoff-boundary-algebra`의 **원시 상태 수치 기반**이다. 이론의 무한 점유 공간과 유한 계산기의 역할을 구별한다. coherent 상태 builder와 Poisson tail 인증, 한영 원고·Worker·교수 화면은 후속 관문이다.

## 공간·기저·입력

계산기는 1…2개 순서 있는 boson 모드와 각 최대 점유 Cᵢ=0…3을 받는다. 국소 차원은 Cᵢ+1이고 전체 차원은 그 곱(1…16)이다. 두 모드의 색인은 n₀(C₁+1)+n₁이며 n₀를 먼저 적는다. 서로 다른 모드는 보통 tensor product로 결합한다. fermion의 점유 permutation 부호를 가져오지 않는다.

`learning-boson-fock-input.ts`는 수치 엔진을 가져오지 않는 별도 입력 모듈이다. 정확히 52개 소유 정수, 절댓값 10⁶ 이하, 양의 공통 Hamiltonian 분모, 비영 Gaussian-integer 진폭을 요구한다. 비활성 모드·차원 밖 진폭은 0이다. 잘못된 probe·cutoff, accessor·추가 키·prototype·reflection을 거부한다. descriptor에서 canonical +0을 만들고 동결하며 caller getter를 실행하지 않는다.

원시 상태 z와 노름 N=z†z를 보존하고 관측은 N으로 나눈다. 유효 입력의 N은 양의 정수다. 진공은 비영 벡터이며, a|0〉=0이라는 연산 결과를 입력 상태나 재정규화할 상태로 바꾸지 않는다.

## √n 계수와 유한 교환자

정규화 기저에서 aᵢ|…nᵢ…〉=√nᵢ|…nᵢ−1…〉이고 생성자는 그 adjoint다. 유한 투영에서는 aᵢ†|…Cᵢ…〉=0이다. 따라서

[aᵢ,aᵢ†] = I − (Cᵢ+1)Pᵢ,top,

〈[aᵢ,aᵢ†]〉 = 1 − (Cᵢ+1)pᵢ,top.

서로 다른 모드의 교환자는 0이다. 같은 모드의 교환자는 trace가 0이며 유한 차원에서 I와 같을 수 없다. C=0에서도 식을 유지한다. 그때 a=a†=0, P_top=I다. top 점유 상태의 교환자 기대는 −C이며, C=3의 (|0〉+|3〉)/√2에서는 −1이다. 이를 계산 오차로 보정하지 않는다.

각 모드 쌍의 실제 행렬, 위 projected identity의 잔차, 무한 CCR과의 차이, annihilation/creation 교환자와 trace를 별도로 반환한다. selected probe는 L Rz·R Lz·차이와 원래 정규화 입력에 대한 작용·노름 비율을 반환한다. 결과를 자동 재정규화하거나 측정 instrument 없는 확률로 부르지 않는다.

## 정확 수 체계와 표시 구간

`learning-boson-algebraic.ts`는 Q(√2,√3)의 [1,√2,√3,√6] 좌표를 유리수로 유지한다. 합·곱·나눗셈·켤레와 **정확 부호**를 제공한다. 나눗셈은 √3, 이어 √2에 대한 대수적 conjugation으로 유리수 분모를 만든다. 부호는 Q(√2) 부호와 u²−3v² 비교로 결정하며 Number나 허용오차를 쓰지 않는다.

출력 real은 네 정확 `coefficients`, `exactSign`, 표시용 `observation`을 포함한다. √2·√3·√6은 정수 제곱근으로 바깥 구간을 만든다. 기본 512비트에 계수 크기에 따른 guard precision을 더해 큰 상쇄에서도 절대 정밀도를 유지한다. 원시 상태·잔차를 √N으로 나눈 좌표는 별도 구간이다. 이 좌표를 정확한 대수 계수인 것처럼 표시하지 않는다. 음수·영·축퇴·대각 여부는 정확 계수로 결정한다.

## Hamiltonian과 관측

분모로 나눈 계수로 다음 유한 투영 Hamiltonian을 조립한다.

H = e₀I + Σᵢ[ωᵢnᵢ + Uᵢnᵢ(nᵢ−1)/2 + fᵢaᵢ† + fᵢ*aᵢ] + h a₀†a₁ + h*a₁†a₀ + Vn₀n₁.

h·f는 복소수, 나머지는 실수다. 두 모드 항은 한 모드에서 비활성이다. H·Hz·E·E₂·분산·잔차·항별 에너지, 전체 ρ·국소 reduced density/purity·γᵢⱼ=〈aᵢ†aⱼ〉·평균 점유·전체 입자 수 분포를 반환한다. N̂ 보존의 실제 [H,N̂]와 현재 상태가 sharp인지를 구별한다. drive는 N̂를 보존하지 않을 수 있지만 상태를 자동 진화시키지 않는다.

무차원 X=(a+a†)/√2, P=(a−a†)/(i√2)를 사용한다. 평균·분산·symmetrized covariance와 Robertson–Schrödinger determinant를 계산한다. 경계는 |〈[a,a†]〉|²/4를 사용하며 무조건 1/4로 고정하지 않는다. [FAU/Max Planck 연습 자료](https://mpl.mpg.de/fileadmin/user_upload/Exercise_1.pdf)의 연산자·quadrature convention을 대조했다. 유한 경계식과 구체적인 반례는 위 투영 행렬에서 직접 유도·검산한다.

## 유한 스펙트럼의 의미

점유 기저에서 H가 대각일 때만 에너지를 열거하고 축퇴 공간·바닥 확률·gap·누출 경계를 준다. 상수 H의 gap은 null이며 비대각 H의 바닥 판정은 null이다. 영분산은 고유상태만 인증한다. 모든 바닥 판정의 범위는 **현재 유한 투영 공간**이다. 예를 들어 H=−n의 유한 바닥 −C는 무한 공간에 바닥이 있다는 근거가 아니다. cutoff 증가에 따른 수렴, 물리 단위 복원, 일반 무한 연산자의 정의역은 별도 이론·검증이 필요하다.

## 남은 coherent 준비 관문

coherent 상태는 [Stuttgart 대학 자료](https://www.itp3.uni-stuttgart.de/downloads/Adv._Quantum_Theory_Engl_ws11.12/ex9.pdf)의 무한 급수 αⁿ/√n!·exp(−|α|²/2)와 대조한다. 후속 builder는 유리수 α와 0≤μ=|α|²≤16을 정확 검사하고, 무한 상태·절단된 비정규화 상태·재정규화 상태를 구별해야 한다. 버린 Poisson 확률과 그 구간 인증을 제공하기 전에는 이 단원의 수치 가족 전체를 준비 완료로 세지 않는다. 현재 raw-state 모델에 이 인증이 있다고 주장하지 않는다.
