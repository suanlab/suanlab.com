# Fermion Fock·parity 수학·입출력 계약

2026-11-09. 예정 단원 `fermion-fock-parity-operator-algebra`의 수치 기반이다. 일반 이론의 입자·모드 수에는 상한을 두지 않으며, 이 계산기는 1…4개 순서 있는 모드의 정확 행렬 실험을 제공한다. 상세 한영 본문·Worker·학습 화면·reader 연결은 후속이다.

## 기저와 연산 순서

점유 기저는 |n₀…nₘ₋₁〉=(c₀†)^n₀…(cₘ₋₁†)^nₘ₋₁|vac〉이고 n₀가 최상위 비트다. 오른쪽 연산자가 먼저 작용한다. cᵢ는 해당 점유가 1일 때 이를 0으로 바꾸고, 앞선 모드의 점유 합이 홀수이면 −1을 곱한다. 생성자는 그 adjoint다. 이 convention은 [공식 Jordan–Wigner 설명](https://quantumai.google/openfermion/tutorials/jordan_wigner_and_bravyi_kitaev_transforms)의 기저·CAR와 대조했다.

각 모드 쌍의 {cᵢ,cⱼ†}=δᵢⱼI, {cᵢ,cⱼ}=0과 생성자 CAR를 **전체 행렬**로 검사한다. 부호 없는 비교 모형은 독립 spin ladder이며 fermion의 대체 표현으로 제시하지 않는다. 진공은 입자가 없지만 노름이 1인 상태다. 같은 모드를 두 번 생성한 영벡터는 유효한 연산 결과이며 입력 물리 상태로 재정규화할 수 없다.

## 소유 입력과 수치 표현

`learning-fermion-fock-input.ts`는 수치 엔진을 가져오지 않는다. 정확히 74개 소유 정수 필드, 절댓값 10⁶ 이하, 양의 공통 Hamiltonian 분모와 비영 복소 상태를 요구한다. 모드는 1…4, sector는 0=formal/1=even/2=odd다. 비활성 모드의 항·차원 밖 진폭은 0이어야 한다. 선택 sector 밖의 진폭과 accessor·추가 키·다른 prototype·잘못된 reflection은 거부한다.

검증한 각 descriptor에서 canonical +0 snapshot을 만들고 동결한다. caller 프로퍼티를 다시 읽지 않는다. 노름이 다른 원시 상태를 받아 물리 관측을 N=z†z로 나눈다. 행렬·평균·분산·확률은 복소 유리수로 정확 계산한다. z/√N과 잔차/√N 좌표만 바깥 512비트 구간이며 표시용 Number를 항등식·영 판정에 사용하지 않는다.

## Hamiltonian과 출력의 뜻

공통 분모로 나눈 계수를 μᵢ,hᵢⱼ,Δᵢⱼ,Uᵢⱼ라 쓰면

H = e₀I + Σᵢμᵢnᵢ + Σᵢ<ⱼ(hᵢⱼcᵢ†cⱼ + hᵢⱼ*cⱼ†cᵢ + Δᵢⱼcᵢ†cⱼ† + Δᵢⱼ*cⱼcᵢ + Uᵢⱼnᵢnⱼ).

h·Δ는 일반 복소수, 나머지는 실수다. pairing adjoint는 cⱼcᵢ로 순서를 뒤집는다. 이 Hamiltonian은 even이며 parity를 보존한다. pairing은 입자 수를 보존하지 않을 수 있다. 홀수 source, 일반적인 네 모드 산란, 시간 진화나 최적화는 제공하지 않는다.

출력은 H,H z,E,E₂,Var(H),r=Hz−Ez,항별 에너지,전체 ρ,γᵢⱼ=〈cᵢ†cⱼ〉,Fᵢⱼ=〈cᵢcⱼ〉,〈cᵢ〉,점유/입자 수 분포와 연결 상관이다. γ는 Hermitian이고 F는 반대칭이다. **이 색인 convention에서 hopping 에너지는 2Re(hᵢⱼγᵢⱼ)**다. γ의 전치를 생략한 Tr(hγ)로 바꾸지 않는다. 〈cᵢ†cⱼ†〉=Fⱼᵢ*다. γ만으로 전체 입자 수 분포나 얽힘을 결정하지 않는다.

전체 N̂와 P=(−1)^N̂에 대해 실제 [H,A]의 Frobenius 제곱과 현재 상태의 평균·분산을 별도로 준다. 현재 상태가 sharp인 것과 Hamiltonian의 보존 법칙은 다르다. 선택 probe의 forward=L Rz, reverse=R Lz와 그 합, 원래 정규화 입력에 대한 작용과 노름 비율을 반환한다. 결과를 별도 재정규화하지 않으며 측정 instrument 없는 확률로 부르지 않는다.

## Parity 가정과 비교 상태

[Szalay 등의 리뷰](https://arxiv.org/abs/2006.03087)는 fermion의 subsystem 표현과 parity superselection을 qubit과 구별한다. 여기서는 parity superselection을 물리 가정으로 명시한다. 순수 입력의 even·odd 확률 pₑ,pₒ와 ρ−(ρ+PρP)/2의 Frobenius 제곱을 직접 계산한다. 후자는 2pₑpₒ이며 0일 때만 그 순수 입력이 이 가정과 양립한다.

sector=0은 even–odd 순수 coherence를 **형식적 반례**로 허용한다. 이것을 물리적으로 허용된 순수 상태로 표현하지 않는다. 〈cᵢ〉가 전부 0이어도 parity coherence는 남을 수 있다. 원래 ρ와 parity twirl ρ̄=(ρ+PρP)/2를 함께 출력하며, twirl을 입력의 자동 수정으로 쓰지 않는다. twirl의 purity·에너지·홀수 평균을 별도로 계산한다. even H의 에너지는 같지만 상태는 달라질 수 있다. qubit의 보통 partial trace를 fermionic mode 얽힘으로 해석하는 출력은 제공하지 않는다.

## 스펙트럼의 범위와 빈 판정

H가 점유 기저에서 대각일 때만 모든 기저 에너지를 정확 열거한다. full Fock 바닥 에너지·축퇴도·projector 무게 p₀·누출 1−p₀·다음 서로 다른 에너지까지의 gap을 준다. gap이 있으면 누출 상한 min(1,(E−E₀)/gap)과 자르지 않은 비율을 구별한다. 상수 H의 gap은 null이다.

even/odd 및 고정 입자 수 부분공간의 최솟값도 **별도** 반환한다. 각 probability는 전체 입력에 대한 projector 무게로 조건부 재정규화 확률이 아니다. sector=0에서는 선택 parity 바닥 판정이 null이다. full Fock 바닥과 선택 sector의 바닥을 혼동하지 않는다. 비대각 H의 스펙트럼·바닥 판정은 null이며, 정확 영분산은 고유상태임만 인증한다.

## 독립 검증 관문

별도 Python 생성기는 제품 코드를 가져오지 않고 SymPy Jordan–Wigner Kronecker 행렬·H²·바깥 밀도·실제 교환자와 mpmath250자리 정규화를 사용한다. 45개 준비에서 γ와 I−γ의 모든 주 principal minor의 비음성도 확인한다. 테스트는 정확 유리수 endpoint 일치, 정규화 참조 대조 여유 10⁻²⁰⁵·max(1,|x|), 구간 폭 10⁻¹²⁵·max(1,|x|)를 요구한다. 자세한 결과와 검증 범위는 [기반 검증 기록](quantum-fermion-fock-foundation-status.md)에 있다.
