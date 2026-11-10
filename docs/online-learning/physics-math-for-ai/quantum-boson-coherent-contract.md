# Coherent 급수·유한 투영·Poisson tail 계약

2026-11-09. `boson-fock-cutoff-boundary-algebra`의 coherent 준비를 [raw-state cutoff 기반](quantum-boson-fock-contract.md)에 연결한다. 무한 이론의 상태와 계산기에 전달하는 유한 상태를 구별한다. 아직 한영 상세 본문·Worker·교수 화면·reader 등록은 하지 않았다.

## 소유 입력과 정확 계수

`learning-boson-coherent-input.ts`는 수치 엔진 없는 별도 입력 모듈이다. raw-state 입력의 차원·Hamiltonian·probe 20필드에 α의 실수/허수 분자·양의 분모를 모드마다 더한 **26개 소유 정수**를 받는다. 각 정수의 절댓값은 10⁶ 이하다. 모드는 1…2, 각 최대 점유 cutoff는 0…3이다. 활성 모드마다

αᵢ=(aᵢ+ibᵢ)/dᵢ, μᵢ=(aᵢ²+bᵢ²)/dᵢ², 0≤μᵢ≤16

을 BigInt로 정확 검사한다. 비활성 α는 0/1이고 해당 Hamiltonian 항은 비활성이다. μ=0도 유효하다. getter·추가 키·다른 prototype·잘못된 reflection은 거부하고 검증한 descriptor로 canonical +0·동결 snapshot을 만든다. 이 범위는 계산기의 인증 범위이며 이론과 교육과정의 분량 제한이 아니다.

coherent 급수는 [Stuttgart 강의 자료](https://www.itp3.uni-stuttgart.de/downloads/Adv._Quantum_Theory_Engl_ws11.12/ex9.pdf)의 정의와 같은 convention으로

|α〉=exp(−μ/2) Σₙ≥₀ αⁿ/√n! |n〉

를 사용한다. 유한 계수 zₙ=αⁿ/√n!는 공통 exp 인자를 제외하고 **Q(√2,√3)**에 정확히 유지한다. 예를 들어 α=1,C=3이면 [1,1,√2/2,√6/6]이고 Z₃=8/3이다. 이 계수의 √n을 Number로 바꾸지 않는다.

## 세 상태와 서로 다른 오차

P는 모든 모드의 직사각형 점유 cutoff projector다. Z_C(μ)=Σₙ₌₀ᶜ μⁿ/n!이고, 여러 모드에서 z=⊗ᵢzᵢ, N=z†z=∏ᵢZ_Cᵢ(μᵢ)다.

| 출력 | 정의 | 노름·오차의 의미 |
| --- | --- | --- |
| `infiniteState` | 무한 coherent 급수의 곱 | 정규화된 무한 상태를 급수와 α·μ로 정의한다. 유한 배열 전체가 무한 상태라고 표시하지 않는다. |
| `unnormalizedProjection` | P|α₀,…〉=exp(−Σμᵢ/2)z | 노름 제곱 S=exp(−Σμᵢ)N. 원래 무한 상태까지 거리 제곱은 버린 확률 1−S다. |
| `normalizedProjection` | z/√N | 노름 1. 원래 무한 상태와 fidelity **제곱**은 S, overlap은 √S, 거리 **제곱**은 2(1−√S)다. |

재정규화는 명시적인 별도 상태이며 버린 확률을 없애지 않는다. cutoff0의 재정규화 상태는 진공이지만 α≠0의 원래 coherent 상태와 같지 않다. 유한 행렬 exp(αa_C†−α*a_C)|0〉를 원래 무한 급수의 투영으로 대신하지 않는다.

## Poisson 인증과 정밀도

한 모드에서 원래 무한 상태의 확률은 exp(−μ)μⁿ/n!이며 retained probability는 exp(−μ)Z_C(μ)다. 버린 확률은

Pr(n>C)=γ(C+1,μ)/C! = 1−exp(−μ)Z_C(μ).

여기서 γ는 **lower incomplete gamma**다. [NIST DLMF 8.4.7·8.4.9·8.4.11](https://dlmf.nist.gov/8.4)과 방향을 대조했다. 조건부 유한 확률 μⁿ/(n!Z_C)와 원래 무한 확률은 별도로 반환한다. 두 모드의 retained probability는 모드별 값의 곱이며 tail을 단순히 더하지 않는다.

제품 코드는 기존 `createTranscendentalContext(768)`의 방향성 유리수 지수 구간을 사용한다. μ를 1/8 이하로 range reduction하고, 양의 Taylor 급수와 해석적 나머지 상한을 계산한 뒤 역수·제곱으로 exp(−μ)를 인증한다. 내부 guard는 32비트이고 출력은 바깥 512비트 구간이다. α의 μ는 정확 유리수로 전달하며 Math.exp를 사용하지 않는다.

공통 지수 인자·확률·거리·정규화 진폭은 인증 구간이고 유한 급수 계수·분포 계정은 정확하다. 알려진 [0,1] 확률 범위와의 교집합은 구간 축소이며 상태의 자동 보정이 아니다. μ=10⁻¹²,C=3의 약 4.17×10⁻⁵⁰ tail도 하한이 양수인 구간으로 남는다. 모드별 retained 곱과 공동 지수 방식의 차이는 0을 포함하는 구간으로 검사하며, 독립 반올림된 구간을 정확 같은 값으로 강제하지 않는다.

## 유한 연산자와 물리 비교

재정규화 상태를 공통 finite engine에 전달해 H·에너지·분산·잔차·국소 밀도·γ·점유 분포·quadrature·실제 경계 CCR를 계산한다. prepared-state bridge는 정확 계수·차원·양의 유리수 노름을 검사한다. 파생 상태에 가짜 Gaussian-integer 입력 필드를 표시하지 않고 원래 coherent 입력만 최상위 `parameters`로 반환한다. 기존 raw-state API와 결과는 유지한다.

무한 상태의 a|α〉=α|α〉는 유한 투영 뒤 일반적으로 성립하지 않는다. p_top=μᶜ/(C!Z_C)에 대해 실제 잔차를 계산하고

‖(a_C−α)ψ_C‖²=μp_top, 〈[a_C,a_C†]〉=1−(C+1)p_top

을 정확 검산한다. 무한 X/P 평균·분산, 전체 입자 수 평균/분산과 같은 normally ordered Hamiltonian의 무한 coherent 기대 에너지를 유한 값과 별도 제시한다. 유한 cutoff의 바닥 판정은 무한 공간의 안정성이나 바닥 존재를 인증하지 않는다. 결합 H를 지정해도 이 곱 준비 상태를 자동 진화시키지 않는다.

## 독립 기준과 남은 관문

새 생성기는 제품 모듈 없이 SymPy의 coherent 급수와 기존 독립 국소 Kronecker/H²/partial trace 기준을 사용한다. Poisson tail은 mpmath250자리 **incomplete gamma**로 대조하며 제품의 지수 Taylor 구현을 공유하지 않는다. 정확 endpoint와 방향성 대조 여유 10⁻²⁰⁵·max(1,|x|), 폭 10⁻¹²⁵·max(1,|x|)를 검사한다. [검증 기록](quantum-boson-coherent-foundation-status.md)은 Node 수치 기반에 한정된다. Worker·한영 원고·실제 reader·생산 자산 관문은 남아 있다.
