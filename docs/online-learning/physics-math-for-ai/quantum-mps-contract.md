# MPS 수축·환경·gauge 수학·입출력 계약

2026-11-09. 예정 단원 `matrix-product-state-contraction-gauge`의 수치 기반이다. [다체계 다섯 단원 설계](quantum-many-body-authoring-plan.md)의 네 번째 가족이며 한영 원고·Worker·교수 화면·reader 등록은 후속이다.

## 공간·색인·소유 입력

1…4개 qubit의 열린 MPS를 계산한다. D₀=Dₙ=1이고 내부 bond 차원은 각각1…4다. 물리 비트 n₀가 최상위이며

ψₛ₀…ₛₙ₋₁ = A₀⁽ˢ⁰⁾ A₁⁽ˢ¹⁾ … Aₙ₋₁⁽ˢⁿ⁻¹⁾

은 1×1 행렬의 값이다. 각 Aᵢ⁽ˢ⁾의 크기는 Dᵢ×Dᵢ₊₁이다. 출력 배열은 `[site][physical][left][right]`, 입력 키는 예를 들어 `t1Imag203`으로 site1·left2·physical0·right3을 지정한다. padding을 활성 차원으로 해석하지 않는다.

`learning-mps-input.ts`는 수치 엔진을 가져오지 않는다. **269개 소유 정수**의 절댓값10⁶ 이하, 양의 공통 tensor 분모, 활성 bond와 비활성 항0, gauge 범위를 검사한다. 원시 텐서는 일반 복소 유리수다. descriptor를 한 번씩 읽어 canonical +0 snapshot을 만들고 동결한다. accessor·추가 키·다른 prototype·잘못된 reflection은 거부하며 caller getter를 호출하지 않는다. factory도 반복 전에 유한 sites 범위를 검사한다.

구조적으로 유효한 텐서가 비영 물리 상태를 보장하지 않는다. 각 tensor가 비영이어도 전체 곱의 경로가 상쇄되거나 연결되지 않을 수 있다. 전체 노름이0이면 원시 진폭·환경·rank·gauge 계정은 유지하고 `physical=null`, `physicalStateDefined=false`로 반환한다. 정규화나 물리 관측을 만들지 않고 다른 상태로 보정하지 않는다.

## 수축과 bra/ket convention

제품 구현은 prefix를 공유하면서 왼쪽부터 진폭을 수축한다. cut의 prefix 계수 행렬을 L, suffix 계수 행렬을 B라 하면 ψ의 계수 배열은 C=LB다. 환경은

Eₐᵦ=Σ_prefix conj(L_prefix,a)L_prefix,b = (L†L)ₐᵦ,

Rₐᵦ=Σ_suffix conj(B_a,suffix)B_b,suffix = (B*Bᵀ)ₐᵦ

이다. 노름은 Σₐᵦ EₐᵦRₐᵦ이며 색인을 바꾼 Tr(ER)로 대체하지 않는다. 모든 cut과 양 끝에서 이 closure가 직접 진폭 노름 N=z†z와 정확히 같아야 한다. 오른쪽 환경은 이 bra/ket 정의의 virtual Gram이며 정규화된 물리 reduced density가 아니다.

국소 밀도는 환경 안에 `|bra〉〈ket|`를 삽입해 계산하고 별도의 전체 진폭 partial trace와 대조한다. 국소 XYZ·purity와 모든 site 쌍의 아홉 Pauli 조합·연결 상관도 계산한다. X/Y/Z는 Pauli convention이며 Y=−2Imρ₀₁이다. 특히 오른쪽 |0〉+i|1〉의 Y=+1을 확인한다. transfer 수축의 배경은 [Schollwöck §4.2.1](https://arxiv.org/html/1008.3477v2)을 참고하되 위 색인과 항등식은 직접 검산한다.

## 가역 gauge와 바뀌는 양

한 bond에서 A→AG, B→G⁻¹B를 적용한다. 일반 가역 gauge의 상태 보존은 [Schollwöck 식57](https://arxiv.org/html/1008.3477v2)의 convention과 대조했다. 계산기는 다음 구조의 정확 gauge를 제공한다.

G=D_scale(row,λ)·(I+ηE_row,column), λ∈Q\{0}, η∈Q+iQ.

row≠column에서는 E²=0이므로 inverse를 정확 계산한다. row=column이면 η=0인 scale만 허용한다. λ 분모와 η 공통 분모는 양수다. 음수 scale과 복소 shear도 허용한다. `gaugeBond=-1`이면 identity 제어값만 허용하며 존재하지 않는 inverse나 covariance를 만들지 않는다. 한 site는 이 모드다.

전체 진폭·N·정규화 물리 관측·cut coefficient rank는 보존되어야 한다. 환경은 선택 cut에서 E′=G†EG, R′=conj(G⁻¹)R(G⁻¹)ᵀ로 변한다. 두 covariance의 실제 행렬 잔차를 계산한다. 텐서 좌표 제곱 길이 Σᵢₛ‖Aᵢ⁽ˢ⁾‖²와 local Gram은 변할 수 있다. GHZ₃의 예는 물리 N=2를 보존하면서 텐서 길이를6에서19/2로 바꾼다. 이를 시간 진화나 물리 상태의 변화로 표현하지 않는다.

## Rank·정규형·비용의 범위

각 연속 cut의 전체 계수 배열 C를 만들고 복소 유리수 Gaussian elimination으로 rank를 정확 계산한다. N>0일 때 이것은 Schmidt rank이며 해당 bond 차원을 넘지 않는다. N=0이면 coefficient rank는0이지만 물리적 Schmidt rank는 null이다. singular values·Schmidt weights·entropy·최적 절단은 다음 가족에서 다룬다.

각 site에서 ΣₛAₛ†Aₛ와 ΣₛAₛAₛ†를 실제 계산하여 identity와의 Frobenius 잔차·left/right isometry 여부를 반환한다. 한 site의 isometry만으로 전체 상태가 정규화됐다고 하지 않는다. SVD/QR로 자동 canonicalization을 수행하지 않는다. 정규형을 만드는 절차는 [리뷰 §4.4](https://arxiv.org/html/1008.3477v2)의 별도 알고리즘과 구별한다.

`cost`는 활성 복소 tensor 항 수, padding128, 전체 진폭2ⁿ, prefix 및 norm-transfer 알고리즘의 dense summand slot 수를 표시한다. 실제 시간·최소 연산 수·bit complexity·항상 압축된다는 보장은 아니다. 전체 진폭을 출력하면 지수적 배열이 필요하다. 작은 계의 큰 bond MPS는 dense state보다 항이 많을 수 있다. 임의 크기·열역학적 극한·모든 연구 알고리즘의 성능으로 일반화하지 않는다.

## 수치·독립 검증 관문

진폭·환경·관측·rank·gauge inverse는 정확 복소 유리수다. 정규화 좌표만 √N을 바깥512비트 구간으로 나눈다. 영 판정·rank·canonicality에 Number나 허용오차를 쓰지 않는다.

별도 생성기는 제품 모듈 없이 **모든 virtual 경로의 명시적 합**, 직접 prefix/suffix Gram, dense Pauli 행렬과 partial trace를 사용한다. mpmath250자리 정규화와 정확 endpoint를 대조한다. 방향성 참조 여유는10⁻²⁰⁵·max(1,|x|), 폭 관문은10⁻¹²⁵·max(1,|x|)다. [검증 상태](quantum-mps-foundation-status.md)는 수치 기반에 한정된다. DMRG·MPO·시간 진화·바닥 상태·Schmidt 절단·실제 reader가 구현됐다는 근거로 세지 않는다.
