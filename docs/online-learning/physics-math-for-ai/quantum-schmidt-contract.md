# Schmidt cut·가중치·최적 절단 수학 계약

2026-11-09. 예정 단원 `schmidt-cuts-bond-rank-truncation`의 수치 기반이다. [다체계 다섯 단원 설계](quantum-many-body-authoring-plan.md)의 마지막 가족이며, 상세 한영 본문·Worker·학습 화면·reader 등록은 후속이다.

## 소유 입력과 cut

`learning-schmidt-input.ts`는 수치 엔진을 가져오지 않는다. **35개 소유 정수**를 받아2…4개 순서 있는 qubit·선택 cut·rank budget0…4·최대16개 복소 원시 진폭을 정의한다. 정수 절댓값10⁶ 이하, 비영 상태, 비활성 진폭0을 요구한다. descriptor에서 한 번의 canonical +0·동결 snapshot을 만들고 accessor·추가 키·prototype·reflection 오류를 거부한다. factory도 반복 전에 sites를 검사한다.

모든 연속 cut k=1…n−1을 계산한다. n₀가 최상위 비트이며 원시 상태를 C∈C^(2ᵏ×2ⁿ⁻ᵏ)로 reshape한다. N=Σ|Cᵢⱼ|²를 보존하고 물리 관측은 C/√N에서 계산한다. `selectedCut`은 선택 화면의 귀속이며 다른 cut의 계산을 생략하지 않는다.

## 물리적 밀도 행렬의 방향

계수 Cᵢⱼ는 |i〉_L|j〉_R의 **ket 진폭**이다. 성분으로 partial trace를 하면

ρLᵢᵢ′ = Σⱼ Cᵢⱼ conj(Cᵢ′ⱼ)/N = (CC†/N)ᵢᵢ′,

ρRⱼⱼ′ = Σᵢ Cᵢⱼ conj(Cᵢⱼ′)/N = (CᵀC*/N)ⱼⱼ′.

C†C/N은 ρR의 복소 켤레이며 같은 양의 스펙트럼을 갖지만 같은 물리적 행렬은 아니다. 원문의 Gram 표기를 전치 없이 ket 밀도로 복사하지 않는다. 예를 들어 C=[1,i]이면 ρR₀₁=−i/2이고 Y=+1이다. 스펙트럼만 맞는지 확인하는 검사로는 이 부호 오류를 찾을 수 없다.

양쪽 전체 물리 밀도와 영 고유값 수를 제공한다. 스펙트럼은 **작은 쪽**(동일 차원이면 left)의 행렬에서만 계산하며 차원은 최대4다. 선택한 쪽을 기록하고 더 큰 쪽의 영 고유값 padding을 구별한다. coefficient rank는 복소 유리수 Gaussian elimination으로 정확 계산하며 작은 수를0으로 자르지 않는다.

## 가중치·projector·entropy

[Schollwöck §4.1.1](https://arxiv.org/html/1008.3477v2)의 SVD·Schmidt 관계, Frobenius 최적 근사와 재정규화의 구별을 참고한다. 이 모델의 λᵢ는 정규화된 ρ의 가중치이며 singular value는 **C/√N의** σᵢ=√λᵢ다. 원시 C의 singular value와 혼동하지 않는다.

정확 characteristic polynomial을 square-free factorization·Sturm 구간 분리로 풀고 multiplicity를 보존한다. 각 서로 다른 λ의 **전체 고유공간 projector**를 다항식으로 계산한다. 축퇴 고유공간 안에서 임의의 벡터를 선택하지 않는다. 양의 가중치 multiplicity의 합이 exact coefficient rank와 같아야 한다.

출력은 λ·σ, multiplicity·영 가중치 여부·작은 쪽 물리 projector, 정확 purity Trρ²·linear entropy와 S=−Σλlogλ의 nats/bits다. λ=0의 entropy 기여는0이다. 전체 좌우 purity도 정확 비교한다. entropy는 입력이 순수한 경우의 bipartite entanglement이며 혼합 상태의 일반 얽힘 척도로 확대하지 않는다.

## At-most-rank 절단의 의미

`retainedRank`는 **최대 허용 rank**다. 실제 양의 유지 rank는 min(budget,Schmidt rank)이며 nullspace에 가짜 Schmidt 벡터를 더하지 않는다. 내림차순 가중치 중 유지 합을 Sₖ, 버린 합을 ε=1−Sₖ라 하면 다음 양을 구별한다.

| 양 | 값 / 조건 |
| --- | --- |
| 최적 비정규화 rank≤k 근사의 거리 제곱 | ε |
| 그 근사의 노름 제곱 | Sₖ |
| 재정규화한 최적 상태와 원래 상태의 fidelity **제곱** | Sₖ, 단 Sₖ>0 |
| 재정규화한 최적 상태까지 거리 **제곱** | 2(1−√Sₖ), 단 Sₖ>0 |

같은 cut의 순수 상태와 Hilbert/Frobenius 노름에 대한 결과다. 여러 cut을 차례로 절단한 MPS의 전역 최적성을 주장하지 않는다.

cutoff가 양의 축퇴 group을 나누면 최적 오차·Sₖ·재정규화 거리는 정의되지만 최적 상태는 유일하지 않다. 이때 `boundaryDegenerate=true`, `chosen=null`로 반환한다. 허용 rank를 임의로 늘려 group 전체를 유지하거나 한 eigenvector를 골라 문제를 숨기지 않는다. group을 온전히 유지하면 projector와 절단된 물리 상태가 다시 유일해질 수 있다.

budget0이면 유일한 비정규화 근사는 영벡터·오차1이다. 정규화 상태·fidelity·정규화 거리는 null이다. 모든 양의 support를 유지하면 exact trace1로 Sₖ=1을 인증한다. projector는 그 양의 support를 표시하며 nullspace의 임의 선택을 포함하지 않는다.

## 선택 가능한 상태의 실제 계산

양의 cutoff를 가르는 축퇴가 없으면 유지 projector P로 실제 절단 계수를 계산한다.

- smallerSide=left: C_ret=P C.
- smallerSide=right: **C_ret=C Pᵀ**. 물리 ket의 오른쪽 작용이므로 C P나 C P†로 대체하지 않는다.

원시 C_ret, 원래 정규화 입력에 대한 C_ret/√N, 재정규화한 C_ret/√(NSₖ)를 별도로 준다. 실제 제곱 오차·유지 노름·잔차 직교성과 projector의 Hermiticity/idempotence를 대조한다. 숫자 표시의 `value=null`인 작은 성분도 구간으로 확인한다. 부동소수점0으로 바꾸지 않는다.

## 정확도와 검증 범위

밀도·rank·purity·polynomial은 정확 복소 유리수다. roots는4096비트 분리, projector 계산은2048비트 작업 구간, 로그는768비트와32 guard, 표시 출력은 바깥512비트다. √λ·재정규화의 제곱근도 방향성 구간이다. multiplicity·rank·영 가중치·축퇴 선택에 tolerance나 Number를 사용하지 않는다.

projector completeness·density reconstruction·trace multiplicity·고유방정식·purity 계정과 선택 절단의 실제 오차를 인증 구간으로 검사한다. 구간은0을 포함하고 폭10⁻¹²⁵ 이하이어야 한다. 독립 기준은 SymPy exact characteristic/rank/root multiplicity와 **mpmath250자리 Hermitian eigenspace**를 사용하여 제품의 polynomial projector와 구별한다. 참조 대조 여유는10⁻²⁰⁵·max(1,|x|), 폭 관문은10⁻¹²⁵·max(1,|x|)다.

[검증 기록](quantum-schmidt-foundation-status.md)은 Node 수치 기반이다. 자동 MPS 압축·DMRG·시간 진화·무한 계의 얽힘 scaling, 한영 원고·Worker·실제 reader 검증은 별도 작업이다.
