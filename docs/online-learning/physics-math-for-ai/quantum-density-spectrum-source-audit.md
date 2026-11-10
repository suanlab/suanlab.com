# Mixed qubit spectrum·entropy·dephasing: 수치 계약과 검산

2026-11-08. `qubit-density-spectrum-entropy-dephasing`의 유도·범위다. 일반 d의 density/entropy 이론·부분추적·연구 심화는 qubit 계산만으로 완료 처리하지 않는다.

## 상태와 스펙트럼

11개 소유 정수는 공통 분모를 가진 Bloch 좌표3개·복소 probe2개·끝점 eta·표본 수다. 모든 정수 절댓값은10⁶ 이하이며 분모는 양수다. |r|²≤1을 BigInt로 정확히 비교한다. 범위 밖 좌표를 ball 안으로 잘라 넣지 않는다.

rho=(I+r·sigma)/2의 순도는(1+|r|²)/2, determinant는(1−|r|²)/4다. s=|r|이면 고유값은(1±s)/2다. 작은 고유값은 직접적인1−s 대신(1−|r|²)/(2(1+s))로 계산한다. 정확한 순수 상태에서만 이 값을0으로 분류한다. 작은 양의 determinant를 tolerance로0에 붙이지 않는다.

s>0의 사영은(I±r·sigma/s)/2다. s=0에서는 고유값1/2·rank2·사영I로 합치며 임의의 고유방향을 선택하지 않는다. eigen-equation·사영 완전성/멱등성/Hermiticity·rho 복원을 구간으로 확인한다.

## 엔트로피와 기저

S=−Σλ lnλ는 nats, S/ln2는 bits다. 정확히0인 고유값의 log는 null이고 entropy 항은 x ln x의 극한으로 정확히0이다. log0을 평가하거나 epsilon으로 대체하지 않는다. 완전 혼합의 entropy bits1은 구조적으로 증명된 값이다. 일반 고유값·로그·entropy는 방향성 구간으로 계산한다.

작업 연산은768bit, 보고 구간은 바깥 방향512bit다. 로그는 기존 양의 유리수 로그의800bit 작업/768bit 경계와 단조성을 사용한다. 양의 고유값 구간 양끝에 로그를 적용한다. 제곱근·로그의 부정확성을 보존하며 출력 상태를 재정규화하지 않는다. 폭 기준은 max(1,|참조|)×10⁻¹²⁵다.

probe로 만든 정규직교 U에서 U†rho U를 정확 유리수로 계산한다. purity와 spectrum은 유지되지만 diagonal entropy와 C_basis=S(diagonal rho)−S(rho)는 기저에 의존한다. 실제 관측량 n·sigma·두 사영·평균/분산과 기저의 대각 확률도 직접 대조한다. 관측값은 무차원 Pauli의±1이며 물리적 spin±ℏ/2가 아니다.

## Channel과 표본 경로

E_eta(rho)=((1+eta)/2)rho+((1−eta)/2)ZrhoZ, −1≤eta≤1이다. 두 계수는 모두 비음이고 합은1이다. Kraus는 √((1+eta)/2)I와 √((1−eta)/2)Z이며 completeness·직접 작용을 확인한다. 음의 eta도 유효한 완전 양성·trace 보존·unital map이다.

Choi의 인덱스는 output⊗input, 2*output+input이다. unnormalized Choi의 trace는2이며 고유값은0,0,1−eta,1+eta다. 상태의 trace1과 혼동하지 않는다. Choi의 output partial trace=I와 선택 입력에 대한 작용을 정확히 확인한다. |eta|=1에서는 rank1·unitary, 나머지에서는 rank2다. 선택한 I/2가 고정이라는 사실만으로 channel이 항등이라고 판단하지 않는다.

표본은 eta(u)=1+(eta_end−1)u를 순회한다. eta_end<0이면 eta0 지점을 정확 유리수로 넣는다. 양의 쪽에서 entropy가 증가하다 음의 쪽에서 줄어들 수 있으므로 이 u를 물리 시간이나 Markovian law로 부르지 않는다. |+X⟩에서 eta−1의 끝점은 entropy0이지만 초기 상태와 직교한다. 그 지점의 squared Uhlmann fidelity는0, trace distance는1이다.

## 거리와 독립 참조

qubit trace distance는 |r−r0|/2다. squared Uhlmann fidelity는 Tr(rho rho0)+2√(det rho det rho0)이며 root fidelity와 구별한다. Hilbert-Schmidt 차이 제곱은 별도의 양이다. 표준 기저의 diagonal이 보존되므로 coherence 감소와 초기 대비 entropy 증가가 같음을 구간으로 확인한다.

독립 참조는 SymPy의 Kraus operator sum·symbolic eigenvalues·직접 trace와 mpmath250dps의 일반 Hermitian eigensolver·고유공간 사영·행렬 제곱근·singular-value trace norm을 사용한다. 생산 Bloch 갱신·안정화한 작은 고유값 식·로그 함수를 재사용하지 않는다. fidelity도 두 행렬 제곱근 곱의 singular values 합의 제곱으로 계산한다.

배경은 [Preskill Chapter3](https://www.preskill.caltech.edu/ph229/notes/chap3.pdf), [Chapter10 §10.2](https://www.preskill.caltech.edu/ph219/chap10_6A_2022.pdf), [IBM channel representations](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/representations-of-channels)을 열어 확인했다. 유한 입력·부호 있는 eta 경로·안정화 식·기저 비교·인덱스·정밀도와 참조 계약은 이 교재의 구현 설계다.
