# MPS 응답의 수학 귀속과 검증 범위

2026-11-09. `readMpsSuccessForInput`은 입력에서 전체 예상 응답을 독립 계산하고, descriptor-only 소유 복사와 요청 ID·종류·입력 key를 대조한다. 반환값은 깊이 동결한 복사다. 생산 MPS 계산기·transfer recurrence·Gaussian rank·정규화 root engine을 검증기의 oracle로 가져오지 않는다.

## 수축·환경·물리 관측

1..4개 qubit와 내부 bond1..4의 열린 사슬에서 각 물리 bit열의 **모든 가상 경로**를 합한다. Prefix L과 suffix B를 별도로 구성해 상태, E=L†L, R=conj(B)Bᵀ와 모든 절단의 ΣE_ab R_ab=N을 계산한다. 오른쪽 환경을 밀도 행렬이나 왼쪽 환경으로 바꿀 수 없다. 복소수 방향·bra/ket 순서가 중요하다.

국소 밀도는 조밀 상태의 직접 partial trace, Bloch·purity는 그 밀도, 모든 쌍의 Pauli 및 connected correlation은 조밀 Kronecker 연산자로 계산한다. 생산 transfer 기반 관측 경로와 구별된다. 비영 국소 텐서가 서로 상쇄해 전역 영벡터가 될 수 있다. 이때 raw algebra와 coefficient rank0은 유지하지만 normalized physical과 Schmidt rank는 null이다.

## Rank·gauge·정규형

각 절단의 coefficient matrix C에서 더 작은 Gram(C C† 또는 C†C)의 정확한 양의 주 소행렬식으로 rank를 구한다. 최대 Gram 차원은4다. Bond dimension이 rank나 압축률을 보장하지 않는다. 각 텐서의 ΣA†A·ΣAA†와 identity 잔차로 좌우 isometry를 따로 확인한다.

G=D S와 G⁻¹=S⁻¹D⁻¹를 입력의 비영 실수 scale·복소 shear로 만들고 인접 텐서를 A G와 G⁻¹B로 변환한 뒤 **다시 수축**한다. 상태·norm 보존과 환경 covariance, 좌표 norm 변화 및 정규형 진단을 모두 대조한다. Gauge가 물리 상태를 보존한다는 이유로 텐서·환경 좌표의 잘못된 응답을 수락하지 않는다.

## 정규화 구간과 소유 구조

각 비영 성분 a/√N의 부호와 제곱 구간 포함 관계를 유리수로 검사한다. 폭은10⁻¹²⁵ max(1,|a|/√N) 이하이며, 비교를 제곱해 별도 sqrt 없이 검사한다. 기존 raw |a| 기준을 실제 정규화 값 기준으로 강화했다. 큰 raw 좌표에서 지나치게 넓은 구간과 작은 N의 합법 입력을 각각 검증한다.

모든 키·배열 길이·숫자 metadata·계약 문구를 대조한다. Getter·get trap을 호출하지 않고 소유 descriptor를 복사하며 symbol·희소 배열·추가 필드·상속 record·순환을 거부한다. 노드 예산500000, 깊이64의 기존 경계는 유지한다.

## 한계와 다음 단계

독립 fixture29개와 응답 변조·브라우저 Worker로 수치 및 전달을 검증한다. SVD·자동 정규형·Schmidt spectrum·절단·DMRG·시간 진화·바닥 상태·무한계·압축 보장을 제공하지 않는다. Schmidt는 별도 응답 종류이며 아직 남아 있다. 실제 reader와 취소 controller·반응성·생산 Worker 자산은 별도 통합 검증이 필요하다. 현재 검증은 독립 preview-origin Blob Worker이며 홈페이지 등록이나 공개 배포의 증거가 아니다.
