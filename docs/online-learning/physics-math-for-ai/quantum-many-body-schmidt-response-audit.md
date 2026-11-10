# Schmidt 전체 응답의 소유 구조와 수학 귀속 감사

2026-11-09. `readSchmidtSuccessForInput`은 소유 입력의 비영 Gaussian-integer 상태에서 모든 연속 절단을 다시 구성한다. 생산 Schmidt·Hilbert·행렬·Ritz·root·로그 엔진을 검증기에 가져오지 않는다. [독립 근·로그 인증](quantum-many-body-schmidt-spectral-audit.md)을 전체 응답에 연결했다.

## 정확한 입력과 물리 부분계

2..4개 qubit의35개 정수 필드에서 상태와 norm을 구하고, 조밀 joint density를 직접 partial trace해 물리 rhoLeft·rhoRight를 계산한다. 오른쪽 밀도는 CᵀC*/N이며 coefficient Gram C†C/N과 전치 관계다. 작은 부분계를 명시하고, 정확한 주 소행렬식으로 rank와 characteristic polynomial을 계산한다. 생산 Gaussian elimination과 trace 기반 characteristic 계산을 oracle로 쓰지 않는다.

원시 상태·정규화 좌표·norm·계수 행렬·밀도·purity·linear entropy·zero multiplicity·전체 절단·선택 절단 index를 입력과 대조한다. 선택한 절단만 검증하거나 입력과 무관한 spectrum을 수락하지 않는다.

## Spectrum·projector와 독립 구간 산술

정확한 square-free factor와 Descartes 인증으로 모든 고유값·중복도를 구한다. 생산 Sturm isolator의 결과를 받아 검증하는 대신 입력 다항식에서 독립 계산한다. 각 고유값 그룹의 Lagrange 보간 polynomial 계수를 구성하고 행렬 Horner 연산으로 projector를 평가한다. 생산 계산기의 행렬 factor 연쇄와 경로를 구별한다. 퇴화 그룹은 하나의 projector로 유지하며 임의의 eigenvector를 고르지 않는다.

새 구간 산술은3072비트 dyadic grid에서 모든 부호의 곱·나눗셈·제곱·행렬 연산을 바깥 방향으로 반올림한다. 고정 분모를 유지해 반복 gcd 비용을 줄인다. Sqrt는 정수 Newton 반복과 정확한 제곱 비교로 하한·상한을 만든다. 음수 범위의 sqrt와0을 포함한 divisor는 거부한다. 이 검증 정밀도와 생산 응답에 명시한4096/2048/768/512비트 계약은 별개다.

## 절단·퇴화·정규화

양의 고유값 그룹을 큰 순서로 budget 안에서 할당한다. 양의 퇴화 그룹을 일부만 남기는 경계는 최적 error·fidelity가 존재해도 chosen projector·state가 null이다. Rank budget0은 유일한 비정규화 영벡터를 가지며 normalized optimum·fidelity·distance는 null이다. Budget이 실제 rank 이상이면 양의 support 전체와 retained mass1을 사용한다.

선택 상태가 유일하면 왼쪽 projector는 P C, 오른쪽은 C Pᵀ로 작용한다. Raw coefficient·정규화 입력 기준 진폭·다시 정규화한 진폭·직접 norm/error를 각각 계산한다. Discarded weight, normalized fidelity²와2(1−sqrt(retained weight)) 거리를 서로 바꾸지 않는다. Entropy는 독립 인증한 −Σ multiplicity·λ·lnλ이며 bits/nats도 각각 대조한다.

## 관측 구간·잔차·소유 구조

각 방향성 관측은 독립 인증 구간 전체를 포함해야 하고 폭은10⁻¹²⁵ max(1,인증 값 크기) 이하여야 한다. Probability는 [0,1] 안에 있어야 한다. Projector 완전성·재구성·고유 방정식·정규직교·절단 오차 수지는 수학적0을 포함하는 좁은 구간으로 검사한다. 정확 density 잔차는 정확0이어야 한다. 넓은 구간이나 unresolved metadata로 값 대조를 우회하지 않는다.

Descriptor-only 복사 후 전체 key·배열 길이·metadata·정밀도·계약 문구·request ID·kind·input key를 대조하고 깊이 동결한 복사를 반환한다. Caller getter/get trap을 실행하지 않는다. Symbol·추가 필드·희소 배열·상속 record·순환과 hostile reflection을 거부한다. 기존500000노드·깊이64의 예산을 유지한다.

## 남은 통합

독립26준비와 변조 검사·전체 종류 Native Worker 검증은 수치 응답의 근거다. 취소 controller/hooks·reader 반응성·시각화·실험 안내·퀴즈·저장·실제 Next Worker 자산·정적 reader는 별도로 통합해야 한다. 이 단계에서 새 학습 단원 등록·홈페이지 배포를 완료했다고 세지 않는다. 다중 절단의 전역 최적성·자동 MPS 압축·DMRG·무한계를 주장하지 않는다.
