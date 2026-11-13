# Boson 동적 절단·MPS transfer 자료 대조

2026-11-13. 기존 두 단원의 한영 본문·풀이를 보존하고 독립 연구 절을 추가한다.

## Boson: 가정과 직접 유도

[Tong 외, Towards provably accurate quantum simulations of large-scale quantum systems](https://arxiv.org/pdf/2110.06942), Quantum 6, 816 (2022)는 초기 상태의 정적 절단과 시간 전개의 절단 정확도를 구별하는 일차 자료다. 본문의 경계 잔차·Duhamel 적분은 명시한 Hamiltonian에서 직접 유도한다. 논문의 더 강한 cutoff scaling 정리를 이 예제에 적용한 것으로 표시하지 않는다.

자기수반 H와 정의역 안의 유한 절단 궤적, 적분 가능한 잔차를 가정한다. HC=PHP이며 잘린 quadrature의 제곱으로 바꾸지 않는다. 구동 oscillator에서는 잔차가 오직 C+1 경계로 나가므로 누적 sqrt(pC) 적분이 벡터 오차를 제어한다. 초기 투영 오차는 별도로 더한다. 유한 궤적의 top 확률과 전체 coherent 상태의 Poisson 꼬리는 다르다.

전체 coherent 해의 전역 위상을 포함하여 벡터 차이를 계산한다. Trace distance에서는 이 위상이 사라진다. C=1의 정확한 유한 귀환에서 top 확률이 0이어도 전체 상태와의 trace distance는 약 0.9897이다. 누적 상계는 0으로 돌아가지 않으며 항상 날카로운 상계라는 주장도 하지 않는다. 유계 관측량 오차와 무한 number 연산자의 오차를 구별한다.

구간 전체의 확률 상한이 있어야 적분의 인증된 상계를 얻는다. 점 표본·SciPy quadrature와 그림의 수치 곡선은 방향성 구간 인증이 아니다. 검산은 Poisson 꼬리를 더한 안정적인 벡터 차이로 상계를 비교하며 그림에서의 작은 음수 roundoff 처리와 구별한다.

## MPS: 명시적 유한 family와 transfer

[Schollwöck, The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/pdf/1008.3477), §4.2.2는 transfer를 통한 상관 계산의 일차 자료다. 본문의 두 행렬·유한 경계·Markov 제곱근 진폭 family는 직접 제시하고 곱하여 검산한다. 확률이 고전 Markov 법칙을 따른다는 사실로 순수 상태를 고전 혼합 상태와 동일시하지 않는다.

E(I)=I, E(Z)=lambda Z와 고정된 dual 행렬 ell을 구별한다. Ell은 virtual 고정점이며 자동으로 physical site 밀도행렬이 아니다. 명시된 양쪽 경계가 각각 한 번의 transfer 후 정상 경계가 되는 것도 확인한다. CZZ(r)=lambda^r은 해당 사이트가 존재하는 유한 사슬의 결과다. 표의 r=4는 N>=5를 택한 독립 예제이며 기존 reader의 N<=4 제어를 확장한 것이 아니다.

Xi=-1/log|lambda|는 0<|lambda|<1에서 쓴다. Lambda=0과 GHZ endpoint는 별도로 다룬다. Bond dimension 2가 고정되어도 상관 길이는 발산할 수 있다. 부모 Hamiltonian을 정하지 않았으므로 transfer gap에서 물리 에너지 gap을 추론하지 않는다.

Column-vectorization에서 S=conj(G) tensor G이고 transfer의 similarity와 양쪽 경계·삽입 연산자를 함께 변환한다. Condition number는 수치 표현에 영향을 주지만 정확한 물리 상태는 보존한다. Jordan의 다항식 인자는 일반 블록에 대한 조건부 설명이며 이 family에 그런 블록이 있다고 주장하지 않는다. 관측량의 mode overlap도 상관에 관여한다.

## 재현

- CI 선언 NumPy·SciPy·SymPy 환경에서 `python3 scripts/learning/verify-quantum-representation-research.py`: 유한 행렬 전개와 전체 coherent 해, Schroedinger 방정식, MPS 직접 수축·모든 ZZ 쌍·복소 gauge를 독립 비교한다. 유리수 각도·transfer 특성다항식은 SymPy로 확인한다.
- `npx tsx --test scripts/learning-quantum-representation-research.test.ts`: 한영 수식·Fraction 코드·정확한 표·KaTeX와 원본문 보존을 검사한다.
- Matplotlib·Noto Sans CJK 환경에서 `python3 scripts/learning/generate-quantum-representation-research.py`: 기존 blue/orange/green 팔레트의 한영 SVG 네 개를 재생성한다. 수치 곡선과 정확 상계·점근선을 구별한다.
- 정적 미리보기에서 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-representation-research.cjs`: 한영 reader와 기존 boson/MPS 계산·안내·저장 복원을 검증한다. 시간 적분·infinite MPS·정규화 또는 Schmidt truncation 알고리즘을 새 UI 기능으로 추가한 것은 아니다.
