# 순차 압축·회로 미분 자료와 가정 대조

2026-11-13. 기존 Schmidt·회로 두 단원의 한영 본문과 원래 풀이를 보존하고 연구 절을 더한다. 원래 51영역·509주제를 유지하며 새 연구 대조와 기존 입문 완료를 따로 집계한다.

## 순차 SVD: 어떤 절차의 오차인가

[Schollwöck, The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/pdf/1008.3477), §§4.1.3·4.5.1은 SVD로 상태를 MPS로 만들고 절단하는 일차 자료다. §4.5.1의 압축 설명은 뒤 단계가 앞선 절단에 의존함을 지적한다. 본문은 유한 정규화 tensor에 왼쪽부터 한 번 시행하는 TT-SVD를 정확히 명시한다. 중간 residual은 정규화하지 않으며 positive budget, 실제 가능한 유지 차원의 최솟값, 직교정규 prefix 삽입을 사용한다.

실제 중첩 물리 projector를 고정한 뒤 버린 조각의 상호 직교를 증명한다. 따라서 현재 residual에서 버린 singular value 제곱의 합은 실제 최종 비정규화 오차 제곱과 같다. 원래 상태의 각 cut 꼬리는 contraction 논증으로 그 손실의 상한이 된다. 두 종류의 꼬리를 같다고 하지 않는다. 최종 한 번의 정규화에서 fidelity 제곱·trace distance·정렬된 Hilbert 거리를 따로 계산한다.

모든 cut rank 제한을 만족하는 비정규화 최적 오차와 비교한 sqrt(n−1) 보증은 직접 부등식으로 유도한다. 정규화된 최적화나 에너지 최소화와 혼동하지 않는다. 3,2,2,1의 sweep 오차 1/2보다 작은 547/1125를 주는 명시적 product 후보는 전체 최적성이 실패하는 실제 반례다. 그 후보도 전역 최적이라는 주장은 하지 않는다. 축퇴에서 선택 결과는 달라도 실제 선택의 인증은 유효하다. 기존 reader의 split-degeneracy 미선택 정책을 바꾸지 않는다.

## 회로 미분: 어떤 생성자와 등장 위치인가

[Schuld 외, Evaluating analytic gradients on quantum hardware](https://arxiv.org/pdf/1811.11184), §II·Theorem 1은 두 생성자 고유값에서의 parameter-shift 일차 자료다. 본문은 고정 입력·관측량과 한 번 등장하는 exp(−i theta G)를 가정하고 commutator·단일 주파수 전개로 공식의 계수와 이동량을 직접 확인한다. Ry 생성자는 Y/2이므로 이동 pi/2와 계수 1/2를 사용한다. 작은 간격의 차분이 아니다.

공유 변수의 등장 위치별 편미분과 고전 손실의 연쇄법칙은 별도로 적용한다. 동시에 모든 등장 위치를 이동하거나 기댓값 대신 비선형 손실을 이동하면 실패하는 예를 제시한다. Controlled Ry의 생성자는 0, ±1/2를 가져 단일 주파수 규칙이 일반적으로 맞지 않는다. 제어 X의 cos(theta/2) 반례를 계산한 뒤, 주파수 1/2·1 양쪽을 미분하는 네 평가 공식을 직접 유도한다. 이 추가 공식이 위 논문에 그대로 주어졌다고 인용하지 않는다.

측정 분산은 독립 Pauli ±1 shot 두 그룹을 전제로 한다. 알려진 양의 분산에서 실수 shot budget을 연속 최적화한 배분식이며 정수 횟수의 정확한 최적 배분으로 표시하지 않는다. 예비 추정·영분산·반올림을 구별한다. 매개변수 의존 잡음, 훈련 확장성이나 barren plateau를 이 작은 실험으로 인증하지 않는다.

## 재현과 시각화

- CI 선언 NumPy·SciPy·SymPy 환경에서 `python3 scripts/learning/verify-quantum-compression-gradient-research.py`: 실제 순차 SVD·prefix projector·최종 진폭·직접 trace norm·모든 bond rank를 검사한다. 직접 dense 회로 미분·단일/다중 주파수 이동과 독립 binomial 결과의 정확 분산을 비교한다.
- `npx tsx --test scripts/learning-quantum-compression-gradient-research.test.ts`: 원본문 hash·원래 풀이와 한영 수식·Fraction 표·KaTeX·등록 범위를 검사한다. 작은 부동소수점 잔차는 방향성 구간 인증이 아니다.
- Matplotlib·Noto Sans CJK에서 `python3 scripts/learning/generate-quantum-compression-gradient-research.py`: 한영 SVG 네 개를 재생성한다. 모든 곡선은 명시한 해석적 공식이고 shot variance는 Monte Carlo 표본이 아니다. 기존 blue/orange/green 팔레트를 유지한다.
- 정적 미리보기에서 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-compression-gradient-research.cjs`: 기존 한 cut Schmidt·고정 H/S/T 회로의 실제 계산과 예측 해설·안내·퀴즈·저장 복원을 확인한다. 순차 압축·연속 미분·hardware sampling 제어를 새 UI 기능으로 표시하지 않는다.
