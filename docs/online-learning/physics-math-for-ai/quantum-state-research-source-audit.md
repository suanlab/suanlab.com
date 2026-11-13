# 양자 상태·사영 오차 연구 확장의 자료 대조

2026-11-13. 기존 [양자 상태 자료 대조](quantum-state-source-audit.md)와 두 원고의 내적·길이·정규화 규약을 유지한다.

| 자료 | 확인한 내용 | 본문의 적용과 한계 |
|---|---|---|
| [Rabei·Arvind·Mukunda·Simon, Bargmann invariants](https://arxiv.org/pdf/quant-ph/9907080) | §II–III의 순환 내적·대표 위상 상쇄·양의 내적 정렬과 음의 argument 규약 | Φ=Arg B와 정렬 후 −Φ를 구별한다. 특정 Hamiltonian·단열 진화는 주장하지 않는다. |
| [NIST DLMF §18.9](https://dlmf.nist.gov/18.9) | Legendre 점화식과 Jacobi 계열 미분 공식 18.9.15 | 기존 shifted Legendre 규약을 유지한다. 미분 제곱 적분은 부분적분·직교성으로 직접 유도하고 1–24차 정확 유리수로 별도 확인한다. |
| [Phan Thành Nam, Functional Analysis II](https://www.math.lmu.de/~nam/LectureNotesFA2021.pdf) | Neumann Laplacian의 이차형식 정의역 H¹과 연산자 정의역의 구별 | 다항식의 기울기 형식을 계산하며 Neumann 강한 연산자의 정의역에 속한다고 가정하지 않는다. Dirichlet 형식과 혼용하지 않는다. |

국소 ray 계량, qubit의 Gram 행렬식·Bloch 삼중곱, 위치 TV 경계와 에너지 반례는 본문에서 직접 유도한다. 논문의 일반 결과를 이 예제의 수치 계산 근거로 대신하지 않는다. 독립 검산은 정규화 복소 벡터의 내적과 정확한 다항식 계수 계산에서 출발한다.

기존 16정수 입력·두 상태/기저 계산과 조각별 affine 파동함수 실습은 그대로다. 세 상태 순환·다항식 준비·에너지 그림은 독립 연구 예제다. 새 시간 전개·Hamiltonian·측정 instrument 조작을 구현했다고 표시하지 않는다.
