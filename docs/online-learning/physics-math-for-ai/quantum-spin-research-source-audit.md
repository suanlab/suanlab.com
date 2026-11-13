# Spin 회전 정보·coherent/cat alias 자료 대조

2026-11-13. 기존 [spin 원문과 수치 근거](quantum-spin-source-audit.md)의 능동 회전·반각·metric 규약을 유지한다.

- [Paris, Quantum Estimation for Quantum Technology](https://arxiv.org/pdf/0804.2981): §2의 Fisher 정보·반복 측정 경계, §4.1 식26–27의 순수 상태 metric/생성자 분산을 읽었다. 본문은 같은 알려진 준비·고정 검출기·단일 각도의 국소 모델을 지정하고, 양의 확률·미분 가능성·국소 불편성 아래 Cramér–Rao 경계를 유도한다. 유한 표본 달성·전역 식별·임의 혼합 상태 모델을 보장하지 않는다.
- [MIT8.321 Lecture19](https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/c491464fb065aa4cc8a99cc562475183_MIT8_321F17_lec19.pdf): 능동 회전의 부호·각운동량 생성자·자기 라벨과 ladder 규약을 대조했다. 같은 j의 단일 irreducible 표현을 사용하며 다입자 자원 해석을 추가하지 않는다.

Spin1/2의 검출기 정보 경계는 Bloch 접선·Cauchy–Schwarz로 직접 유도한다. 기울어진 검출기의 유리수 반각 예제는 영 확률 없이도 일차 정보가 사라짐과 전역적으로 같은 히스토그램을 주는 각도를 함께 보여 준다. 정칙 식의0/0 경계에 pseudocount를 넣지 않는다.

높은 spin의 coherent 상태는 이항정리와 기존 polynomial metric으로 정규화한다. 두 끝점 중첩은 j>0에서만 사용한다. 직접 계산한 곡률·fringe·ray 주기는 국소 민감도와 전역 alias를 구별한다. j=0 singlet은 따로 처리하며 큰 quantum 정보만으로 실제 준비·검출·총 비용의 우위를 주장하지 않는다.

독립 검산은 일반 행렬 exponential과 밀도행렬 미분에서 순수 상태 SLD를 만들고 직접 측정 확률·score를 구한다. 정규직교 제곱근 ladder에서 coherent/cat 상태와 모멘트·fringe·대표 부호·ray alias를 다시 계산한다. 본문 유리수 표는 별도로 정확하게 실행한다. 부동소수점 대조는 기존 생산 계산기의 방향성 구간 인증과 구별한다.

기존 원본문과 풀이를 그대로 보존한다. 두 준비는 지원 범위 j≤4의 원시 입력으로 표현할 수 있지만, 새로운 추정기·시행 표본·Fisher 정보·임의 coherent projector 제어를 구현했다고 표시하지 않는다.
