# Euclidean 양자 커널: 일차 자료와 집필 범위

2026-11-10. 경로적분 네 처리 중 두 번째 `euclidean-quantum-kernels-and-time-slicing`를 준비한다. 첫 Brownian bridge 단원은 실제 교재에 등록했다. 이번 단원은 자료 검토·유한 모형 유도 검산 단계이며 원고·생산 계산기·Worker·학습 화면은 아직 없다.

## 직접 확인한 자료

- [Andreas Wipf, Euclidean Path Integral §6.1](https://www.tpi.uni-jena.de/qfphysics/homepage/wipf/lectures/pfad/pfad6.pdf): 인쇄43–45쪽의 자기수반·아래로 유계인 Hamiltonian, 허수 시간 semigroup, 자유입자와 Mehler 커널 식6.9–6.10을 확인했다. 자료의 ℏ=1 규약을 본 단원에서는 복원한다.
- [Wipf, Path Integrals in Statistical Mechanics §8.1](https://www.tpi.uni-jena.de/qfphysics/homepage/wipf/lectures/pfad/pfad8.pdf): 인쇄72–73쪽의 커널 대각 적분·trace·주기 경로와 조화진동자 에너지 합 식8.1–8.4,8.6–8.8을 확인했다. 온도는 β=1/(k_B T),τ=βℏ로 선언한다. 추출된 식8.5의 분모 표기가 아니라 앞 장의 Mehler 식으로 대각 적분을 독립 유도한다.
- [Christian Grosche, An Introduction into the Feynman Path Integral §III.1–III.2](https://arxiv.org/pdf/hep-th/9302097): 인쇄33–35쪽의 자유 Gaussian 합성과 조화진동자 이차 작용·고정끝 변동을 확인했다. 이 자료의 중점 potential 규약을 아래 사다리꼴 potential과 같은 유한 격자로 취급하지 않는다.

[Siu A. Chin의2023논문](https://arxiv.org/abs/2310.18809)은 제목·초록만 확인했다. 본문 HTML 접근은 실패했으므로 고차 분할이나 논문 내 식을 읽고 검증한 근거로 세지 않는다. 이번 유한 행렬·닫힌 사슬·분할 오차 식은 직접 유도했다.

## 학습자가 구분할 대상

1. 양끝 고정 커널과, 그 커널로 조건화한 내부 경로 확률 분포.
2. 양의 열핵, 보존적 Markov 전이밀도, 양자 Born 확률, 정규화한 열적 밀도 행렬.
3. 커널의 대각 원소와 대각 전체 적분인 partition function.
4. 정확한 유한 시간 분할 적분과 목표 Hamiltonian의 연속 시간 커널.
5. 대칭 분할과 오른쪽 끝점 분할, 대각 관측에서는 드러나지 않는 끝점 오차.
6. 고정 N의 유한 항등식, 선택 예제의 수치 오차율, 일반 potential의 연산자 수렴 정리.

## 명시적으로 제외하는 주장

자유입자를 무한 직선에서 계산할 때 열적 trace는 발산한다. 영 모드를 삭제한 뒤 같은 물리계의 유한 분배함수라고 부르지 않는다. 임의 potential·경계·자기장·비가환 연산자에 대한 무조건적인 Wick 회전·Trotter 수렴이나 균일 오차 보증도 제공하지 않는다. 열핵의 양성만으로 Born 확률 또는 전체 질량1인 전이 법칙이라고 선언하지 않는다.

## 제작 순서

[유한 모형과 검산](euclidean-kernels-foundation-status.md)→일반 대학원생용 한영 상세 원고·손계산→정확분수 계산과 독립 검산→실제 Worker 수명 관리→기존 홈페이지 스타일의8개념 그림·예측 실습·해설 퀴즈→실제 Interactive Book 등록 순서다. 이번 준비를 등록 단원 수에 더하지 않는다.
