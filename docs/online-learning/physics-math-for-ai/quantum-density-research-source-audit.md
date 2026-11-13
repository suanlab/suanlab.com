# 유한 표본 tomography·시간 dephasing 자료 대조

2026-11-13. 기존 [ensemble 수치 계약](quantum-density-ensemble-source-audit.md), [spectrum·channel 계약](quantum-density-spectrum-source-audit.md)과 [실제 교재 통합](quantum-density-integration-status.md)을 보존한다.

- [Hradil, Quantum state estimation](https://arxiv.org/pdf/quant-ph/9609012): 선형 역산의 양성 문제와 Born 확률에 대한 우도 함수, 식1·3을 읽었다. 본문의 세 축 binomial 계수·물리적 제약·불균등 시행 비교는 직접 구성하며 원 논문의 일반 복원 알고리즘을 구현하지 않는다.
- [Rivas·Huelga·Plenio, Entanglement and non-Markovianity of quantum evolutions](https://arxiv.org/pdf/0911.4270): 식1–2의 비음의 율·CPTP 전파 합성 배경을 확인했다. 본문은 지정한 실수 dephasing 가족의 CP-divisibility를 직접 검사한다. 임의의 모든 비-Markov 정의를 서로 동치로 취급하지 않는다.
- [Breuer·Laine·Piilo, Measure for the degree of non-Markovian behavior](https://arxiv.org/pdf/0908.0238): 식1–2의 trace distance와 CPTP 수축을 확인했다. 이 가족의 ±X 쌍이 드러내는 회복을 설명하며, 한 쌍으로 모든 일반 channel의 divisibility를 특성화한다고 주장하지 않는다.

Raw Pauli 평균의 모멘트와 confidence box는 지정한 독립 표본 모델에서 직접 유도한다. 중심화 로그 모멘트생성함수의 이차 미분≤1, 지수 꼬리 최적화와 union bound를 사용한다. 신뢰영역은 raw 추정에 중심을 두고 Bloch ball과 교차하며 적합 상태로 재중심화하지 않는다. 방사형 거리 사영·음의 로그 우도·rank의 통계적 의미를 구별한다.

Dephasing은 η(s)≠0의 가역 중간 map과 η(s)=0의 같은 입력 논증을 분리한다. 고정 무작위 주파수 모델은 한 시행 안에서 부호를 유지하며, 구간마다 새로 추출하는 모델과 다르다. γ의 계수2·시간 단위·비가역 영점·음의 개별 CPTP 채널을 명시한다. 기존 u slider를 물리 시계로 바꾸지 않는다.

독립 검산은 정확 binomial 합과 일반 3좌표 제약 최적화, 직접 Choi·합성·4×4 unitary 환경 부분추적·특이값 trace norm을 사용한다. 본문의 경계 호 이분법이나 scalar 거리 식을 다시 호출하지 않는다. 부동소수점 표·비교를 기존 생산 계산기의 방향성 구간 인증과 구별한다. 원본문과 풀이를 보존하고 새로운 표본·MLE·환경·시간율 UI를 구현했다고 표시하지 않는다.
