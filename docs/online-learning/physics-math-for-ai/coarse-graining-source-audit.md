# Coarse graining 원문과 직접 유도의 범위

2026-11-08. 기존 등록 내용은 보존하고 동역학적 집계·기억·유효 기술을 추가로 준비한다.

## 확인한 1차 자료

1. Peter Buchholz, *Exact and ordinary lumpability in finite Markov chains*, Journal of Applied Probability31(1),59–75(1994), [출판사 원문 페이지](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/exact-and-ordinary-lumpability-in-finite-markov-chains/2DC748F09D80BEEB03CCF18036E149D7). 출판사 서지와 초록을 확인했다. 본문 전체를 읽었다고 주장하지 않는다. 역사적·용어 배경의 자료이며, 이 교재의 모든 초기 법칙에 대한 CT=AC 조건과 세 상태 출구 확률 동치는 직접 증명한다. 논문의 exact/ordinary 용어를 본문 확인 없이 이 교재의 strong 조건에 일괄 대응시키지 않는다.
2. Gouasmi, Parish, Duraisamy, *A priori estimation of memory effects in reduced-order models of nonlinear systems using the Mori–Zwanzig formalism*(2017), [공개 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5627381/). 서론과2(a)절의 선형 분할·숨은 변수 해·기억식 도입을 읽었다. 연속 선형계의 정확한 소거를 다음 가족의 비교 근거로 삼는다. 비선형 근사 방법 전체를 구현했다고 주장하지 않는다.

## 직접 유도와 구현상의 구별

첫 가족은 연속 물리계에 시계를 임의로 붙이는 대신, 명시한 이산 세 상태의 법칙을 정확히 전파한다. 행렬은 열 확률 관례다. C와 R에서 H=I−RC를 만들고 p=Rm+h를 대입해 거시/숨은 갱신을 얻는다. 숨은 갱신의 반복 대입으로 초기 준비 항과 이산 기억 커널을 직접 계산한다. 이 유한 선형대수 항등식은 본문 접근이 제한된 논문의 정리를 대신 인용한 것이 아니다.

기억 커널의 성분과 h는 부호가 있고, 전체 합이 정확 법칙을 재구성한다. 임의의 noise를 넣지 않는다. 독립 참조는 Python Fraction으로 세 성분을 직접 갱신하고, 숨은 좌표 h=p₀−α(p₀+p₁)의 스칼라 전파로 커널을 구한다. 생산 코드는3차원 사영 행렬을 사용한다. 서로의 코드를 import하지 않는다.

정상 법칙 보존과 모든 초기화의 동적 닫힘, 초기 lifting 일치와 이후의 불변성, 지역적 누락 항과 전역 근사 오차를 구별한다. 음수 근사·긴 기억·부호 교대·표시 범위 아래 비영 계수를 제어 사례로 남긴다. 특정 작은 예제의 성공을 비선형 사영·일반 Markov 극한·재규격화 이론 전체의 검증으로 확대하지 않는다.
