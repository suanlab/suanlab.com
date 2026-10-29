# 교육과정의 근거 자료와 논문 독해 관리

확인일: 2026-10-26. 아래는 교육과정의 범위·정의·연구 연결을 검토하는 시작 자료다.
지식 영역의 구성과 학습 경로는 이 교재의 설계이며 특정 책의 목차를 그대로 옮긴 것이 아니다.
자료를 추가할 수 있고 아래 목록에 전체 참고 문헌 수의 상한을 두지 않는다.

## 수학과 학습의 기반

| 원문 | 이 설계에서 참고하는 범위 |
| --- | --- |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | 선형·기하·미적분·확률과 모델을 연결하는 교육적 출발점 |
| [MIT Matrix Calculus for Machine Learning and Beyond](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/) | 행렬·벡터 공간의 미분과 큰 계산의 연결 |
| [Convex Optimization — Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) | 볼록 문제·조건·쌍대·해법·응용의 범위 |
| [Stanford Modern Statistical Learning](https://web.stanford.edu/class/stats315a/) | 통계·최적화·학습 이론을 연결하는 연구 수준의 범위 |
| [Deep Learning — 저자 제공 온라인판](https://www.deeplearningbook.org/) | 수학 기반에서 신경망·표현·확률 모델·연구로 이어지는 연결 |

각 자료의 모든 정리와 증명을 이번 설계 단계에서 검토했다는 뜻은 아니다. 실제 단원은
관련 정의와 정리의 정확한 위치·가정·표기를 확인하고 별도의 유도·예제·검산을 작성한다.

## 물리학의 기반

- [David Tong의 이론물리 강의 원문 목록](https://davidtong.org/teaching/): 역학·전자기·양자·유체·통계·장 등 기초부터 연구 수준까지의 연결 범위.
- [MIT Statistical Physics I 강의계획](https://www.ocw.mit.edu/courses/8-044-statistical-physics-i-spring-2013/pages/syllabus/): 확률·통계역학·열역학의 연결.
- [OpenStax Newton's Second Law](https://openstax.org/books/university-physics-volume-1/pages/5-3-newtons-second-law): 현재 벡터·힘 시범 원고의 법칙·단위 검토.

각 물리 분야의 집필에서는 해당 강의·교재·논문의 구체적인 장과 문제를 연결한다.
유효 이론·근사·실험 조건을 명시하고 AI에 적용하는 수학적 대응은 별도로 검토한다.

## AI 동작과 연구를 연결하는 원 논문

| 원문과 확인 버전 | 교육과정에서 연결할 개념 |
| --- | --- |
| [Attention Is All You Need, v7](https://arxiv.org/abs/1706.03762v7) | attention·표현·조건부 모델과 계산 구조 |
| [Score-Based Generative Modeling through SDEs, v2](https://arxiv.org/abs/2011.13456v2) | score·확률과정·전방/역방향 생성·수치해법 |
| [Flow Matching for Generative Modeling, v2](https://arxiv.org/abs/2210.02747v2) | 벡터장·분포 경로·연속 생성 |
| [Neural Ordinary Differential Equations, v6](https://arxiv.org/abs/1806.07366v6) | 연속 상태·ODE 해법·미분 |
| [Geometric Deep Learning, v2](https://arxiv.org/abs/2104.13478v2) | 군·대칭·그래프·기하와 모델 구조 |
| [Physics Informed Deep Learning Part I, v1](https://arxiv.org/abs/1711.10561v1) | 방정식·관측·물리 제약과 학습 |
| [Fourier Neural Operator, v3](https://arxiv.org/abs/2010.08895v3) | 함수 공간·연산자·Fourier와 PDE 모델 |

이번 단계는 원문의 제목·공개 정보·초록을 이용해 관련 학습 범위를 확인한 것이다.
위 논문의 증명 전체나 실험 성능을 재현 완료한 것으로 표시하지 않는다.
성능 수치를 다른 문제의 일반적인 보장으로 사용하지 않는다.

## 단원별 문헌 기록 양식

- 정의·정리·법칙 또는 연구 질문과 연결할 단원 ID.
- 원문 제목·저자·공개/출판 정보·직접 링크·확인 버전·확인 날짜.
- 실제로 읽은 절·식·정리·그림·실험과 해당 주장의 가정.
- 교육적 재구성, 직접 유도, 재현 코드와 원문 결과의 차이.
- 검토 상태: 범위 확인 / 본문 독해 / 유도 검토 / 재현 / 조건 변경 실험.
- 반례·수정·상충 결과와 후속 단원의 필요성.

신규 연구를 읽을 때마다 필요한 개념을 지식 지도에 추가한다. 논문의 새 이름을 나열하는 데서
끝내지 않고 어떤 수학·물리·모델 동작을 배워야 주장을 이해하고 검증할 수 있는지 기록한다.

## 수·함수·좌표 기초 확장의 참고 자료

- **OpenStax, Algebra and Trigonometry 2e** — [출판사 안내](https://openstax.org/details/books/algebra-and-trigonometry-2e), 확인 2026-10-27.
  수 체계·방정식·함수·지수/로그·삼각·수열의 추가 읽기 자료다. 원고의 작은 계산과 증명은 직접 작성했으며,
  출판사 안내 확인을 교재 전체의 모든 증명·연습 검토나 실험 재현으로 표시하지 않는다.
- **BIPM, The International System of Units, 9th edition, updated 2026** — [공식 SI Brochure 안내](https://www.bipm.org/en/publications/si-brochure), 확인 2026-10-27.
  단위 표기·접두어·차원 구분의 참고 자료다. 이 확장에서는 공식 안내의 판·갱신 정보를 확인했다.
  cm→m·면적·속력 예제는 독립 계산이며 실제 측정·계측 장치의 교정 재현이 아니다.
- **MIT, Mathematics for Computer Science, Spring 2015** — 기존 등록 자료를 점근 표기의 보충 읽기로 연결했다.
  Big-O의 일차 비용 증명서와 양화사 반박은 본문에서 직접 유도한다. 유한 실행 시간 측정으로 보편 상한을 입증했다고 주장하지 않는다.


## 이산수학 확장의 참고 자료 (2026-10-28)

- [MIT Mathematics for Computer Science, Spring 2015](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/):
  공식 강좌 안내의 정의·증명·그래프·상태 기계·세기 범위를 확인했다. 기존 자료의 보충 읽기로 연결한다.
- [MIT Theory of Computation, Fall 2020](https://ocw.mit.edu/courses/18-404j-theory-of-computation-fall-2020/):
  공식 강좌 안내의 정규언어·계산 가능성 범위를 확인했다. 두 DFA의 구성과 불변식 증명은 직접 작성했다.
- 위 확인은 공식 안내의 범위 확인이다. 교재 전체 독해·모든 증명 검토·강의 과제 재현으로 표시하지 않는다.
  작은 조합·그래프·ANF·궤적 계산은 본문 조건에서 독립적으로 유도하고 유한 범위를 완전 열거한다.


## 선형 구조 보완의 참고 자료 (2026-10-28)

- [MIT Linear Algebra, Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/):
  공식 안내의 벡터공간·행렬식·고유값·유사·정부호 행렬 범위를 확인했다. 교재 전체나 모든 강의 독해 완료를 뜻하지 않는다.
- 작은 행렬의 랭크–영차원 논증, 면적·trace 불변량, PSD·Loewner 조건, 열 우선 벡터화 항등식은
  본문에서 직접 유도했다. Kronecker 구조와 의사역행렬은 명시한 성분 전개·SVD 논증으로 검산하며
  이 참고 강좌 안내를 세부 항등식의 직접 인용이나 모든 수치 실험 재현으로 표시하지 않는다.


## 해석학·다변수 미적분 확장의 자료 (2026-10-28)

- [MIT Real Analysis, Fall 2020](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/): 공식 안내의
  수열·급수·연속·미분·Riemann 적분·함수열·균등성·극한 교환 범위를 확인해 보충 읽기로 연결했다.
- [MIT Multivariable Calculus, Fall 2010](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/):
  공식 안내의 미분·벡터장·선적분·면적분·Gauss·Stokes 구성을 확인했다.
- 확인은 공식 강좌 범위의 확인이다. 모든 강의·증명을 독해하거나 과제를 재현한 것으로 표시하지 않는다.
  ε–δ 예제·꼬리 상한·봉우리 반례·작은 행렬 미분·벡터장·HVP 계산은 본문에서 독립 구성했다.
  π²/6 평가는 알려진 값으로 사용하며 현 단원에서 그 정확한 평가까지 증명했다고 주장하지 않는다.


## 복소해석 확장의 참고 자료 (2026-10-28)

- [MIT Complex Variables with Applications, Spring 2018](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/):
  공식 안내의 정칙 함수와 조화함수·적분·Fourier/Laplace 응용 범위를 확인해 보충 읽기로 연결했다.
- 확인은 공식 강좌 범위의 확인이며 모든 강의·과제·증명을 독해하거나 재현했다는 뜻이 아니다.
  극형식·CR·원 적분·유수·로그 경로·Hermitian·Wirtinger·신호 예제는 본문의 명시적 식으로 독립 구성했다.
- Cauchy·유수 정리·해석적 연속의 일반 이론은 조건과 해석을 소개한다. 작은 수치 구적을 일반 정리의 증명으로 표시하지 않는다.

## 텐서·다중선형대수 원문 (2026-10-28)

- [Kolda–Bader, Tensor Decompositions and Applications (2009)](https://www.kolda.net/publication/TensorReview.pdf): CP·Tucker와 다중선형 표기, 일반 텐서 근사에서 행렬 직관의 한계.
- [Oseledets, Tensor-Train Decomposition (2011)](https://doi.org/10.1137/090752286): TT 코어·내부 랭크와 펼침 행렬의 관계.
- [NumPy 공식 broadcasting 문서](https://numpy.org/doc/stable/user/basics.broadcasting.html): 뒤쪽 축 정렬과 크기 일치·singleton 규칙.

직교 좌표 항, 세 코어의 게이지, 세 행렬의 수축 비용 예제는 교재에서 별도로 구성했다.
분해 최적화·TT-SVD·GPU 벤치마크를 실행한 결과로 표시하지 않는다.

## 추상대수 입문 자료 (2026-10-29)

- [Judson, Abstract Algebra: Theory and Applications](https://judsonbooks.org/abstract-algebra-theory-and-applications/): 군·환·체, 준동형, 몫, 다항식·유한체와 군 작용의 후속 읽기.
- [Etingof 외, Introduction to Representation Theory](https://math.mit.edu/~etingof/replect.pdf): 대수·표현·아이디얼·몫과 유한군 표현을 심화하는 자료.

원고의 작은 잉여류·두 cubic 법 다항식·D4 좌표·C4 행렬 평균 예제는 별도로 구성해 검산한다.
일반 분류 정리, 다항식 시스템 소거기나 전체 연구 논문의 재현 결과를 뜻하지 않는다.

## 확률 기초 자료 (2026-10-29)

- [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/): 확률공리·조건부·단일/다중 확률변수·극한정리의 후속 읽기.
- [Harvard Statistics 110](https://stat110.hsites.harvard.edu/): Blitzstein–Hwang 교재와 조건부·분포·극한정리 강의 자료의 공식 입구.
- [Kingma–Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): 재매개화에 의한 연속 잠재변수 목적함수 경사 추정의 원 논문.

네 점 사건, 두 이진 변수, 단위원 변환, 이항 질량과 이차 손실 예제는 교재에서 별도로 구성했다.
VAE 전체 학습·논문 재현을 실행한 결과로 표시하지 않는다.
