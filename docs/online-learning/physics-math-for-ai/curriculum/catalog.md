# AI를 위한 물리와 수학 지식 영역 상세 색인

이 색인은 최초 등록 범위이며 영역·장·절의 개수에 상한을 두지 않는다. 연구 문제에 필요한
개념이 생기면 선수 관계와 검증 기준을 갖춘 단원을 계속 추가한다. 현재 상태는 교육과정 설계다.

영역의 선수 목록은 진입에 필요한 해당 절을 가리킨다. 앞 영역 전체를 끝내야 다음 영역의
모든 내용을 읽을 수 있다는 뜻이 아니다. 구체적인 필수 선수 절은 단원 집필 때 지정한다.

각 영역은 아래 학습 내용을 여러 장·절로 풀어 집필한다. 정의만 나열하지 않고 유도·증명,
조건·반례, 계산·시각화, AI 동작 분석, 연구 과제를 함께 제공한다.

[전체 설계로 이동](../../physics-math-for-ai.md) · [AI 동작과 연구 경로](ai-research-paths.md)

## 영역 색인

- [논리·집합·증명](#logic-proof)
- [수·함수·좌표·기초 기하](#elementary-math)
- [이산수학과 조합 구조](#discrete-math)
- [선형대수와 행렬 분석](#linear-algebra)
- [미적분과 실해석](#analysis)
- [다변수·벡터·행렬 미적분](#matrix-calculus)
- [복소수 해석과 복소 계산](#complex-analysis)
- [텐서와 다중선형대수](#tensor-methods)
- [추상대수와 대수적 구조](#abstract-algebra)
- [확률과 확률변수](#probability)
- [측도론적 확률과 극한](#measure-probability)
- [통계적 추정과 실험](#statistics)
- [Bayes 추론과 잠재변수](#bayesian-inference)
- [정보이론과 압축](#information-theory)
- [고차원 확률과 무작위 행렬](#high-dimensional)
- [확률 그래프 모델과 구조적 추론](#graphical-models)
- [부호·계산과 알고리즘 정보](#coding-algorithmic-information)
- [함수해석과 연산자](#functional-analysis)
- [근사이론·커널·표현력](#approximation-kernels)
- [조화해석과 신호 처리](#harmonic-analysis)
- [미분기하·위상·데이터 형상](#geometry-topology)
- [군·대칭·표현론](#symmetry-groups)
- [볼록해석과 제약 최적화](#convex-optimization)
- [비볼록·확률·대규모 최적화](#stochastic-optimization)
- [변분법과 기하적 최적화](#variational-geometry)
- [최적수송과 분포의 기하](#optimal-transport)
- [수치계산과 자동 미분](#numerical-computing)
- [상미분·편미분방정식](#differential-equations)
- [확률과정·확률미분방정식](#stochastic-processes)
- [과학계산과 수치 시뮬레이션](#numerical-simulation)
- [알고리즘·계산복잡도·계산 구조](#algorithms-complexity)
- [통계학습·일반화·강건성](#learning-theory)
- [인과추론과 식별 가능성](#causality)
- [제어·최적제어·강화학습의 수학](#control-decision)
- [조합 최적화와 형식적 추론](#discrete-optimization)
- [게임이론과 다중 행위자 학습](#game-theory)
- [프라이버시·공정성·강건성의 수학](#statistical-privacy)
- [측정·단위·물리 모델링](#physical-modeling)
- [뉴턴 역학과 강체 운동](#classical-mechanics)
- [해석역학과 구조 보존](#analytical-mechanics)
- [파동·진동·음향·광학](#waves-optics)
- [전자기학과 전자기장](#electromagnetism)
- [연속체·유체·탄성·수송](#continuum-physics)
- [열역학과 평형 통계역학](#thermo-statmech)
- [비평형·확률열역학·복잡계](#nonequilibrium)
- [양자역학과 양자정보](#quantum-physics)
- [장이론·임계현상·다중스케일](#field-multiscale)
- [상대론과 시공간 기하](#relativity)
- [물질·계산 장치·물리적 자원](#matter-computation)
- [역문제·동화·실험 설계](#inverse-problems)
- [수학·물리 기반 AI 연구 방법](#research-methods)

## 수학의 언어와 기초

<a id="logic-proof"></a>
### 논리·집합·증명
식별자: `logic-proof`. 선수 영역: 처음 시작할 수 있는 영역.

**학습 내용:** 명제와 술어, 전칭·존재 양화사, 집합·함수·관계, 동치·순서·사상, 직접·대우·모순 증명, 수학적 귀납법, 반례 구성, 필요·충분조건, 정의·정리·가정·결론의 구분.

**성취 기준:** 정리의 양화사와 가정을 읽고 증명 또는 반례로 주장을 검토한다.

**AI 연결:** 모델 보장 조건, 논문 증명, 형식 추론과 프로그램 검증.

**시각화·실험:** 집합 포함·함수 대응과 반례를 조작하는 논리 그래프.

<a id="elementary-math"></a>
### 수·함수·좌표·기초 기하
식별자: `elementary-math`. 선수 영역: `logic-proof`.

**학습 내용:** 정수·유리수·분수·비율·단위 변환, 실수·복소수, 방정식·부등식, 다항·유리함수, 지수·로그, 삼각함수, 함수 합성·역함수, 좌표·거리·각도, 수열·급수의 도입, 단위와 차원, 점근 표기.

**성취 기준:** 식·그래프·좌표를 서로 바꾸어 설명하고 정의역·단위·범위를 확인한다.

**AI 연결:** 활성화·확률·손실 함수 읽기, 표현 공간과 모델 크기 이해.

**시각화·실험:** 정의역과 함수 매개변수를 바꿔 그래프·좌표·기울기를 관찰.

<a id="discrete-math"></a>
### 이산수학과 조합 구조
식별자: `discrete-math`. 선수 영역: `logic-proof`, `elementary-math`.

**학습 내용:** 세기·순열·조합, 이항계수·포함배제, 점화식·생성함수, 그래프·트리·DAG, 연결성·경로·순환, Boolean 대수, 오토마타·형식언어, 부분순서·격자, 유한 상태와 전이.

**성취 기준:** 유한한 구조와 상태 공간을 정의하고 개수·경로·전이를 계산한다.

**AI 연결:** 탐색, 토큰·문법, 그래프 신경망, 구조적 추론.

**시각화·실험:** 트리 확장·조합 수·상태 전이와 계산 경로.

## 선형 구조와 해석

<a id="linear-algebra"></a>
### 선형대수와 행렬 분석
식별자: `linear-algebra`. 선수 영역: `elementary-math`.

**학습 내용:** 벡터·벡터공간, 기저·차원·랭크·영공간, 선형사상·행렬, 내적·노름·직교·투영, 선형 방정식·최소제곱, 행렬식·trace, 고유값·고유벡터, 대칭·양의 준정부호 행렬, SVD·의사역행렬, 이차형식·행렬 부등식, Kronecker 곱.

**성취 기준:** 선형층·투영·분해를 유도하고 특이성·조건·차원을 설명한다.

**AI 연결:** 신경망 선형층, 회귀, PCA, 임베딩, 저랭크 적응.

**시각화·실험:** 기저·격자·점 구름의 변환, 고유방향, 특이값과 정보 손실.

<a id="analysis"></a>
### 미적분과 실해석
식별자: `analysis`. 선수 영역: `elementary-math`, `logic-proof`.

**학습 내용:** 극한·연속·완비성, 미분·평균값정리, 적분·미적분 기본정리, Taylor 전개와 나머지, 수열·급수·함수열, 점별·균등 수렴, compactness·연속성, 볼록성의 도입, 적분과 극한의 교환 조건.

**성취 기준:** 변화율·누적·근사의 의미와 수렴 조건을 설명하고 적용이 실패하는 반례를 만든다.

**AI 연결:** 손실 근사, 학습 수렴 가정, 함수 근사와 연속 모델.

**시각화·실험:** 함수 확대·접선·누적 면적과 수렴 실패 사례.

<a id="matrix-calculus"></a>
### 다변수·벡터·행렬 미적분
식별자: `matrix-calculus`. 선수 영역: `linear-algebra`, `analysis`.

**학습 내용:** 전미분·선형화, 편미분·방향미분, gradient·Jacobian·Hessian, chain rule, 암묵함수·역함수 정리, 행렬·trace·logdet 미분, 벡터장·발산·회전, 선·면·체적 적분, Gauss·Stokes 정리, JVP·VJP·고차 미분.

**성취 기준:** 스칼라·벡터·행렬 함수의 미분 형상과 연쇄법칙을 유도하고 수치로 검산한다.

**AI 연결:** 역전파, attention, 정규화, 자동 미분, 연속 생성모델.

**시각화·실험:** 접평면·벡터장·Jacobian 변환과 계산 그래프의 미분 전달.

<a id="complex-analysis"></a>
### 복소수 해석과 복소 계산
식별자: `complex-analysis`. 선수 영역: `analysis`, `linear-algebra`.

**학습 내용:** 복소평면·극형식, 정칙성·Cauchy-Riemann 조건, 복소 적분·Cauchy 정리, 유수, 해석적 함수·특이점, 해석적 연속, 복소 선형대수, Wirtinger 미분, 복소 신호와 위상.

**성취 기준:** 복소 표현·위상을 계산하고 정칙 함수와 실수값 목적함수의 미분을 구분한다.

**AI 연결:** Fourier 계산, 복소 신경망, 파동·양자 AI.

**시각화·실험:** 복소 위상 회전과 실수·허수 좌표의 변환.

<a id="tensor-methods"></a>
### 텐서와 다중선형대수
식별자: `tensor-methods`. 선수 영역: `linear-algebra`.

**학습 내용:** 텐서의 차수·축·shape, 지표와 수축, 다중선형사상, broadcasting과 연산 의미, CP·Tucker 분해, tensor train, tensor network, 저랭크 근사, 대칭·반대칭 텐서, 저장·계산 복잡도.

**성취 기준:** 축의 역할을 구분하고 텐서 연산·분해의 수학과 비용을 설명한다.

**AI 연결:** 배치·헤드·채널, 다중모달 결합, 모델 압축, 양자 상태 근사.

**시각화·실험:** 축·수축·분해 네트워크와 작은 텐서의 복원.

<a id="abstract-algebra"></a>
### 추상대수와 대수적 구조
식별자: `abstract-algebra`. 선수 영역: `logic-proof`, `discrete-math`, `linear-algebra`.

**학습 내용:** 군·환·체, 준동형·동형, 부분구조·몫구조, 다항식·유한체, module, 대수적 대칭, 다항식 시스템, invariant의 구성, 대수적 표현의 적용 범위.

**성취 기준:** 구조 보존 사상과 대수적 제약을 정의하고 작은 예제에서 계산한다.

**AI 연결:** 대칭·표현, 대수적 추론, 코드·구조 학습의 연구 기반.

**시각화·실험:** 연산·몫구조·유한체와 구조 보존 사상.

## 확률과 추론

<a id="probability"></a>
### 확률과 확률변수
식별자: `probability`. 선수 영역: `analysis`, `linear-algebra`.

**학습 내용:** 사건·확률공리, 조건부 확률·독립·Bayes, 이산·연속 분포, 밀도·질량·누적분포, 기대·분산·공분산, 다변량 Gaussian, 변수변환·Jacobian, 결합·주변·조건부 분포, 대수의 법칙·중심극한정리, 재매개화.

**성취 기준:** 확률·밀도·표본을 구분하고 조건부 예측과 분포 변환을 계산한다.

**AI 연결:** 확률적 모델, 샘플링, likelihood, 생성과 불확실성.

**시각화·실험:** 분포·표본·조건부 단면과 변수변환의 체적.

<a id="measure-probability"></a>
### 측도론적 확률과 극한
식별자: `measure-probability`. 선수 영역: `probability`, `analysis`.

**학습 내용:** sigma 대수·측도, Lebesgue 적분, 거의 확실한 성질, 수렴정리·우세수렴, Fubini·Tonelli, 확률측도·기대, 조건부 기대, Radon-Nikodym 도함수, 약수렴·확률수렴, 확률과 적분의 교환 조건.

**성취 기준:** 기댓값·극한·미분의 교환에 필요한 가정을 검토하고 정리를 적용한다.

**AI 연결:** 확률 학습 이론, 생성 경로, 연속 확률 모델의 정당화.

**시각화·실험:** 수렴 방식과 질량 이동을 작은 반례와 수치 표로 비교.

<a id="statistics"></a>
### 통계적 추정과 실험
식별자: `statistics`. 선수 영역: `probability`, `matrix-calculus`.

**학습 내용:** 표본·모수·추정량, 편향·분산·일치성, likelihood·MLE·MAP, 지수족·충분통계량, 신뢰구간·가설검정, Bootstrap·재표집, 회귀·잔차, 점근 정규성과 delta method, 검정력·다중 비교, 실험 설계·오차 추정.

**성취 기준:** 추정과 불확실성을 계산하고 데이터 분할·검정·반복 실험의 조건을 설명한다.

**AI 연결:** 모델 평가, ablation, 성능 차이 검증, 데이터 효율 연구.

**시각화·실험:** 표본·추정 분포·구간·검정력의 반복 실험.

<a id="bayesian-inference"></a>
### Bayes 추론과 잠재변수
식별자: `bayesian-inference`. 선수 영역: `statistics`.

**학습 내용:** 사전·우도·사후, 공액성·사후 예측, 계층 모델, evidence, 잠재변수·EM, 변분 추론·ELBO, importance sampling, Laplace 근사, 모델 선택, 예측 불확실성·보정.

**성취 기준:** 관측과 가정을 분리해 사후·예측·근사 오차를 설명한다.

**AI 연결:** VAE, Bayes 신경망, 탐색과 불확실성, 잠재 표현.

**시각화·실험:** 관측으로 변하는 사후와 잠재 공간·ELBO의 항별 변화.

<a id="information-theory"></a>
### 정보이론과 압축
식별자: `information-theory`. 선수 영역: `probability`.

**학습 내용:** 자기정보·entropy, 조건부 entropy·상호정보량, cross entropy·KL, Jensen 부등식, 데이터 처리 부등식, 부호화·압축·MDL, rate-distortion, Fano 유형 경계, Rényi 정보량, bits·nats와 정규화.

**성취 기준:** 학습 목적과 정보량을 유도하고 확률·밀도·로그 밑·정규화의 차이를 구분한다.

**AI 연결:** cross-entropy 학습, 표현·압축, 변분 목적, 정보 기반 일반화.

**시각화·실험:** 분포 간 차이·부호 길이·병목·복원 오차.

<a id="high-dimensional"></a>
### 고차원 확률과 무작위 행렬
식별자: `high-dimensional`. 선수 영역: `probability`, `linear-algebra`, `analysis`.

**학습 내용:** concentration, sub-Gaussian·tail bounds, Hoeffding·Bernstein 유형 부등식, covering·packing, 고차원 기하, Johnson-Lindenstrauss, random matrix spectrum, Gaussian width, random features, 차원의 저주와 유효 차원.

**성취 기준:** 차원이 커질 때의 현상과 확률적 경계를 가정·확률 수준과 함께 해석한다.

**AI 연결:** 과매개화, random projection, 임베딩, 일반화와 스케일 분석.

**시각화·실험:** 고차원 거리 집중의 투영과 고유값 분포 비교.

<a id="graphical-models"></a>
### 확률 그래프 모델과 구조적 추론
식별자: `graphical-models`. 선수 영역: `bayesian-inference`, `discrete-math`.

**학습 내용:** Bayes network·Markov random field, factor graph, 조건부 독립·d-separation, variable elimination, junction tree, belief propagation, HMM·상태 공간 모델, 구조·모수 학습, energy·partition function, 정확·근사 추론의 조건.

**성취 기준:** 분포의 인수분해를 유도하고 추론의 계산량과 근사 조건을 설명한다.

**AI 연결:** 구조적 예측, 잠재변수, message passing, 시계열 추론.

**시각화·실험:** 확률 그래프·메시지·주변 분포와 근사 오차.

<a id="coding-algorithmic-information"></a>
### 부호·계산과 알고리즘 정보
식별자: `coding-algorithmic-information`. 선수 영역: `information-theory`, `discrete-math`, `linear-algebra`.

**학습 내용:** 채널·용량, 오류정정부호, 유한체와 선형 부호, 압축과 복원, algorithmic complexity, Kolmogorov 복잡도의 계산 불가능성, MDL·모형 선택, 계산 제한과 근사, 통신 복잡도·분산 표현.

**성취 기준:** 통계적 정보량·기술 길이·계산 복잡도를 구분하고 보장 조건을 확인한다.

**AI 연결:** 압축·분산 학습, 언어와 예측, 효율·일반화 연구.

**시각화·실험:** 코드·채널 잡음·오류 복구와 기술 길이 비교.

## 함수 공간과 구조

<a id="functional-analysis"></a>
### 함수해석과 연산자
식별자: `functional-analysis`. 선수 영역: `linear-algebra`, `analysis`, `measure-probability`.

**학습 내용:** 노름·Banach·Hilbert 공간, bounded·compact 연산자, 함수 공간의 쌍대, 직교 분해, spectral theorem의 적용 조건, distribution, 약도함수·Sobolev 공간, 적분 연산자, 유한·무한 차원 근사.

**성취 기준:** 함수 간 사상과 약한 해를 정의하고 이산 근사와 원래 문제를 구분한다.

**AI 연결:** kernel, Gaussian process, neural operator, PDE 학습.

**시각화·실험:** 함수 기저·연산자 작용·격자 해상도에 따른 근사.

<a id="approximation-kernels"></a>
### 근사이론·커널·표현력
식별자: `approximation-kernels`. 선수 영역: `functional-analysis`, `probability`.

**학습 내용:** 다항·spline 근사, 보편 근사 정리의 조건, RKHS·Mercer, representer theorem, kernel regression, Gaussian process, random feature 근사, NTK와 극한 조건, 표현력과 샘플 효율, 근사·추정·최적화 오차.

**성취 기준:** 표현 가능성과 실제 학습 가능성을 구분하고 근사 오차를 분석한다.

**AI 연결:** 신경망 표현력, kernel과 feature learning, operator 근사.

**시각화·실험:** 기저 추가·커널 폭·모델 규모와 근사 오차.

<a id="harmonic-analysis"></a>
### 조화해석과 신호 처리
식별자: `harmonic-analysis`. 선수 영역: `analysis`, `linear-algebra`, `complex-analysis`.

**학습 내용:** Fourier series·transform, convolution theorem, Laplace·Z transform, sampling·aliasing, DFT·FFT, STFT·wavelet, 필터·주파수 응답, Parseval 관계, 공간·주파수 localization, inverse filtering.

**성취 기준:** 시간·공간 표현과 주파수 표현을 연결하고 표본화·경계 조건을 설명한다.

**AI 연결:** CNN, 음성·영상, spectral bias, Fourier feature와 operator.

**시각화·실험:** 파동 중첩·필터·스펙트럼·aliasing을 직접 조작.

<a id="geometry-topology"></a>
### 미분기하·위상·데이터 형상
식별자: `geometry-topology`. 선수 영역: `matrix-calculus`, `analysis`, `linear-algebra`.

**학습 내용:** 위상·연속성·연결성, 다양체·chart, 접·여접공간, metric·geodesic, connection·curvature, 미분형식, embedding·immersion, homology·persistent homology, 위상적 안정성의 조건, 좌표 독립 표현.

**성취 기준:** 좌표와 내재적 구조를 구분하고 공간의 거리·곡률·위상을 계산한다.

**AI 연결:** manifold learning, geometric optimization, TDA, 표현 공간.

**시각화·실험:** 다양체·접공간·최단 경로와 점 구름의 위상 변화.

<a id="symmetry-groups"></a>
### 군·대칭·표현론
식별자: `symmetry-groups`. 선수 영역: `linear-algebra`, `geometry-topology`.

**학습 내용:** 군·군 작용, invariance·equivariance, 표현·불가약 표현, Lie group·Lie algebra, SO(3)·SE(3), 생성자·지수사상, gauge의 도입, 대칭과 제약, invariant feature, tensor representation.

**성취 기준:** 입력 변환에 대한 출력의 변화 법칙을 정의하고 등변성을 검증한다.

**AI 연결:** equivariant networks, 분자·3D·로봇 모델, geometric deep learning.

**시각화·실험:** 회전·이동·반사와 모델 출력의 대응.

## 최적화와 계산

<a id="convex-optimization"></a>
### 볼록해석과 제약 최적화
식별자: `convex-optimization`. 선수 영역: `matrix-calculus`, `analysis`.

**학습 내용:** 볼록집합·함수, subgradient, 최적성 조건, Lagrangian·KKT, 약·강 쌍대성과 조건, 선형·이차·반정부호 계획, proximal 연산, regularization, interior-point, duality gap.

**성취 기준:** 문제를 정식화하고 최적성·쌍대성의 가정과 해의 보장을 설명한다.

**AI 연결:** regularized learning, SVM, constrained fitting, 안전 제약.

**시각화·실험:** 목적함수·제약·쌍대 변수와 해의 이동.

<a id="stochastic-optimization"></a>
### 비볼록·확률·대규모 최적화
식별자: `stochastic-optimization`. 선수 영역: `convex-optimization`, `probability`, `numerical-computing`.

**학습 내용:** gradient descent·SGD, smoothness·학습률 조건, momentum·adaptive methods, saddle·plateau, mini-batch noise, variance reduction, distributed optimization, bilevel·hypergradient, implicit differentiation, convergence와 stationary point 구분.

**성취 기준:** 갱신 규칙을 유도하고 수렴·안정성·비용을 적용 조건 안에서 비교한다.

**AI 연결:** 대규모 신경망 학습, 미세조정, meta-learning, optimizer 연구.

**시각화·실험:** 지형·갱신·확률 궤적과 학습률 경계.

<a id="variational-geometry"></a>
### 변분법과 기하적 최적화
식별자: `variational-geometry`. 선수 영역: `matrix-calculus`, `functional-analysis`, `convex-optimization`, `geometry-topology`.

**학습 내용:** 함수의 변분, Euler-Lagrange, 제약 변분, Fisher 정보, natural gradient, mirror descent, Bregman divergence, Riemannian optimization, implicit objective, 함수 공간의 최적화.

**성취 기준:** 좌표 갱신과 함수·분포의 최적화를 구분하고 metric에 따른 방향을 계산한다.

**AI 연결:** natural gradient, policy optimization, 에너지·분포 모델 설계.

**시각화·실험:** 같은 목적함수의 다른 metric과 갱신 경로.

<a id="optimal-transport"></a>
### 최적수송과 분포의 기하
식별자: `optimal-transport`. 선수 영역: `probability`, `convex-optimization`, `functional-analysis`.

**학습 내용:** Monge·Kantorovich 문제, 수송 계획·쌍대, Wasserstein 거리, entropic regularization·Sinkhorn, 동적 수송·연속방정식, displacement interpolation, 분포의 gradient flow, 정규화·이산화 오차.

**성취 기준:** 분포 간 이동·거리·경로를 계산하고 수송 목적과 생성 목적의 차이를 설명한다.

**AI 연결:** 생성·정합·도메인 적응, flow matching, distribution optimization.

**시각화·실험:** 질량 이동·coupling·수송 경로와 정규화 변화.

<a id="numerical-computing"></a>
### 수치계산과 자동 미분
식별자: `numerical-computing`. 선수 영역: `linear-algebra`, `analysis`.

**학습 내용:** 부동소수점·반올림, overflow·underflow·logsumexp, conditioning·stability, 선형·비선형 해법, sparse·iterative solver, 보간·수치 적분, forward·reverse AD, finite difference·gradient check, mixed precision·quantization error, 계산·메모리 비용.

**성취 기준:** 정확한 식과 실제 계산 결과의 차이를 분석하고 미분·해법을 검산한다.

**AI 연결:** 안정적인 softmax·학습, AD, low precision, 대규모 모델 구현.

**시각화·실험:** 정밀도·조건수·미분 방식에 따른 오차와 비용.

## 동역학과 추론

<a id="differential-equations"></a>
### 상미분·편미분방정식
식별자: `differential-equations`. 선수 영역: `matrix-calculus`, `analysis`, `linear-algebra`.

**학습 내용:** 상태·초기값·경계값, 존재·유일성 조건, 선형·비선형 ODE, phase portrait·고정점, 안정성·분기·혼돈, PDE의 유형, 확산·파동·수송, 특성곡선, 약한 해, 보존 법칙과 경계 조건.

**성취 기준:** 연속 모델의 조건과 해의 성질을 설명하고 초기·경계 조건을 구분한다.

**AI 연결:** Neural ODE, world models, 물리 모델, scientific AI.

**시각화·실험:** 상태 공간·궤적·분기와 장의 시간 변화.

<a id="stochastic-processes"></a>
### 확률과정·확률미분방정식
식별자: `stochastic-processes`. 선수 영역: `measure-probability`, `differential-equations`.

**학습 내용:** Markov chain·process, Poisson 과정, martingale, Brownian motion, Itô·Stratonovich, SDE, Fokker-Planck, 시간 역전의 조건, ergodicity, MCMC·Langevin·HMC, 확률적 제어.

**성취 기준:** 표본 경로와 분포의 진화를 구분하고 확률적 해법과 역과정의 조건을 검토한다.

**AI 연결:** score-based diffusion, Monte Carlo 추론, stochastic control.

**시각화·실험:** 확률 궤적 집합·분포·score와 시간 방향.

<a id="numerical-simulation"></a>
### 과학계산과 수치 시뮬레이션
식별자: `numerical-simulation`. 선수 영역: `differential-equations`, `numerical-computing`.

**학습 내용:** Euler·Runge-Kutta·implicit methods, local·global error, stiffness, symplectic integrator, finite difference·volume·element, spectral methods, mesh·boundary handling, CFL 유형 안정성 조건, multigrid·multiscale, adjoint·민감도, 수렴 검증.

**성취 기준:** 시뮬레이션의 근사 오차와 학습 오차를 분리하고 격자·시간 간격 수렴을 확인한다.

**AI 연결:** 미분 가능한 시뮬레이션, PDE 학습 데이터, solver 설계.

**시각화·실험:** 해석해·격자·적분 방법·해상도를 바꾸는 오차 실험.

## 학습 이론과 결정

<a id="algorithms-complexity"></a>
### 알고리즘·계산복잡도·계산 구조
식별자: `algorithms-complexity`. 선수 영역: `discrete-math`, `linear-algebra`.

**학습 내용:** 점근 시간·공간, 분할정복·동적계획, graph search·flow·cut, randomized·approximation algorithm, 계산 가능성·reduction, 복잡도와 하한, sparse·parallel computation, 메모리 계층·communication cost, 분산 계산, 복잡도와 경험적 속도의 차이.

**성취 기준:** 모델·알고리즘의 계산량과 구현 비용을 구분하고 병목과 근사 전략을 분석한다.

**AI 연결:** attention 효율, 추론·검색·계획, 압축과 시스템 연구.

**시각화·실험:** 연산 DAG·메모리 이동·그래프 탐색·비용 모델.

<a id="learning-theory"></a>
### 통계학습·일반화·강건성
식별자: `learning-theory`. 선수 영역: `statistics`, `high-dimensional`, `convex-optimization`, `approximation-kernels`.

**학습 내용:** risk·ERM, approximation·estimation·optimization errors, VC·Rademacher, PAC·PAC-Bayes, algorithmic stability, implicit bias, interpolation·double descent, calibration·uncertainty, distribution shift·robust optimization, sample complexity, 이론 경계와 관측 scaling 관계.

**성취 기준:** 훈련 성능과 일반화 보장을 분리하고 정리의 조건·경계·실험을 비교한다.

**AI 연결:** 과매개화, 데이터 효율, 일반화, 강건성과 scaling 연구.

**시각화·실험:** 모델·표본 크기·분포 변화와 오차 분해.

<a id="causality"></a>
### 인과추론과 식별 가능성
식별자: `causality`. 선수 영역: `statistics`, `discrete-math`.

**학습 내용:** SCM·causal DAG, 교란·selection, 관측·개입, do-calculus의 조건, counterfactual, identifiability, instrumental variable, mediation, causal discovery의 가정, 불변성과 transportability, off-policy confounding.

**성취 기준:** 상관·개입·반사실적 주장을 구분하고 식별 불가능한 예제를 구성한다.

**AI 연결:** 모델 해석·인과 표현, 정책 평가, 분포 이동 연구.

**시각화·실험:** 동일한 관측 분포와 다른 개입 결과를 만드는 구조.

<a id="control-decision"></a>
### 제어·최적제어·강화학습의 수학
식별자: `control-decision`. 선수 영역: `differential-equations`, `probability`, `convex-optimization`.

**학습 내용:** state-space, controllability·observability, Kalman filtering, Lyapunov 안정성, LQR·MPC, dynamic programming·Bellman, HJB·Pontryagin, MDP·POMDP, bandit·exploration, policy gradient·off-policy estimation, constrained·robust control.

**성취 기준:** 상태·관측·행동·보상을 정의하고 추정·제어·학습의 역할과 보장을 구분한다.

**AI 연결:** 강화학습, 로봇, planning, world model, 안전 제어.

**시각화·실험:** 피드백·관측 잡음·정책·제약과 폐루프 궤적.

<a id="discrete-optimization"></a>
### 조합 최적화와 형식적 추론
식별자: `discrete-optimization`. 선수 영역: `algorithms-complexity`, `convex-optimization`.

**학습 내용:** 정수계획·relaxation, matching·cut·flow, submodularity, SAT·SMT·제약 충족, branch and bound, 근사 보장, 증명 탐색·type checking, 프로그램 합성, soft relaxation과 정확 해의 차이.

**성취 기준:** 이산 제약 문제를 정식화하고 휴리스틱·근사·정확 해의 차이를 검증한다.

**AI 연결:** planning, 추론·코드 생성, 신경 기호 모델, 구조 탐색.

**시각화·실험:** 제약·탐색 트리·relaxation·증명 객체의 검증.

<a id="game-theory"></a>
### 게임이론과 다중 행위자 학습
식별자: `game-theory`. 선수 영역: `probability`, `convex-optimization`.

**학습 내용:** 전략·보수·혼합전략, Nash equilibrium, minimax·saddle, 반복 게임·regret, no-regret learning, cooperative·Markov game, mean-field game, mechanism·incentive, adversarial dynamics, 해의 존재와 수렴의 구분.

**성취 기준:** 상호작용의 목적과 균형을 정의하고 학습 동역학의 안정성과 보장 조건을 구분한다.

**AI 연결:** GAN, 다중 에이전트 RL, 경쟁·협력·정렬 연구.

**시각화·실험:** 전략 공간·보수 지형·왕복 궤적과 평균 전략.

<a id="statistical-privacy"></a>
### 프라이버시·공정성·강건성의 수학
식별자: `statistical-privacy`. 선수 영역: `statistics`, `information-theory`, `convex-optimization`.

**학습 내용:** differential privacy의 정의, epsilon·delta·구성, 민감도·잡음, 정보 누출과 추론, fairness 지표·식별 조건, 지표 간 양립 조건, robust statistics, adversarial perturbation, 분포적 강건성, 수학적 보장과 현실적 평가의 차이.

**성취 기준:** 서로 다른 보장의 정의와 가정을 명시하고 정확도·불확실성·제약의 trade-off를 검토한다.

**AI 연결:** private learning, robustness, 신뢰 가능한 학습 방법 연구.

**시각화·실험:** 잡음·민감도·결정 경계·집단별 지표의 변화.

## 물리 모델의 기초

<a id="physical-modeling"></a>
### 측정·단위·물리 모델링
식별자: `physical-modeling`. 선수 영역: `elementary-math`, `linear-algebra`, `analysis`.

**학습 내용:** SI 단위·차원, 측정·불확실성, 계·경계·관측, 초기·경계 조건, 근사·유효 범위, 무차원화·차원해석, 보존과 제약, 대칭과 모형, 매개변수와 관측 가능성.

**성취 기준:** 어떤 물리량을 어떤 가정으로 모델링하는지 명시하고 단위·스케일을 검산한다.

**AI 연결:** 물리 priors, 과학 데이터, 센서·실험 설계, scientific AI.

**시각화·실험:** 단위·축·모형 가정을 바꾸며 측정과 예측 비교.

<a id="classical-mechanics"></a>
### 뉴턴 역학과 강체 운동
식별자: `classical-mechanics`. 선수 영역: `physical-modeling`, `differential-equations`.

**학습 내용:** 힘·질량·가속도, 운동량·충격량, 일·에너지, 각운동량·토크, 보존 법칙, 마찰·구속, 중심력, 회전·관성텐서, 강체·접촉, 초기 상태의 역할.

**성취 기준:** 힘·운동·에너지를 연결하고 모델에 포함한 외력과 경계를 설명한다.

**AI 연결:** 로봇·접촉·동작 예측, 물리 시뮬레이션과 world model.

**시각화·실험:** 합력·궤적·충돌·에너지 전달과 강체 회전.

<a id="analytical-mechanics"></a>
### 해석역학과 구조 보존
식별자: `analytical-mechanics`. 선수 영역: `classical-mechanics`, `variational-geometry`.

**학습 내용:** 일반화 좌표, 작용·정상 작용, Lagrange 방정식, Hamiltonian·정준 변수, Poisson bracket, symplectic geometry, Noether 정리, 구속·보존량, Hamilton-Jacobi, 보존 구조의 수치 근사.

**성취 기준:** 운동 방정식을 변분으로 유도하고 보존 구조와 해법의 관계를 검토한다.

**AI 연결:** Hamiltonian·Lagrangian networks, HMC, 구조 보존 scientific ML.

**시각화·실험:** 상 공간·보존 곡면·에너지와 symplectic 궤적.

## 장과 연속체

<a id="waves-optics"></a>
### 파동·진동·음향·광학
식별자: `waves-optics`. 선수 영역: `differential-equations`, `harmonic-analysis`, `classical-mechanics`.

**학습 내용:** 조화진동·감쇠·공진, 결합 진동·정상모드, 파동 방정식, 중첩·간섭·회절, 분산·군속도, 편광, 기하·파동 광학, Fourier optics, 관측·샘플링·센서.

**성취 기준:** 파의 진화와 관측된 신호의 관계를 유도하고 유효한 근사 범위를 설명한다.

**AI 연결:** 음성·영상, inverse imaging, 물리 센서·photonic AI.

**시각화·실험:** 파동 중첩·모드·간섭·렌즈와 영상 형성.

<a id="electromagnetism"></a>
### 전자기학과 전자기장
식별자: `electromagnetism`. 선수 영역: `physical-modeling`, `matrix-calculus`, `differential-equations`.

**학습 내용:** 전하·전류·장, 정전·정자기, 전위·벡터 퍼텐셜, Maxwell 방정식, 유도·전자기파, 장의 에너지·Poynting 흐름, 매질·경계, gauge freedom, 회로·센서와 연속장, 유효 모델.

**성취 기준:** 장·퍼텐셜·관측량을 구분하고 방정식·경계·에너지 수지를 연결한다.

**AI 연결:** 전자기 역문제, 센서·통신, 장 예측·회로 학습.

**시각화·실험:** 장선·퍼텐셜·파동·물질 경계와 에너지 흐름.

<a id="continuum-physics"></a>
### 연속체·유체·탄성·수송
식별자: `continuum-physics`. 선수 영역: `physical-modeling`, `differential-equations`, `numerical-simulation`.

**학습 내용:** 질량·운동량·에너지 보존, continuum 가정, 응력·변형, Euler·Navier-Stokes, 열·물질 확산, 반응수송, 구성방정식, 무차원 수, 다중물리 결합, 난류·폐쇄 문제, 보존형 이산화.

**성취 기준:** 구성 관계·경계·해상도·닫힘의 역할을 구분하고 보존·오차를 확인한다.

**AI 연결:** neural operator, surrogate, 재료·유체 scientific AI.

**시각화·실험:** 유동·탄성체·온도장·경계와 여러 해상도.

## 열·통계·비평형

<a id="thermo-statmech"></a>
### 열역학과 평형 통계역학
식별자: `thermo-statmech`. 선수 영역: `probability`, `physical-modeling`, `analysis`.

**학습 내용:** 온도·열·일, 열역학 법칙, entropy·free energy, 미시·거시 상태, ensemble, partition function, Boltzmann·Gibbs 분포, 상전이·질서변수, 상관·fluctuation, 상세평형, 유한계와 열역학 극한.

**성취 기준:** 미시 확률과 거시 물리량의 관계를 계산하고 정보 entropy와 단위·조건을 구분한다.

**AI 연결:** energy-based model, sampling, partition function, 학습 동역학의 통계물리적 관점.

**시각화·실험:** 입자·상태 점유·분포·상전이와 자유에너지.

<a id="nonequilibrium"></a>
### 비평형·확률열역학·복잡계
식별자: `nonequilibrium`. 선수 영역: `thermo-statmech`, `stochastic-processes`.

**학습 내용:** master equation, Fokker-Planck, 확률 흐름, detailed balance와 그 위반, fluctuation-dissipation, linear response, entropy production, stochastic thermodynamics, synchronization·critical dynamics, coarse graining·유효 기술.

**성취 기준:** 평형·정상상태·비평형 흐름을 구분하고 비유와 검증된 모델의 범위를 표시한다.

**AI 연결:** 확률적 학습·생성, 비평형 추론, 복잡한 시스템 모델링.

**시각화·실험:** 확률 전류·온도·감쇠·잡음과 정상 분포.

## 양자·장·다중스케일

<a id="quantum-physics"></a>
### 양자역학과 양자정보
식별자: `quantum-physics`. 선수 영역: `complex-analysis`, `linear-algebra`, `probability`, `functional-analysis`.

**학습 내용:** 상태·Hilbert 공간, 관측·연산자, unitary evolution·Schrödinger, spin·angular momentum, density matrix, uncertainty·measurement, tensor product·entanglement, variational principle, many-body·tensor network, quantum circuit·channel, quantum information.

**성취 기준:** 복소 확률진폭과 고전 확률을 구분하고 상태·측정·얽힘을 계산한다.

**AI 연결:** 양자 상태 학습·제어, 신경망 파동함수, quantum ML 연구 경로.

**시각화·실험:** 복소 위상·Bloch 구·측정 분포와 얽힘의 제한된 투영.

<a id="field-multiscale"></a>
### 장이론·임계현상·다중스케일
식별자: `field-multiscale`. 선수 영역: `analytical-mechanics`, `thermo-statmech`, `functional-analysis`, `symmetry-groups`.

**학습 내용:** 고전장·작용, 대칭·gauge, perturbation, correlation function, path integral의 정의·근사, statistical field theory, criticality·universality, renormalization·coarse graining, effective theory, scale separation.

**성취 기준:** 미시·거시·유효 기술의 관계를 설명하고 형식적 계산과 정당화된 결과를 구분한다.

**AI 연결:** multiscale learning, 물리·고에너지 AI, 학습과 coarse graining의 연구적 연결.

**시각화·실험:** 스케일 변화·상관 길이·모드 제거와 유효 모형.

<a id="relativity"></a>
### 상대론과 시공간 기하
식별자: `relativity`. 선수 영역: `geometry-topology`, `classical-mechanics`, `electromagnetism`.

**학습 내용:** 특수상대론·Lorentz, Minkowski metric, causal structure, 상대론적 운동학, 곡률·공변 미분, geodesic, Einstein 방정식의 구조, 시공간의 관측, 좌표와 물리량, 유효 근사.

**성취 기준:** 좌표 의존성과 불변량을 구분하고 상대론적 문제의 물리적 가정을 설명한다.

**AI 연결:** 천체·중력 scientific AI와 시공간 데이터 연구의 선택 경로.

**시각화·실험:** 시공간 절단·light cone·geodesic과 좌표 변환.

<a id="matter-computation"></a>
### 물질·계산 장치·물리적 자원
식별자: `matter-computation`. 선수 영역: `quantum-physics`, `thermo-statmech`, `electromagnetism`, `algorithms-complexity`.

**학습 내용:** 고체·격자·phonon, 전자·band, 반도체·회로·센서, analog·photonic computing, 잡음·비이상성, 지연·대역폭·메모리 이동, 에너지·열 비용, Landauer 원리의 조건, 하한과 실제 시스템 측정, 물리적 구현의 검증.

**성취 기준:** 알고리즘 비용과 하드웨어·열·통신 비용을 구분해 측정과 모델을 연결한다.

**AI 연결:** 효율적 AI 하드웨어, neuromorphic·physical computing, 에너지 연구.

**시각화·실험:** 회로·물질 상태·메모리 이동·에너지 수지.

## 연구로의 통합

<a id="inverse-problems"></a>
### 역문제·동화·실험 설계
식별자: `inverse-problems`. 선수 영역: `statistics`, `numerical-simulation`, `convex-optimization`.

**학습 내용:** forward·inverse map, ill-posedness, 식별 가능성, regularization, Bayesian inverse problem, sensitivity·adjoint, data assimilation, optimal experimental design, uncertainty propagation, 관측·방정식 잔차, model discrepancy.

**성취 기준:** 관측에서 모델을 복원하는 가정과 한계를 밝히고 새로운 관측으로 가설을 검증한다.

**AI 연결:** PINN, inverse design, differentiable simulation, scientific discovery.

**시각화·실험:** 희소·잡음 관측과 여러 설명 가능한 모델·추정 불확실성.

<a id="research-methods"></a>
### 수학·물리 기반 AI 연구 방법
식별자: `research-methods`. 선수 영역: `logic-proof`, `statistics`, `numerical-computing`.

**학습 내용:** 문제·가설·정식화, 선행 연구와 새 기여, 가정·정리·증명·반례, 기준선·공정 비교, ablation·negative control, seed·불확실성·검정, 누출·식별·재현, 단위·극한·보존·수렴 검사, 이론·경험·가설 구분, 실패 결과와 연구 기록.

**성취 기준:** 문헌의 주장을 재구성하고 검증 가능한 새 질문·모형·실험·증명을 설계한다.

**AI 연결:** 새 모델·학습 규칙·추론 방법·scientific AI 연구 전반.

**시각화·실험:** 같은 현상을 설명하는 가설·모형·검증 결과의 비교.

## 확장과 집필 원칙

이 목록의 항목 하나가 한 주나 한 장을 뜻하지 않는다. 필요한 분량으로 개념·정리·예제·실험을
분리하고, 새 연구의 선수 개념을 발견하면 기존 ID를 유지하면서 하위 단원을 추가한다.
정의·증명·물리 법칙과 관련 논문 원문은 [근거 자료 관리](sources.md)에 따라 단원별로 연결한다.
