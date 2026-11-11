# 변분법과 기하적 최적화 원주제 대조

2026-11-12 기준. `variational-geometry`의 10개 원주제를 한영 원고·기존 실습·실행 예제와 대조했다. 먼저 보강한 두 주제에 이번 여덟 주제를 더해 누적 25영역·245/509개 원주제의 대조 기록을 연결한다. 요청 당시 입문 48개 및 후속 입문 362/362 완료와 이번 연구 심화를 구분한다.

## 먼저 보강한 두 주제 (`aa93832`)

| 원주제 | 단원 | 논증·실행·직관 |
|---|---|---|
| 함수의 변분 | functional-directional-variations | H₀¹ 도함수 노름에서 균일 이차 나머지로 Fréchet 미분 증명; L2에서는 불연속인 고주파 반례; 유리수 나머지와 주파수별 비교표 |
| Euler–Lagrange | euler-lagrange-boundaries | 약한 잔차의 쌍대 노름과 에너지 오차의 등식; 내부·자연 경계 잔차를 함께 쓴 인증; 내부 잔차 0인 잘못된 해와 289개 다항 후보 검산 |

기존 설명·실습·연습 해설을 모두 보존하고 연습 앞에 절을 추가했다. 첫 예제는 α=0에서 π²를 분리해 유리수로 에너지를 계산하므로 ε=10⁻²⁰의 이차 나머지도 보존한다. 고주파 함수족과 다항 오차족은 별도의 실행 예제이며, 기존 두 sine mode 실습에 새 조절기를 연결했다고 주장하지 않는다.

자유 경계 인증은 정확한 연속 연산자와 왼쪽 본질적 경계값을 만족하는 후보를 가정한다. 양 끝 고정 문제에는 다른 시험공간을 사용하며, 본질적 경계값 위반·근사 구적·비강제 문제에는 추가 오차 분석이 필요함을 설명한다. 작은 표본 훈련 손실을 연속 문제의 정확도 인증과 동일시하지 않는다.

## 첫 두 주제의 근거와 검증

기존 [MIT 변분법 강의](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf)의 함수 미분·내적·경계항 구별을 확인했다. 추가한 노름별 연속성 반례·잔차 인증과 작은 유리수 예제는 본문에서 직접 유도한다. 강의가 이 추가 정리를 모두 증명했다고 귀속하지 않는다.

네 원고의 이전 본문·연습 보존과 immutable fixture의 SHA를 대조하고, 새 SHA를 reviewed ledger에 연결했다. 관련 계산·원고 보존 검사 65개와 콘텐츠 검증이 통과했다. 한영 네 원고의 Python 실행 결과·표시 수식 일치를 확인했고, 독립 다항식 합성·정확 적분으로 에너지 289건과 약한 잔차 pairing 578건을 검산했다.

최종 정적 빌드 1,893개 경로와 export 1,888개 HTML의 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 한영·명암·320/768/1440px 24개 상태에서 접근성 위반·브라우저 오류가 0이다. 표·코드의 키보드 가로 스크롤, 안내 계산 6개·퀴즈 4개·완료 저장 2개를 확인했다. 모바일 표의 긴 제목은 짧은 수식·용어로 바꾸고, 최종 한영 표 네 화면을 직접 확인했다.

작업 소스 1,882개·학습 원고 1,300개·이번 릴리스 파일 8개의 후보 byte 일치를 확인했다. 전체 CI 및 공개 배포 완료는 로컬 검사와 구별한다.

## 이어서 보강한 여덟 주제

| 원주제 | 단원 | 논증·직관·실행 |
|---|---|---|
| 제약 변분 | isoperimetric-constrained-variations | 면적 함수의 전사성과 kernel 분해로 승수 존재 증명; 최적값의 한계 비용; ∫y²=0의 정칙성 실패 |
| Fisher 정보 | bernoulli-fisher-coordinates | score 공분산과 Cauchy–Schwarz의 추정 분산 하한; 이항 count 정확 열거와 표본 평균의 등호 |
| Natural gradient | natural-gradient-coordinate-steps | 손실 볼록성의 흐름·확률 Euler 오차 상계; 같은 identity damping의 좌표 불변성 실패와 metric 변환 |
| Mirror descent | entropic-mirror-simplex | 유계 비용의 기울어진 분산·log-normalizer 상계; 변화하는 전체 비용의 누적 regret과 예측/관측 순서 |
| Bregman divergence | bregman-generator-geometry | 투영 일차 조건과 방향을 명시한 Pythagoras 부등식; 비정규화 entropy 질량 항과 simplex 정규화 |
| Riemannian optimization | circle-riemannian-descent | 원 목적의 곡률 상계와 유효 retraction 보폭; 엄격한 유한 보폭 감소·정지성 및 극대점의 한계 |
| Implicit objective | implicit-adjoint-objectives | 상태·수반 잔차의 결합 hypergradient 인증; 수반 잔차 0이어도 상태 오차가 남는 유리수 예제 |
| 함수 공간 최적화 | function-space-gradient-metrics | 양의 κ·ℓ에서 주파수에 균일한 일반화 곡률과 에너지 수축; L2 불안정·κ=0의 비균일성 |

기존 여덟 주제의 본문·연습을 보존하고 한영 16개 원고에 동일 코드의 표준 라이브러리 예제를 추가했다. 이번 코드와 먼저 만든 두 예제를 합하면 이 영역에 독립 Python 예제 10종이 있다. 온라인 실험과 고주파 전처리 연구의 한영 SVG 네 개는 실제 실습의 조절기를 확장했다고 주장하지 않으며, 별도 예제의 시간 순서·보폭·범위를 그림과 본문에 밝힌다.

## 수치 계산과 해석의 보완

Fisher 질의는 요청한 변화량 Δ로 로그 나머지를 평가해 p=1/2,Δ=10⁻²⁰의 KL 약 2×10⁻⁴⁰을 보존한다. 별도로 반올림한 p+Δ가 p와 같을 수 있다. 이는 같은 실수 질의의 다른 수치 평가이며 저장된 같은 확률 두 개의 KL이 양수라는 주장이 아니다. 점 좌표·기존 곡선은 원래 확률 좌표로 유지한다.

Natural flow는 p+(r−p)(−expm1(−η))로 평가하여 η=0에서 입력 p를 정확히 반환한다. 같은 실제 흐름의 평가식을 바꾼 것이고, 좌표별 Euler 보폭이나 metric을 변경하지 않았다. 두 변경의 경계·회귀 검사를 추가했다.

## 이번 근거와 최종 검증

추정 하한은 [MIT의 효율적 추정 강의](https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/15224973dfc1c2b4287804c8712681f7_MIT14_381F18_lec6.pdf), natural-gradient 불변성의 범위는 [Martens의 저자 논문](https://www.jmlr.org/papers/v21/17-678.html), 생성 함수·mirror 설정은 [Nemirovski 강의](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf), 다양체 기울기·복귀의 배경은 [Boumal의 저자 교재](https://www.nicolasboumal.net/book/)와 대조했다. 작은 유리수·분포 열거·원·모달 예제와 특화된 오차 인증은 본문에서 직접 유도하며 원 논문의 실험 재현으로 제시하지 않는다.

관련 계산·원고 보존 검사 67개, 타입·lint·콘텐츠 검증이 통과했다. 이 영역의 한영 Python 20개 블록을 다시 실행하여 코드 SHA·출력·표시 수식 일치를 확인했다. 140자리 Decimal 독립 계산의 KL 16건·flow 36건에서 최대 KL 상대 오차는 4.09×10⁻¹⁵, flow 절대 오차는 5.96×10⁻¹⁷ 이하다.

1,893개 경로의 정적 빌드와 1,888개 export HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 한영·320/768/1440px·명암 98개 상태에서 접근성 위반·브라우저 오류가 0이며, 상호작용 12묶음·안내 30개·퀴즈 20개·이전 계산 25개·원주제 연결 10개를 확인했다. 정수 초안의 오류 표시는 blur 뒤 확인하고, 모형값 보존·Escape 복원도 검증한다. 최종 320px 과학 그림 네 개를 직접 확인했다.

후보와 작업 소스 1,882개·학습 원고 1,300개·이번 릴리스 파일 27개·설정 8개의 byte 일치 및 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구별한다.

## 보존과 후속 범위

이전 원고의 설명·연습과 기존 열 실습의 기저·조절기·퀴즈·안내를 보존한다. 새로 변경한 16개 원고도 reviewed ledger에 과거 fixture SHA·현재 SHA·이 기록을 연결하고 immutable fixture 네 개는 바꾸지 않는다.

일반 비선형·무한 제약의 승수 정리, 특이 통계 모형, 신경망의 근사 Fisher 해결기, bandit·확률 oracle, 비볼록 투영, 일반 다양체의 전역 수렴, 근사 연산자의 PDE 수반 오차와 비이차 함수 공간 문제는 후속 심화 범위다. 이번 열 원주제 대조가 해당 분야의 모든 연구 이론을 완성했다는 뜻은 아니다. 다음 영역은 최적수송과 분포의 기하 8개 원주제다.
