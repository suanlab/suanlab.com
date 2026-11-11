# 비볼록·확률·대규모 최적화 원주제 대조

2026-11-12 기준. `stochastic-optimization`의 10개 원주제를 기존 한영 원고·실습·독립 계산과 대조했다. 기본 `stochastic-optimization` 단원의 세 target·분산 8/3 모형도 읽고 보존했다. 새 유한합 모형은 네 target·분산 5σ²/2로 서로 다른 실험이다. 영역 연결 단원은 11개, 원주제는 10개다. 누적 대조는 24영역·235개 원주제이며 전체 51영역·509개 이론·연구 심화 완료를 뜻하지 않는다.

## 원주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`, 계산은 `src/lib/learning-optimization.ts`다.

| 원주제 | ID | 보강한 논증·예제 |
|---|---|---|
| GD·SGD | finite-sum-gd-sgd | 감소 보폭의 조건부 거리 점화·supermartingale 논증, 1/k 표본 평균의 정확 분포 |
| Smoothness·학습률 | smoothness-learning-rate | Descent 합의 비볼록 정상성 상계, PL과 비유일 최적해의 강볼록성 구별 |
| Momentum·adaptive | momentum-adaptive-updates | Cayley/Hurwitz의 heavy-ball 안정 구간 증명, 회전 좌표의 Adam 보폭 반례 |
| Saddle·plateau | saddles-flat-plateaus | 국소 선형 탈출 시간과 정확한 0 초기화, 이산 quartic 점근 꼬리 증명 |
| Mini-batch noise | minibatch-gradient-noise | 한 epoch의 음의 교차 batch 공분산, 24순열의 정확 두 보폭 분산 |
| Variance reduction | variance-reduced-gradients | SVRG 이차 모멘트·평균 anchor의 epoch 수축 상계와 실제 성분 평가 비용 |
| Distributed | distributed-local-drift | Broadcast-gradient 보정의 정확 round map·고정점·추가 정보 비용 |
| Bilevel·hypergradient | bilevel-unrolled-hypergradients | 유한 합 미분의 Kq^(K−1) 항, 정확한 상태와 다른 미분, Neumann 근사의 범위 |
| Implicit differentiation | implicit-solution-sensitivity | 근 잔차/최소 도함수의 오차 상계, 역민감도와 고정 근 adjoint 잔차 영향 |
| Convergence·stationarity | convergence-stationarity-limits | 매개변수 단위별 gradient 허용 오차, 곡률을 함께 변환한 목적·거리 bound |

## 독립 코드와 그림

표준 라이브러리 Python 10종을 한영에 같은 코드로 제공한다. 표본 평균은 정수 합 분포를 합성해 64개 추출까지 정확 MSE를 확인한다. Fraction으로 PL 예제·epoch 공분산·보정 round·유한 hypergradient·implicit 잔차·좌표 단위를 검산한다. Quartic 꼬리는 80자리 Decimal로 10,000회 실행하며 finite control 범위 밖의 별도 예제임을 밝힌다.

SVRG 코드는 실제 표본 epoch와 평균 anchor를 실행한다. Epoch당 4+2m개의 성분 기울기를 직접 세며, 기대 수축 상계를 한 경로의 매 epoch 보장으로 오해하지 않는다. 저장된 anchor 기울기를 재사용하는 다른 비용 모형과도 구별한다. Broadcast 보정은 fresh full gradient를 쓰는 결정론 예제이며 SCAFFOLD 전체나 실제 네트워크 시스템을 구현했다고 주장하지 않는다.

한영 SVG 네 개는 iid 평균 오차/MSE에서 상수 보폭과 1/k 보폭을 비교하고, 상태 오차와 민감도 오차의 다른 속도 및 한 보폭의 정확 상태/부정확한 해 미분을 보여준다. 폭 440, title·desc·대체 텍스트와 인접 표·실행 코드를 제공한다. 모멘트 그래프는 한 seeded 경로의 통계나 신경망 경험 결과가 아니다.

## 수치 계산 보완

기존 SGD는 target 1+σz를 먼저 만들면서 σ=10⁻²⁰의 잡음을 잃었다. 현재는 별도 중심화한 오차 점화식으로 경로를 그리며, 원래 매개변수 상태는 별도로 계산한다. 둘은 같은 실수 모형의 서로 다른 부동소수점 평가다. Seed 0, η=1/2,초기 1에서 첫 오차 −10⁻²⁰을 유지하고 표시 매개변수는 1로 반올림될 수 있다. σ=0의 두 오차 경로는 같은 연산으로 일치시킨다.

SVRG는 x=0,anchor=10⁻²⁰에서 반올림한 추정량의 표본 분산 0 대신 해석적인 1.25×10⁻⁴⁰을 유지한다. 표시 추정량과 평균은 별도의 수치 진단이다. 정확해 hypergradient는 outer 잔차를 (2−target−target·lambda)/(1+lambda)로 평가해 target=2,lambda=10⁻²⁰의 약 4×10⁻²⁰을 보존한다. 유한 unroll을 정확한 해의 미분으로 바꾸지 않는다.

작은 η에서 spectral factor나 round A가 1로 반올림돼도 수학적 엄격 수축을 놓치지 않도록, quadratic은 0<ηL<2, local round는 0<m1+m2<4를 사용한다. 이는 기계 상태가 매번 실제로 움직인다는 보장이 아니며 유한 정밀도 정체를 원고에서 별도 설명한다.

## 출처·보존·후속 범위

[Bottou·Curtis·Nocedal의 저자 논문](https://arxiv.org/abs/1606.04838)과 기존 MIT SGD 자료를 기준으로 smoothness·확률 가정을 대조했다. SVRG의 epoch 구조는 [Johnson·Zhang의 원 논문](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html), 지역 학습의 가정은 [Stich](https://arxiv.org/abs/1805.09767), control variate의 연구 연결은 [SCAFFOLD 원 논문](https://proceedings.mlr.press/v119/karimireddy20a.html)을 따른다. Inexact hypergradient의 오류 관리 배경은 [Pedregosa의 저자 논문](https://arxiv.org/abs/1602.02355)이다. 본문의 작은 유리수·단위·회전·정확 분포 예제는 별도로 구성했으며 원 논문의 경험 결과 재현이 아니다.

기존 설명·연습을 보존하고 두 한국어 원고의 수식 마침표/if-when 차이 세 곳만 한영 수식 일치를 위해 정리했다. 변경 원고 20개의 과거 fixture SHA·현재 SHA·이 기록을 ledger에 명시하고 네 immutable fixture는 바꾸지 않았다. 기본 세 target 단원은 수정하지 않았다.

일반 adaptive-method 수렴·반례, 확률적 negative-curvature 탈출 복잡도, 비동기·압축·부분참여 local SGD, 고차 bilevel·비특이성 실패와 완전한 inexact-gradient 수렴 조건은 후속 연구 범위다. 이를 모두 완료했다고 선언하지 않는다. 다음 영역은 변분법과 기하적 최적화의 10개 원주제다.

## 검증

선택한 계산·원고 보존 검사 67개가 통과했다. 한영 20개 Python 블록의 코드 SHA·출력과 수식 일치를 확인했다. 별도 120자리 Decimal 점화식으로 정확한 binary64 입력·같은 정수 PRNG를 대조한 48개 중심화 경로의 최대 scaled error는 2.02×10⁻¹⁶이다.

타입·lint·콘텐츠 검증과 1,893개 경로의 정적 빌드가 통과했다. 내보낸 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사를 통과했다. 실제 export에서 한영·320/768/1440px·명암 98개 화면 상태의 접근성 위반과 브라우저 오류가 0이며, 상호작용 12묶음·안내 30개·퀴즈 20개·이전 계산 22개·교육과정 11개 연결을 확인했다. 최종 320px 그림 네 개도 직접 확인했다.

후보와 작업 소스 1,882개·학습 원고 1,300개·릴리스 파일 31개·설정 8개의 byte 일치 및 SVG 네 개의 실제 export 일치를 확인했다. 전체 CI 검사 완료나 공개 배포 완료는 이 로컬 검증과 구별해 기록한다.
