# 통계학습·일반화·강건성 원주제 대조

2026-11-12 기준. `learning-theory`의 11개 원주제를 한영 원고·실습·독립 예제에 대조한다. 입문 48개 및 후속 입문 362/362 완료와 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-theory.ts`다.

| 원주제 | 단원 ID | 보강한 논증·독립 실행 |
|---|---|---|
| Risk·ERM | finite-risk-empirical-minimization | Approximate ERM oracle 부등식·realizable consistency의 빠른 보증·표본 완전 열거 |
| 오차 분해 | approximation-estimation-optimization-risk | Early-stopping filter·기대 bias/variance·중심화된 작은 population 효과 |
| VC·Rademacher | vc-threshold-rademacher | Random-walk 최솟값·classifier/loss 복잡도 변환·기대 ERM 보증의 양화 |
| PAC·PAC-Bayes | finite-posterior-pac-bayes | Gibbs 변분 항등식·prior family 선택 비용·작은 이차 KL |
| 안정성 | replace-one-ridge-stability | Ghost observation 교환 증명·정확한 기대 gap·작은 이웃 변화 |
| Implicit bias | linear-nullspace-implicit-bias | SPD preconditioning·inverse-metric 최소 거리·피타고라스와 불변량 |
| 보간·Double descent | minimum-norm-interpolation-risk | 조건부 bias/variance·Schur complement와 chi-square inverse moment·무한 기대 |
| Calibration·불확실성 | calibration-brier-bin-uncertainty | 조건부 기대 사영·Brier 항등식·진단 병합과 실제 점수 변경 |
| Shift·강건 최적화 | group-shift-total-variation-dro | 추정 반경과 shift의 삼각부등식·연속 coin minimax·importance 분산 |
| Sample complexity | binomial-sample-complexity | 두 Bernoulli 세계의 검정 환원·KL data processing·필요/충분 표본 수 |
| Scaling 해석 | finite-window-learning-scaling | Fourth moment·nested-prefix 공분산·zero 위험과 log 선택 효과 |

언어별 비교표와 표준 Python 블록 11개, 한영 Gaussian 기대 위험·calibration SVG 네 개를 추가했다. Gaussian 기대 그림은 비절단 정확한 역행렬 모형의 평균이고 현재 SVD cutoff를 쓴 단일 design 곡선이 아니다. Calibration 그림의 상수 predictor는 실제 점수 변경이며 bin만 합친 진단과 구분한다.

## 계산 보완과 한계

Population quadratic은 선언한 x+c*x² 모형의 알려진 중심과 곡률로 평가한다. 실제 생성 표본·ERM 동률 규칙·GD recurrence는 유지한다. 근사·추정·signed 최적화 효과·empirical 간극을 동일한 quadratic의 중심화된 항등식으로 계산해 공통 noise floor를 직접 빼며 작은 효과를 지우지 않는다. 직접 평가한 population과 뺄셈은 `expandedPopulation`·`expandedEffects`로 남긴다.

Gibbs KL은 정규화 log-ratio의 convex generator를 쓰고 작은 구간에서 이차부터의 급수를 계산한다. 음수 KL을 단순히 0으로 clip하지 않으며 직접 q*log(q/pi) 합은 `expandedKL`에 보존한다. 정규화한 prior와 분석한 Gibbs의 진단으로, 표시 확률의 마지막 자릿수보다 작은 tilt를 구별할 수 있다.

Ridge의 이웃 변화는 입력 label 차이에 대한 해석적 weight 변화와 affine loss 차이를 사용한다. 반올림된 두 loss나 weight를 빼지 않으며 실제 direct subtraction은 `expandedDifference`에 남긴다. 이들은 binary64 해석 진단이고 측정한 표시 숫자의 차이나 arbitrary-precision 인증이 아니다.

Preconditioner·prior family·Gaussian 분포 평균·두 세계 검정은 독립 유도와 코드이다. 해당 모형을 실제 조절기에 추가했다고 주장하지 않는다. Seed 생성기의 유한 집합을 IID 정리의 확률 공간으로 취급하지 않는다. 기존 binomial scaled-precision 경계 정책도 유지한다.

## 출처와 보존

배경 범위는 [Understanding Machine Learning 저자 자료](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/), [MIT 6.867](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/), [Belkin 등의 double descent 연구](https://arxiv.org/abs/1812.11118), [Hastie 등의 ridgeless 연구](https://arxiv.org/abs/1903.08560)와 대조했다. 추가 유리수·표본 열거·finite Gaussian moment 유도와 그림은 독립 구성이다. 원 논문의 전체 실험이나 고차원 점근 정리를 재현했다고 주장하지 않는다.

스물두 원고의 손계산 절 앞에 보강하고 기존 prefix·손계산 이후 suffix를 보존했다. Immutable fixture 네 개는 수정하지 않고 이전 SHA·검토 SHA·이 기록을 reviewed ledger에 연결했다. 추가 표시 수식과 Python은 한영 대응을 검사한다.

## 검증

관련 계산·원고 보존 검사 76개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 22개 블록을 실행하고 추가 표시 수식·코드의 대응 및 기존 본문 prefix·손계산 suffix 보존을 확인했다.

별도 650자리 Decimal 참조에서 Gibbs KL 756건·이웃 변화 216건·위험 분해 252건을 대조했다. Scaled error는 |계산−참조|/max(1,|참조|)로 정의한다. 최대 scaled error는 KL 7.2×10⁻¹⁵·ridge 진단 2.3×10⁻¹⁷·quadratic 진단 2.3×10⁻¹⁶ 미만이다. 절댓값 10⁻³⁰⁰보다 큰 참조의 최대 상대 오차는 각각 1.5×10⁻¹⁴·2.0×10⁻¹⁶·9.5×10⁻¹⁶ 미만이다. 직접 계산에서 KL이 0/음수였던 양의 효과 118건, 직접 손실 차이가 0이었던 비영 변화 131건을 보존했다. 이는 시험한 binary64 입력과 명시한 해석적 진단의 대조이며 전체 실수 영역의 interval 인증이 아니다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 110개 상태는 한영 320/768/1440px 72개·영문 desktop dark 12개·경계 25개·영역 연결 1개다. Interactive Book 영역의 접근성 위반과 uncaught 브라우저 오류는 0이며 상호작용 13묶음·안내 33개·퀴즈 22개·이전 계산 51개·영역 연결 11개를 확인했다.

작은 양의 근사 효과·Gibbs KL·이웃 변화가 실제 readout에도 유지되는지 검사했다. 정수 초안 blur 오류·모형 보존·Escape 복원, 설정 저장·언어 전환·완료 상태, 원고 표·코드의 키보드 스크롤과 320px 그림 네 개도 확인했다. 소스 1,882개·학습 원고 1,300개·릴리스 파일 33개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

무한 class의 일반 고확률 bound·localized complexity, 데이터 의존 prior의 정당화, 일반 비선형 안정성·implicit bias, 비등방·비정규 design의 보간, adaptive calibration과 의존 평가, unknown shift의 식별·신뢰 집합, 실제 neural scaling의 실험 설계는 후속 연구 심화 범위다. 원주제 대조는 해당 분야의 모든 이론 완료를 뜻하지 않는다. 이번 대조로 누적 32영역·316개, 후속 19영역·193개이며 다음은 인과추론과 식별 가능성 11개다.
