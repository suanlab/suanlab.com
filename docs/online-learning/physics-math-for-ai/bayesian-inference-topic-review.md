# Bayes 추론 원주제 대조와 사후 의존성·근사·evidence 심화

2026-11-11에 Bayes 추론과 잠재변수의 10개 원주제를 연결된 한영 11개 reader 및 `learning-bayes.ts`·`learning-foundations.ts`와 대조했다. 공통 중심 추론, Gaussian mean-field의 양방향 KL, 중요도 추정량별 목적과 지지, 고정 z의 모형 evidence를 보강했다. 51영역 전체의 일반 이론·연구 심화 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다.

| 원주제 | 원고 ID | 확인한 정의·가정·논증 |
|---|---|---|
| 사전·우도·사후 | bayesian-inference, finite-bayes-updates | Beta 밀도와 유한 질량, 관측 사건·정규화, 사후 요약·예측, 지지 0과 민감도 |
| 공액성·사후 예측 | conjugate-posterior-predictive | Beta-binomial 적분, 공유 θ로 생기는 공분산, plug-in과 모형 적합성 |
| 계층 모델 | hierarchical-partial-pooling | 고정 초모수의 제곱 완성, 미지 공통 중심의 정규 사후·전공분산, 새 그룹 |
| evidence | marginal-likelihood-evidence | 순서·개수 표현, 순차 예측 곱, Gaussian 표본평균 밀도 비율·두 극한 |
| 잠재변수·EM | latent-mixtures-em | 정확 E/M 단계의 범위, 고정 분산·가중치, 대칭 정지점·라벨 식별 |
| 변분 추론·ELBO | variational-elbo | 하한·방향·지지, Gaussian 대각 족 전역 최적, 주변·조건부 분산의 차이 |
| importance sampling | importance-sampling | 비편향 일반 평균·편향 가능 비율, V와 CLT 계수 A, 목적별 지지·최적 제안 |
| Laplace 근사 | laplace-posterior-approximation | 내부 모드·곡률, 정확 Beta 모멘트, 지지 밖 질량·Jacobian·좌표 의존성 |
| 모델 선택 | bayesian-model-comparison | 동일 자료 표현·모형 사전 odds, Bayes factor와 후보 집합의 적합성 구분 |
| 예측 불확실성·보정 | predictive-uncertainty-calibration | 전분산, 별개 고정 평가 라벨의 ECE·Brier, 빈·표본·분포 이동의 범위 |

## 새 논증과 실행 예제

공통 중심 μ~N(0,s²), 알려진 τ²와 그룹 관측 분산 1/n_j를 사용했다. 중심 사후 분산 S와 데이터 가중치 w_j에서 그룹 공분산은 δ_jk r_j+(1−w_j)(1−w_k)S다. 두 그룹 예제의 S=5/13, 그룹별 분산 14/65, 공분산 1/65, 새 그룹 모수 분산 18/13을 결합 3×3 정밀도 역행렬로 독립 검산한다. 중심만 추론하며 τ의 불확실성까지 포함한다고 주장하지 않는다.

단위 주변분산·상관 ρ인 이변량 Gaussian 목표에 대각 Gaussian q를 사용했다. 로그 밀도 평균을 전개하고 평균 이차식·분산의 미분으로 KL(q||p)의 전역 최적 분산 1−ρ²와 간격 −log(1−ρ²)/2를 유도했다. 역방향은 분산 1이며, 이 예제에서는 최소 KL 숫자가 같아도 최적 분포가 다르다. 특이한 |ρ|=1은 제외한다. 기존 Bernoulli 화면과 별도 계산임을 명시했다.

유한 세 상태에서 일반 중요도 평균의 단일 표본 분산 V와 자기정규화 CLT 계수 A를 구분했다. Cauchy–Schwarz로 두 목적의 제안 차이를 유도했다. q_SN=(13/35,9/70,1/2)의 최소 A=49/100이며 일반 분산 V는 101/100이다. 일반 평균의 형식적 최적 q_*=(0,3/13,10/13)는 해당 함수의 적분에는 지지 문제가 없지만, 같은 제안의 자기정규화 극한은 13/8로 틀린다. 모든 양의 q_ε에서는 V=μ²ε/(1−ε)이고 0은 달성되지 않는 하한이다. N=2의 아홉 순서쌍을 분수로 열거한다. 기존 지지 설명도 정규화·자기정규화 목적에 해당하도록 명확히 했다.

Gaussian 관측 분산 1, M0의 θ=0, M1의 적절한 N(0,τ²) 사전에서 표본평균 밀도 비율을 직접 유도했다. r=nτ², z=√n·표본평균일 때 BF_10=(1+r)^−1/2 exp[z²r/(2(1+r))]이다. 고정 z·τ²의 n→∞, 고정 n·z의 τ²→∞에서 BF_10→0이지만, 비영 관측 평균을 고정한 경로는 다른 극한이다. 적절한 사전의 극한과 부적절한 사전의 정규화 삭제를 구분했다. 사후 척도로 변수만 바꾼 직접 밀도 Simpson 적분으로 수치를 검산한다.

## 시각화

`public/assets/learning/bayes-mean-field-{ko,en}.svg`는 ρ=0.9 목표의 Cholesky 좌표 (2cos t,2ρcos t+2√(1−ρ²)sin t), KL(q||p) 최적 원의 반지름 2√0.19, 역방향 원의 반지름 2를 같은 축에 그린 원본 벡터다. 한영 범례·대체 텍스트를 제공하고 기존 reader의 반응형 이미지 스타일을 따른다. 각 곡선은 자기 분포의 Mahalanobis 반지름 2이며 95% 포함률 그림으로 설명하지 않는다.

## 출처와 후속 범위

- [Stan User’s Guide, Hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression): 공유 초모수를 추론하는 계층 구조의 배경. 위 Gaussian 예제와 공분산은 직접 유도했다.
- [Blei 등, Variational Inference §2.3·Figure 1](https://arxiv.org/html/1601.00670v9#S2.SS3): mean-field 족과 상관 Gaussian 배경. 단위 주변분산·역방향 수치는 로그밀도에서 직접 계산했다.
- [Art Owen, Chapter 9 §9.1–9.3](https://artowen.su.domains/mc/Ch-var-is.pdf): 일반·자기정규화 표집의 서로 다른 지지·분산 목적. 지정한 세 상태의 분수와 q_ε는 직접 계산했다.
- [PyMC, Bayes Factors and Marginal Likelihood](https://www.pymc.io/projects/examples/en/2022.01.0/diagnostics_and_criticism/Bayes_factor.html): evidence의 모형 의존성과 사전 민감도 배경. Gaussian 비율과 극한은 직접 유도했다.

일반 사후 일치성·Bernstein–von Mises의 조건과 실패, 미지 분산·비중심 계층 계산, 일반 변분족의 오차 인증, 적응·연속 상태 표집, 정량 Laplace 나머지, 선택 후 모형 비교와 실제 데이터 보정은 후속 연구 심화다. 이번 유한 예제 대조를 그 이론 전체의 완료로 계산하지 않는다.

## 검증 기록

- `learning-bayes.test.ts` 26개와 `learning-manuscript-preservation.test.ts` 7개, 기존 Beta-Bernoulli 계산 1개 및 연결된 11개 reader의 한영 원고·수식 검사 11개가 통과했다. 합계 45개이며 선택 실행에서 제외한 검사는 통과 수에 넣지 않았다. 그림 추가 뒤 영향받은 렌더링 1개와 보존 7개도 다시 통과했다.
- 네 예제의 한영 Python 블록 8개를 실행했다. 최종 원고의 코드 해시가 실행 기록과 일치한다. 분수 3×3 역행렬, 양방향 Gaussian KL, N=2의 아홉 표본쌍 및 아홉 Gaussian 밀도 적분을 검산했다.
- 한영 SVG의 타원 꼭짓점 241개씩을 목표 이차식에 대조하고 작은 원의 분산 0.19도 확인했다. 한영 320px 원고 배치와 최종 정적 페이지의 그림을 직접 확인했다. 두 SVG의 빌드된 바이트가 원본과 일치한다.
- Node 20에서 `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 1,893개 경로를 생성했고 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 검사했다. 1,300개 한영 원고와 1,882개 `src/` 파일이 빌드 사본과 일치한다.
- `SUANLAB_TEST_BASE=http://127.0.0.1:43299 npm run check:bayes`가 최종 산출물에서 통과했다. 11개 reader와 교재 화면의 한영 320/768/1440px·영어 다크 모드·특수 경계 등 88개 상태, 13개 상호작용 묶음을 확인했다. 33개 안내 예제·22개 확인 문제, 입력 초안·Escape·초기화, 저장·언어 전환·완료 상태, 목차·검색, 필요한 키보드 스크롤과 그림 로딩, 기존 대표 계산이 통과했다. 자동 접근성 위반과 JavaScript 오류는 없었다.
- 첫 브라우저 실행에서는 초기화 직후 `NumberField`의 비동기 초안 반영 전에 기본값을 단정한 스크립트가 실패했다. 기본 입력값 전체의 반영을 기다리도록 검증 스크립트를 수정했다. 유한·기존 Bayes 단원의 기능을 먼저 재확인한 뒤 전체 검사를 처음부터 통과했다. 제품 입력 코드는 변경하지 않았다.

원시 기록은 git에서 제외한 `.runtime/learning/bayes-*.log`, `bayes-browser.json`, `bayes-research-python.json`에 있다. 전체 테스트 모음과 공개 배포 성공 여부는 GitHub Actions 결과로 별도 확인한다. 이번 선택 검사를 전체 모음의 성공으로 보고하지 않는다.
