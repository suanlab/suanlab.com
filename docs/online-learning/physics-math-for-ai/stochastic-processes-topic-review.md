# 확률과정·확률미분방정식 원주제 대조

2026-11-12 기준. `stochastic-processes`의 11개 원주제를 한영 원고·실습·독립 예제에 대조한다. 기존 seed `stochastic-processes`는 별도로 보존하므로 영역 연결은 12개다. 기존 입문 48개 및 후속 입문 362/362 완료와 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-stochastic.ts`다.

| 원주제 | 단원 ID | 보강한 논증·실행 |
|---|---|---|
| Markov 과정 | markov-transitions-generators | Uniformization·행 질량·Poisson 혼합과 underflow support |
| Poisson | poisson-arrivals-counts | 조건부 이항·도착 순서통계량·의존 변화·rate 우도 |
| Martingale | martingale-stopping-compensation | 비유계 정지의 반례·조화 탈출 확률·균등 적분 가능성 실패 |
| Brownian motion | brownian-covariance-variation | Chi-square·Chernoff·dyadic Borel–Cantelli·무한 total variation |
| Itô·Stratonovich | ito-stratonovich-sums | 예측 가능 적분의 등거리 구성·미래 integrand 반례·유한 walk 완전 열거 |
| SDE | coupled-ou-sde-discretization | 가산 OU의 강한 RMS 차수 1·국소 bridge와 전체 innovation 오차·정상 bias |
| Fokker–Planck | fokker-planck-gaussian-flux | Gaussian 상대 엔트로피·소산·지수 감소와 중심화 flux |
| 시간 역전 | reverse-time-score-conditions | Probability-flow 반 인자·정확 Gaussian 수송·주변 법칙과 경로 법칙의 구분 |
| Ergodicity | ergodic-mixing-time-averages | 정상 공분산·유한 창 분산·음의 상관과 유효 표본 수·구조적 mixing |
| MCMC·Langevin·HMC | mcmc-langevin-hamiltonian | Leapfrog 수정 이차식·정확한 에너지 오차·안정성과 수용 보정 |
| 확률적 제어 | stochastic-lqg-bellman | Adapted 정책의 완전제곱 성능 항등식·비음수 준최적 비용 |

언어별 비교표·표준 Python 블록 11개와 한영 과학 SVG 네 개를 추가했다. QV 그림은 이론 상계이며 빈도 측정이 아니다. Leapfrog 그림은 정확 유리수 제안의 에너지이며 실제 수용 chain으로 제시하지 않는다. 기존 본문·연습·개념·안내·퀴즈·난수 seed 및 유한 창 관례를 보존한다.

## 계산 보완

Bayes 역전은 양의 joint 인자가 남아 있는데 곱이 underflow한 경우 로그 스케일링으로 조건부 행을 계산한다. 실제 영 support는 비운다. 주변 질량은 여전히 binary64로 표시하므로 반올림된 0과 수학적 영을 구분한다.

Gaussian flux는 SDE drift·diffusion 계수에서 중심화하고 곱의 미분으로 공간 도함수를 계산한다. 별도로 계산하는 밀도 시간 미분과 대조하며 작은 평균 이동의 flux를 보존한다. Mixing flag는 흡수·교대의 구조적 경우에서 판정해 표시 고유값의 반올림과 분리한다.

LQG 초과 비용은 완전제곱 Bellman 잔차의 비음수 합으로 구한다. 최적 정책의 정확한 0과 아주 작은 양의 차이를 보존하며 직접 총비용 뺄셈은 `expandedGap`으로 남긴다. 기존 stats 순서·조절기·모형 범위는 바꾸지 않는다. 비음수 항등식은 정확한 대수 관계이며 binary64 범위 아래의 underflow나 입력 반올림까지 제거하는 interval 최적성 인증은 아니다.

## 출처와 보존

확률 구성 배경은 [MIT 확률과정](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/), [Lawler 저자 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf)를 확인했다. 역 SDE·probability flow는 [Song 등](https://arxiv.org/abs/2011.13456), HMC는 [Betancourt](https://arxiv.org/abs/1701.02434), 제어 조건은 [Boyd 강의](https://ee363.stanford.edu/archive/lectures/stoch_lqr.pdf)와 대조했다. 추가한 특화 유리수·Decimal·반례·그림은 독립 구성으로, 원 논문 결과나 일반 확률 패키지의 성능 재현이 아니다.

스물두 원고의 연습 앞에 내용을 추가하고 기존 prefix·연습 이후 suffix를 보존한다. Immutable fixture 네 개는 바꾸지 않고 이전 SHA·검토 SHA·이 기록을 reviewed ledger로 연결한다.

## 검증

관련 계산·원고 보존 검사 67개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 22개 블록을 실행하고 코드·표시 수식의 대응을 확인했다. 별도 정확 유리수 계산으로 LQG 초과 비용 243건과 underflow support 3건을 대조했으며 비용 차이의 최대 상대 오차는 4.5×10⁻¹⁶ 미만이다. 650자리 Decimal 참조에서 작은 평균 이동 flux·공간 도함수 24입력의 최대 상대 오차는 1.6×10⁻¹⁶ 미만이다.

최종 소스로 1,893개 경로를 정적 빌드하고 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사를 통과했다. 실제 export의 한영·320/768/1440px·명암 111개 상태에서 접근성 위반·브라우저 오류가 0이며, 상호작용 13묶음·안내 33개·퀴즈 22개·이전 계산 39개·영역 연결 12개를 확인했다. 정수 초안의 blur 오류·모형 보존·Escape 복원, 표·코드의 키보드 스크롤과 320px의 과학 그림 네 개도 확인했다.

원본 스물두 원고의 본문 prefix·연습 suffix를 다시 비교했다. 소스 1,882개·학습 원고 1,300개·릴리스 파일 33개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

일반 jump·Lévy·거친 경로, 비전역 Lipschitz SDE의 존재·폭발·수치 수렴, 비Gaussian 밀도의 entropy·비평형 전류, 상태 의존 확산의 시간 역전·학습 score 오차, 일반 target의 MCMC 수렴과 부분 관측·제약 제어는 추가 심화 범위다. 원주제 대조가 해당 분야의 모든 연구 이론 완료를 뜻하지 않는다. 이번 검증으로 누적 29영역·284개, 후속 22영역·225개이며 다음은 과학계산과 수치 시뮬레이션 11개다.
