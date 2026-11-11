# 고차원 확률 원주제 대조와 net·JL·모멘트·특징 오차 심화

2026-11-11에 고차원 확률과 무작위 행렬의 10개 원주제를 한영 10개 reader와 `learning-high-dimensional.ts`에 대조했다. 연속 구면 net의 결정적 제어, 최적 Chernoff 두 꼬리, 유한 Gaussian 이차 모멘트, 유한 Gram·ridge 오차를 보강했다. 초기 15영역의 147개 원주제 대조 범위를 마무리하지만, 전체 51영역의 일반 이론·연구 심화 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다.

| 원주제 | 원고 ID | 확인한 정의·가정·논증 |
|---|---|---|
| concentration | concentration-finite-averages | 독립 부호 평균·분산, 정확 포함 꼬리·경계, Chebyshev와 Hoeffding 비교 |
| sub-Gaussian·tail bounds | sub-gaussian-tail-bounds | 모든 λ의 중심 MGF proxy·Chernoff, 원자·Gaussian, 작은 log cosh 계산 |
| Hoeffding·Bernstein | hoeffding-bernstein-bounds | 독립 Bernoulli의 범위·분산·중심 상계, 퇴화 분포, 격자 사건 |
| covering·packing | covering-packing-nets | 유한 꼭짓점·엄밀 packing, 연속 구면 부피 논증·대칭 이차형식 net 보정 |
| 고차원 기하 | high-dimensional-geometry | 공 내부 반지름·껍질, 구면 등방성·제곱 거리, 실제 자료 모형과 구분 |
| Johnson–Lindenstrauss | johnson-lindenstrauss-projections | 고정 점·제곱 거리, Gaussian χ² MGF·두 Chernoff 비율, 쌍 union·적응 선택 한계 |
| random matrix spectrum | random-matrix-spectra | 비중심 Gram·수치 영점·MP 기준, 유한 trace 모멘트·랭크·세 절점 적분 |
| Gaussian width | gaussian-width-geometry | 축·구면의 supremum, 기대 폭·chi recurrence·근사 적분, 폭과 표본 수의 구분 |
| random features | random-fourier-features | paired map·PSD·점별 분산, 고정 유한 쌍 오차·Gram 연산자·ridge 계수 |
| 차원의 저주·유효 차원 | curse-effective-dimension | 정확 BigInt 격자, 공분산 effective rank·참여 랭크·ridge 차원의 서로 다른 정의 |

## 보강한 논증과 계산

전체 Euclidean 구면의 ε-net은 부피 비교로 크기 (1+2/ε)^d 이하로 만들 수 있다. 실수 대칭 A에서 두 이차형식의 차이를 2ε||A||로 제어해 Q_N≤||A||≤Q_N/(1−2ε)를 증명했다. 분모 조건 ε<1/2를 명시한다. 연속 원의 n등분 net 반지름은 2sin(π/(2n))이며, A=((2,1),(1,−1)), n=8,16,64에서 Q와 인증 상한을 정확 고유값 노름에 대조한다. 유한 꼭짓점을 덮는 기존 실습과 별도 문제다.

독립 Gaussian 투영의 χ²_m MGF에서 Chernoff 최적화로 a_+=(ε−ln(1+ε))/2와 a_-=(−ε−ln(1−ε))/2를 얻었다. 고정 점의 L쌍에 L(exp(−ma_+)+exp(−ma_-))를 적용한다. N=4,L=6,δ=.05에서 두 꼬리 상계의 충분 정수 차원 367·102·46을 기존 단순 상계의 421·132·74와 비교한다. ε는 .25,.5,.8이며 최소는 지정 상계를 만족하는 정수에 대한 것이지 실제 최소 임베딩 차원이 아니다. MGF 수치 적분과 영 공간의 적응 선택 반례를 함께 제시했다.

이상적 iid Gaussian의 S=AᵀA/m에서 E tr S=d, E tr(S²)=d+(d²+d)/m, Var(tr S/d)=2/(dm)을 독립 2차·4차 모멘트로 유도했다. tr(S²)와 (tr S)²를 구분하고 trace만으로 스펙트럼을 제어하지 못하는 I₃·diag(3,0,0)의 한영 벡터 그림을 추가했다. 3점 Gaussian 적분 절점의 곱을 분수로 열거하여 지정한 다항식 모멘트를 검산한다. 이 3점 법칙이 Gaussian의 전체 분포나 랭크 확률까지 재현한다고 주장하지 않는다.

paired RFF의 cos 평균에 Hoeffding을 적용하고 고정 L쌍의 union bound로 m≥(2/η²)ln(2L/δ)를 얻었다. 정확한 단위 대각을 사용해 Gram 오차 ||E||≤(N−1)η, PSD와 resolvent로 ridge 계수 오차 ≤(N−1)η||y||/λ²를 유도했다. λ와 Nλ 관례, 학습 계수와 새 점 예측·일반화를 구분했다. 고정 Python 주파수 접두열 m=2,8,32와 λ=.1,1,10의 선형계 검산은 JavaScript seed의 동일 값이나 모든 seed에서의 단조 개선을 주장하지 않는다.

## 수치 수정

작은 x에서 기존 log cosh 식은 비슷한 상수를 빼서 상대 오차가 커졌다. 예를 들어 x=10⁻⁸에서 기존 값은 약 1.11022×10⁻¹⁶이지만 올바른 이차 주항은 5×10⁻¹⁷이다. |x|<1에서 log1p(2sinh²(x/2))를 사용하고 큰 |x|의 overflow 회피 식은 유지했다. 양·음 λ, 작은 양의 값과 정확한 0, 분기점 양쪽을 회귀 검사한다. 반각의 곱은 작은 값의 제곱보다 2배를 먼저 적용해 중간 underflow를 줄이고, Gaussian 이차식도 반곱을 먼저 적용했다. 표현 가능한 subnormal 양의 값은 유지하고 표현 범위 아래의 선택 로그 MGF는 —·수치 범위 안내와 점 보류로 구분한다. 임의 정밀도나 모든 반올림 경계의 인증은 아니다.

## 출처와 후속 범위

- [Vershynin, §5.2.2, Lemmas 5.2–5.4](https://arxiv.org/html/1011.3027#S2.SS2): net·연산자 제어의 배경. 양의 보정 분모 조건을 명시하고 2×2 예제를 직접 계산했다.
- [Stanford CS 368 Lecture 4](https://cs368-stanford.github.io/spring2022/lectures/lecture4-notes.pdf): Gaussian 투영·χ²·고정 유한 점 union의 배경. 두 최적 지수와 정수 표는 제곱 오차 관례에서 직접 계산했다.
- [Vershynin, §5.3.1·5.4.3](https://arxiv.org/html/1011.3027): 유한·점근 Gaussian 행렬의 배경. trace 모멘트와 3점 적분은 직접 전개했다.
- [Rahimi–Recht (2007), §3](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf): Fourier feature 배경. 원문의 다른 feature 관례를 그대로 복사하지 않고 paired cos 평균 범위에서 상수를 구했다. Gram·ridge 섭동은 직접 유도했다.

일반 empirical process·chaining, 비등방·의존·heavy-tail 공분산, 극단 고유값의 정량 정리, 연속 영역의 uniform RFF·학습 일반화, 구조적 유효 차원의 정확한 표본 복잡도는 후속 연구 심화다. 초기 147개 대조를 전체 509개 이론의 완료로 계산하지 않는다.

## 검증 기록

- 고차원 계산 검사 28개와 기존 원고 보존 검사 7개, 선택한 한영 reader 렌더링 검사 10개를 통과했다. 마지막 수치 안내 변경 후 해당 reader도 다시 검사했다. 전체 테스트 모음을 실행한 결과와 혼동하지 않는다.
- 네 보강 단원의 한영 Python 코드 8개를 실행하고 최종 코드 SHA와 출력이 기록과 같은지 다시 확인했다. 320px에서 한영 표 8개와 최종 정적 SVG 2개의 배치를 확인했다.
- `npm run check:high-dimensional`은 실제 정적 빌드에서 한영·320/768/1440px·다크 모드 등 83개 접근성·배치 상태, 12개 동작 그룹, 안내 예제 30개와 퀴즈 20개를 통과했다. 작은 양의 log MGF와 표현 범위 아래 입력, 저장·언어 전환·초기화·기존 계산 회귀를 포함한다.
- `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`를 통과했다. 빌드는 1,893개 경로를 생성했으며 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 빌드 후보의 소스 1,882개·한영 학습 원고 1,300개와 릴리스 파일·설정이 작업 저장소와 같은지 대조했다. 공개 반영 여부는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
