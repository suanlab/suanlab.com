# 확률 원주제 대조와 변환·조건부·극한·gradient 심화

2026-11-11에 확률의 10개 원주제를 한영 원고, `learning-probability.ts` 및 `learning-foundations.ts`의 계산에 대조했다. 분포 변환, 공동 Gaussian 예측, CLT의 가정, 재매개화 추정량의 분산을 증명·비교표·실행 예제로 보강했다. 전체 51영역의 일반 이론·연구 심화는 계속 진행한다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다.

| 원주제 | 원고 ID | 확인한 정의·가정·논증 |
|---|---|---|
| 확률공간·사건·확률공리 | probability-spaces-events | 유한 전집합의 σ-대수, 가산 가법성, 사건 마스크, 확률 0과 공집합의 구분 |
| 조건부확률·독립성·Bayes | conditional-probability-independence | 조건 방향·양의 분모, 네 칸 개수, 독립과 서로소, 영 사건과 미정의 확률법칙 |
| 확률변수·분포 | probability-distributions | 지표 합의 독립·동일 p 가정, 이항 계수와 정규화, 모형 질량과 실제 빈도 |
| 질량·밀도·누적분포 | mass-density-cdf | 점 질량의 CDF 점프, 우연속, 절대연속 가정, 특이연속·혼합 분포의 존재 |
| 기대·분산·공분산 | expectation-variance | 모멘트 적분 가능성·단위, 모집단과 표본, 제곱 손실 최적 상수와 손실 의존성 |
| 다변량 Gaussian | multivariate-gaussians | 양의 스케일·준정부호, 타원의 확률 의미, 독립 잔차·Schur complement, 최적 예측과 비Gaussian 반례 |
| 변수변환·Jacobian | random-variable-transformations | 음의 배율·역분기 합, ReLU 혼합 법칙, 삼각 변환의 역행렬식·곡선 지지집합 |
| 결합·주변·조건부 분포 | joint-marginal-conditional | 합산과 재정규화, 영 행·전체 영 개수, 이진 특수성과 일반 무상관 반례 |
| 대수의 법칙·중심극한정리 | laws-large-numbers-clt | Chebyshev와 고정 오차, 특성함수 CLT 논증, 변하는 p의 희귀 사건 반례, CDF와 질량의 단위 |
| 재매개화 | reparameterization-gradients | 고정 경로·기대 gradient, 사차 모멘트와 분산, 동일 비용 antithetic 비교, 문턱 함수의 미분 교환 실패 |

## 분포 변환과 기준 측도

Uniform[−1,1]에 ReLU를 적용하면 원점 질량 1/2와 Uniform(0,1) 절반의 혼합 법칙을 얻는다. CDF·점 질량 표, 구간 확률 5/8과 1/8의 차이, 평균 1/4·분산 5/48을 제시했다. Dirac 측도를 통상적인 Lebesgue 밀도와 구분했다.

독립 단위 균등 입력의 (u,v)=(x1,exp(x1)x2)에는 역 Jacobian과 0<u<1,0<v<exp(u)의 지지집합이 모두 필요하다. 지지집합을 예전 직사각형으로 두면 질량 1−exp(−1)만 남는다. 정확한 혼합 질량과 명시적인 삼각 변환의 부동소수점 역행렬·정규화를 별도로 검산한다. 기존 실습의 선형·제곱 CDF와 일반 flow 구현을 혼동하지 않도록 범위를 명시했다.

## 공동 Gaussian과 최적 예측

독립 표준정규 구성에서 Y=βX+η를 유도하여 조건부 법칙과 Schur complement를 증명했다. 제곱 적분 가능한 임의 h에 대한 오차 분해로 βX가 모든 허용 예측기 중 최적임을 보였다. 공분산 [[4,3],[3,9]]의 β=3/4, 잔차 분산 27/4와 조건 X=2의 평균 3/2를 정확한 분수로 검산했다.

X가 −1,0,1 균등이고 Y=X²−2/3인 법칙은 대각 공분산 diag(2/3,2/9)를 갖지만 종속이다. 같은 공분산의 공동 Gaussian과 달리 실제 조건부 분산은 0이다. 선형 오차 2/9와 비선형 오차 0의 차이로 공분산만으로 Gaussian 조건부를 적용할 수 없음을 보였다.

## 고정 법칙 CLT와 희귀 사건의 차이

고정 p의 중심 Bernoulli 특성함수를 Taylor 전개하고 독립 합의 n제곱과 Lévy 연속성 정리로 Gaussian 극한을 얻었다. 사용한 연속성 정리의 측도론적 증명과 균일 CDF 오차율은 여기서 증명했다고 계산하지 않는다.

각 행에서 p_n=1/n을 택하면 표준화 횟수의 −1/2 CDF는 (1−1/n)^n→exp(−1)이며 정규 CDF 약 0.308538과 다르다. 평균의 분산은 (n−1)/n³로 0에 가므로 집중과 비정규 표준화가 양립한다. n=5·20·100의 표와 n=1000까지의 정확한 분수 계산, 고정 p 특성함수의 수치 예시를 분리했다. 기존 조절기의 n≤30 범위를 확장했다고 주장하지 않는다.

## gradient의 모멘트·비용·미분 교환

m=μ−1에서 경로 gradient 분산은 (σ²,m²+2σ²)이다. 사차 잡음 모멘트 3이 필요하므로 ±1 두 노드의 기대 gradient 일치는 실제 Gaussian 분산의 검증이 아니다. 같은 두 번 평가 비용으로 독립 평균과 antithetic 평균을 비교했다. 후자의 scale 분산은 2σ²이고 m²>2σ²일 때만 개선된다. 기준 (μ,σ)=(0,1)에서는 3/2에서 2로 악화된다.

Bernoulli 문턱 1{U<θ}는 기대 미분 1과 거의 모든 경로의 미분 기대 0을 갖는다. 폭 h·높이 1/h 차분의 정확한 적분으로 미분 교환의 지배 조건이 실패함을 설명했다. 코드는 다항식 Gaussian 모멘트를 분수로 계산하며 무작위 표집이나 VAE 학습 실험은 아니다.

## 출처와 후속 범위

- [Pishro-Nik §4.1.3](https://www.probabilitycourse.com/chapter4/4_1_3_functions_continuous_var.php): 사건 역상·역분기와 변수변환 조건.
- [Pishro-Nik §5.3.2](https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php): 독립 표준정규 구성과 조건부 Gaussian.
- [Pishro-Nik §7.1.2](https://www.probabilitycourse.com/chapter7/7_1_2_central_limit_theorem.php): 고정 i.i.d.의 유한 비영 분산 CLT.
- [Sheffield, MIT 18.175 Lecture 15](https://math.mit.edu/~sheffield/175/Lecture15.pdf), Continuity theorems·Moments, derivatives, CLT 절: 연속성 정리·모멘트와 CLT.
- [Kingma–Welling §§2.3–2.4, v11](https://arxiv.org/html/1312.6114v11): 미분 가능한 잡음 구성과 경로 추정. 논문의 학습 실험을 재현한 것은 아니다.

위 표·작은 행렬·희귀 사건·동일 비용 분산·문턱 반례는 명시한 법칙에서 직접 유도했다. 일반 측도 분해·조건부 확률의 구성, 연속성 정리의 완전 증명, 삼각 배열 CLT와 정량 오차율, 일반 가역 신경망, 실제 gradient 추정·학습 성능은 후속 연구 심화다. 10개 원주제 대조를 전체 확률 이론의 완료로 계산하지 않는다.

## 검증 기록

- 확률 계산 19개·기존 이항/기대분산 계산 2개·원고 보존 7개, 총 28개 검사를 통과했다. 네 단원의 한영 변경 8개는 명시한 검토 해시만 허용한다.
- 10개 단원의 한영 이론·수식 렌더링 검사를 통과했다. 한영 Python 코드 8개를 실행하고, 문장 끝 마침표를 정규화한 표시 수식의 일치를 확인했다.
- Node 20 정적 빌드와 타입 검사를 통과했다. 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical 검사도 통과했다.
- 기존 기초 단원 두 개까지 포함한 10개 단원을 한영 320·768·1440px, 영문 다크 모드와 영역 검색으로 검사했다. 78개 화면 상태와 12개 상호작용 묶음, 30개 안내 실험·20개 퀴즈를 통과했으며 접근성 위반과 페이지 오류는 0이다.
- 개념 단계·목차·언어 전환·입력 저장·완료 상태·영 개수·특이 Gaussian·변환 경계와 정수 초안 오류/계산 보존/Escape 복원을 확인했다. SVD·TT·통계 추정·정규화의 기존 기준값도 확인했다.
- 직접 확인한 320px gradient 비교표의 좁은 열 줄바꿈을 수정했다. 숫자 묶음에 inline math를 쓰고 열 제목을 짧게 했다. 이후 해당 단원의 원고 렌더링·보존 검사를 다시 통과하고 최종 빌드에서 한영 320·768·1440px와 한영 320px 다크 모드 8개 상태를 추가 검사했다. 표의 숫자 셀 9개·헤더·필요한 키보드 스크롤과 접근성을 확인했고 최종 영문 모바일 화면을 직접 확인했다.
- 최종 표 수정 후 정적 링크 검사를 다시 통과했다. 최종 빌드의 1,300개 한영 원고는 작업본과 일치한다.

증거는 무시 경로 `.runtime/learning/probability-{research-python.json,calculation-tests.log,foundation-tests.log,render-tests.log,typecheck.log,browser.json,browser.log,build-final.log,export-final.log}`와 `probability-gradient-{render-final.log,final-browser.json,final-browser.log,prose-en-mobile-final.png}`, `probability-preservation-final.log`에 남겼다. 전체 회귀 검사와 공개 배포는 push 이후 GitHub Actions 결과로 별도 확인한다.
