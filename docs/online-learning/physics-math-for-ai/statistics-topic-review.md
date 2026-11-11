# 통계학 원주제 대조와 포함률·재표집·점근·설계 심화

2026-11-11에 통계학의 10개 원주제를 연결된 한영 12개 reader와 `learning-statistics.ts`·`learning-foundations.ts`의 계산에 대조했다. 포함률 완전 열거, 최댓값 bootstrap 반례와 모수적 비교, 고차 delta 및 국소 대안, 정수 예산의 군집 설계를 보강했다. 전체 51영역의 일반 이론·연구 심화는 계속 진행한다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다. 10개 주제에 기존 기초 단원 두 개가 함께 연결되어 reader는 12개다.

| 원주제 | 원고 ID | 확인한 정의·가정·논증 |
|---|---|---|
| 표본·모수·추정량 | samples-parameters-estimators | 고정 관측·모수·표집법칙의 구분, 낮은 비트 관측, 순서와 개수 사건, 불가능한 모형 |
| 편향·분산·일치성 | bias-variance-consistency | 정확 위험 분해, 고정 축소의 점별·전체 모수 일치성, 비편향이지만 불일치인 예 |
| likelihood·MLE·MAP | statistical-estimation, likelihood-mle-map | 고정 자료·binomial 계수, 경계 MLE, plug-in SE 한계, Beta의 모드·평균과 좌표 의존성 |
| 지수족·충분통계량 | exponential-families-sufficiency | 고정 지지의 내부 자연모수, 충분성의 모형 상대성, null 조건과 수치 underflow의 차이 |
| 신뢰구간·가설검정 | confidence-intervals-tests | 알려진 σ의 Gaussian pivot, 이항 Wald·score·정확 꼬리 포함률, 검정 역상과 보수성 |
| Bootstrap·재표집 | bootstrap-resampling | 경험법칙과 표집법칙, 27개 행 열거·중복·분위수, 최댓값의 비소멸 atom과 모형 기반 pivot |
| 회귀·잔차 | regression-residuals-diagnostics | 정규방정식·절편, rank 자유도·leverage, 최소 노름 선택과 계수 식별의 차이 |
| 점근 정규성·delta method | asymptotic-delta-method | 스칼라 고차 Taylor 극한, 가측성·국소 매끄러움, Gaussian 모멘트, 변하는 평균의 상대 분산 오차 |
| 검정력·다중 비교 | power-multiple-comparisons | 지정 대립의 power, 독립 전 귀무 FWER와 일반 Bonferroni 상계, 선택 절차의 범위 |
| 실험 설계·오차 추정 | experimental-design-error, research-methods | 대응·군집의 표집 단위, 랜덤 효과 공분산, 균형 정수 예산, 합성 대응 t 구간·계획·누출 |

## 포함률을 표집법칙 전체에서 비교하기

기존 알려진 σ의 정규 구간과 별도로 n=10,p=1/20의 Bernoulli 표집을 사용했다. Wald·Wilson score에는 z=49/25=1.96이라는 통상적인 약 95% 임계값을 명시하고, 정확 꼬리에는 α=1/20을 사용한다. 구간 포함은 부등식으로 검사하므로 유한 포함률을 정확한 분수로 구한다.

Wald는 k=1,2,3에서만 참 p를 포함하여 포함률 약 0.400235이고, score는 k=0,1에서 약 0.913862다. 정확 양측 꼬리는 k=0,1,2에서 약 0.988496이다. 두 이산 꼬리 기각 사건과 합집합 상계로 일반적인 포함률 ≥1−α를 증명했다. 101점 모수 검사는 이 증명을 대신하지 않는다. score의 내부 분산과 경계 이차식 평가를 구분했다. 기존 영어의 임계값 초과 주어도 |Z|=2로 명확히 했다.

## 최댓값 재표집의 실패와 모형 기반 비교

iid Uniform(0,θ)의 R_n=n(θ−M_n)/θ 생존함수는 (1−t/n)^n이다. 경험 재표집의 n(M_n−M_n*)/M_n에는 원점 질량 1−(1−1/n)^n→1−exp(−1)이 남는다. 모집단 지수 극한의 연속점인 원점에서 CDF가 다르므로 일치적 근사가 실패한다.

관측 (1/5,1/2,4/5)의 27개 재표본은 pivot 0,9/8,9/4를 19,7,1번 만든다. 반면 Uniform(0,M_n)에서 표집한 모수적 정규화 pivot은 참 R_n과 유한 n에서도 같은 법칙이다. 후자는 모형의 타당성을 가정한 결과이며 다른 모집단에서도 유효하다는 주장은 아니다.

## 고차 delta와 줄어드는 모수 근방

가측 g:R→R의 국소 C^r, 첫 비영 도함수 r차와 √n 추정량 극한을 가정했다. Taylor 나머지 계수의 확률수렴과 주항의 확률적 유계성으로 n^(r/2) 변환 극한을 유도했다. 모멘트 수렴까지 자동으로 따른다고 주장하지 않는다.

정확한 Gaussian 행 모형 μ_n=c/√n에서 n·표본평균²은 (Z+c)²이고, n² 분산은 2+4c²다. 일차 근사는 4c²이므로 c=1/2에서 상대 분산은 항상 1/3이다. 고정 비영 평균의 점근 결과를 원점 근방의 균일 보장으로 바꾸지 않도록 했다. 코드의 분수 모멘트와 고정 값의 거듭제곱 척도 검산을 일반 분포 증명과 구분했다.

## 독립 군집과 반복의 비용 배분

단위 분산 모형 Y_ji=μ+U_j+ε_ji에서 Var(U)=ρ,Var(ε)=1−ρ로 두고 독립성을 명시했다. 균형 C군집·m반복의 평균 분산은 [ρ+(1−ρ)/m]/C이며 Gaussian 가정 없이 공분산으로 유도했다.

선형 비용 a+bm, 예산 T의 연속 완화 최적 반복 수는 sqrt(a(1−ρ)/(bρ))이다. 제약과 정수화를 별도로 적용했다. a=9,b=1,T=120,ρ=1/4,C≥2에서 가능한 m=1,…,51을 모두 열거하면 m=6,C=8,분산 3/64가 유일한 최적이다. 연속 값 sqrt27을 반올림한 m=5는 분산 1/20으로 더 크다. 전체 공분산 합과 상관 민감도를 분수로 검산한다. 결론은 지정한 균형 설계·비용·공분산 모형 안의 것이며 불균형 표집의 일반 최적해는 아니다.

## 출처와 후속 범위

- [NIST §7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm): Wilson 역검정과 정확 이항 꼬리 구간.
- [Bickel–Freedman (1981), 원문 PDF](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/BickelFreedman81.pdf), §6 Counter-example 2, 인쇄 p.1210; [DOI](https://doi.org/10.1214/aos/1176345637): 최댓값·모수적 pivot 비교. 스캔 원문을 직접 확인했다.
- [MIT 18.655 Lecture 16](https://ocw.mit.edu/courses/18-655-mathematical-statistics-spring-2016/2ca793c6ea6c4a6fece39c0da3440ba2_MIT18_655S16_LecNote16.pdf), 정리 5.3.3·분산 논의: Taylor 기반 분포와 모멘트 근사의 범위.
- [MIT 18.650 강의 자료](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/pages/lecture-slides/): 원주제의 추정·검정·회귀 배경. 이항 포함률 분수, 세 점 열거, 고차·국소 Gaussian과 균형 비용 배분은 명시한 모형에서 직접 계산·유도했다.

일반 bootstrap 일치성 조건과 대안, 벡터 고차 delta·정량 균일성, 선택 후 추론·의존 자료 구간, 일반 불균형·정보성 표집 설계와 실제 벤치마크 성능은 후속 연구 심화다. 10개 원주제 대조를 전체 통계 이론의 완료로 계산하지 않는다.

## 검증 기록

- `learning-statistics.test.ts` 23개, `learning-manuscript-preservation.test.ts` 7개, 기존 기초 계산 2개와 연결된 12개 reader의 한영 원고·수식 검사 12개가 통과했다. 합계 44개이며 선택 실행에서 제외된 검사는 통과 수에 넣지 않았다.
- 네 확장 예제의 한영 Python 블록 8개를 실행했다. 최종 원고에서 추출한 코드 해시가 실행 기록과 일치한다. 분수 포함률·27개 재표집·Gaussian 모멘트·51개 정수 설계를 확인했다.
- Node 20에서 `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 1,893개 경로를 생성했으며 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 검사했다.
- 최종 정적 산출물에 `SUANLAB_TEST_BASE=http://127.0.0.1:43299 npm run check:statistics`를 실행했다. 12개 reader와 교재 화면의 한영 320/768/1440px·영어 다크 모드·특수 경계 등 95개 상태, 14개 상호작용 묶음이 통과했다. 36개 안내 예제·24개 확인 문제, 입력 초안과 Escape 복구, 저장·언어 전환·완료 상태, 표·코드의 필요한 키보드 스크롤과 기존 대표 계산을 확인했다. 자동 접근성 위반과 JavaScript 오류는 없었다.
- 처음 실행은 기존 미리보기 서버 종료로 연결이 끊겨 실패했다. 서버를 재시작한 뒤 전체 브라우저 검사를 처음부터 통과했다. 실패를 제품 오류 수정이나 통과 검사로 계산하지 않았다.
- 1,300개 한영 원고, 1,882개 `src/` 파일과 생산 설정이 빌드 사본과 일치한다. 최종 이항 포함률 표의 한영 모바일 화면도 직접 확인했다.

원시 실행 기록은 git에서 제외한 `.runtime/learning/statistics-*.log`, `statistics-browser.json`, `statistics-research-python.json`에 있다. 전체 테스트 모음과 공개 배포의 성공 여부는 별도 GitHub Actions 결과로 확인한다. 이번 선택 검사의 통과를 전체 모음의 통과로 보고하지 않는다.
