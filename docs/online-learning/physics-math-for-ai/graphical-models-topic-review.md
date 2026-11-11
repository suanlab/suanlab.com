# 확률 그래프 모델 원주제 대조와 구조·추론·학습 심화

2026-11-11 기준, `graphical-models`의 10개 원주제를 한영 원고와 `learning-graphical.ts`에 대조했다. 기존 단원을 보존하면서 정확성 조건의 증명·반례·독립 계산을 보강했다. 초기 15영역의 147개에 이어 이 영역을 포함한 대조 범위는 16영역·157개이며, 전체 51영역·509개 이론과 연구 심화의 완료를 뜻하지 않는다.

## 원주제와 확인 위치

원고 경로는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`이다.

| 원주제 | ID | 정의·가정·보강한 근거 |
|---|---|---|
| Bayes network·Markov random field | bayesian-markov-factorization | 유한 DAG sink 정규화 귀납, 순환 조건부 곱의 반례, 양의 collider와 moralization |
| factor graph | factor-graphs-products | bipartite scope, 소거의 분배법칙 불변식, 메시지 배율과 Z 복구, 영 evidence |
| 조건부 독립·d-separation | d-separation-colliders | 경로 차단·faithfulness, 후손 관측의 정확분수 표, 2×2 행렬식, 작은 정보량 |
| variable elimination | variable-elimination-orders | 비정규화 주변 불변식·fill-in, sum/max의 부등식과 marginal MAP 반례 |
| junction tree | junction-tree-calibration | separator 절단 증명·running intersection, 구조가 틀리면 calibration만 맞는 반례·그림 |
| belief propagation | tree-belief-propagation | 부분나무 귀납, 단일 field chain의 tanh·Z 닫힌식, 거리와 동기 라운드 |
| HMM·상태 공간 모델 | hmm-forward-backward | αβ evidence 불변식·γ/ξ, tied EM 갱신·Jensen, Kalman 완전제곱의 조건 |
| 구조·모수 학습 | graph-structure-parameter-learning | MLE 우도 차=N×경험 MI, BIC의 단위·방향 한계, Laplace 정규성 조건 |
| energy·partition function | energy-partition-functions | log Z Hessian·공분산·열용량, 두 에너지 준위 검산, 에너지 학습의 두 평균 |
| 정확·근사 추론의 조건 | exact-approximate-inference | triangle의 even-edge 전개·Z·상관, BP pair와 독립 곱 구분·그림, 고정점의 한계 |

## 독립 예제와 시각화

8종 Python 코드가 한영 원고에 같은 내용으로 들어간다. 후손 관측·sum/max 반례·잘못된 클러스터·HMM은 분수로 계산하고, 나무 tanh·BIC·에너지 모멘트·cycle 상관은 닫힌식과 독립 열거 또는 수치 차분을 대조한다. 유한차분은 정리의 증명과 구분한다.

후손 D를 정확도 4/5로 관측하면 P(D=1)=31/50, P(A=1|D=1)=37/62, 공분산은 −111/1922다. 이 확장 예제는 화면에 네 번째 노드 제어를 추가한 것은 아니다. Marginal MAP의 양의 결합 표는 max_A sum_B=.59와 sum_B max_A=.69를 구분한다.

Junction tree 반례는 w=1{A=B}1{A=C}와 가운데 상수 인자를 사용한다. AB−BC−AC가 A를 둘로 복사한 잘못된 합은 Z=4이고 원래는 Z=2다. 가운데 BC에 equality 인자를 추가하는 다른 모형으로 혼동하지 않는다. Separator 주변은 둘 다 공정해 calibration만으로 잘못된 구조를 발견할 수 없다. AB−ABC−AC로 확장하면 A의 연결을 복구한다.

HMM은 prior를 공정하게 고정하고 r,s만 tied 학습한다. 관측 0011에서 한 번의 정확 EM은 (.8,.9)→(.674656,.909637), evidence .058276→.068564로 갱신한다. 일반 transition 행의 영 방문 분모·전역 최적점·일반화 보장은 별도 문제다.

Triangle의 h=0에서 Z=8cosh³(J)(1+t³), 정확 pair 상관=(t+t²)/(1+t³), BP pair 상관=t, 독립 평균 곱=0이다. 두 새 한영 SVG는 running intersection과 세 상관의 차이를 그린다. 기존 화면이 독립 평균 곱을 BP pair로 잘못 부른다고 주장하지 않는다.

## 수치 수정

약한 의존에서 H(A)+H(C)−H(A,C)는 크기 1의 항을 빼므로 양의 작은 값을 잃었다. Chain의 ε=.5−2⁻²⁰에서 기존 계산은 0이지만 정보량은 약 9.55×10⁻²⁴ bits다. 독립점 근방에서 모형의 정확한 대수적 2×2 행렬식과 KL의 비음수 나머지 급수를 사용해 차이를 보존한다. Chain·fork의 무조건부와 collider의 두 조건 사건을 검사하고, d-separated·정확 독립인 0은 그대로 구분한다. 이 범위에서 급수의 인수 크기를 제한하며 일반 임의 분포나 임의 정밀도의 수치 인증으로 확대하지 않는다.

## 출처와 후속 범위

- [Stanford CS228: directed representation](https://ermongroup.github.io/cs228-notes/representation/directed/), [VE](https://ermongroup.github.io/cs228-notes/inference/ve/), [junction trees](https://ermongroup.github.io/cs228-notes/inference/jt/): 표현·소거·메시지·running intersection의 배경. 수치 반례와 독립 예제는 직접 구성했다. 관측 DAG를 인과 모형으로 자동 해석하지 않는다.
- [Rabiner (1989), §III](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf): forward/backward·reestimation 배경. 여기의 tied 두 모수 EM·정확분수 예제는 직접 유도했다.
- [Schwarz (1978)](https://sites.stat.washington.edu/courses/stat527/s14/readings/ann_stat1978.pdf): 차원 비용의 점근 배경. 고정 후보·정규성 가정과 화면의 경계 MLE를 구분한다.

일반 causal discovery·식별, latent/singular 구조 학습, 비선형·비Gaussian filtering/smoothing, 대규모 treewidth 최적화, 일반 loopy BP 수축 정리, Bethe 자유에너지·loop 보정과 Monte Carlo의 혼합·오차 증명은 후속 심화다. 기존 입문 완료를 이 연구 범위의 완료로 계산하지 않는다.

## 검증 기록

- 그래프 계산 검사 27개와 기존 원고 보존 검사 7개, 선택한 한영 reader 렌더링 검사 10개가 통과했다. 전체 테스트 모음의 통과로 확대하지 않는다.
- 최종 원고의 Python 코드 16개를 재실행하고 코드 SHA·출력을 대조했다. 한영의 절 수·표시 수식·실행 코드가 일치하며 320px에서 새 표 16개와 SVG 4개의 배치를 확인했다.
- 작은 정보량 24개 입력을 정확분수의 결합 분포와 200자리 Decimal 로그로 독립 검산했다. 독립점 근방·분기 양쪽의 최대 상대 차이는 약 4.72×10⁻¹² 이하였으며, 표시 막대의 반올림과 모형의 작은 양의 정보량을 구분한다.
- `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 1,893개 빌드 경로와 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- `npm run check:graphical`의 최종 실행이 한영·320/768/1440px·다크 모드의 82개 접근성·배치 상태와 12개 동작 그룹을 통과했다. 안내 예제 30개·퀴즈 20개, 저장·언어 전환·초기화·불가능/극소 HMM 관측·작은 정보량·기존 계산을 포함한다.
- 표의 숫자가 중간에서 끊기지 않도록 셀의 `overflow-wrap`을 normal로 바꾸고, 긴 표의 기존 키보드 스크롤을 유지했다. 새 표 16개와 기존 net·Gaussian spectrum·ELBO 세 reader의 320px 배치를 확인했다. 회귀 검사는 viewport 변경의 레이아웃 반영을 기다린 뒤 판정한다.
- 빌드 후보의 소스 1,882개와 한영 원고 1,300개, 설정·릴리스 파일 및 SVG 4개의 내보낸 바이트를 대조했다. 공개 반영 완료는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
