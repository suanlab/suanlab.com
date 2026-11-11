# 인과추론·식별 가능성 원주제 대조

2026-11-12 기준. `causality`의 11개 원주제를 한영 원고·실습·독립 예제에 대조한다. 입문 48개 및 후속 입문 362/362 완료와 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 기존 계산은 `src/lib/learning-causality.ts`·`learning-causal-research.ts`·`learning-causal-graphs.ts`다.

| 원주제 | 단원 ID | 보강한 논증·독립 실행 |
|---|---|---|
| SCM·DAG | binary-structural-causal-model | Truncated factorization·역위상 정규화·의존 외생 입력의 한계 |
| 교란·선택 | confounding-selection-reversal | Augmented functional·double-robust 곱 잔차·nuisance와 식별 가정 |
| 관측·개입 | observation-intervention-policy | Odds 개입·Support 보존·가중치 경계·정책 미분 |
| Do-calculus | do-calculus-graph-conditions | Front-door 단계 유도·안쪽 모집단 평균·정확한 joint 검산 |
| 반사실 | abduction-action-counterfactual | Fréchet 결합 경계·benefit/necessity·무작위 관측의 비식별 |
| 식별 가능성 | observational-equivalence-causal-bounds | 단조 반응의 sharp 경계·추가 가정의 충돌·추정 불확실성과 식별 |
| IV | instrumental-moment-assumptions | Principal stratum·Wald LATE 증명·전체 ATE와 구분 |
| 매개효과 | natural-controlled-mediation | Mediation formula 가정·동일 관측과 다른 nested 평균의 반례 |
| 발견 | gaussian-markov-discovery | 재매개변수화로 같은 Gaussian 법칙 구성·개입으로 방향 구분 |
| 불변·이전 | invariant-mechanism-transport | Label 없는 target의 불변성 비식별·층별 민감도 corner와 sharpness |
| Off-policy | hidden-confounding-off-policy | 일관성만의 정책 가치 경계·baseline 개선의 비인증·overlap 한계 |

언어별 비교표와 표준 Python 블록 11개, 한영 double-robust 편향·반사실 결합 SVG 네 개를 추가했다. 그림은 알려진 모집단 functional과 호환 결합의 비교이며 표본 오차 막대나 추정한 confidence band가 아니다. 기존 본문·연습·개념·안내·퀴즈·조절기 범위를 보존한다.

## 계산 보완과 구분

새 `learning-causal-probability.ts`는 Bernoulli log factor와 log-weight 정규화를 제공한다. SCM 및 정책 질의의 관측 조건부·반사실 posterior·backdoor·source transport·marginal policy 조건부를 raw 곱의 0과 구분해 계산한다. 양의 factor 곱이 binary64 아래로 underflow해도 Support를 없애지 않는다. 구조적 0 factor는 계속 제외하고 실제 빈 셀은 미정이다. 원래 atom·cell·joint mass는 보존한다.

반사실의 `logEvidence`는 유한한 극소 증거와 불가능한 증거를 구분한다. Raw evidence가 0으로 반올림돼도 posterior-defined가 1일 수 있다. Raw weight나 확률이 표현되지 않는다는 사실도 숨기지 않으며, oracle로 미관측 셀을 채우거나 floor를 더하지 않는다.

매개효과는 중심 잡음을 적분한 해석식으로 계산하고 반올림된 atom 차이 평균은 `expanded`에 남긴다. IV는 저장된 binary64 계수의 분자·분모를 정확한 dyadic BigInt 다항식으로 만든 뒤 ties-to-even으로 최종 비율을 binary64에 반올림한다. 작은 곱과 큰 항의 상쇄를 모두 보존한다. 직접 나눈 rounded moment는 `expandedIV`, 보통의 인수분해 계산은 `factoredIV`에 남긴다. 정확히 0인 분모·표현 불가능한 최종 비율은 계속 미정이다. 정확한 중간 산술은 이 작은 다항식과 저장 입력에 한정되며 도구 타당성·표본 추론·전체 실수 영역의 인증이 아니다.

## 출처와 보존

배경은 [Pearl의 do-calculus 원전](https://proceedings.mlr.press/r0/pearl95a.html), [Hernán·Robins 저자 자료](https://miguelhernan.org/whatifbook), [Peters 등의 저서](https://mitpress.mit.edu/9780262344296/elements-of-causal-inference/), [Kennedy의 incremental intervention](https://arxiv.org/abs/1704.00211), [Kallus·Zhou의 정책 연구](https://papers.neurips.cc/paper_files/paper/2018/hash/3a09a524440d44d7f19870070a5ad42f-Abstract.html)와 대조했다. 추가 유리수 모집단·완전 열거·반례·그림은 독립 구성이다. 해당 논문의 전체 추정기·longitudinal efficiency·강건 최적화기를 재현했다고 주장하지 않는다.

스물두 원고는 각각 기존 손계산·확인·완전 수치 예제 절 앞에 보강하고 이전 prefix·해당 절 이후 suffix를 보존한다. 단원별 보존 경계는 실행 기록의 anchor와 일치한다. Immutable fixture 네 개를 수정하지 않고 이전 SHA·검토 SHA·이 기록을 reviewed ledger에 연결한다. 추가 표시 수식·Python의 한영 대응을 검사한다.

## 검증

관련 계산·원고 보존 검사 82개와 타입·lint·콘텐츠 검증이 통과했다. 한영 Python 22개 블록을 실행하고 추가 표시 수식·코드의 대응 및 기존 본문 prefix·단원별 보존 경계 이후 suffix를 확인했다. 기존 독립 ancestral moralization 대조는 네 노드 DAG 543개·분리 질의 13,032개를 포함하며 local do-calculus 검사는 그래프·규칙·증거·기제 864개 조합을 검사한다.

별도 650자리 Decimal 참조에서 반사실 1,680건, 관측·backdoor·정책·이전 672개 조합, 매개효과 1,296건, IV 1,620건을 대조했다. Scaled error는 |계산−참조|/max(1,|참조|)이며 전체 최대는 2.9×10⁻¹⁴ 미만이다. 절댓값 10⁻³⁰⁰보다 큰 참조의 최대 상대 오차는 1.3×10⁻¹³ 미만이다. Raw evidence가 0이었던 양의 증거 89건, 직접 차이 평균이 0이었던 비영 매개효과 scalar 1,926개, 직접 IV 비율이 0이었던 비영 효과 90건을 보존했다.

IV는 추가로 Python의 정확한 Fraction 비율과 float 반올림을 대조해 1,620건 중 유한 결과 1,270건이 일치하고 분모 0·overflow 350건을 미정으로 유지함을 확인했다. Node 회귀 검사에는 binary midpoint의 ties-to-even, subnormal·음의 0·overflow 경계를 포함했다. 이 보증은 저장된 binary64 입력의 작은 다항식에 한정되며 임의의 실제 자료나 인과 가정 자체의 인증이 아니다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 실제 export의 116개 상태는 한영 320/768/1440px 72개·영문 desktop dark 12개·경계 31개·영역 연결 1개다. Interactive Book 영역의 접근성 위반과 uncaught 브라우저 오류는 0이며 상호작용 13묶음·안내 33개·퀴즈 22개·이전 계산 62개·영역 연결 11개를 확인했다.

희귀한 SCM 관측·정책 질의·증거·source cell·logging action의 양의 Support와 작은 IV·매개효과가 실제 readout에도 유지되는지 검사했다. 정수 초안 blur 오류·모형 보존·Escape 복원, 설정 저장·언어 전환·완료 상태, 원고 표·코드의 키보드 스크롤과 320px 그림 네 개도 확인했다. 소스 1,883개·학습 원고 1,300개·릴리스 파일 35개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

일반 identification·transportability 알고리즘, treatment-induced confounding과 longitudinal g-method, 효율적·cross-fitted 추정과 동시 추론, 미지 sensitivity parameter의 근거, 약한 도구의 finite-sample inference, 잠재 원인·피드백을 포함한 발견 및 실제 정책 최적화는 후속 심화 범위다. 원주제 대조는 모든 인과 이론의 완료를 뜻하지 않는다. 이번 대조로 누적 33영역·327개, 후속 18영역·182개이며 다음은 제어·최적제어·강화학습의 수학 11개다.
