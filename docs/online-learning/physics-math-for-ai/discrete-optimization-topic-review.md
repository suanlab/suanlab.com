# 조합 최적화·형식적 추론 원주제 대조

2026-11-12 기준. `discrete-optimization`의 원주제 9개와 한영 단원 9개를 대조한다. 요청 당시 입문 48개 및 후속 입문 362/362의 완료를 전체 51영역·509개 연구 심화 상태와 구분한다.

## 원주제와 보강 범위

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`다. 기존 정의·실습·손계산·연습 해설을 유지하고 연습 절 앞에 연구 확장을 넣는다.

| 원주제 | 단원 ID | 추가한 논증·독립 실행 |
|---|---|---|
| 정수계획·relaxation | binary-knapsack-integer-relaxation | 비음수 가격의 쌍대 상한·원문제 일치·등식 모드의 승수 부호 |
| Matching·cut·flow | bipartite-matching-cover-flow | Hall 최대 결핍 공식·종료 도달 집합·512개 그래프 독립 열거 |
| Submodularity | coverage-submodularity-greedy | 오차 있는 marginal oracle·적응 질의의 균일 가정·가산 손실 |
| SAT·SMT·CSP | sat-smt-integer-difference | 비엄격 정수 차분계의 실수/정수 실행 가능성·구성적 witness·쌍대 모순 |
| Branch-and-bound | branch-bound-knapsack-certificates | 정수 목적값 격자의 조기 인증·실수 상한과 값 인증의 구분 |
| 근사 보장 | knapsack-value-scaling-fptas | 예측 점수 오차와 알고리즘 근사 오차를 분리한 참 가치 하한 |
| 증명 탐색·type checking | typed-lambda-proof-search | Peirce의 두 세계 Kripke 반모형·타입 규칙 건전성·Boolean 타당성과 구분 |
| 프로그램 합성 | counterexample-guided-polynomial-synthesis | 잡음 보간·sharp 외삽 증폭·절점 간격과 계수 오차 |
| Soft relaxation·정확해 | entropy-softmin-discrete-choice | 동률 접근과 냉각의 결합 극한·희귀 엔트로피의 정규화 항 |

9개 표준 Python 예제를 한영에 동일하게 추가했다. 정확 분수·부분집합·그래프·오차 패턴 열거와 Kripke 의미 평가를 사용한다. Hall 결핍 빈도와 Gibbs 결정 경계는 한영 SVG 네 개다. 그래프 개수는 이름 붙인 512개 그래프의 열거이며 실제 데이터의 경험 분포가 아니다. 보조 오차 모형·Kripke 반모형·격자 인증을 기존 조절기의 구현이라고 보고하지 않는다.

## 수치 수정

`learning-combinatorial-soft.ts`는 정규화를 최소 선택지 개수와 나머지 질량으로 나누고 `log1p`로 계산한다. `1+w`가 1로 반올림돼도 표현 가능한 양의 로그 정규화를 보존한다. 로그 확률과 `logPartition + expectedExcess/temperature` 엔트로피에 적용한다. 희귀 이진 선택지의 기존 계산은 정규화 기여를 빠뜨려 `gap/temperature=300`에서 상대 오차 약 0.33%를 보였다.

원래 확률·총비용·분산·선택과 조절기 범위를 유지한다. 평균 오차는 목표에서의 가중 편차를 보상 합산한 뒤 제곱한다. 반올림된 평균으로 구한 오차는 `expandedMeanLoss`에 보존한다. 유일 최적점과 불규칙 동률에서 두 큰 편차가 소거될 때도 표현 가능한 작은 평균 잔차와 오차를 유지한다. 비용 차이 결과창은 반올림한 총비용의 뺄셈 대신 분산 항등식을 사용한다. 해당 뺄셈은 `expandedCostGap`으로 보존하여, 두 총비용이 같게 반올림돼도 표현 가능한 양의 차이를 숨기지 않는다. `expandedEntropy`는 이전 산술, `expectedExcess`는 최소 비용 위의 기대 초과량, `softMinimumCorrection`은 최소 비용에 더하는 정규화 보정이다. 총비용에서 작은 보정이 반올림되거나 확률이 underflow할 수 있다는 사실을 숨기지 않는다. 해석적 0온도 모드는 계속 동률의 균일 분포이며 새 진단도 입력 온도에 독립이다. 수치 개선은 arbitrary-precision 인증이나 Gibbs 평균의 이산 실행 가능성 증명이 아니다.

## 출처·보존

[MIT flow optimality](https://courses.csail.mit.edu/6.854/21/Notes/n07-flow_computation.html), [Stanford branch-and-bound](https://web.stanford.edu/class/ee364b/lectures/bb_notes.pdf), [PLFA의 구성적·고전 논리](https://plfa.github.io/Negation/)와 기존 각 원고의 primary reference를 유지·대조했다. 추가 쌍대 후보·오차 부등식·Kripke 반모형·정확 열거·그림은 독립 구성이다. 제한된 예제의 성공을 임의 solver의 완전성이나 모든 생성 프로그램의 정확성으로 확대하지 않는다.

18개 원고의 이전 prefix와 연습 절 이후 suffix를 보존한다. Immutable fixture 네 개는 수정하지 않고 기존 SHA와 검토 SHA를 reviewed ledger에 연결한다. 한영 표시 수식·실행 코드·표 수치와 독립 출력의 대응을 확인한다.

## 검증

- 조합 최적화·원고 보존 테스트 **215개 통과**, 실패 0개. 기존 정확 열거·별도 type checker·유한 CEGIS·근사 품질 검사를 유지하고 희귀 정규화·반사·underflow·온도 독립성·작은 비용 차이·평균 오차 회귀 테스트 여섯 개를 추가했다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content`, `npm run build`, `npm run check:export` 통과. **1,893개 정적 경로·1,888개 HTML**의 링크·자산·앵커·사이트맵을 검사했다.
- 한영 320/768/1440px·야간·경계·설정·안내·퀴즈를 포함한 브라우저 검사 **117개 상태·11개 상호작용 그룹·27개 안내 예제**를 통과했다. 새 표·코드 키보드 스크롤, SVG 로드·대체 텍스트, 언어 전환·정수 초안·저장·초기화·완료 표시 및 앞 단원 계산을 확인했다. 접근성 위반·페이지 오류는 없었다.
- 9개 Python 예제의 한영 18개 블록을 실행하고 코드·표시 수식·출력 대응과 기존 prefix/suffix 보존을 확인했다. Hall 공식은 512개 그래프를 독립 열거했고 엔트로피·정규화·초과량·평균·평균 오차·비용 차이·보정·모든 로그 확률은 **120개 설정을 Decimal 800자리**로 별도 대조했다.
- 한영 그림 네 개의 렌더링과 실제 export의 320px 화면을 확인했다. source·원고·설정·릴리스 파일 및 SVG export의 byte 일치를 검사했다. 검증 출력만 이전 출력 백업을 남겨 로컬 홈페이지에 설치한다.

위 항목은 이번 변경의 선별 검증이며 전체 저장소 CI 완료를 대신하지 않는다. 공개 홈페이지 반영은 push 이후 GitHub Pages 배포 상태로 별도 확인한다.

## 후속 범위

일반 다면체의 integrality·cutting plane, 가중·비이분 매칭, matroid/비단조 submodular 최적화, 실제 DPLL(T)·증명 인증서, branch heuristic 성능, 일반 FPTAS bit complexity, 의존 타입·정상화와 완전성, 무한 문법 CEGIS·통계 검증 및 Gumbel/straight-through 편향은 추가 심화 범위다. 원주제 대조는 전체 분야 완성을 뜻하지 않는다. 이번 기록으로 누적 35영역·347개이며 후속 16영역·162개를 이어간다. 다음은 게임이론·다중 행위자 학습 10개 원주제다.
