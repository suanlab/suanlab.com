# 볼록해석·제약 최적화 원주제 대조

2026-11-11 기준. `convex-optimization`의 10개 원주제를 기존 한영 원고·실습·독립 계산에 대조했다. 기본 ridge 단원 `regularization`도 읽고 기존 SVD·역문제 연결을 보존했다. 영역의 연결 단원은 11개이지만 원주제는 10개다. 누적 대조 범위는 23영역·225개 원주제이며 전체 51영역·509개의 일반 이론·연구 심화 완료를 뜻하지 않는다.

## 원주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`, 계산은 `src/lib/learning-convex.ts`다.

| 원주제 | ID | 보강한 논증·예제 |
|---|---|---|
| 볼록집합·함수 | convex-sets-jensen | Epigraph·하반연속성, 닫힌 볼록집합의 사영 존재·유일성과 분리 초평면 |
| Subgradient | subgradient-support-lines | Fenchel 등호, relative-interior 합 규칙과 −sqrt(x)+singleton indicator 반례 |
| 최적성 | constrained-first-order-optimality | 사영 경사 고정점의 동치, L-smooth·단계 크기의 감소 증명 |
| Lagrangian·KKT | lagrangian-kkt-multipliers | 값 함수 도함수 −승수, 활성 전환·제약 배율·단위와 민감도 |
| 약·강 쌍대성 | duality-slater-counterexamples | 엄격 Slater의 perturbation 지지 논증, 승수 미달성과 nonclosed value function 구별 |
| LP·QP·SDP | linear-quadratic-semidefinite-programs | Conic dual 유도, PSD trace 비음수와 trace-zero 행렬 상보성 |
| Proximal | proximal-operators-moreau | Fenchel 역관계의 Moreau 분해, 강볼록 부분경사 단조성의 수축률 |
| Regularization | lasso-ridge-regularization | 상관 설계 lasso의 쌍대 가능점과 gap ball, 충분한 안전 screening 조건 |
| Interior point | barrier-central-path-newton | 중심점 m/t 간극의 가정, scalar self-concordance와 decrement damping |
| Duality gap | primal-dual-gap-certificates | 근사 내부 Lagrangian 평가의 위험과 전역 곡률로 보정한 하한 |

## 독립 예제·그림

Python 10종을 한영에 같은 코드로 제공한다. Fraction으로 분리 초평면·사영 단계의 감소·승수 민감도·SDP trace/상보성·Moreau 분해·lasso dual feasibility·보정 하한을 검사한다. 부분미분 반례의 slope sequence와 perturbation의 명시적 feasible sequence를 제공한다. 유한 수치 목록을 무한 영역의 증명 대신 사용하지 않는다.

장벽 Python은 decrement 기반 보폭이며 기존 실습의 fraction-to-boundary/Armijo 알고리즘과 구별한다. 상관 설계 lasso도 기존 항등 설계 제어와 별도다. 수치 bound가 안전하려면 가능성·gap·엄격 margin을 신뢰할 수 있어야 하며, 인과·통계적 변수 선택 보장으로 해석하지 않는다.

한영 SVG 네 개는 연속이지만 유한 지지 기울기가 없는 값 함수와 끝점 점프로 양의 gap이 생기는 값 함수를 비교하고, proximal/dual 분해와 매끄러운 envelope를 보여준다. 폭 440, title·desc·대체 텍스트와 인접 표·실행 코드를 제공한다. 불가능한 u<0에 유한 함수값을 그리지 않고 열린·닫힌 끝점과 하반연속 hull을 구별한다.

## 계산 보완

인접 입력 x=1,y=1+2⁻⁵²,t=1/2에서 기존 이차 gap은 약 2.22×10⁻¹⁶, 로그 gap은 약 −1.11×10⁻¹⁶이었다. 실제 값은 각각 2⁻¹⁰⁶, 약 2⁻¹⁰⁷이다. 이차·절댓값은 해석적 항등식으로, 로그는 작은 혼합 가중치를 기준으로 한 log1p 및 국소 급수로 계산한다. 두 값이 양수라는 사실과 별도로 표시 높이는 반올림된다. 양수의 subnormal 끝점도 단순 곱의 부호나 반올림된 로그 차이만으로 판정하지 않는다.

경계 목적 초과값은 거의 같은 두 높이를 빼는 대신 최적점 변위로 평가한다. a=1,C=[−1,0],x=−10⁻²⁰에서 약 10⁻²⁰을 보존한다. Proximal 도함수도 해당 해석 공식으로 계산해 y=1,lambda=10⁻²⁰의 작은 힘을 유지한다. KKT의 세 항은 보상 합산한다. 가능한 쌍의 양수 gap이 0으로 underflow하면 gap·거리 bound와 해당 suboptimality 진단을 함께 숨기며 정확한 0 최적해와 구별한다. 불가능 쌍의 대수적 상쇄를 range 오류로 오인하지 않는다.

모든 부동소수점 출력이 형식적으로 outward-rounded certificate라는 주장은 하지 않는다. 정확한 분수 예제와 수학적 정리를 구별하고, 기계 수준의 보증에는 interval/outward 연산이 추가로 필요함을 원고에 명시했다.

## 근거·보존·후속 범위

[Boyd·Vandenberghe의 저자 교재](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)의 볼록성·쌍대·barrier 가정과 exercise 5.21의 문제를 확인했다. [Parikh·Boyd의 저자 논문](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)은 proximal·Moreau 관계의 근거다. [Fercoq·Gramfort·Salmon의 원 논문](https://proceedings.mlr.press/v37/fercoq15.html)은 간극 기반 lasso screening 연구를 제공한다. 본문의 작은 유리수 행렬·민감도·screening·Newton 예제는 별도로 구성한 것으로 논문의 경험 결과 재현이 아니다.

20개 변경 원고의 과거 fixture SHA·현재 SHA·이 기록을 보존 ledger에 연결하고 네 immutable fixture는 바꾸지 않았다. 기존 기본 ridge 원고는 수정하지 않았다. 이후 범위는 일반 cone의 self-concordant barrier와 복잡도 증명, 대규모 solver의 안정성·검증 연산, 식별·통계적 선택 보장과 실제 데이터 실험이다. 이를 끝내기 전에 전체 연구 교육과정의 완료를 선언하지 않는다.

## 검증

- 계산·한영 렌더링·안내 예제·원고 보존의 선택 테스트 66개 통과. 전체 CI 테스트의 통과를 뜻하지 않는다.
- Python 20개 블록의 실행 출력·SHA를 확인했다. 한영 수식·제목 수·언어 순도와 기존 20개 원고의 추가 절 앞뒤 원문 보존을 대조했다.
- 정확 binary64 입력의 800자리 Decimal 로그 계산 30개 사례와 비교했다. 인접 입력·작은 혼합 가중치·극소 양수 입력을 포함하며 최대 상대 오차는 약 3.33×10⁻¹⁶이었다. 경계 목적 초과·proximal 도함수·보상 KKT·underflow 인증의 별도 회귀 검사도 통과했다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content` 통과.
- Node 20 격리 후보의 `npm run build` 통과: 경로 1,893개 생성. `npm run check:export`로 HTML 1,888개의 내부 링크·자산·앵커·사이트맵·canonical 검사 통과.
- `npm run check:convex`: 한영 320/768/1440px, 영어 다크 모드, 경계값·접근성 등 99개 상태와 동작 12개 묶음 통과. 안내 예제 30개·퀴즈 20개, 설정 저장·언어 전환·정수 초안/Escape 복구, 기본 ridge를 포함한 교육과정 링크 11개 및 이전 계산 19개의 회귀를 확인했다. 수집된 페이지 실행 오류와 Axe WCAG 위반은 0개다.
- 최종 정적 출력의 한영 SVG 네 개를 320px에서 캡처·검토했다. 소스 1,882개·학습 원고 1,300개·반영 파일 32개·설정 8개의 후보 byte 일치와 네 SVG 출력 byte 일치를 확인했다.
- commit/push와 공개 배포는 구분한다. 공개 홈페이지 반영 완료는 해당 GitHub Pages 작업 성공으로 별도 확인한다.
