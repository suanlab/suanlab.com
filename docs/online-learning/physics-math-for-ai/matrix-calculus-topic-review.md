# 행렬 미적분 원주제 대조와 미분 가정 심화

2026-11-11에 행렬 미적분의 10개 원주제를 한국어·영어 원고, 실습 계산 식과 대조했다. 이전의 세 연구 예제 보강에 이어 전미분과 방향미분의 구분, 거리 기준에 따른 최급강하를 확장했다. 원주제의 내용 대조이며 일반 정리의 모든 증명과 연구 재현까지 완료했다는 뜻은 아니다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/`의 각 ID에 해당하는 `.ko.md`와 `.en.md`다.

| 원주제 | 원고 ID | 확인한 가정·논증·계산 |
|---|---|---|
| 전미분·선형화 | jacobians, chain-rule | 모든 작은 이동의 나머지 극한, 다항식 나머지 상한, 고정 방향과 휘어진 접근의 차이 |
| 편미분·방향미분 | gradients, jacobians | 고정하는 변수, D_vL=gᵀv의 미분 가능성 조건, 연속이고 모든 방향미분이 있지만 전미분은 없는 반례 |
| gradient·Jacobian·Hessian | gradients, jacobians, higher-order-autodiff | 벡터·선형 사상·이차 미분의 형상, 계량에 따른 gradient 표현, 대칭과 양의 정부호의 구분 |
| chain rule | chain-rule, backpropagation | 작동점의 국소 미분 합성, 유한 변화의 나머지, 공유 경로 합산, 다섯 매개변수의 동시 갱신 |
| 암묵함수·역함수 정리 | implicit-inverse-functions | C¹·가역 Jacobian의 국소 결론, x³의 역과 도함수, 고정점 수반 식과 반복 수렴의 별도 조건 |
| 행렬·trace·logdet 미분 | matrix-trace-logdet-derivatives | Frobenius 규약, 공유 성분 b의 계수, SPD 차분 정의역, LLᵀ의 자유 삼각 매개변수 미분 |
| 벡터장·발산·회전 | vector-fields-divergence-curl | trace와 curl 성분, 전단의 회전, 화살표와 궤적, 퍼텐셜의 영역 조건 |
| 선·면·체적 적분 | line-surface-volume-integrals | ds와 dr의 방향, 접선 일과 법선 유출, 곡면 외적과 체적 Jacobian, R=0의 퇴화 극한 |
| Gauss·Stokes 정리 | gauss-stokes-theorems | 경계·법선 방향, 구·적도 원판 계산, 독립 경계 수치 적분, 내부 특이점 반례 |
| JVP·VJP·고차 미분 | jacobians, higher-order-autodiff | 출력×입력 Jacobian, 유클리드 쌍대 항등식, 상수 방향 HVP, Rademacher trace 기대값과 분산 |

`learning-calculus.ts`의 연쇄·Jacobian·뉴런 계산, `learning-simulations.ts`의 양의 이차 손실과 갱신, `learning-analysis.ts`의 여섯 행렬 미적분 모형도 원고와 대조했다. ReLU 원점의 선택한 계산 규약과 고전적 도함수를 구분하고, 경계 수치 적분을 같은 해석 값의 복사로 취급하지 않는지 확인했다.

## 전미분의 양화사와 반례

실습 F=(xy,x²+y)의 나머지 (dₓdᵧ,dₓ²)는 노름이 √2‖d‖² 이하이므로 모든 작은 방향에서 정규화한 오차가 0으로 간다. 별도 함수 f=x³y/(x⁴+y²), f(0,0)=0은 |f|≤|x|/2로 원점 연속이다. 각 고정 방향의 미분은 0이지만 (t,t²)에서는 f=t/2이어서 전미분 나머지의 정규화 극한이 1/2이다. 원점 밖 x축의 y 편미분 1/x도 연속 편미분 충분조건이 실패함을 보여준다.

고정 직선과 곡선의 세 분수 비교표, 음수 t·좌표 방향을 포함한 Python 검산을 한영으로 추가했다. 유한 열거와 모든 방향에 대한 극한 논증의 역할을 구분했다. 기존 대화형 실습의 함수를 새 반례로 바꾸지는 않았다.

## 거리 기준과 최급강하

Cauchy–Schwarz로 유클리드 최급강하를 증명한 뒤, 양의 정부호 M의 내적에서 gradient가 M⁻¹g임을 유도했다. 같은 M-길이 예산 아래 음의 유클리드 gradient의 정규화와 실제 최급강하를 비교했다. g=(4,2), M=diag(4,1)의 두 기울기는 −10/√17과 −√8이다. 세 대각 계량에서 단위 길이·최소 기울기·원래 손실의 중앙차분을 독립적으로 계산했다. 국소 방향과 유한 보폭의 전역 수렴은 별도 조건이다.

[Boyd·Vandenberghe 저자 공개 PDF](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)의 §9.4.1, 인쇄 476쪽에서 이차 노름의 최급강하를 확인했다. 비교표·실행 예제·반례는 교재에서 직접 구성했다. 이전의 고정점·trace 추정·Cholesky 예제와 원전 확인 범위는 [별도 기록](matrix-calculus-research-examples.md)을 따른다.

## 실행 검증

계산·입력 경계 검사 54개와 원고 보존 검사 7개, 총 61개가 통과했다. 행렬 미적분의 10개 등록 단원을 선택한 원고 공통 검사도 통과했으며 다른 661개 검사는 이 선택 실행에 포함하지 않았다. 두 언어의 수식·MathML·목차와 영어의 한글 혼입을 확인했다. 새 Python 예제 네 개도 실행했고, 표시 수식은 문장 끝 구두점을 제외하고 한영이 일치했다. 실행 근거는 `.runtime/learning/matrix-calculus-python.json`이다.

`SUANLAB_TEST_BASE=http://127.0.0.1:43299 node scripts/check-matrix-calculus-core.cjs`로 핵심 네 단원의 영어·한국어, 320/1440px, 영어 밝은 테마·한국어 어두운 테마 조합 16개를 검사했다. 전체 개념 단계, 안내 예제 12개, 해설 문제 8개, 입력·완료 저장 복원, 새 표·수식·코드와 넘치는 코드 블록의 키보드 스크롤을 확인했다. 실행 오류와 WCAG A/AA 위반은 0건이었다. 근거는 `.runtime/learning/matrix-calculus-core-browser.json`이다. 나머지 여섯 단원은 `1d0ee37`에서 `check:analysis`의 118개 상태 검사에 포함되었고 이번에는 해당 원고와 계산 모형을 변경하지 않았다.

정적 빌드와 TypeScript 검사가 성공했고, 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical 검사를 통과했다. 검토 원고 해시 목록에는 이번 네 한영 파일을 명시적으로 추가했다. 현재 24개 검토 원고에 대해 추가 문구·미등록 기준 해시를 거부하는 보존 검사를 유지한다. 역사적 fixture는 변경하지 않았다. 공개 반영은 GitHub Pages 배포 결과로 확인한다.

## 이어갈 범위

초기 15영역·147개 중 원주제 대조는 이번 10개를 더해 59개다. 다른 9영역의 88개가 남았으며 다음은 복소해석의 9개 원주제다. 전체 51영역의 연구 심화는 계속 진행한다. 일반 역함수·암묵함수 정리와 Gauss·Stokes의 완전한 일반 증명, 추가 고급 문제와 신경망 연구의 전체 실험 재현은 이번 대조의 완료 범위에 포함하지 않는다.
