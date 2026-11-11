# 함수해석과 연산자 원주제 대조

2026-11-11 기준. `functional-analysis`의 9개 원주제를 한영 원고와 `learning-functional.ts`에 대조했다. 기존 설명·실습·연습을 보존하면서 완비성, 연산자 정의역, 연속 스펙트럼과 정칙화의 조건을 보강했다. 이 기록을 포함한 범위는 18영역·175개 원주제이며 전체 51영역·509개 일반 이론과 연구 심화의 완료를 뜻하지 않는다.

## 원주제별 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`이다.

| 원주제 | ID | 정의·가정·보강 근거 |
|---|---|---|
| 노름·Banach·Hilbert | normed-banach-hilbert | 좌표 극한으로 ℓ² 완비성 증명, c00 반례, 차원별 노름 비교 상수 |
| 유계·콤팩트 연산자 | bounded-compact-operators | 대각 연산자의 콤팩트성 필요충분조건, 닫힌 비유계 연산자·수반 정의역·graph 노름 |
| 쌍대와 함수 평가 | dual-functionals-evaluation | 실수 Hilbert Riesz 표현, 평균 센서 노름, 1차원 H¹ 대표의 점 평가 경계 |
| Hilbert 직교 사영 | orthogonal-hilbert-projections | 최소화 열·평행사변형으로 존재 증명, 직교·유일성, 비직교 Gram 계산 |
| 스펙트럼 정리 조건 | spectral-theorem-conditions | 곱셈 연산자의 resolvent·스펙트럼, 실제 고유벡터 없는 근사 고유벡터 |
| 분포와 시험 함수 | distributions-test-functions | H′=δ, |x|″=2δ의 부분적분 증명, 국소 적분 함수로 δ 표현 불가, 실제 C∞ 시험 함수 |
| 약도함수와 Sobolev | weak-derivatives-sobolev | 모든 시험 함수의 항등식, |x|ᵅ의 W¹ᵖ 임계값, 절대연속 대표·trace와 H¹₀ |
| 적분 커널 연산자 | integral-kernel-operators | Hilbert–Schmidt 경계·콤팩트성, 점 구적과 유계 cell 사영의 구분, 정확한 2×2 오차 |
| 유한·무한 차원 근사 | finite-infinite-approximations | Lax–Milgram·Céa 논증, 에너지/유클리드 사영 비교, 절단 역문제의 최악 잡음 |

## 독립 예제와 그림

Python 8종을 한영에서 같은 코드로 실행한다. 비직교 사영은 정확분수로 잘못된 계수의 오차 제곱 784/135를 비교한다. 분포 예제는 compact C∞ bump에 Simpson 적분을 적용하며, 유한 시험값을 모든 시험 함수에 대한 증명의 대용으로 부르지 않는다. Sobolev 예제는 α=3/4의 p=1·2·3·4 경계를 비교한다.

복소 L²[0,1]의 곱셈 연산자 M=x에서 λ=.5 근처의 정규화 구간 함수는 L² 노름 1을 유지하지만 잔차와 적분 질량은 줄어든다. 한영 그림은 이 함수를 단위 질량 Dirac 열과 구분한다.

매끄러운 입력의 중점 구적 오차 O(N⁻²)와 모든 L² 입력의 유계 cell 연산자 오차 O(N⁻¹)를 구분한다. 점 평가를 모든 L² 동치류의 유계 함수로 간주하지 않는다. 절단 역문제는 δ=.05에서 N=2·4·8·16·32를 비교한다. 한영 그림의 전체 오차는 각 N에 대해 다른 최악 잡음 δe_N을 쓰는 envelope이며, 한 번 고정한 데이터의 오차 곡선이 아니다. N=32는 코드·그림 예제로, 기존 실습 제어의 최대 N=16과 구별한다.

## 수치 수정

m=8,x=10⁻⁵의 기하급수 꼬리를 두 거의 같은 값의 차로 구하면 약 10⁻¹⁶의 가짜 오차가 나왔다. 닫힌 꼬리 식으로 약 9.77×10⁻⁴⁹를 계산한다. 중점 커널 구적도 두 계수 차 대신 정확한 bias −ab/(12N²)를 사용한다. a=b=10⁻⁸,N=16의 오차는 약 3.26×10⁻²⁰이다.

수학적으로 양수지만 binary64 범위 아래인 오차는 —와 수치 범위 안내로 표시한다. x=0 또는 a·b 중 하나가 정확히 0인 경우의 진짜 영 오차는 유지한다. 꼬리 오차 범위 때문에 실습의 다항식 값 점을 지우지 않는다.

## 출처와 후속 범위

- [MIT 18.102 강의 자료](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/pages/lecture-notes-and-readings/): Banach·Hilbert·유계 연산자의 배경. 완비성·대각 정의역·곱셈 스펙트럼 계산은 본문의 모델에 맞춰 직접 전개했다.
- [MIT Lecture 17: Minimizers and Riesz](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/resources/lecture-17-minimizers-orthogonal-complements-and-the-riesz-representation-theorem/): 사영·표현 정리 배경. 비직교 Gram 예제는 직접 적분했다.
- [MIT Lecture 20: Compact Operators and Spectrum](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/resources/lecture-20-compact-operators-and-the-spectrum-of-a-bounded-linear-operator-on-a-hilbert-space/), [Lecture 22: Compact Self-adjoint Spectral Theorem](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/resources/lecture-22-the-spectral-theorem-for-a-compact-self-adjoint-operator/): 콤팩트·스펙트럼 정리의 조건을 확인하는 자료.
- [MIT 18.155: Test Functions](https://ocw.mit.edu/courses/18-155-differential-analysis-fall-2004/resources/section6/): 분포의 시험 함수 배경. 점프·모서리의 미분과 실제 bump 계산은 별도로 전개했다.
- [Imperial finite element course §5](https://finite-element.github.io/L5_convergence.html): 약도함수·Sobolev·coercivity·Céa 배경. 본문은 2×2 Galerkin과 대각 역문제의 계산을 추가한다.

일반 Banach 쌍대 표현·약위상·반사성, 비유계 연산자의 일반 spectral measure·semigroup, 고차원 Sobolev embedding·trace, 비대칭/비coercive PDE 안정성과 일반 정칙화 parameter 선택은 후속 심화다. 이 모델들의 증명과 유한 검증을 해당 전체 분야의 완료로 확대하지 않는다.

## 검증 기록

- 함수해석 계산 검사 29개·기존 원고 보존 검사 7개·선택한 한영 reader 렌더링 검사 9개를 통과했다. 이 45개 검사를 전체 테스트 모음의 통과로 확대하지 않는다.
- Python 코드 16개를 최종 SHA·출력과 대조해 재실행했다. 9개 단원의 한영 절 수·표시 수식·실행 코드가 일치하며 영어 원고의 한글과 두 언어 원고의 일본어 가나가 없다. 320px에서 표 18개와 최종 내보내기 SVG 4개의 배치를 확인했다.
- 기하급수 꼬리·중점 구적의 10개 입력을 정확분수와 1,100자리 Decimal로 독립 대조했다. 실제 0·표현 가능한 극소 양수·양수의 범위 손실을 구분했다.
- 실제 정적 빌드의 경계 선행 검사는 16개 상태·9개 동작 그룹이 통과했다. `npm run check:functional`의 한영·320/768/1440px·다크 모드 87개 접근성·배치 상태, 11개 동작 그룹, 안내 예제 27개·퀴즈 18개를 통과했다. 표·코드의 키보드 스크롤, 정수 입력 draft·Escape 복구, 저장·언어 전환·초기화, 이전 계산 회귀도 확인했다.
- `npm run lint`, `npm run build`, `npm run typecheck`, `npm run validate:content`, `npm run check:export`가 통과했다. 빌드 1,893개 경로와 내보낸 1,888개 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 소스 1,882개·한영 원고 1,300개·설정 8개·릴리스 파일 29개를 빌드 후보와 바이트 대조했고, 내보낸 SVG 4개도 원본과 같다. 공개 반영 완료는 commit/push 이후 GitHub Pages 배포 결과로 별도 확인한다.
