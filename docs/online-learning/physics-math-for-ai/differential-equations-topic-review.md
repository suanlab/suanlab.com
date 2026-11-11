# 상미분·편미분방정식 원주제 대조

2026-11-12 기준. `differential-equations`의 10개 원주제를 한영 원고·실습·실행 예제에 대조한다. 기존 seed 단원 `differential-equations`는 별도 보존하므로 영역 연결은 11개다. 입문 48개와 후속 입문 362/362 완료, 전체 51영역·509개 원주제의 연구 심화 상태를 구분한다.

## 원주제와 보강 내용

원고는 `content/learning/physics-math-for-ai/{ID}.{en,ko}.md`, 계산은 `src/lib/learning-differential.ts`다.

| 원주제 | 단원 ID | 보강한 논증·독립 실행 |
|---|---|---|
| 초기·경계·상태 | initial-boundary-state-data | 에너지 유일성·Green kernel·경계 오차와 잔차의 supremum 인증 |
| 존재·유일성 | existence-uniqueness-waiting | Picard 수축·Gronwall 잔차 상계·국소 Lipschitz와 유한 시간 탈출 |
| 선형·비선형 ODE | linear-nonlinear-ode-flows | 시간 합성과 초기값 중첩의 차이·affine 보정·로지스틱 차이의 인수분해 |
| 위상·평형 | phase-portraits-fixed-points | 안정 비정규 행렬의 과도 증폭·정확한 Lyapunov 행렬 |
| 안정·분기·혼돈 | stability-bifurcation-chaos | 퍼텐셜 감소·임계 대수 감쇠·일차원 주기 궤도 부재·사상의 예외 궤도 |
| PDE 유형 | pde-principal-symbol-types | Congruence inertia·좌표별 coercivity 상수·유형과 조건성 구분 |
| 확산·파동·수송 | diffusion-wave-advection-modes | 주기 에너지 추정·Fourier 합·역방향 열연산의 L2 불연속 |
| 특성선 | characteristics-gradient-breakdown | 유계 tanh 데이터의 gradient 폭발·Riccati 식·역 label 민감도 |
| 약한·entropy 해 | weak-entropy-riemann-solutions | 분포 jump 계수·점성 진행파·팽창 shock 배제와 극한 정리의 범위 |
| 유한체적·경계 | finite-volume-boundary-conservation | L1 수축·이항 전이의 정확한 모멘트·수치 확산과 CFL |

한영 원고 각각에 비교표와 표준 Python 실행 블록을 추가했다. 비정규 반지름과 이항 확산의 한영 SVG 네 개는 별도 해석 예제를 그리며, live 실습에 없는 기능을 제공한다고 주장하지 않는다. 기존 본문·연습·개념·안내·퀴즈·초기값과 경계 관례를 보존한다.

## 계산 수정

로지스틱 analytic 중첩 차이는 expm1과 통분한 인수분해로 계산해 작은 비선형 효과를 유지한다. 세 흐름을 따로 뺀 expanded 진단도 남긴다. 위상평면의 고립성은 구조화된 두 행렬의 비영 계수로 판정해 determinant underflow의 잘못된 분류를 막는다.

PDE 유형은 저장된 binary64 계수를 2⁻¹⁰⁷⁴의 정수배로 표현한 정확한 determinant 부호로 판정한다. 극소 계수 및 두 곱이 같은 값으로 반올림되는 경우도 구별한다. 표시 determinant·고유값은 여전히 부동소수점 진단이며, 저장 수의 분류를 불확실한 물리 계수의 인증으로 해석하지 않는다. 기존 stats 순서와 조절기 범위는 유지한다.

## 출처와 보존

배경은 [MIT ODE](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/), [MIT PDE](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/), 저자의 [Clawpack Burgers 교재](https://www.clawpack.org/riemann_book/html/Burgers.html)와 대조했다. 추가한 특화 증명·유리수·Decimal 예제·그림은 독립 구성이다. 원 논문이나 범용 PDE 패키지의 성능 재현으로 주장하지 않는다.

기존 스무 원고의 연습 앞에 내용을 추가했으며 원본 prefix와 연습 이후 suffix를 보존한다. Immutable fixture 네 개는 수정하지 않고 이전 SHA·검토 SHA·이 문서를 reviewed ledger에 연결한다.

## 검증

관련 계산·원고 보존 검사 60개와 타입·lint·콘텐츠 검증이 통과했다. 정확한 유리수 기준으로 PDE 유형 463건을 대조했고, 650자리 Decimal 기준으로 로지스틱 차이 144건을 비교해 최대 상대 오차 4.0×10⁻¹⁶ 미만을 확인했다. 한영 Python 20개 블록을 직접 실행하고 코드·표시 수식의 언어별 일치를 검증했다.

1,893개 경로 정적 빌드와 1,888개 HTML의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. Picard 표기가 Markdown 상대 링크로 해석되는 문제를 한영 수식 표기로 고친 뒤 최종 빌드·검사를 다시 통과했다. 실제 export의 한영·320/768/1440px·명암 102개 상태에서 접근성 위반·브라우저 오류가 0이며, 상호작용 12묶음·안내 30개·퀴즈 20개·이전 계산 35개·영역 연결 11개를 확인했다. 정수 초안의 blur 오류·모형 보존·Escape 복원, 표·코드 키보드 스크롤과 320px의 최종 과학 그림 네 개도 확인했다.

소스 1,882개·학습 원고 1,300개·릴리스 파일 31개·설정 8개의 후보 byte 일치와 SVG 네 개의 export 일치를 확인했다. 전체 CI 및 공개 배포 완료는 이 로컬 검증과 구분한다.

## 후속 범위

일반 비자율·비매끄러운 ODE, 무한차원 진화 방정식, 가변계수·고차원 PDE의 경계 적정성, 일반 conservation system의 entropy 수렴·고차 유한체적과 적응 격자는 후속 심화 범위다. 단순화한 모형의 원주제 대조를 해당 분야의 모든 연구 이론 완료로 해석하지 않는다. 이번 검증으로 누적 28영역·273개, 남은 근거 통합은 23영역·236개이며 다음은 확률과정·확률미분방정식 11개다.
