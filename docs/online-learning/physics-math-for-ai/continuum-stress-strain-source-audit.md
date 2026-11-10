# 응력·변형: 자료 검토와 계산 준비

이후 [계산 모델·상세 한영 원고 검증](continuum-stress-strain-foundation.md)을 마쳤다. 아래 준비 단계 설명은 해당 기록 당시 기준이며 홈페이지 통합은 다음 단계다.

2026-11-06. 연속체 가정 단원의 홈페이지 검증과 병행해 다음 입문 주제의 자료와 예제를 검토했다. 아래 두 단원은 집필·계산 준비 상태이며 실제 모델·상세 한영 원고·등록·주제 완료를 뜻하지 않는다.

## 확인한 1차 자료

1. David M. Parks, MIT 2.002, [Elements of Continuum Elasticity](https://www.ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/f7b561fbe8762ff73da9236e6c0fff73_lec7.pdf). 변위 기울기·소변형률, traction·응력과 평형, 구성 관계를 구분하는 구성을 확인했다.
2. MIT 2.080J, [Lecture 2: The Concept of Strain](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/71034688fba0ff26d25653a4bd236a4a_MIT2_080JF13_Lecture2.pdf), §§2.1–2.2. 변형률의 정의가 하나가 아니라는 점, 변위 기울기의 대칭·반대칭 분해와 미소 회전 예제를 대조했다. 이 자료의 변위 기울기 기호를 유한 변형 기울기와 혼동하지 않는다.
3. University of Alberta, [Cauchy Stress Tensor](https://engcourses-uofa.ca/stress/cauchy-stress-as-a-linear-map/). 면 법선에서 traction으로 가는 선형 사상, 힘/면적과 응력 성분, 통상적인 Cauchy 연속체의 각운동량 조건에 따른 대칭성을 확인했다.
4. University of Alberta, [Strain Measures](https://engcourses-uofa.ca/displacement-and-strain/strain-measures/). 소변형률의 선형화 범위, 방향별 길이·각도 변화와 Green 변형률의 유한 회전 처리 차이를 확인했다.

자료의 일반 관계를 아래 선택된 이차원 예제에 적용한다. 모든 재료의 구성 법칙, FEM 해, 변형과 응력의 실제 측정 일치를 인증하는 모형은 아니다.

## 먼저 고정할 기호와 단위

기준 좌표 X와 현재 좌표 x를 구분한다. x=chi(X), u=chi−X, H=grad_X u, F=I+H로 표기한다. 소변형률 epsilon=(H+H^T)/2와 Green–Lagrange E=(F^T F−I)/2를 분리한다. F를 H의 별칭으로 사용하지 않는다.

Cauchy 응력 sigma의 단위는 Pa=N/m²이며 현재 면의 단위 법선 n에 대한 traction은 t=sigma n이다. 힘은 실제 면적을 곱한다. 이차원 그림에서도 깊이 B와 면적·부피 단위를 명시한다. 인장을 양으로 두며 정수압 압축은 sigma=−pI다. 공학 전단 gamma와 tensor 전단 epsilon12의 배수 관계도 따로 설명한다.

## 계획한 두 단원

- `cauchy-traction-stress-balance`: 대칭 이차원 응력, 법선·전단 traction, 법선 반전, 회전한 성분, 실제 경계 힘과 모멘트, affine 응력의 발산 및 균형 체적력. 정적 평형과 개별 면의 비영 힘을 구분한다.
- `finite-deformation-small-strain-rotation`: 양의 주 신장과 전단에 유한 회전을 결합한 변형, 격자·선분·면적의 실제 변화, 정확한 길이와 소변형 근사, 유한 강체 회전의 음성 대조, 좌표·관측 회전과 객관성.

## 독립 검증 계획

실제 면의 traction을 별도로 구적해 순힘·모멘트를 구하고 체적 적분과 비교한다. 국소 발산식을 독립 기준에 복사하지 않는다. 변형된 두 점의 거리와 삼각형 면적으로 방향 신장·면적을 계산한다. 별도 고정밀 좌표 회전으로 변환 법칙과 작은 변화·반올림을 확인한다. 같은 식의 대입 잔차를 독립 인증으로 부르지 않는다.

[집필 설계](continuum-stress-strain-authoring-plan.md) · [남은 전체 순서](remaining-next-steps.md)
