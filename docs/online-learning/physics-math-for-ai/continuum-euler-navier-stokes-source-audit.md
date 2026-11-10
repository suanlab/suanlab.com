# Euler·Navier–Stokes: 자료 검토와 다음 계산 준비

이후 [계산 모델·상세 한영 원고 검증](continuum-euler-navier-stokes-foundation.md)을 마쳤다. 홈페이지 통합은 미완료이며 아래 준비 단계 설명은 당시 기록이다.

2026-11-06. 응력·변형의 홈페이지 검증과 병행해 다음 주제를 준비했다. 아래는 자료 검토와 실습 설계이며 모델·한영 원고·홈페이지 등록·주제 완료를 뜻하지 않는다.

## 확인한 1차 자료

- NASA Glenn Research Center, [Euler Equations](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/euler-equations-2/). 질량·운동량의 결합, 비점성 가정과 점성·경계층에서의 한계, 상태 관계와 계산적 접근의 필요성을 확인했다. 페이지의 정상 이차원 예시를 모든 비정상 압축성 유동의 완전한 방정식이라고 부르지 않는다.
- NVIDIA Warp, [Taylor–Green example](https://github.com/NVIDIA/warp/blob/main/warp/examples/fem/example_taylor_green.py), `exact_velocity`, `exact_pressure`와 약형 관련 주석을2026-11-06에 확인했다. 단위 정사각형의 정확한 속도·압력 감쇠와 수치 솔버를 대조하는 검증 구성을 참고한다. 제품의 시간 적분기·FEM 코드를 가져오거나 그 결과를 이 교재의 검증으로 바꾸지 않는다. main 브랜치의 가변 자료이므로 실제 모델 작성 시 식과 버전을 다시 대조한다.

## 두 실습의 범위

`linearized-euler-acoustic-characteristics`는 균일 배경 유속 U, 밀도 rho0, 음속 c에서 선형화한 Euler 섭동을 다룬다. 질량·속도·압력 섭동을 함께 이동시키고 U−c,U,U+c의 세 정보를 분리한다. 접촉 모드의 p′−c²rho′는 압력 단위의 entropy-like 변수이며 열역학적 엔트로피 자체라고 표시하지 않는다. 선택한 선형계의 정확한 해를 임의 진폭의 비선형 Euler 해라고 부르지 않는다.

`taylor-green-navier-stokes-viscous-decay`는 일정 밀도의 비압축성·주기적 이차원 유동에서 비선형 대류, 압력, 점성 감쇠를 함께 비교한다. 선택할 위상에서 u=−U cos(kx)sin(ky) exp(−2nu k²t), v=U sin(kx)cos(ky) exp(−2nu k²t), 압력 변화는 −rho U²(cos(2kx)+cos(2ky)) exp(−4nu k²t)/4다. 이 식은 구현 전 직접 대입과 독립 미분·적분으로 재검증한다. 3차원 난류·일반 초기경계값·충격파 솔버를 주장하지 않는다.

## 검증 준비

### 모델 작성 시 원본 재대조

2026-11-06에 NVIDIA의 실제 파일을 다시 읽고 변경 이력을 고정했다. [고정 커밋의 원본](https://github.com/NVIDIA/warp/blob/fba9f925eb2942d6242c1fd5c7442b5688246160/warp/examples/fem/example_taylor_green.py)은 속도 지수−8pi²nu t와 압력 지수−16pi²nu t, 두 성분과 압력의 부호가 본문의 k=2pi·U=rho=1 경우와 일치한다. 파일 SHA-256은 `912c5d64f9d9da438c6167294e879d3a7741884aac57ace813ae5b4ccdd40ad0`이다. 원본과 조회 기록은 `.runtime/learning/continuum-euler-navier-stokes-foundation/`에 보관했다. 해당 예제의 수치 경계 조건과 이 교재의 주기 셀을 동일시하지 않는다.

실제 섭동 장의 차분과 유한 창 수지를 비교하고 선형계 잔차와 같은 장의 비선형 Euler 잔차를 분리한다. Taylor–Green은 실제 속도·압력의 공간/시간 차분, 입자 ODE, 한 주기 셀의 질량·운동 에너지·점성 소산 적분을 별도로 계산한다. 압력 기울기를 제거한 대조는 같은 실제 장에서 수행한다. 작은 비영 점성·변화에는 고정밀 참조를 준비하며 공유 해석 잔차를 일반 오차 인증으로 표시하지 않는다.

[집필 설계](continuum-euler-navier-stokes-authoring-plan.md) · [남은 전체 순서](remaining-next-steps.md)
