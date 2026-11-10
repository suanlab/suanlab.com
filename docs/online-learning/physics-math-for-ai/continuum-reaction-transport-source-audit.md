# 반응수송: 다음 자료 검토

2026-11-06. 열·물질 확산의 화면 검증을 준비하며 다음 주제의 자료를 확인했다. 새 반응 모델·원고·등록·주제 완료를 뜻하지 않는다.

## 확인한 1차 자료와 범위

- Eymard, Hilhorst, Olech, [Numerical approximation of a reaction-diffusion system with fast reversible reaction, arXiv:0808.0251v1](https://arxiv.org/html/0808.0251v1), 초록과 서론의 결합 방정식·Neumann 경계 조건을 확인했다. 가역 반응과 확산의 결합, 빠른 반응 극한과 수치 근사의 구별을 참고한다. 이 논문의 유한 영역·다른 확산계수·일반 반응률을 아래 무한 직선의 일차 반응 예제와 동일시하거나 논문의 수렴 정리를 새 코드의 검증으로 사용하지 않는다.
- Ablowitz, Zeppetella, [Explicit solutions of Fisher's equation for a special wave speed](https://link.springer.com/article/10.1007/BF02462380), Bulletin of Mathematical Biology41,835–840(1979). 출판사 초록에서 무차원 특수 속도±5/√6과 논문의 범위를 확인했다. 전체 논문은 구독 범위여서 읽은 것으로 주장하지 않는다. ScienceDirect 주소는403으로 열리지 않았고 Springer 초록으로 확인했다. 계획할 logistic-square 프로필의 차원 복원과 PDE 대입은 별도로 직접 검증해야 한다.

## 두 집필 방향

첫 단원은 같은 물질의 두 표지 상태 A↔B가 같은 이류·확산을 겪는 일차 가역 반응을 다룬다. 각 상태는 생성·소멸하지만 합의 질량은 보존된다. 질량 농도를 사용할 때 두 상태의 같은 질량 기준을 명시하고 임의의 화학양론을 그냥 더하지 않는다.

둘째 단원은 Fisher–KPP의 비선형 성장·확산 전선을 다룬다. 성장 원천이 있는 finite-window 수지와 ‘원천 누락’ 대조를 구분한다. 선택한 특수 전선 속도를 모든 초기조건이 선택하는 보편 속도나 최소 속도라고 부르지 않는다. 왼쪽의 비영 plateau 때문에 무한 직선 전체의 질량은 유한하지 않으며 유한 창의 저장과 원천·경계를 비교한다.

[집필 설계](continuum-reaction-transport-authoring-plan.md)
