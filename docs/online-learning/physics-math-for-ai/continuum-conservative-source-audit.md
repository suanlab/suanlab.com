# 보존형 이산화 자료 검토

후속 상태: [두 단원의 홈페이지 통합 검증](continuum-conservative-authoring-status.md)을 마쳤다. 아래 내용은 자료·설계·기반 단계의 기록이며 현재 다음 순서는 연속체 전체 영역 종합 점검이다.

2026-11-06. 연속체 입문의 마지막 주제를 준비한다. 자료 검토 이후 [두 단원 계산·원고 기반](continuum-conservative-godunov-foundation.md)을 검증했다. 실습 화면과 등록은 아직 미완료다.

## 확인한 일차 자료

- [Clawpack Riemann Book의 유한체적 설명](https://www.clawpack.org/riemann_book/html/Approximate_solvers.html)에서 셀 평균, 공유 수치 플럭스, 망원경 합과 일관성, Godunov 인터페이스 값을 확인했다. 같은 플럭스를 인접 셀이 반대 부호로 사용한다는 구조와 충격파에서 보존형이 필요한 이유를 구분한다.
- [같은 저자의 선형 이류 장](https://www.clawpack.org/riemann_book/html/Advection.html)에서 경계 유출입의 적분 수지와 정확한 이동 해를 확인했다. 새 예제의 비균일 체적·공유 플럭스·이차 노름은 별도로 유도한다.
- [Burgers 장](https://www.clawpack.org/riemann_book/html/Burgers.html)에서 Rankine–Hugoniot 속도, 충격파/희박파와 entropy 선택을 확인했다. 보존성만으로 적합한 약해가 선택되는 것은 아니다. 수학적 entropy와 열역학적 엔트로피를 자동으로 동일시하지 않는다.

- [Tadmor의 Acta Numerica 2003 논문](https://math.umd.edu/~tadmor/pubs/files/Tadmor_Acta2003.pdf) 식3.8–3.11 및7절에서 수치 entropy 플럭스·잠재함수와 시간 이산 보정을 확인했다. 공간 반이산의 소산과 전진Euler의 양의 보정을 구분하고, 자체 Burgers 특수화는 별도 기호식으로 대조한다.

- [Burgers 근사 해법 장](https://www.clawpack.org/riemann_book/html/Burgers_approximate.html)에서 Roe의 평균 속도·보존 플럭스와 transonic 희박파의 실패, entropy fix의 필요성을 확인했다. 새 계산은 미보정 Roe를 명시적으로 비교한다.

## 기존 교재와의 구분

기존 `finite-volume-boundary-conservation`은 양의 속도·균일 셀·실제 upwind 갱신·경계 수지·CFL을 제공한다. `weak-entropy-riemann-solutions`는 약해·entropy·Riemann의 기초다. 새 예제는 이를 반복하는 대신 비균일 체적의 질량 가중치, 공유 면 플럭스와 이산 노름, 실제 비선형 Godunov 갱신과 entropy가 다른 보존형 후보를 다룬다.

참고 문헌의 일반 이론이 새 구현의 독립 인증은 아니다. 실제 체적·플럭스·갱신 배열과 별도 적분·유리수·고정밀 참조를 대조한 후 화면을 검증해야 한다.
