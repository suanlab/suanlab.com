# 다중물리 결합 자료 검토

2026-11-06 후속 상태: [홈페이지 통합 검증](continuum-multiphysics-authoring-status.md)을 마쳤다. 아래는 해당 단계의 설계·검증 당시 기록이다.

2026-11-06. 무차원 수 다음 주제의 자료 검토다. 이후 [계산 모델·학생용 한영 원고 검증](continuum-multiphysics-foundation.md)을 마쳤으며 실습 화면 연결은 미완료다.

## 열·탄성의 역방향 결합

[P. Kelly, Solid Mechanics Part IV §2.2](https://pkel015.connect.amazon.auckland.ac.nz/SolidMechanicsBooks/Part_IV/Chapter_2_Applications/PDF/02_Thermomech_Applications_02_Thermoelasticity.pdf)의 열전도·열역학 관계와 작은 변형 열탄성을 검토했다. PDF10–11쪽의 식2.2.60은 변형이 온도 변화에 기여하는 결합 항을,14쪽의 식2.2.80은 선형화한 열전도 관계를 제시한다. 온도를 받아 열팽창 응력만 계산하는 모델과 이 역방향 항까지 가진 모델을 구분한다. 이후 교재의1차원 계수·주기 모드는 별도 자유에너지와 실제 적분에서 직접 유도한다.

[MOOSE의 Basic Thermal/Mechanical Coupling](https://mooseframework.inl.gov/modules/combined/tutorials/introduction/thermomech_step01.html)은 열·역학 구성요소와 상호작용을 같은 모델에 연결하는 구현 예로 확인했다. 여러 변수를 동시에 푼다는 사실만으로 모든 물리적 역결합 항이 포함되었다고 판정하지 않는다. 우리의 상호 에너지 교환 검증은 선언한 방정식에서 별도로 수행한다.

## 열·전기 수송과 에너지 포트

[COMSOL의 Thermoelectric Effect 이론](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_theory.07.083.html)에서 Seebeck·Peltier·Thomson과 Joule 가열의 구분을 확인했다. 식 이미지도 흰 배경에서 직접 확인했다: P=ST,열 플럭스 q=−k∇T+PJ,전류 J=−σ(∇V+S∇T)다. 여기서 P는 Peltier 계수이고 전력 기호가 아니다. 새 원고에서는 Π와 전력 P를 구분한다.

상수 S의 일차원 막대에서 전류 연속·열 플럭스·전기 일의 국소 수지를 직접 유도할 예정이다. 단일 막대의 단자 조건과 닫힌 열전 회로 전체를 동일시하지 않는다. 절대온도·전위차 부호·접촉 기준·열저장/정상 조건을 선언하고,열전달을 전부 Joule 가열로 묶지 않는다.

## 남은 검증

현재 자료는 설계의 근거다. 두 특수 가족의 방정식·단위·초기/경계조건·독립 고정밀 행렬 지수/미분/물리 적분과 실제 브라우저를 검증해야 한다. 작은 변형의 이차 가용 에너지와 실제 전체 내부 에너지를 구분하며 범용 다중물리 해석기·실제 장치 성능의 완료를 주장하지 않는다.

[두 단원 설계](continuum-multiphysics-authoring-plan.md) · [자체 유도](continuum-multiphysics-derivation-notes.md)
