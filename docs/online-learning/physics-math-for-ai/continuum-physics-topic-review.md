# 연속체·유체·탄성·수송 11원주제 연구 대조 순서

2026-11-13. 전자기학의 [20단원 연구 종합 대조](electromagnetism-research-audit.md)에 이어 연속체의 **11원주제·22단원·44한영 원고**를 기존 순서대로 검토한다. 기존 [입문 종합 감사](continuum-domain-final-audit.md)는 출발 근거이며 연구 심화 완료와 구분한다.

[보존식](continuum-conservation-research-status.md)·[연속체 가정](continuum-assumptions-research-status.md)·[응력/변형](continuum-stress-research-status.md)·[유동](continuum-flow-research-status.md)·[확산](continuum-diffusion-research-status.md)·[반응수송](continuum-reaction-research-status.md)·[구성방정식](continuum-constitutive-research-status.md)까지 각 배치의 관련 검사·새 빌드·16화면을 검증했다. 현재 순차 보강은 **14/22단원**이며 전체 **42영역·415주제 대조/9영역·94주제 후속**을 유지한다.

## 원주제와 실제 등록 단원

각 행은 `src/data/learning-curriculum.ts`의 원주제와 두 단원이다. 등록 순서·실습·기존 원본문을 보존하고 아래 근거와 실제 원고를 먼저 읽는다. 기록이 이미 존재하는 주제를 미집필로 다시 세지 않는다.

| 원주제 | 기존 단원 ID | 기존 통합 근거 |
|---|---|---|
| 질량·운동량·에너지 보존 | `moving-control-volume-mass-transport`, `compressible-flow-momentum-energy-balance` | [보존식](continuum-conservation-authoring-status.md) |
| continuum 가정 | `particle-kernel-continuum-fields`, `continuum-scale-separation-counting` | [연속체 가정](continuum-assumptions-authoring-status.md) |
| 응력·변형 | `cauchy-traction-stress-balance`, `finite-deformation-small-strain-rotation` | [응력·변형](continuum-stress-strain-authoring-status.md) |
| Euler·Navier–Stokes | `linearized-euler-acoustic-characteristics`, `taylor-green-navier-stokes-viscous-decay` | [유동](continuum-euler-navier-stokes-authoring-status.md) |
| 열·물질 확산 | `fourier-heat-diffusion-mode-budget`, `gaussian-fick-diffusion-spreading` | [확산](continuum-diffusion-authoring-status.md) |
| 반응수송 | `reversible-reaction-advection-diffusion-budget`, `fisher-kpp-reaction-diffusion-front` | [반응수송](continuum-reaction-transport-authoring-status.md) |
| 구성방정식 | `kelvin-voigt-maxwell-memory-dissipation`, `neo-hookean-uniaxial-constitutive-objectivity` | [구성식](continuum-constitutive-authoring-status.md) |
| 무차원 수 | `advection-diffusion-peclet-boundary-layer`, `reaction-diffusion-damkohler-penetration` | [무차원 수](continuum-dimensionless-authoring-status.md) |
| 다중물리 결합 | `thermoelastic-mode-reciprocal-energy-exchange`, `thermoelectric-bar-seebeck-peltier-joule-balance` | [다중물리](continuum-multiphysics-authoring-status.md) |
| 난류·폐쇄 문제 | `reynolds-ensemble-stress-closure-defect`, `gaussian-filtered-burgers-subgrid-transfer` | [난류·폐쇄](continuum-turbulence-authoring-status.md) |
| 보존형 이산화 | `nonuniform-volume-shared-flux-mass-norm`, `godunov-burgers-conservative-shock-entropy` | [이산화](continuum-conservative-authoring-status.md) |

## 첫 보강: 미분이 없는 위치와 보존만으로 부족한 판단

첫 단원은 움직이는 불연속 밀도의 저장 질량을 양의 부분으로 적분하고, 경계 교차를 구분한 상대 유입과 비교한다. 점프 속도만 잘못된 장이 점별 고전 잔차를 통과하면서 계면 원천을 남기는 음성 대조를 다룬다. 순간 기울기 부재와 유한 시간 수지는 구분한다.

둘째 단원은 별도 이상 기체 점프에서 세 Rankine–Hugoniot 조건과 엔트로피 부호를 유도한다. 같은 상태의 좌우 반전은 세 보존을 통과하지만 엔트로피 조건을 실패한다. Galilean 변환에서는 속도·운동량·총에너지·점프 궤적을 함께 바꾼다. 기존 매끄러운 affine 실습을 충격파 해법으로 소개하지 않는다. 상세 범위는 [보강 설계](continuum-conservation-research-plan.md)를 따른다.

응력·변형 두 단원은 기존 본문을 보존하고 방향 관측의 식별성과 공간 변형장의 적합성을 검증했다. Euler/Navier–Stokes의 유입 경계와 기준 해·오차 에너지까지 검증했다. 열·물질 확산의 엔트로피 생성과 Gaussian 분포 비교도 검증했다. 반응수송의 공간 조성비와 Fisher–KPP 전파 상·하계도 검증했다. 구성방정식의 주파수·하중 경로 식별성도 검증했다. 다음은 무차원 수의 격자 Peclet 해석과 Robin 공급 저항이다.

## 순차 검증과 완료 기준

- 각 원주제의 가정·단위·논증·직관·손계산·독립 실행 예제·시각화·해설을 한영 원고와 실제 실습에 대조한다. 수식 문자열 일치만으로 번역 의미를 검증했다고 하지 않는다.
- 기존 원본문과 여섯 연습 해설을 보존하고 새 연구 질문·해설을 추가한다. 연구 절 제거로 원본문을 복원하며 다른 영역의 연구 입력·실습 등록을 유지한다.
- 각 배치의 소스·계산·원본 그림을 검토하고 같은 최종 빌드에서 한영·반응형·명암·접근성·개념·안내·퀴즈·저장/완료를 확인한다. 선언한 보충 모델을 기존 자유 실험의 새 제어로 표시하지 않는다.
- 22단원 순차 보강 후 영역 전체 종합 대조를 별도로 수행한다. 개별 배치 통과를 11원주제 전체 감사로 대신하지 않는다. 그 전에는 영역 완료 집계를 올리지 않는다.

연속체 이후에는 열역학·평형 통계역학부터 [남은 9영역의 순서](domain-topic-review-index.md)를 유지한다. 입문 362/362와 요청 당시 48개 완료, 전체 51영역·509원주제 및 연구 심화의 범위를 함께 보존한다.
