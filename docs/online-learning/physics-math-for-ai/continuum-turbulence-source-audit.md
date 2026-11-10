# 난류·폐쇄 문제 자료 검토

2026-11-06 후속 상태: [홈페이지 통합 검증](continuum-turbulence-authoring-status.md)을 마쳤다. 아래는 해당 단계 당시의 설계·검증 기록이다.

2026-11-06. 다중물리 결합 다음 주제의 자료다. 이후 [두 계산 가족·한영 원고](continuum-turbulence-foundation.md)를 검증했다. [두 실습의 화면·홈페이지 통합](continuum-turbulence-interaction-plan.md)은 미완료다.

## 확인한 일차 자료

- [INL/MOOSE의 RANS 이론](https://mastodon.inl.gov/modules/navier_stokes/rans_theory.html)에서 Reynolds 분해·평균 운동량식·Reynolds 응력과 Boussinesq 가정을 검토했다. 평균 속도에 공분산 응력을 연결하려면 추가 모델이 필요하다. 여기의 자체 예제는 상수 밀도이며,자료의 소프트웨어 예제를 보편적으로 검증된 예측기로 해석하지 않는다.
- [NASA CFPOST의 응력 정의](https://www.grc.nasa.gov/www/winddocs/cfpost/appc.html)의 난류 응력 항·밀도 인자·eddy viscosity 관례를 확인했다. 압축성/Favre 표기를 상수 밀도 Reynolds 평균과 섞지 않으며,공분산 R과 운동량식의−ρR 부호를 구분한다.
- [University of Washington ME544 LES 강의 노트](https://courses.washington.edu/mengr544/handouts/hd-les.pdf)의5–8쪽에서 정규화 필터·미분 교환 조건·Gaussian/투영 필터·전달함수를 검토했다. Gaussian 필터는 일반적으로 멱등이 아니므로 필터한 잔차의 평균을0으로 놓지 않는다. 원문의 전달함수 exp(−k²Δ²/24)는 표준편차ℓ를 쓰는 exp(−k²ℓ²/2)와ℓ=Δ/√12로 대응한다.24쪽의 잔여 응력과 Galilean 불변성도 확인했다.

## 자체 특수화와 제한

Reynolds 단원은 명시한 유한 위상 앙상블의 발산0장으로 평균·공분산·강제 항을 직접 구한다. 정상 난류의 실제 시간 통계나 외력 없는 채널DNS라고 부르지 않는다. 같은 평균에 다른 공분산이 가능하다는 주장과 같은 전체 경계값/강제 문제의 해가 여러 개라는 주장은 다르다.

필터 단원은 주기 Gaussian 필터와 강제 Burgers 두 모드의 정확한 잔여 응력·에너지 교환을 유도한다. NS의 응력 발산과 달리 Burgers의u²/2플럭스에는1/2인자가 필요하다. 양의 필터 커널의 분산 비음성과 부호 있는 에너지 전달을 구분한다. 두 모드로 관성 구간이나 실제3차원 난류를 재현했다는 주장을 하지 않는다.

[두 단원 집필 설계](continuum-turbulence-authoring-plan.md) · [자체 유도](continuum-turbulence-derivation-notes.md)
