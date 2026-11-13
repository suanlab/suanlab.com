# 열역학·평형 통계역학 자료 대조 준비

2026-11-06. 첫 주제의 [집필 계획](thermo-statmech-authoring-plan.md)과 [직접 유도](thermo-statmech-derivation-notes.md)를 위한 기록이다. 새 제품·원고·화면의 검증 완료나11주제 전체의 문헌 점검 완료를 뜻하지 않는다.

## 확인한 일차 자료와 관례

[David Tong의 Statistical Physics 강의 페이지](https://davidtong.org/teaching/statistical-physics/)에서 저자 제공 원문을 확인했다. 이전 Cambridge 주소는 현재 이 페이지로 이동한다. 첫 가족의 근거는 [4장 Classical Thermodynamics](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf)의 온도·영법칙·제1법칙·가역 과정·엔트로피 논의다. 열과 일의 전달 과정, 상태함수와 경로량, 가역 경로에 따른 엔트로피 정의를 대조했다.

원문의 계가 받는 일과 여기서 사용할 계가 외부에 하는 일은 부호가 반대다. 이 교재의 식은 W_by=−W_on, ΔU=Q−W_by로 통일한다. 준정적이라는 조건만으로 가역성을 결론 내리지 않는다. 두 유한 물체 사이의 온도차에 의한 열전달과 가역 이상기체 경로의 저장고 조건도 구별한다.

원문은 집중 열접촉의 실제 시간 상수나 이번 두 경로의 제품 구현을 제공하지 않는다. 일정 열용량·선형 열연결, 단원자 이상기체·일정 몰 정적 열용량과 선택한 경로는 여기서 명시한 특수화다. 그 식과 적분은 따로 유도·검산한다. 단순한 자기 수지 일치를 독립 정확도 인증으로 부르지 않는다.

## 상수·단위와 남은 자료 점검

SI 단위에서는 온도K, 에너지J, 열용량J/K, 몰 열용량J/(mol K), 압력Pa, 부피m³를 구분한다. R=N_Ak_B의 SI 정의와 실제 수치 표현은 [NIST 상수표](https://physics.nist.gov/cuu/Constants/Table/allascii.txt)를 기준으로 확인한다. 수학적 상수의 정의와 코드의 이진 근사값을 구별한다.

[1장 Fundamentals](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf)는 이후 앙상블·분배함수의 후보 자료다. 이번 준비에서 그 전체 내용이나 뒤의 기체·상전이 장을 모두 읽었다고 주장하지 않는다. 해당 주제 집필 때 측도·축퇴도·유한계·극한·상호작용 관례를 실제로 읽고 추가한다. 기존 두 수준계 단원을 보존하며, 물리적 열역학과 AI의 지수 점수 사이에는 상태 에너지·저장고·단위라는 조건이 필요함을 유지한다.


## 2026-11-13: 두 수준계의 온도 추정 보강

[MIT 앙상블 자료 문제 1](https://web.mit.edu/esfaki/www/8.08_files/canonical.pdf)의 정준 에너지 분산–열용량 항등식과 [Tong 1장 1.3.1–1.3.2절](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf)의 고정 스펙트럼 정준 모멘트 관례를 첫 연구 보강에 대조했다. 여기서는 kB=1이고 C는 SI 열용량/kB에 대응한다. 두 자료를 통계적 추정기의 성능 인증으로 인용하지 않는다.

독립·동일 온도 평형 준비의 오차 없는 준위 표지라는 관측 모형을 추가했다. 그 이항 우도에서 로그 온도 점수·정보를 직접 유도하고, 미지 간격의 공통 척도 영방향과 g=1 상대 민감도 최대점을 별도로 계산한다. 불편 추정량에 관한 정보 하한을 유한 표본 MLE의 정확도 보장으로 바꾸지 않는다. 상관·검출 잡음·상호작용은 이번 관측 모형 밖이다. 자세한 진행과 후속 범위는 [연구 심화 진행표](thermo-statmech-research-progress.md)를 따른다.
