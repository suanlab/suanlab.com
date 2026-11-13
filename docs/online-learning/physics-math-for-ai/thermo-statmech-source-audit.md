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


## 2026-11-13: 열접촉·이상기체 경로의 역문제

[Tong 4장](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf)의 4.1 계·열접촉, 4.2 상태 에너지와 경로 전달량, 4.3 가역성·엔트로피 구분을 이번 두 보충의 관례와 대조했다. 원문의 계에 하는 일과 이 교재의 계가 하는 일의 부호 차이를 유지한다. 집중 열접촉의 선형 동역학이나 새 경로 형상을 원문이 제공했다고 소개하지 않는다.

열접촉에서는 무입력 온도 기록의 공통 척도 불변성과 로그 변화율 Jacobian을 직접 유도했다. 절대 열입력을 추가하면 두 물체의 바깥 경계가 더 이상 단열이 아님을 명시하며, 열원 온도 조건을 포함한 엔트로피 수지를 따로 쓴다. 전기적 일을 열의 엔트로피 유입으로 대체하지 않는다.

이상기체에서는 로그 부피 위치의 두 다항 형상에 대해 실제 p dV와 가역 δQ/T를 적분했다. 총 Q와 W의 종속성·중간 탐침의 영 민감도·1/4 탐침의 비영 민감도는 이 선언한 가족의 유도다. 지정된 시간 일정의 연쇄 법칙을 물리적 열전달 법칙이나 유한 속도 가역 장치의 실현성으로 바꾸지 않는다. [검증 기록](heat-work-research-status.md)에서 계산·원문·그림·실제 reader 근거를 확인한다.


## 2026-11-13: 유한 열원·장치 복원·크기와 저온의 동시 극한

[Tong 4장](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf)의 Carnot·엔트로피·기준 온도 자유 에너지·제3법칙 관례와 [1장](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf)의 고정 스펙트럼 정준 모멘트를 대조했다. 새 유한 열원 경로, 비순환 끝점 자원 항과 독립 복사본의 N,T 동시 수열은 여기서 추가한 직접 유도이며 원문에 그 사례의 구현이나 검증이 있다고 주장하지 않는다.

고정 저장고와 유한 열용량 물체의 경계, 장치 순환 복귀와 복원에 드는 일, 고정 N에서 미분한 열용량과 N까지 바꾸는 수열 미분을 각각 구별한다. Poisson 개수 극한은 이항 확률의 유한 곱과 꼬리 상한으로 설명한다. 독립 복사본의 평형 확률 조건을 상호작용 물질의 상전이 또는 유한 시간 냉각 비용으로 확대하지 않는다. 범위와 실제 검사 근거는 [열역학 법칙 연구 보강](thermodynamic-laws-research-status.md)에 연결한다.


## 2026-11-13: 자유 에너지와 명시적 동역학

[Tong 1장의 정준 분포·자유 에너지](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf)와 [4장 4.2–4.4절의 일·엔트로피·환경 퍼텐셜](https://davidtong.org/pdfs/teaching/statistical-physics/statphys4.pdf) 관례를 이번 보충에 대조했다. 두 상태 준위 변경과 상세평형 전이율, 관성·점성 피스톤과 같은 온도조절기로 돌아가는 소산 열은 이 보충에 추가한 가정이다. 자료가 그 속도·수치 적분·실제 장치의 유한 열전달까지 인증한다고 소개하지 않는다.

급격한 에너지 변경 직후의 초과 일을 즉각적인 생성 엔트로피로 부르지 않고 이후 고정 목표 완화와 구분한다. 피스톤은 위치만이 아닌 운동 에너지까지 가용 에너지에 포함하며 기체 단독 열과 기체·피스톤의 온도조절기 순열을 구별한다. 결정론적 거시 근사를 미시 피스톤의 정준 요동 법칙으로 확대하지 않는다. [자유 에너지 연구 보강 기록](free-energy-research-status.md)에 직접 유도·검산·reader 검증을 연결한다.


## 미시·거시 연구 보강: 전이와 반복 관측

Levin–Peres의 저자 공개 [Markov Chains and Mixing Times](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf)의 1.3절(비주기성과 lazy 전이), 1.5절(정상 분포), 1.6절의 상세평형 명제를 대조했다. 정지 확률을 넣은 유한 껍질 교환은 여기서 직접 정의·열거한 예이며, 상태 수만으로 실제 동역학을 도출하지 않는다.

[MIT 6.441 정보량 정의와 성질](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf)을 조건부 정보·KL 자료처리·chain rule의 배경으로 사용한다. 세 상태 투영의 기억·독립/공유 센서 비교는 명시한 유한 결합 법칙에서 직접 계산한다. 정보량을 물리 열비용으로 해석하지 않으며, 기존 reader 계산기에 없는 새 동역학을 조작 가능한 기능으로 소개하지 않는다. [검증 범위](micro-macro-research-status.md)를 따른다.


## Ensemble 연구 보강: 관측·구동·매개변수 방향

[Tong의 통계물리 1장](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf) 1.2–1.3절의 고정 총량·정준 준비, 1.3.2절의 에너지 요동, 1.4.1절의 입자 교환 정규화·응답을 대조했다. 유한 껍질의 쌍 관측 역산과 오차 척도는 직접 세기로, 주기 구동은 새로 지정한 양의 전이율로, 응답 행렬의 랭크는 교정한 두 에너지의 네 상태 합으로 유도했다. 교재 배경의 평형 관계만으로 완화율·시계·장치 비용·미지 에너지 기준의 식별을 얻었다고 주장하지 않는다. 세부 유한 검산과 reader 범위는 [ensemble 검증 기록](ensembles-research-status.md)을 따른다.


## 분배함수 연구 보강: 추정·주변화·꼬리

[Tong 1장 1.3절](https://davidtong.org/pdfs/teaching/statistical-physics/statphys1.pdf)의 정준 합·에너지 요동, [5장 5.2절](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf)의 Ising Hamiltonian을 배경으로 대조했다. [Owen의 중요도 표집 9장](https://artowen.su.domains/mc/Ch-var-is.pdf) 도입과 9.1절의 출발 분포·우도비 정의 및 무한 분산 주의점을 확인했다. 유한 재가중 이차 모멘트·독립 묶음·로그 편향, 열린 세 스핀 합·조건부 에너지 분산·Fisher 손실, 기하 꼬리의 정확한 이차 모멘트 조건은 본문에서 지정한 모형으로 직접 계산한다. 미시 에너지를 온도 의존 축약 퍼텐셜과 바꾸거나 절단 적분을 원래 전체 분산으로 바꾸지 않는다. 범위와 재현은 [분배함수 검증 기록](partition-research-status.md)을 따른다.


## Boltzmann·Gibbs 연구 보강: 관측 설계와 표집 정확도

[Berkeley 지수족 자료](https://stat210a.berkeley.edu/fall-2024/reader/exponential-families.html)의 평균·공분산 미분 관계, [Stanford Gibbs 자료](https://ermongroup.github.io/cs323-notes/probabilistic/gibbs/)의 조건부 갱신·Metropolis 구성, [Wibisono 2018 예제 1–2](https://proceedings.mlr.press/v75/wibisono18a/wibisono18a.pdf)의 Gaussian 정확 전이·ULA 정상 편향을 대조했다. 두 교정 온도의 기준 오즈·에너지 역산과 관측 우도, 상태 의존 선택의 반례·구성별 선택 비 보정은 여기서 지정한 모형으로 직접 유도한다. Gaussian 표는 무차원 정밀도와 알고리즘 단계를 사용한다. 정규화·조건부 정확성·반복 안정성·목표 보존·물리 시간의 의미를 서로 대신하지 않는다. 상세 검산 범위는 [Boltzmann·Gibbs 검증 기록](boltzmann-research-status.md)을 따른다.


## 상전이·질서변수 연구 보강: 임계 요동·유한 가중치·계면

[Cerf·Gorny 서론 인쇄 2–3쪽](https://www.imo.universite-paris-saclay.fr/~raphael.cerf/papers/cwsoc.pdf)의 일반화 Curie–Weiss 임계 극한과 [Tong 5장, 특히 5.5절](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf)의 Landau·기울기 범함수·영역벽 배경을 대조했다. 전자의 뒤쪽 자기조직화 모형을 현재 이진 준비와 혼동하지 않는다. 이진 엔트로피의 꼬리 제어·Gamma 모멘트, 구속 계수 측도의 이산 Laplace 진폭·b=v/N 혼합, 6차 공존 0→m₀ 계면은 본문에서 직접 유도한다. Tong의 사차 벽 해를 6차 계면의 해로 인용하지 않는다. 구형 방울은 작은 양의 벌크 구동과 등방 3차원 얇은 계면을 새로 가정해 면적·부피를 계산한 근사다. 공존 계면 해·곡면 근사·동역학적 속도는 서로 대신하지 않는다. [검증 기록](phase-transition-research-status.md)의 유한 검산 범위와 전체 범위 미완료 표시를 유지한다.


## 상관·요동 연구 보강: 모수·공간 모드·추정량을 명시하기

[Berkeley 지수족 자료](https://stat210a.berkeley.edu/fall-2024/reader/exponential-families.html)의 평균·공분산 미분, [Tong 5.3.1절](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf)의 주기 Ising 전달 법칙, [Shalizi 6–6.1절](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf)의 역공분산 GLS를 대조했다. 네 상태 세 특징의 행렬식·로그 대비·방해 점수 사영, 유한 고리의 허용 Fourier 파수·합 규칙·홀짝 저온 극한, 정상 AR(1)의 정밀도·경계 가중치·분산 차이는 지정한 모형에서 직접 유도한다. 독립 평형 표본의 추론과 물리적 고정 K 응답, 공간 파수와 시간 주파수, 보통 평균과 알려진 공분산의 GLS를 구분한다. 극한 정규분포의 공분산을 유한 MLE 공분산으로 오인하지 않고, 특이 끝점은 별도 경로로 검산한다. [검증 기록](correlations-research-status.md)의 유한 범위와 전체 교육과정 미완료 표시를 유지한다.
