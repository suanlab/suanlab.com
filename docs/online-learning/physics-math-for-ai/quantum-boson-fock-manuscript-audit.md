# Boson cutoff·coherent 원고의 가정과 검산

2026-11-09. 대상은 `boson-fock-cutoff-boundary-algebra`의 등록 전 한영 상세 원고다. 실제 content loader·Worker·reader 화면의 검증으로 세지 않는다.

## 설명과 계산의 범위

직교정규 모드, 정규화된 Fock 기저와 오른쪽 연산자 우선 convention을 고정했다. 무한 연산자의 곱은 유한 Fock 합 또는 필요한 모멘트가 존재하는 상태에 적용한다. 수치 모형의 한두 모드·최대 점유0…3은 예제의 범위이며 교육과정의 범위가 아니다.

Ladder 계수√n을 노름으로 유도하고 유한 교환자 `I−(C+1)P_top`의 trace가0임을 설명한다. Robertson–Schrödinger bound에는 실제 유한 교환자를 넣는다. 원래 X/P의 제곱을 투영한 것과 투영한 X/P를 제곱한 것은 다르며, 차이는 각각 `(C+1)P_top/2`다. 따라서 유한 분산을 원래 quadrature의 분산으로 해석하지 않는다.

무한 coherent 상태, 비정규화 투영과 재정규화 상태를 구분한다. Retention S는 normalized fidelity의 **제곱**이며 비정규화 오차 제곱은1−S, 정규화 거리 제곱은2(1−√S)다. Poisson tail은 lower incomplete gamma로 계산하며, tiny μ에서1−retention의 반올림만 사용하지 않는다. 유한 소멸자의 nilpotency와 nonzero α의 고유상태 잔차도 설명한다.

## 독립 기준과 대표 예제

제품 모듈을 가져오지 않는 `scripts/learning/generate-boson-fock-authoring-references.py`가 SymPy의 유한/확장 ladder 행렬·H²·quadrature 곱과 mpmath250자리 incomplete gamma를 계산한다. 기준은 `scripts/fixtures/boson-fock-authoring-reference.json`이다.

- C=0…3의 전체 ladder·교환자·진공/상단 관측, 경계 superposition의 음의 교환자 평균을 대조한다.
- `|2,2〉` hopping은 cutoff(3,3)에서 분산12, (3,2)에서6이다. 평균0만으로 경계 효과를 판단하지 않는다.
- α=1,C=3에서 Z=8/3, p_top=1/16, 평균 number15/16, number 분산207/256, 고유상태 잔차 제곱1/16이다.
- 유한 X/P 분산39/128·9/16과 embedded 상태의 원래 X/P 분산55/128·11/16을 별도로 대조한다.
- μ=0,1,16,10⁻¹²의 여섯 cutoff 준비와 두 모드의 곱 retention을 인증 구간과 대조한다. 곱의 차는 초월함수의 구간 차이이므로 좁은 구간이0을 포함하는지 확인한다.

원래 연산자의 embedded 관측, 재조합 oscillator Hamiltonian, finite displacement의 cos/sin 비교와 bounded-observable error는 **해석적 비교**다. 계산기의 새 입력 모드로 주장하지 않는다. 유한 최소를 무한 `H=−n`의 안정성 증명으로 확대하지 않는다. 작은 tail이 unbounded number의 오차를 일률적으로 제한하지 않는 반례도 포함했다.

## 출처와 검증 한계

[FAU Genes/Reitz 연습 자료](https://mpl.mpg.de/fileadmin/user_upload/Exercise_1.pdf)의 oscillator·quadrature·coherent 정의, [Stuttgart 연습 자료](https://www.itp3.uni-stuttgart.de/downloads/Adv._Quantum_Theory_Engl_ws11.12/ex9.pdf)의 coherent/displacement 문제와 [DLMF §8.4](https://dlmf.nist.gov/8.4)의 incomplete-gamma 항등식을 기준으로 삼았다. 예제 해설은 별도로 유도했으며 외부 정답지를 검증했다고 주장하지 않는다.

Strict KaTeX/GFM, 한영 수식·문제 번호·참고 링크 일치, 표의 ket 구분자, 영어 한글 부재와 분량을 검사했다. 구조 검사는 학습 효과의 증거가 아니다. 새11검사·기존161회귀·타입·lint와 기준 JSON의 재생성 바이트 일치를 확인한다. 실제 브라우저·Worker·세 안내·저장·Three.js·접근 가능한 실험 표·등록은 후속이다.
