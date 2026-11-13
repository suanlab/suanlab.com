# 유한 spin 응답·Majorana 공분산 자료 대조

2026-11-13. 기존 spin chain·fermion Fock의 한영 본문을 보존하고 연구 절을 추가한다. 일반 비대각 입력의 미계산 ground 분류를 새 eigensolver로 바꿔 표시하지 않는다.

## 유한 spin의 직접 유도

두 site·distinct edge 하나·Pauli 계수를 고정하고 Bell 기저에서 Q=XX의 두 섹터를 직접 대각화한다. h>0 바닥은 비축퇴이며 국소 Z 평균0이다. 전체 M의 susceptibility는 Φ−로의 단일 전이 행렬 원소와 분모 R−J로 계산한다. Site당 응답과 전체 응답의2배 인자·에너지 역수 단위를 명시한다.

h=0의2차원 바닥과 h>0에서의 접근을 구별한다. 중복도 포함 ordered first gap과 다음 distinct level까지의 gap은 축퇴점에서 다르다. 종방향 b에서 h=0의 cusp와 h>0의 고립 가지 rounding을 비교하는 것이 이 절의 avoided-crossing 맥락이다. h 방향의 서로 다른 Q 섹터를 같은 섹터의 준위 반발로 해석하지 않는다. 두-spin의 비가환 극한이나 큰 응답으로 무한계의 자발적 대칭 깨짐·임계 scaling을 증명하지 않는다.

선형 응답은 비축퇴 ground의1차 섭동으로 직접 유도하며, 4|b|<gap은 min–max에 의한 충분 고립 조건이다. 이 조건만으로3차 오차의 수치 계수가 알려진다고 주장하지 않는다. 그림의 작은 장 점근선은 정확 곡선과 다른 선이며 큰 장에서 그대로 외삽하지 않는다.

## Fermion 일차 자료와 부호

- [Kitaev, Unpaired Majorana fermions in quantum wires](https://arxiv.org/pdf/cond-mat/0010440), 식 (1)–(3)와 unpaired Majorana 논의: Hermitian CAR 조합과 실수 반대칭 계수의 quadratic Hamiltonian이 출발점이다. 본문은 두 모드의 가운데 결합·바깥 결합을 원래 c 생성 순서로 직접 전개한다. 유한2모드 축퇴를 긴 wire의 locality·공간 분리·지수적 보호 정리로 인용하지 않는다.
- [Bravyi, Lagrangian representation for fermionic linear optics](https://arxiv.org/pdf/quant-ph/0404180), §IV의 Gaussian 정의·식 (17)의 Wick 관계와 covariance 순수 조건, canonical transformations: 이점 자료로 높은 모멘트를 정하는 주장은 Gaussian 클래스 아래의 것이다. 본문은 Γ=i⟨[m,m]⟩/2·vacuum Γ01=−1을 자체 행렬 규약으로 검산한다. 다른 ladder/covariance 규약의 부호를 섞지 않는다.

Quadratic Heisenberg 방정식과 Γ의 orthogonal congruence는 임의 초기 밀도에 대해 성립한다. Gaussian성이 있어야 이 정보로 전체 상태를 재구성할 수 있다. 네 모드 pure even cat과 I16/16은 같은 Γ=0이지만 ⟨n0n1⟩·Var(N)가 다르다. 이는 한 쌍의 비교 상태이며 두 상태가 모두 비Gaussian이라는 뜻이 아니다. Density interaction의 시간 미분은 수 연산자×Majorana를 포함해 상수 계수 선형 closure를 벗어난다.

Odd zero 연산자는 반대 parity 섹터를 연결하는 수학적 연산자다. Parity superselection 아래 그 섹터를 자유롭게 중첩한 물리 pure qubit으로 표시하지 않는다. Even 바깥 결합이 splitting을 만들면서 parity를 보존하는 반례를 정확히 계산한다.

## 재현·검산·그림

`python3 scripts/learning/verify-quantum-many-body-research.py`는 CI의 NumPy·SciPy·SymPy 환경에서 직접 Pauli/Jordan–Wigner 행렬과 dense eigensolver·matrix exponential을 사용한다. 실제 고유벡터·부분추적·전이 합·유한 장 차분, quadratic thermal/ground 상태의 Wick 관계, 임의 parity-even 입력의 전체 unitary와 covariance 전개, CAR·짝 섭동·상호작용 교환자를 대조한다. 작은 부동소수점 잔차나 차분 비교는 방향성 구간 인증이 아니다.

본문의 Python 네 셀은 표준 라이브러리 `Fraction`으로 표를 정확히 재현한다. SymPy가 유리수 root와 두 모드 characteristic polynomial을 확인한다. `npx tsx --test scripts/learning-quantum-many-body-research.test.ts`로 한영 계산·표·원본문 보존·실제 renderer를 검사한다.

Matplotlib·Noto Sans CJK에서 `python3 scripts/learning/generate-quantum-many-body-research.py`로 네 SVG를 재생성한다. 두-spin 그림의 축은 h/J·gap/J·Jχ이며 fermion 그림은 같은 covariance의 고차 관측과 finite spectrum을 구별한다. 기존 blue/orange/green 팔레트를 유지한다. 정적 미리보기에서 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-many-body-research.cjs`로 실제 reader·별도 spin/fermion lab·펼침 안내·현재 입력의 에너지/분산을 확인한다. 새 Gaussian·시간 전개·일반 비대각 ground 제어를 추가한 것으로 설명하지 않는다.
