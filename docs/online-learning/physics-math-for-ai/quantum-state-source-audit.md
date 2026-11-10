# 양자 상태·Hilbert 공간 원문 확인

2026-11-08. 공식 강의 노트와 수학 참고문헌을 열어 확인했다. 아래는 개념 근거이며, 새 유한 실험의 수치 결과는 별도 독립 참조로 검산한다. 긴 원문이나 그림을 복제하지 않는다.

- [Preskill, Foundations I: States and Ensembles, §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf): 복소 내적·norm 완비성·ray와 정규화 대표, 전역/상대 위상 구별. 전체 양자 상태를 항상 순수 ray라고 부르지 않도록 혼합 상태의 범위도 확인했다.
- [TU Delft, Hilbert spaces and operators, §8.3–8.4](https://interactivetextbooks.tudelft.nl/mqp/hilbert-spaces-and-operators.html): bra의 켤레와 정규직교 기저 표기. 무한 차원 연산자 정의역은 이 유한 행렬 설명을 그대로 확장하지 않는다.
- [MIT 18.102, Complete Lecture Notes](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/8fb8d5c170f1613151aca71de21027bc_MIT18_102s21_full_lec.pdf): L²·완비성·정규직교 전개, 연속 함수의 조밀성과 다항식 근사의 연결. 이 수학 노트는 첫 인자를 선형으로 두지만 이번 교재는 Preskill의 물리 표기처럼 첫 인자에 켤레를 취한다. norm·완비성 논증은 이 표기 변환과 양립한다.
- [NIST DLMF §18.3](https://dlmf.nist.gov/18.3), [§18.5](https://dlmf.nist.gov/18.5), [§18.18](https://dlmf.nist.gov/18.18): Legendre 정규화·직교성과 명시적 전개를 확인하는 기준. u=(x+1)/2로 이동한 기저는 ∫₀¹P̃mP̃n=δmn/(2n+1)이다.

## 이번 유도와 검산

정수 복소 대표·선택 기저와 전역 phasor, 고정 비정규직교 기저의 Gram 비교는 이번 교육용 준비를 직접 정식화했다. 일반 n차원의 정의와 C² 실험을 구분한다.

연속 가족은 복소 구간별 affine 함수의 norm·확률·모멘트와 shifted Legendre 사영을 직접 적분한다. 생산 코드는 recurrence, 독립 참조는 이항 계수 전개를 사용한다. 계수·bin 확률·직접 오차와 Pythagoras 잔차를 대조하며, 기저 절단 오차와 별도 재정규화 결과를 합치지 않는다.

일반적인 L² 함수의 무한 전개는 조밀성 논증으로 설명한다. 유한 계산 결과를 그 정리의 증명으로 대체하지 않는다. 점 값·Dirac의 일반화 고유벡터·자기수반 연산자의 정의역은 각각 별도의 문제로 다룬다.
