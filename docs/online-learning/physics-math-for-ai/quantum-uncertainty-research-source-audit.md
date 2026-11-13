# 불확정성·공동 측정 연구 확장 자료 대조

2026-11-13. 기존 `finite-state-covariance-uncertainty`와 `qubit-povm-instrument-backaction`의 한영 본문을 보존하며 연구 절을 추가한다. 수식·수치 예제는 아래 가정에서 직접 유도한다.

## 일차 자료와 적용 범위

- [Busch·Heinosaari, Approximate joint measurements of qubit observables](https://arxiv.org/pdf/0706.1415), §4.1의 식 (39)·Proposition 4: unbiased binary qubit 효과의 공동 측정 조건은 두 Bloch 벡터 합·차의 길이 합이 2 이하인 것이다. 본문은 직교 Z/X 방향의 κ²+λ²≤1로 특수화하며, 일반 네 효과의 자유 변수와 삼각부등식으로 필요성을 별도로 증명한다. biased 효과에 대한 충분조건으로 일반화하지 않는다.
- [Ozawa, Universal Uncertainty Principle in the Measurement Operator Formalism](https://arxiv.org/pdf/quant-ph/0510083), §II·§III.C–F: 준비 상태의 표준편차, jointly unbiased 판독의 RMS noise, 채널의 disturbance를 구별한다. 본문의 두 보정 오차 제곱의 곱 ≥1은 특정 직교 qubit 공동 POVM의 모멘트로 직접 계산한 결과다. 이를 임의 장치의 보편적인 error–disturbance 관계로 인용하지 않는다.

## 직접 유도한 내용

여러 Hermitian 관측량의 중심화 행렬·Hilbert–Schmidt 내적으로 복소 Gram의 양성을 보인다. Pauli 곱셈으로 G=V+iC와 회전 공변성을 구하고, z 방향 회전으로 세 고유값·행렬식·쌍별 소행렬식을 계산한다. 두 일반 Pauli 방향의 slack에는 방향 외적의 제곱이 곱해진다. 따라서 qubit 순도 항등식은 보정된 Pauli 연산자·commutator 평균을 전제하며 두 히스토그램만으로 일반 차원의 순도를 판정하지 않는다.

공동 POVM의 네 효과는 양성·주변합·고유값을 직접 검산한다. Lüders Z 뒤 X 판독을 첫 시점으로 끌어온 효과는 X 강도가 √(1−κ²)로 줄어든다. 두 번째 판독도 약하게 만들면 원판 내부를 구현한다. κ=1에서는 역산하지 않고 λ=0을 직접 처리한다. 같은 첫 효과의 prepare instrument는 X 주변이 coin이므로 효과와 instrument를 동일시하지 않는다.

## 계산·독립성·그림

본문 네 Python 셀은 표준 라이브러리 `Fraction`으로 결과표를 정확히 생성한다. 별도 NumPy 검산은 원문 스칼라 공식을 그대로 실행하는 대신 직접 density·Pauli 행렬, Hilbert–Schmidt 열, 제곱근 Kraus 합성, 주변 모멘트 연산자를 구성한다. SymPy가 순수 끝점의 복소/실수 rank를 정확히 확인한다. 부동소수점 잔차 비교는 방향성 구간 인증이 아니다.

과학 그림은 Matplotlib SVG이며 기존 blue/orange/green 팔레트를 따른다. 공동 측정 원판은 이 unbiased 직교 모델의 전 입력 조건이다. 아래 noise 곡선은 양의 두 강도에서만 그리며 보정이 정의되지 않는 끝점을 유한 숫자로 대체하지 않는다. 새 연구 예제는 기존 실습 UI에 없는 일반 다관측량·네 결과 POVM 제어를 제공한다고 주장하지 않는다.

## 재현 명령

`python3 scripts/learning/verify-quantum-uncertainty-research.py`는 CI에 선언된 NumPy·SymPy 환경에서 독립 행렬 검산 JSON을 출력한다. `npx tsx --test scripts/learning-quantum-uncertainty-research.test.ts`는 한영 본문 보존·코드/표 일치·실제 렌더러를 함께 검사한다.

그림을 다시 만들 때는 Matplotlib와 Noto Sans CJK 글꼴이 있는 환경에서 `python3 scripts/learning/generate-quantum-uncertainty-research.py`를 실행한다. 생성기를 테스트의 필수 의존성으로 추가하지 않으며 생성된 네 SVG를 저장소에 포함한다. 브라우저 검사는 정적 미리보기 실행 후 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-uncertainty-research.cjs`로 실행한다.
