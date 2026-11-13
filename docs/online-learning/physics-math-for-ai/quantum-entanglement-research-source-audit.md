# 곱상태 거리·표지 필터·유한 witness 자료 대조

2026-11-13. 기존 `bipartite-pure-schmidt-partial-trace`와 `two-qubit-ppt-correlations-chsh`의 원본문을 유지하고 두 연구 절을 더한다. 원래 일반 차원 순수 상태와 두-qubit 혼합 상태 계산의 차원을 줄이지 않는다.

## 일차 자료의 역할

- [Bennett·Bernstein·Popescu·Schumacher, Concentrating Partial Entanglement by Local Operations](https://arxiv.org/pdf/quant-ph/9511030), Procrustean 논의: 알려진 Schmidt 기저·가중치로 한 쌍의 큰 진폭을 약화하고 성공 표지를 전달하는 농축을 설명한다. 본문은 두 Kraus를 직접 구성해 completeness·성공 확률·실패 가지·무신호성을 유도한다. 대각 단일 필터의 성공 확률 계산을 모든 LOCC의 최적 변환 정리나 점근 수율로 인용하지 않는다.
- [Horodecki 등의 Separability of mixed states](https://arxiv.org/pdf/quant-ph/9605038), Theorem 1·분리 선형 함수 논의: 모든 곱사영에서 비음수인 Hermitian 함수와 분리 상태를 연결한다. 본문의 NPT 방향 witness·최대 Schmidt 가중치 witness·singlet Pauli 분해는 지정한 tensor/부분전치 규약에서 직접 검산한다. 고정 witness의 실패는 분리 인증이 아니다.
- [Hoeffding, Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf), 인쇄 p.15 Theorem 1의 식 (2.3)·공통 [a,b] 재척도화 설명: 독립 유계 변수의 단측 꼬리 상계다. 원 PDF를 내려받아 해당 페이지를 이미지로 읽었다. 본문은 폭 3/2의 witness 점수와 폭 1의 CHSH 승리 점수에 적용하며, 독립·동일한 원천과 사전 고정 시도 수를 명시한다. 데이터 재사용·임의 중단·제한 없는 장치 오차에 이 상계를 그대로 적용하지 않는다.

## 자체 유도와 범위

정규화 계수표의 연산자 노름이 최대 곱벡터 겹침을 준다. 복소 SVD에서 Bob 물리 벡터는 NumPy `Vh`의 해당 행이며, 켤레 위치를 직접 전역 벡터 겹침으로 확인한다. 순수 곱 근사까지 trace distance와 위상 정렬 벡터 거리를 분리한다. 분리 혼합 전체까지의 최소 거리로 바꾸지 않는다. Bell과 고전 혼합의 거리 반례를 포함한다.

필터는 q=0·1/2의 영확률 가지를 정규화하지 않는다. 표지별 비정규화 Bob 상태를 합치면 원래 축약 밀도다. 평균 순수 가지 엔트로피와 입력 엔트로피의 부등식은 binary entropy의 오목성으로 증명한다. 혼합 밀도의 엔트로피를 보편적인 얽힘 척도로 취급하지 않는다.

측정 witness는 축을 무작위로 선택하는 신뢰 Pauli 모형이다. 점수 평균의 신뢰 상한, 상관 보정의 결정론적 오차 여유, 제1종 오류와 탐지 실패 확률을 각각 유도한다. 원천 상태를 데이터에 대한 사후확률로 해석하지 않는다. CHSH 비교는 별도의 사전 고정 iid·설정 독립·전 시도 이진 결과 모형이며 검출 손실·일반 기억·optional stopping을 다루는 증명으로 확대하지 않는다. 학습한 witness는 독립 holdout 검증 또는 적절한 동시 상계가 필요하다.

## 계산·그림·재현

본문의 네 Python 셀은 표준 라이브러리 `Fraction`으로 표와 정확히 같은 분수를 출력한다. 표본 수 표는 설명용 기록이며 측정 데이터나 보장된 결과가 아니다. 그림의 연속 표본 수 곡선은 고정-n 반지름의 비교선이며 시간에 따른 감시·중단 절차가 아니다.

`python3 scripts/learning/verify-quantum-entanglement-research.py`는 CI의 NumPy·SymPy·mpmath 환경에서 직접 복소 SVD·밀도 차이 고유값·회전한 Kraus·부분전치 witness·전체 국소 측정 확률을 구성한다. 이항 분포를 `Fraction`으로 열거하고 고정밀 반지름과 대조한다. 범용 유계 점수 법칙도 검사하므로 모든 이항 매개변수를 물리적 Werner 원천으로 해석하지 않는다. 유한 열거는 일반 확률 정리의 증명이 아니며 행렬 부동소수점 잔차도 방향성 구간 인증이 아니다.

`npx tsx --test scripts/learning-quantum-entanglement-research.test.ts`로 한영 본문 보존·실행 표·실제 렌더러를 검사한다. Matplotlib·Noto Sans CJK 환경에서는 `python3 scripts/learning/generate-quantum-entanglement-research.py`로 기존 blue/orange/green 팔레트의 네 SVG를 재생성한다. 정적 미리보기에서 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-entanglement-research.cjs`로 실제 reader를 검사한다. 새 필터·표본·witness 입력 UI를 추가한 것으로 설명하지 않는다.
