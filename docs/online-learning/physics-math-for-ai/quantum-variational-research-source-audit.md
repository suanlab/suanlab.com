# 스펙트럼 하한·분산 인증·Schur 보정 자료 대조

2026-11-13. 기존 `finite-state-rayleigh-energy-variance`와 `nonorthogonal-ritz-subspace-certificates`의 한영 본문을 보존하고 연구 절을 추가한다. 원래 차원·복소 입력·비직교 metric·축퇴 공간·정확 잔차 계약을 유지한다.

## 일차 자료와 자체 증명

- [Zhu·Argentati·Knyazev, Bounds for the Rayleigh quotient and the spectrum of self-adjoint operators](https://arxiv.org/pdf/1207.3240), §2의 식 (2.3)·(2.4), §5의 바닥 고유공간 논의: 잔차 제곱에 의한 오류 상계에는 빈 스펙트럼 구간과 올바른 영역 식별이 필요하다. 본문은 모든 바닥 밖 에너지의 하한 β와 E<β를 명시하고, (H−λ0)(H−β)의 비음수 기댓값으로 유한 행렬 Temple 하한을 직접 증명한다. 높은 Ritz 값·작은 분산만으로 그 β를 얻었다고 주장하지 않는다.
- [Dusson·Sigal·Stamm, The Feshbach–Schur map and perturbation theory](https://arxiv.org/pdf/2105.02058), Theorem 1.2·식 (1.11)의 보완 공간 역연산자와 고유벡터 복원: 외부 블록의 가역성이 소거의 조건이다. 본문은 일반 유한 retained/exterior 블록 방정식을 직접 풀고 에너지 의존 보정의 부호·Loewner 상계를 유도한다. 이 교재의 두-by-두 비교 상수·유리수 예제·잔차 한 단계 확장을 논문의 전체 섭동 정리나 수렴률로 인용하지 않는다.

바닥 밖 총확률 상계는 바닥·들뜬 두 집단의 분산 분해에서 직접 유도한 단측 분산/Cantelli 형태다. 들뜬 확률0에서는 조건부 평균을 정의하지 않는다. 바닥 공간의 축퇴는 첫 증명에서 허용하며, 기준 Hamiltonian의 섭동으로 β를 구성하는 별도 예제만 중복도 포함 정렬·비축퇴 기준 gap을 전제한다. 순서별 섭동 상계는 min–max와 −ηI≤V≤ηI로 증명한다.

불확실한 E·분산·β의 상계는 양의 최악 분모로 보수적으로 전파한다. Temple 식을 E에 대해 단조라고 가정하지 않는다. 결정론적 enclosure는 통계적 공동 신뢰수준이 아니다. 분모를 해결하지 못하면 이 인증을 보류하고, 음수 추정 분산을 조용히0으로 보정하지 않는다.

## 좌표·물리 공간·근사 오류

Q=BS⁻¹ᐟ²는 같은 시험 공간의 정규직교 frame이며, AV는 원래 좌표 에너지 행렬 B†HB와 다르다. 외부 frame R을 완성한 블록 표현을 Ĥ로 구별한다. 결합 노름과 최소 Ritz 값은 같은 공간의 unitary frame 변경으로 불변이다. Gram 조건 개선이나 raw 벡터 재배율은 빠진 물리 방향을 만들지 않는다.

전체 외부 블록 D≥βI와 β>μ를 정당화해야 두-by-두 비교를 바닥 오류 상계로 읽을 수 있다. 보존 공간에 정확한 들뜬 고유벡터만 있는 반례는 이 순서 조건을 위반한다. 잔차 확장도 잔차가0이면 새 방향을 정의하지 않으며, 비영일 때의 한 단계 에너지 감소가 일반 최적화·학습 가족의 수렴률을 증명하지 않는다.

## 독립 계산과 재현

`python3 scripts/learning/verify-quantum-variational-research.py`는 CI 선언 NumPy·SymPy 환경에서 복소 회전한 Hamiltonian·순수/혼합 밀도·직접 잔차와 두 집단 확률을 만든다. 별도의 Hermitian eigensolver로 전체 스펙트럼을 비교한다. 보완 공간의 직접 선형 해법, 임의 비직교 좌표의 사영 재구성, 확대 공간의 실제 Ritz 고유값과 한 단계 비교값도 대조한다. 부동소수점 잔차는 방향성 구간 인증이 아니며 참 스펙트럼을 아는 oracle 실험이 일반 문제에서 β를 자동으로 찾아주는 알고리즘도 아니다.

본문 네 Python 셀은 `Fraction`으로 표를 정확히 재현한다. SymPy는 두 준위 등호·enclosure 분수·잘못된 cutoff 반례·유리수 Schur root를 따로 확인한다. `npx tsx --test scripts/learning-quantum-variational-research.test.ts`로 한영 계산·표·원본문 보존·실제 renderer를 검사한다.

Matplotlib·Noto Sans CJK 환경에서 `python3 scripts/learning/generate-quantum-variational-research.py`를 실행하면 기존 blue/orange/green 팔레트의 네 SVG를 만든다. 첫 그림은 q=1을 인증 영역에서 제외하고, 둘째는 에너지 오류를 Δ로 나눈 값과 무차원 외부 확률을 다른 패널에 그린다. 정적 미리보기에서 `SUANLAB_TEST_BASE=http://127.0.0.1:43309 node scripts/check-quantum-variational-research.cjs`로 실제 reader를 검사한다. 외부 하한·Schur 반복·자동 확장을 새 UI 제어로 추가한 것은 아니다.
