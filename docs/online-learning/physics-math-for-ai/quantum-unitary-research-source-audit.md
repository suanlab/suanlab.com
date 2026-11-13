# Ray 속도·반복 측정·고정 노출량 분할 자료 대조

2026-11-13. 기존 [unitary 수치 근거](quantum-unitary-source-audit.md)와 [실제 읽기 통합](quantum-unitary-integration-status.md)을 보존한다.

- [Anandan·Aharonov, Geometry of quantum evolution](https://doi.org/10.1103/PhysRevLett.65.1697): 공식 논문 정보와 [저자 측 PDF](https://www.tau.ac.il/~yakir/yahp/yh64)를 확인했다. 스캔 첫 페이지의 식 1–5를 직접 읽어 Schrödinger 미분·에너지 분산·이웃 중첩과 경로 길이의 두 배 규약을 대조했다. 본문은 거리 arccos|내적|, 속도 σ_H/ℏ를 쓰며 직교 거리는 π/2다. 양쪽 규약을 혼용하지 않는다.
- [Preskill §2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf): 닫힌계 전파와 조건부 사영 규칙. 반복 측정의 all-yes 사건은 화면 표본·기록을 버린 비선택 출력과 구별한다. 이상적인 순간 사영 극한이지 실제 측정 장치의 자원 보증이 아니다.

시간 하한은 Cauchy–Schwarz와 ray 길이로 직접 유도한다. 유한 하한이 직교점의 도달 가능성을 보장하지 않는 두 수준 반례를 함께 제시한다. fidelity의 cos² 하한은 각도≤π/2 구간에만 적용한다.

Lie 분할의 Duhamel 보간·unitary 망원합과 대칭 분할의 급수 꼬리 경계는 본문에서 직접 유도한다. 이차/삼차 국소 항과 전역 누적을 구별한다. 대칭 상한은 의도적으로 보수적이며 일반 비유계 미분 연산자 정리나 측정 데이터에 맞춘 추세선으로 확대하지 않는다. 총 pulse 노출량과 물리 시간은 정련해도 고정한다.

독립 검산은 일반 행렬 exponential·특이값·사영 분기 곱을 사용한다. 본문 Pauli 예제의 SU(2) trace 단축식을 재사용하지 않는다. 본문 Python 표는 배정밀도 반올림 예제이며 기존 생산 계산기의 방향성 구간 인증과 구별한다.

기존 영문 표제 1곳과 한영 각 2문장은 이미 등록된 화면을 미래 계획으로 적고 있어 현재 시제로 수정했다. [명시한 5개 문구](../../../scripts/fixtures/quantum-unitary-research-editorial.json)를 역변환한 뒤 연구 절을 제거하면 원래 본문과 정확히 같다. 수식·풀이·기존 조작은 바꾸지 않았다. 반복 사영과 분할 횟수는 독립 연구 예제이며 새 UI 제어로 표시하지 않는다.

CI의 기존 NumPy 전용 설치 목록에 독립 검산용 SciPy·SymPy·mpmath가 빠져 있었다. 현재 대조한 버전을 `scripts/learning/requirements-ci.txt`에 고정하고 Python 3.12 격리 환경에서 설치·관련 검사를 별도로 확인한다. 배포의 나머지 흐름은 바꾸지 않는다.
