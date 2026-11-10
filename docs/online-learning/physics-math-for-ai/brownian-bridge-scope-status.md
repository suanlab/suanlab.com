# Brownian bridge와 시간 분할: 범위·유도식 검증

2026-11-10. [경로적분 입문 범위](path-integrals-primary-source-audit.md)의 첫 처리 `brownian-bridge-time-slicing-measure`를 설계했다. 고정끝 Gaussian 시간 분할을 사용해 측도·밀도 정규화·정련·관측 오차를 구분한다. 아직 원고·생산 계산기·독립 응답 검산·Worker·학습 화면·reader는 구현하지 않았다.

## 검산한 내용

표준 라이브러리 `fractions.Fraction`으로 발생 행렬을 직접 구성하고 Gauss–Jordan 역행렬·행렬식을 계산했다. 분할N=2…32에서 두 매개변수 조합, 총62모형을 검산했다. 공분산의 닫힌 식을 그대로 행렬 역산에 사용하는 방식이 아니다.

- Q의 역행렬, 행렬식, 양끝을 포함한 선형 방정식과 선형 평균을 확인했다. N=2에서는 첫/마지막 내부 노드가 같으므로 두 끝의 항을 더한다.
- Gaussian 밀도들의 정규화 인자와 내부 적분 인자를 결합하면 전체 시간 자유 커널의 prefactor가 된다. 양수의 제곱을 유리수로 비교했으며 π·제곱근·지수의 수치 구간을 검산했다고 주장하지 않는다.
- 사다리꼴 면적의 평균·분산, 각 증분 분산·예상 이차 변동·예상 작용을 확인했다. 끝점 변위의 기여를 거칠기와 혼동하지 않는다.
- 정련15쌍에서 짝수 시각 평균·공분산의 정확한 일치, 홀수 블록의 대각 구조와 Schur 소거, 면적 분산 차 Dτ³/(8N²)을 확인했다.

**695개 유한 항등식 검사를 통과했다.** 이 결과는 연속 경로 측도의 존재 정리나 임의 potential의 Feynman–Kac/Trotter 수렴 증명을 대신하지 않는다. 이후 원고에서 유한 항등식·연속 과정의 정리·수치 근사의 보증을 구분한다.

## 재현과 다음 단계

```bash
python3 scripts/learning/verify-brownian-bridge-derivations.py
```

추가 Python 패키지가 필요 없다. `.runtime/learning/brownian-bridge-foundation/derivations.json`에 각 검사를 기록한다. 다음은 일반 대학원생을 위한 한영 원고·손계산·설명 그림 설계와 정확한 수치 기반이다. 도서 메뉴의 Interactive Book 경로를 사용하고 기존 홈페이지 스타일과 색을 유지한다. 단원 등록 전에는 완료 단원 수에 더하지 않는다.
