## 관측 행에서 공분산 만들기

m×d 행렬 A의 행이 관측이고 열이 변수라고 합시다. S=AᵀA/m은 비중심화한 표본 이차 모멘트입니다. 모형 평균이 0임을 알고 있으므로 여기서는 표본평균을 빼거나 m−1로 나누지 않습니다. 이 구분은 실측 데이터의 covariance 전처리와 중요하게 연결됩니다.

$$
S=\frac1m A^\top A,\qquad v^\top Sv=\|Av\|^2/m\geq0.
$$

## 작은 행렬의 실제 고유값

실습은 고정 seed로 만든 유사 Gaussian 행렬과 Jacobi 대각화를 사용합니다. 파랑 점은 정렬한 유한 표본 고유값입니다. 연결선은 번호별 비교를 위한 것이며 스펙트럼의 확률밀도가 아닙니다. trace는 고유값 합과 같아 검산에 사용할 수 있습니다.

대각화에는 부동소수점 오차가 있습니다. 최대 고유값 규모와 1 중 큰 값의 10⁻¹⁰배 이하 절댓값은 수치 0으로 표시하며 그 허용오차를 문장에 기록합니다. 수치 랭크가 임의의 자료에서 기호적 랭크를 증명하는 것은 아닙니다.

## Marchenko–Pastur의 점근 기준

이상적 iid 중심 0·분산 1 원소에서 m,d가 함께 커지고 d/m→γ>0인 극한을 생각합니다. 적절한 가정 아래 nonzero bulk의 경계는 (1±√γ)²이고, γ>1에서는 0에 질량 1−1/γ가 생깁니다.

$$
\lambda_-=(1-\sqrt\gamma)^2,\quad\lambda_+=(1+\sqrt\gamma)^2,\quad
\text{zero atom mass}=\max(0,1-1/\gamma).
$$

주황 선은 이 점근 경계입니다. 현재처럼 최대 d=8인 작은 행렬의 신뢰구간이나 모든 고유값의 확정 범위가 아닙니다. 그 밖의 값이 나와도 극한 정리의 반례로 볼 수 없습니다.

## 표본 수가 변수 수보다 적으면

rank(AᵀA)≤min(m,d)이므로 d=8,m=3이면 영 공간 차원은 적어도 5입니다. nonzero bulk의 하한이 양수여도 별도의 영 원자를 빼먹으면 안 됩니다. 이상적 Gaussian의 랭크는 거의 확실히 min(m,d)이지만, 현재 화면은 정해진 유사난수 실현과 수치 허용오차로 검사합니다.

## AI에서 읽는 스펙트럼

표현·gradient·공분산의 스펙트럼은 조건수와 저차원 구조를 이해하는 도구입니다. 그러나 실제 학습 데이터의 의존·비등방성·spike 구조는 iid 기준과 다릅니다. normalization, centering, 표본/변수 축, 유한 표본과 극한을 먼저 명시한 뒤 비교해야 합니다.

## 확인 문제와 해설

1. d>m일 때 반드시 생기는 것은? **영 공간입니다.**
2. 주황 경계 밖 값은 정리 위반인가요? **아니요. 점근 기준입니다.**
3. S는 왜 양의 준정부호인가요? **모든 v에서 vᵀSv가 제곱 노름이기 때문입니다.**

추가 읽기: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).

[Vershynin, Introduction to the Non-Asymptotic Analysis of Random Matrices](https://arxiv.org/abs/1011.3027).
