## 연속 군에는 접대수가 있습니다

Lie group은 군이면서 매끄러운 다양체이고 곱·역 연산도 매끄럽습니다. Lie algebra는 항등원의 접공간에 bracket을 더한 구조입니다. SO3의 항등을 지나는 곡선에서 R(t)ᵀR(t)=I를 미분하면 Ωᵀ+Ω=0이므로 so3는 3×3 skew-symmetric 행렬입니다.

벡터 ω는 hat(ω)v=ω×v인 외적 연산자에 대응합니다. 세 skew 독립 성분을 각 생성자 좌표로 읽는 것이며 유한 회전의 합성을 전역적인 보통 벡터 덧셈으로 바꾸는 것은 아닙니다.

$$
\widehat\omega=\begin{pmatrix}0&-\omega_z&\omega_y\\
\omega_z&0&-\omega_x\\-\omega_y&\omega_x&0\end{pmatrix},\qquad
[\widehat a,\widehat b]=\widehat{a\times b}.
$$

## Bracket은 무한소 비가환성을 측정합니다

행렬의 괄호는 [A,B]=AB−BA입니다. A,B가 skew이면 결과도 skew라 대수 안에 닫힙니다. 실험은 A=a hat(ex), B=b hat(ey)로 두고 bracket ab hat(ez)를 실제 행렬 곱으로 계산합니다. 3차원 연산을 평면 회전으로 대신하지 않습니다.

작은 네 회전의 group commutator는 다음과 같습니다.

$$
K(h)=e^{hA}e^{hB}e^{-hA}e^{-hB}
=I+h^2[A,B]+O(h^3).
$$

각 지수를 이차까지 전개하면 상수항 I가 남고 일차항이 지워진 뒤 혼합항 AB−BA가 남습니다. 고정된 유계 A,B에서 (K−I)/h²는 일차 truncation 오차로 bracket에 접근합니다.

## 유한 계산 읽기

a=b=1,h=0.1에서 scaled [1,0] 성분은 약 0.99665, bracket은 1입니다. 전체 scaled 행렬의 Frobenius 오차는 약 0.10011이고 h=0.05에서는 약 0.05001입니다. 성분 하나는 전체 행렬보다 빠르게 수렴할 수 있으므로 그 성분만으로 모든 오차율을 판단하지 마세요.

위 그림은 K로 실제 변환한 3차원 벡터와 기준 벡터입니다. 아래는 선택한 행렬 성분을 h에 따라 그렸고 주황은 bracket 기준입니다. 비등각 투영의 보이는 각도가 실제 회전각은 아닙니다.

## 0과 수치 정밀도는 다른 문제입니다

h=0에서 quotient는 정의되지 않지만 극한과 bracket은 존재합니다. 아주 작은 양수 h에서는 K−I의 소거 오차가 h⁻²로 증폭됩니다. 구현은 h<10⁻⁶의 scaled 값을 숨기고 정밀도 한계를 명시합니다. 이 기준은 수치 표시 정책이며 수학적 정의역 제한이 아닙니다. 기준 곡선은 h≥0.01을 씁니다.

## 직접 풀기와 연구 연결

b=0이면 B=0이라 모든 허용 a에서 K=I, bracket=0입니다. 확인한 뒤 비영 a,b의 h=0.1과 0.05를 비교해 group 효과의 이차항과 scaled 오차의 일차항을 설명하세요.

Lie algebra 갱신을 지수화하면 회전을 군 위에 유지하는 데 도움이 됩니다. 비가환 생성자는 갱신 순서가 중요한 이유도 설명합니다. 수치 실험은 예를 보여주며 특정 숫자를 넘어선 근거는 급수 전개와 대수적 닫힘에서 얻습니다.

추가 읽기: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
