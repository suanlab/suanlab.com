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

## Bracket·나머지항·부동소수점 오차 구별하기

행렬 곱의 결합법칙은 Jacobi 항등식을 보장합니다. 중첩 교환자 세 개를 전개하면 모든 삼차 단항식이 반대 부호와 상쇄됩니다. 따라서 skew 행렬은 임의 비가환 연산을 붙인 벡터 공간이 아니라 Lie algebra를 이룹니다.

고정된 유계 행렬에서 지수 급수를 한 차수 더 곱하면 다음과 같습니다.

$$
K(h)=I+h^2[A,B]+\frac{h^3}{2}[A+B,[A,B]]+O(h^4).
$$

이차항은 h의 짝함수, 삼차항은 홀함수입니다. 따라서 K(h)와 K(−h)를 평균하면 삼차항이 지워집니다. 항등을 빼고 h²로 나눈 중심 quotient의 truncation 오차는 O(h²)이고, 한쪽 quotient는 일반적으로 O(h)입니다. 중심 계산은 별도의 Python 비교이며 실습의 한쪽 quotient와 작은 h 정밀도 정책은 유지합니다.

| 계산량 | 수학적 극한 | Truncation 차수 |
|---|---|---|
| (K(h)−I)/h² | [A,B] | O(h) |
| (K(h)+K(−h)−2I)/(2h²) | [A,B] | O(h²) |
| 극도로 작은 h의 계산된 quotient | 극한만으로 정확성 보장 불가 | Roundoff가 지배할 수 있음 |

코드는 실습의 Rodrigues 함수를 호출하지 않고 행렬 지수 급수를 독립 합산합니다. h를 절반으로 줄여 오차 비가 점근 구간에서 약 2와 4가 되는지 확인하세요. 부동소수점 차분에서는 h를 무한히 줄여도 정확도가 단조롭게 좋아지지 않습니다.

생성자 하나가 0이면 모든 h에서 group commutator는 정확히 항등입니다. 구현은 이 항등식을 직접 평가해 회전과 그 역의 행렬 곱 roundoff가 가짜 scaled 오차로 확대되지 않게 합니다. h=0과 작은 h의 표시 정책은 별도로 유지합니다.

## Python으로 계산 재현하기

```python
from math import sqrt
I=[[float(i==j) for j in range(3)] for i in range(3)]
A=[[0,0,0],[0,0,-1],[0,1,0]]; B=[[0,0,1],[0,0,0],[-1,0,0]]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def exp(M):
    total=[r[:] for r in I]; term=[r[:] for r in I]
    for k in range(1,45):
        term=[[v/k for v in row] for row in mm(term,M)]
        total=[[total[i][j]+term[i][j] for j in range(3)] for i in range(3)]
    return total
def K(h):
    out=I
    for M,s in [(A,h),(B,h),(A,-h),(B,-h)]: out=mm(out,exp([[s*v for v in row] for row in M]))
    return out
AB,BA=mm(A,B),mm(B,A); C=[[AB[i][j]-BA[i][j] for j in range(3)] for i in range(3)]
for h in [.1,.05,.025,.0125]:
    plus,minus=K(h),K(-h)
    forward=sqrt(sum(((plus[i][j]-I[i][j])/h**2-C[i][j])**2 for i in range(3) for j in range(3)))
    centered=sqrt(sum(((plus[i][j]+minus[i][j]-2*I[i][j])/(2*h*h)-C[i][j])**2 for i in range(3) for j in range(3)))
    assert centered < forward
    print(h, round(forward,9), round(centered,9))
```

## 직접 풀기와 연구 연결

b=0이면 B=0이라 모든 허용 a에서 K=I, bracket=0입니다. 확인한 뒤 비영 a,b의 h=0.1과 0.05를 비교해 group 효과의 이차항과 scaled 오차의 일차항을 설명하세요.

Lie algebra 갱신을 지수화하면 회전을 군 위에 유지하는 데 도움이 됩니다. 비가환 생성자는 갱신 순서가 중요한 이유도 설명합니다. 수치 실험은 예를 보여주며 특정 숫자를 넘어선 근거는 급수 전개와 대수적 닫힘에서 얻습니다.

추가 읽기: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
