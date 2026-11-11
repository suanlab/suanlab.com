## 유형을 부르기 전에 Principal Part를 읽는다

두 독립 변수의 상수계수 이차 PDE를 A u_xx+2B u_xy+C u_yy로 씁니다. 낮은 차수 항은 해에 영향을 주지만 principal quadratic form은 바꾸지 않습니다. 혼합항의 2가 중요합니다. 식에 b u_xy라고 적혀 있으면 행렬의 비대각 원소는 b/2입니다.

이 단원은 두 변수의 이차 principal part를 분류합니다. Heat equation의 parabolic이라는 말은 일차 시간·이차 공간 미분이 함께 있는 진화형 분류입니다. 공간 Laplacian만 분류하는 것과 같은 질문은 아닙니다. 독립 변수와 미분 차수를 명시하세요.

## Symbol은 방향에 따른 이차형식이다

단위 normal n=(cos(phi),sin(phi))에 대해 다음을 정의합니다.

$$
Q=\begin{pmatrix}A&B\\B&C\end{pmatrix},\qquad
p(n)=n^TQn,
\qquad D=B^2-AC.
$$

D<0이면 Q가 정부호 또는 음의 정부호여서 elliptic입니다. D>0이면 부정부호인 hyperbolic입니다. D=0이면서 Q가 0이 아니면 rank-one parabolic principal part입니다. 계수가 모두 0이면 이차 principal part 자체가 없으며 rank-one으로 부르면 안 됩니다.

파랑은 normal 각도별 p(n)입니다. Elliptic은 모든 방향에서 같은 비영 부호, hyperbolic은 방향에 따라 부호가 바뀌고 characteristic normal에서 0입니다. Rank-one은 빠진 방향에서 0에 닿습니다. 퇴화 근처에서는 class code만 보지 말고 계수의 정확도와 반올림도 확인해야 합니다.

## 행렬과 Normal을 함께 회전한다

회전 행렬 R로 좌표를 바꾸면 다음과 같습니다.

$$
Q'=R^TQR,\qquad n'=R^Tn,\qquad
(n')^TQ'n'=n^TQn.
$$

주황은 같은 물리 normal을 회전 좌표의 각 phi-theta로 바꾸어 계산한 값입니다. Q만 돌리고 normal 성분을 그대로 두면 다른 물리 방향을 비교하게 됩니다. Trace·determinant는 회전 불변이지만 부동소수점 재계산에는 작은 차이가 남을 수 있습니다.

## 세 가지 형태를 손으로 비교하기

u_xx+u_yy에서는 Q=I라 모든 방향 값이 1입니다. u_xx-u_yy는 cos(2phi)여서 대각 normal에서 0이 됩니다. u_xx만 있으면 cos²(phi)인 비음수 rank-one입니다. 이 방향들은 입자의 시간 궤적이 아니라 미분의 방향입니다.

## 좌표 불변성과 정량적 타원성

가역 선형 좌표변환 x=Jy에서는 principal 행렬이 Q′=J⁻¹QJ⁻ᵀ가 된다. ξ에서의 이차형식은 원래 형식의 J⁻ᵀξ 값이다. 이 사상이 전단사이므로 양·음·영 부분공간의 최대 차원이 유지된다. 유형 불변성을 보이는 inertia 논증이다. 실습의 직교 회전은 그 특수한 경우다. 비직교 좌표변환은 inertia를 유지해도 고유값 크기를 바꾼다.

양의 정부호 Q의 정량적 타원성은 ξᵀQξ≥α‖ξ‖²라는 하한을 요구한다. Q≥αI이면

$$
\xi^TQ'\xi\ge\alpha\|J^{-T}\xi\|^2
\ge\frac{\alpha}{\|J\|_2^2}\|\xi\|^2.
$$

따라서 좌표변환이 PDE 추정의 상수를 나쁘게 만들 수 있다. Q=diag(1,ε), 0<ε≤1은 모두 elliptic이지만 최적 하한 ε는 0으로 간다. 유형명만으로는 비등방 확산과 neural PDE 훈련의 조건성을 알 수 없다.

| ε | 유형 | 최적 Euclidean 하한 |
|---|---|---:|
| 1 | elliptic | 1 |
| 1/10 | elliptic | 1/10 |
| 1/100 | elliptic | 1/100 |

```python
from fractions import Fraction as F
def mm(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def det(A): return A[0][0]*A[1][1]-A[0][1]*A[1][0]
Jinv = [[F(1), F(-2)], [F(0), F(1)]]
JTinv = list(map(list, zip(*Jinv)))
for eps in (F(1), F(1, 10), F(1, 100)):
    Q = [[F(1), F(0)], [F(0), eps]]
    transformed = mm(mm(Jinv, Q), JTinv)
    assert det(transformed) == det(Jinv)**2*det(Q) > 0
    for v in ((F(1), F(0)), (F(0), F(1)), (F(1), F(-1))):
        form = sum(v[i]*Q[i][j]*v[j] for i in range(2) for j in range(2))
        assert form >= eps*sum(x*x for x in v)
    print(eps, transformed)
```

유리수 shear 계산은 퇴화 근처의 반올림된 고유값 대신 congruence와 determinant 배율을 정확히 검산한다. Q=diag(1,−1)의 characteristic normal은 ξ₁²−ξ₂²=0을 만족한다. 상태 공간 궤적이 아닌 symbol 방향이다. Principal 유형만으로 경계조건이나 완전한 적정성 정리가 주어지지도 않는다.

실제 유형은 저장된 binary64 계수를 2⁻¹⁰⁷⁴의 정수배로 나타내 determinant 부호를 정확히 판정한다. Underflow나 반올림 상쇄가 유형을 바꾸는 문제를 막는다. 표시 determinant와 고유값은 부동소수점 진단으로 남는다. 저장 수의 정확한 분류가 불확실한 물리 계수를 인증하는 것은 아니다.

## 직접 풀기: 혼합항 관례

u_xx+u_xy+u_yy에서는 B=1/2, D=-3/4이므로 elliptic입니다. Phi=pi/4에서 값은 1.5입니다. 잘못 B=1을 입력하면 rank-one으로 다른 식을 만든 셈입니다. 올바른 계수를 pi/2만큼 돌려 같은 물리 normal의 값이 유지되는지 검산하세요. 식 전체에 -1을 곱해도 elliptic 유형이 바뀌지 않는 이유도 설명하세요.

더 읽기: [MIT PDE 입문](https://ocw.mit.edu/courses/18-152-introduction-to-partial-differential-equations-fall-2011/).
