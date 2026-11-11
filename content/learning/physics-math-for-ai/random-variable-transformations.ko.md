## 출력 사건의 역상을 구하기

확률변수 Y=g(X)의 분포를 구할 때 먼저 Y≤y가 입력 공간에서 어떤 사건인지 묻습니다. 이것은 역함수 공식보다 기본적인 방법이며 g가 일대일이 아니거나 상수여도 사용할 수 있습니다. 실습은 정수 배율 a에 대해 Y=aX와 Y=aX²를 비교합니다. 선형 모드의 X는 Uniform[0,1], 제곱 모드는 Uniform[−1,1]입니다. 그림은 출력 CDF입니다.

$$
F_Y(y)=P(g(X)\le y).
$$

## 절댓값 Jacobian의 역할

미분 가능한 일대일 변환에서 작은 구간의 질량을 보존하면 출력 밀도에는 역변환 도함수의 절댓값이 곱해집니다. X가 Uniform[0,1]이고 Y=2X이면 폭은 두 배, 밀도는 절반입니다. a가 음수여도 밀도는 음수가 되지 않습니다. 방향 반전은 누적분포의 부등호 방향을 바꾸지만 질량은 양수입니다. 다차원에서는 같은 논리가 역 Jacobian 행렬식의 절댓값으로 확장됩니다.

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
$$

## 제곱 변환은 두 역상을 갖는다

Y=X²에서 0<y<1이면 x=√y와 x=−√y가 같은 출력으로 갑니다. 각 분기의 입력 밀도는 1/2이고 역도함수 크기는 1/(2√y)이므로 두 기여를 합하면 1/(2√y)입니다. 기준 y=0.25에서는 밀도 1, CDF는 √0.25=0.5입니다. 한 분기만 쓰면 전체 질량의 절반을 잃습니다.

$$
f_Y(y)=\sum_{x:g(x)=y}\frac{f_X(x)}{|g^{\prime}(x)|}=\frac1{2\sqrt y},\quad F_Y(y)=\sqrt y\quad(0<y<1).
$$

## 경계·상수 변환과 적용 범위

제곱 모드의 밀도는 0에 접근하며 발산하지만 적분은 유한합니다. y=0에서는 이 유한값 공식을 표시하지 않고 경계 발산 표식을 보여줍니다. 한 점에서 택하는 밀도값은 확률을 바꾸지 않습니다. a=0이면 출력이 항상 0이라 점 질량이 1이며 Lebesgue 밀도가 없습니다. 이런 경우 역 Jacobian으로 0을 나눠서는 안 됩니다. 생성 모델에서 가역 변환의 밀도 계산은 이 조건들을 전제로 하며, 모든 신경망에 그대로 적용되는 공식은 아닙니다.

$$
a=0\implies P(Y=0)=1,\qquad a\ne0:\;f_{aX}(y)=\frac1{|a|}f_X(y/a).
$$

## ReLU는 연속 입력을 혼합 분포로 바꾼다

X가 [−1,1]의 균등분포이고 Y=max(0,X)라고 합시다. 모든 음수 입력은 0으로 가며 이 구간의 입력 확률은 1/2입니다. 양수 입력의 위치는 그대로 남습니다. {Y≤y}의 역상을 구하면 점프와 기울어진 구간을 함께 갖는 CDF를 얻습니다.

$$
F_Y(y)=\begin{cases}0&y<0,\\(1+y)/2&0\le y<1,\\1&y\ge1.\end{cases}
$$

| y | y에서의 CDF | y의 점 질량 |
|---|---|---|
| −1/2 | 0 | 0 |
| 0 | 1/2 | 1/2 |
| 1/4 | 5/8 | 0 |
| 1 | 1 | 0 |

분포는 원점의 단위 점 질량 절반과 Uniform(0,1) 절반의 혼합입니다. 연속 부분의 밀도는 (0,1)에서 1/2이므로 적분하면 1/2만 나옵니다. 나머지 절반이 원점의 질량입니다. Dirac 측도 δ0는 단위 점 질량이며 통상적인 Lebesgue 밀도가 아닙니다. 측도로 쓰면 다음과 같습니다.

$$
P_Y(dy)=\tfrac12\delta_0(dy)+\tfrac12\mathbf1_{(0,1)}(y)\,dy.
$$

예를 들어 P(Y≤1/4)=5/8이지만 P(0<Y≤1/4)=1/8입니다. 기울어진 부분만 미분하면 점프를 잃습니다. 입력의 구간 전체가 한 점으로 붕괴하는 현상은 일대일 Jacobian 공식으로 처리할 수 없습니다. 직접 적분하면 평균 1/4, 이차 모멘트 1/6, 분산 5/48입니다.

## 2차원 Jacobian에도 변환된 지지집합이 필요하다

독립인 Uniform(0,1) 입력 X1,X2를 U=X1, V=exp(X1)X2로 옮깁니다. 역변환과 그 행렬식은 직접 계산할 수 있습니다.

$$
(x_1,x_2)=(u,ve^{-u}),\qquad
Dg^{-1}(u,v)=\begin{pmatrix}1&0\\-ve^{-u}&e^{-u}\end{pmatrix}.
$$

새 좌표의 지지집합은 곡선 경계를 가지며 0<u<1, 0<v<exp(u)입니다. 원래 밀도 1에 역변환의 행렬식을 곱하면 다음과 같습니다.

$$
f_{U,V}(u,v)=e^{-u}\mathbf1_{\{0<u<1,\;0<v<e^u\}}.
$$

u=log 2에서 v의 조건부 구간 폭은 2입니다. v=1의 결합밀도는 1/2이지만 v=3에서는 0입니다. 먼저 v를 적분하면 e⁻ᵘeᵘ=1이고, 이후 0<u<1에서 적분하면 전체 질량 1입니다. 예전 직사각형 0<v<1을 그대로 쓰면 질량 1−e⁻¹만 남습니다. 로그 행렬식이 맞더라도 지지집합이 틀리면 다른 분포입니다. 가역 생성 모형에서 행렬식만으로 밀도가 완성된다고 생각하지 말고 역변환과 지지집합을 함께 확인하세요.

## 혼합 분포와 가역 변환을 따로 실행하기

다음 코드는 혼합 분포에는 정확한 분수, 명시적인 삼각 변환에는 부동소수점 지수를 사용합니다. 구간 질량, 역 Jacobian과 조건부 폭에 따른 정규화를 검사합니다. 현재 페이지의 배율·제곱 변수 CDF 조절기와 별도인 확장 예제이며 조절기에 ReLU 분포나 일반 flow 엔진을 추가한 것은 아닙니다.

```python
from fractions import Fraction as Q
from math import exp, log, isclose

def cdf(y):
    return Q(0) if y < 0 else (1 + min(y, Q(1))) / 2

def interval_mass(a, b):
    atom = Q(1, 2) if a < 0 <= b else Q(0)
    length = max(Q(0), min(b, Q(1)) - max(a, Q(0)))
    return atom + length / 2

for y in [Q(-1, 2), Q(0), Q(1, 4), Q(1)]:
    assert cdf(y) == interval_mass(Q(-2), y)
    print("y, CDF, point mass:", y, cdf(y), Q(1, 2) if y == 0 else 0)
assert cdf(Q(1, 4)) == Q(5, 8)
assert interval_mass(Q(0), Q(1, 4)) == Q(1, 8)
mean, second = Q(1, 4), Q(1, 6)
assert second - mean**2 == Q(5, 48)

def density(u, v):
    return exp(-u) if 0 < u < 1 and 0 < v < exp(u) else 0.0

for x1 in [.25, .5, .75]:
    for x2 in [.25, .75]:
        u, v = x1, exp(x1)*x2
        assert isclose(v*exp(-u), x2, abs_tol=1e-14)
        forward = ((1.0, 0.0), (v, exp(u)))
        inverse = ((1.0, 0.0), (-v*exp(-u), exp(-u)))
        for i in range(2):
            for j in range(2):
                value = sum(inverse[i][k]*forward[k][j] for k in range(2))
                assert isclose(value, float(i == j), abs_tol=1e-14)
        assert isclose(density(u, v)*exp(u), 1.0, abs_tol=1e-14)
u = log(2)
assert isclose(density(u, 1), .5, abs_tol=1e-14)
assert density(u, 3) == 0.0
print("mixed-law mean and variance:", mean, second-mean**2)
print("triangular density:", density(u, 1), density(u, 3))
```

## 추가 문제와 연구 연결

1. Y=1에는 왜 점 질량이 없나요? **역상이 입력 확률 0인 끝점 X=1 하나이기 때문입니다.**
2. 비가역 층에도 통상적인 flow 가능도 공식을 항상 쓸 수 있나요? **붕괴 구간, 여러 분기와 기준 측도를 따로 고려해야 합니다.**
3. u=log 2의 결합밀도가 1/2인데 U의 밀도는 왜 1인가요? **폭 2인 v 지지구간에서 적분하면 주변밀도 1이 됩니다.**

사건의 역상과 역분기 조건은 [Pishro-Nik, §4.1.3](https://www.probabilitycourse.com/chapter4/4_1_3_functions_continuous_var.php)을 참고하세요. 위 ReLU와 삼각 변환은 직접 유도한 예제입니다. [CDF 단원](../mass-density-cdf/)의 점 질량과 밀도 구분을 함께 복습하세요.

## 확인 문제와 해설

1. 선형 모드 a=−2에서 밀도는? **지지구간 (−2,0)에서 1/2입니다.**
2. 제곱 분기가 하나인가요? **아니요. 내부 양의 값에 두 개입니다.**
3. 원점 밀도 발산은 원점 질량이 있다는 뜻인가요? **아니요. 비영 배율에서 점 확률은 여전히 0입니다.**

추가 읽기: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
