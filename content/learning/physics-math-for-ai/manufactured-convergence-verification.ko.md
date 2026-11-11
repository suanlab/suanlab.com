## 그림이 그럴듯한 것보다 강한 검증

해를 모르는 실제 문제만으로 solver를 개발하면 어느 오차가 줄었는지 판단하기 어렵습니다. 제조해는 원하는 매끄러운 u를 먼저 정하고 그 u를 방정식에 넣어 source와 경계를 만듭니다. 현재는 u=sin(πx), -u″=π²sin(πx), 양 끝 0입니다. 이 알려진 문제에서도 정확하지 않으면 더 복잡한 연구 문제의 결과를 신뢰하기 어렵습니다.

세 격자 N,2N,4N에서 실제 FD 시스템을 조립합니다. Direct 모드는 tridiagonal elimination으로 풀고 Jacobi 모드는 0에서 지정한 횟수만 실제 반복합니다. 경계 bias를 일부러 넣어, 틀린 문제를 매우 정확하게 푸는 경우도 비교할 수 있습니다.

## 같은 Norm에서 측정한 차수

$$
E_\infty=\max_i|u_i-u(x_i)|,\qquad
E_2=\sqrt{h\left[\tfrac12e_0^2+\sum_{i=1}^{N-1}e_i^2+\tfrac12e_N^2\right]},\qquad
p_{obs}=\log_2(E_h/E_{h/2}).
$$

Max norm에는 경계도 포함합니다. L2는 연속 적분의 사다리꼴 가중 근사이며 단순 벡터 Euclidean norm과 다릅니다. Norm을 바꾸거나 물리 구간을 바꾼 값끼리 비율을 만들면 차수의 뜻이 사라집니다. 두 연속 비율을 함께 보며 점근 구간인지 확인하세요.

중심 차분은 매끄러운 현재 해에서 약 2차 오차를 기대합니다. 하지만 이는 실제 풀이의 대수 오차와 경계 모델 오류가 충분히 작을 때입니다. 선을 h² 모양으로 미리 정하지 않고 계산한 세 오차로만 그림을 만듭니다.

## 잔차와 오차는 다른 양

대수 잔차 r=b-Au는 이산 식을 얼마나 잘 풀었는지 말합니다. 해 오차 e=u-u_exact는 연속 정답과의 거리입니다. 정확한 이산해에 대한 대수 오차는 A^{-1}r와 연결되지만 이산화와 잘못된 경계 데이터까지 r가 검출해 주지는 않습니다.

왼쪽 경계를 0.1로 바꾸면 다른 경계 문제를 풉니다. Direct solve의 잔차는 반올림 수준이지만 원래 정확해에 대한 오차는 0으로 가지 않습니다. 반대로 Jacobi 반복 수가 너무 적으면 fine 격자의 느린 수렴이 지배해 격자를 줄여도 오차가 잘 줄지 않습니다.

## 잔차를 검증 예산의 한 항으로 바꾸기

1D Dirichlet FD 행렬은 역이 비음수인 M-matrix다. 이산 barrier wᵢ=xᵢ(1−xᵢ)/2는 Aw=1,max w≤1/8을 만족한다. 잔차 상계에 비교 원리를 적용하면 대수 오차 ‖e_alg‖∞≤‖r‖∞/8이다. 경계 불일치 δ와 제조 목표의 일관성 결함 τ를 포함하면

$$
\|u_{computed}-u_{target}\|_\infty
\le |\delta|+\tfrac18(\|r\|_\infty+\|\tau\|_\infty).
$$

잘못된 데이터·선형 풀이·이산화를 분리한다. 이차 방법에서는 대수 잔차 O(h²)가 이 norm에서 맞는 예산으로 충분하며 고정 반복 수가 이를 보장하지는 않는다.

0에서 시작하는 Jacobi와 sine 우변의 정확 이산해에 대한 오차는 그 진폭에 cos(πh)ᵏ를 곱한 것이다. h² 이하로 만들려면 O(h⁻²log(1/h))회가 필요하다. 격자 세분화는 미지수 수뿐 아니라 필요한 relaxation 작업도 늘린다.

| N | 필요한 대수 오차 목표 | 반복 수 |
|---|---|---|
| 8 | h²=1/64 | 아래 계산 |
| 16 | h²=1/256 | 아래 계산 |
| 32 | h²=1/1024 | 아래 계산 |

```python
from fractions import Fraction as F
from math import pi,sin,cos,log,ceil
for N in (4,8,16):
    h=F(1,N);xs=[F(i,N) for i in range(1,N)]
    inverse=[[h*min(x,y)*(1-max(x,y)) for y in xs] for x in xs]
    for x,row in zip(xs,inverse):assert sum(row)==x*(1-x)/2<=F(1,8)
for N in (8,16,32):
    h=1/N;scale=pi*pi*h*h/(4*sin(pi*h/2)**2);rho=cos(pi*h)
    iterations=ceil(log(h*h/scale)/log(rho))
    assert scale*rho**iterations<=h*h
    assert scale*rho**(iterations-1)>h*h
    print(N,iterations,'algebraic error:',scale*rho**iterations)
```

정확한 역행렬 행합으로 소거 해법과 독립적으로 안정 상수를 검산한다. 반복 수 추정은 보조 작업량 계산이며 live의 최대 500회보다 클 수 있다. 일관성 차수가 틀리지 않아도 fine 오차 곡선이 평평해지는 이유다. 실제 부동소수점 잔차를 엄밀한 인증에 쓰려면 산술 오차도 평가해야 한다.

## 손계산과 직접 풀기

정현파가 FD 행렬의 고유벡터이므로 올바른 경계의 이산해는 다음 배율을 가집니다.

$$
u_i=\frac{\pi^2h^2}{4\sin^2(\pi h/2)}\sin(\pi x_i).
$$

분모를 전개하면 배율은 1+π²h²/12+O(h⁴)입니다. 따라서 max 오차의 주도항이 h²이고 관측 차수는 2에 접근합니다. 이 식은 직접 해법 결과와 독립적으로 검산하는 기준입니다.

직접 풀기: 왼쪽 경계 bias δ=0.1이 있을 때 max 오차가 왜 적어도 0.1인가요? 풀이: max norm은 경계 x=0도 포함하며 계산값 δ와 참값 0의 차이가 이미 0.1입니다. 작은 내부 행렬 잔차로 이 경계 불일치를 숨길 수 없습니다.

## 연구에 남겨야 할 검증 기록

AI surrogate나 역문제 연구에서는 PDE 설정, 제조해, source 유도, 경계, norm, grid, 선형 해법 tolerance 또는 반복 수를 함께 기록하세요. 반올림 plateau나 조건수 증폭은 더 작은 h에서 별도 점검합니다. 이 유한 격자 표가 모든 문제의 asymptotic 수렴이나 실험 데이터 모델 타당성을 증명하지는 않습니다. 작은 검증 문제에서 여러 독립 항등식을 맞추는 것은 그보다 큰 주장으로 가기 위한 출발점입니다.

참고: [MIT 수치 PDE 강의](https://ocw.mit.edu/courses/18-336-numerical-methods-for-partial-differential-equations-spring-2009/).
