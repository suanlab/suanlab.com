## 볼록 함수와 접선 사이의 간극

미분 가능한 볼록 생성 함수 φ에 대해

$$
D_\phi(x,y)=\phi(x)-\phi(y)-\phi'(y)(x-y).
$$

볼록성으로 비음수이고 엄격 볼록이면 같은 인자에서만 0이다. 두 번째 인자 y는 접선 기준점, 첫 번째 x는 세로 간극을 재는 위치다. 파란 생성 함수와 주황 접선의 x에서 높이 차이가 divergence다.

Divergence가 자동으로 거리 metric인 것은 아니다. 비대칭일 수 있고 삼각부등식도 실패할 수 있다. 유클리드 거리 제곱의 절반조차 거리 자체의 삼각부등식을 만족하지 않는다.

## 세 생성 함수 비교

실습은 양수 [0.1,3]에서 다음을 사용한다.

$$
\begin{aligned}
\phi(x)=x^2/2 &: D=(x-y)^2/2,\\
\phi(x)=x\log x-x &: D=x\log(x/y)-x+y,\\
\phi(x)=-\log x &: D=x/y-1-\log(x/y).
\end{aligned}
$$

Entropy divergence는 정규화된 확률 벡터에 합하면 선형 질량 항이 사라져 KL이 된다. 양수 스칼라 하나를 정규화된 분포처럼 읽으면 안 된다. Log 생성 함수는 절대 차이보다 상대 비율을 강조한다.

x=2,y=1에서 entropy 순방향은 2log2−1≈0.386294, 역방향은 1−log2≈0.306853이다. 인자를 바꾸면 접선 기준점도 바뀐다. Quadratic은 양쪽 모두 0.5다.

## 국소 곡률과 수치 상쇄

작은 h에서 Dφ(y+h,y)=φ″(y)h²/2+o(h²)다. 곡률은 quadratic에서 1, entropy에서 1/y, log에서 1/y²다. 같은 절대 변화도 기준점에 따라 다른 국소 비용을 낸다.

함수값과 접선을 직접 빼면 작은 양의 간극이 상쇄로 사라질 수 있다. 구현은 가까운 점에서 안정적인 로그 Taylor 나머지를 사용한다. 음수를 몰래 0으로 바꾸거나 수치 0을 서로 다른 인자의 divergence가 진짜 0이라는 증명으로 삼지 않는다.

## 세 점 항등식

정의를 전개하면

$$
D(x,z)-D(x,y)-D(y,z)
=(\phi'(y)-\phi'(z))(x-y).
$$

실습은 양변을 따로 계산한다. Gradient 차이 항 때문에 유클리드 삼각형·Pythagoras 논리를 그대로 재사용할 수 없다. 특정 투영 부등식에는 직교성·최적성 같은 추가 조건이 필요하다.

## 방향을 맞추어 Pythagoras 부등식 복구하기

볼록 집합 C에서 Dφ(v,x)를 최소화하는 z가 미분 가능 영역 안에 존재한다고 하자. 일차 최적성으로 모든 u∈C에 대해 ⟨∇φ(z)−∇φ(x),u−z⟩≥0이다. 이를 세 점 항등식에 넣으면

$$
D_\phi(u,x)-D_\phi(u,z)-D_\phi(z,x)
=\langle\nabla\phi(z)-\nabla\phi(x),u-z\rangle\ge0.
$$

이는 투영 변수를 첫 인자에 둔 투영 부등식이다. 임의 세 점의 삼각부등식이 아니다. C의 볼록성과 최소화 방향이 필요하며, KL 인자를 뒤집거나 비볼록 집합을 쓰면 같은 증명을 그대로 적용할 수 없다.

양수 벡터에서 φ(v)=Σ(vᵢlog vᵢ−vᵢ)를 쓰고 x를 확률 simplex에 투영하자. S=Σxᵢ일 때 해는 z=x/S다. Gradient 차이는 상수 벡터 −log S이고 u와 z 모두 질량 1이므로 u−z와의 내적은 0이다. 따라서 모든 내부 simplex u에 대해 등호다. 비정규화 divergence D(z,x)=S−1−log S에는 질량 항이 포함된다. 이를 버리면 다른 값을 계산한다.

| x | z | D(z,x) |
|---|---|---|
| (2,1) | (2/3,1/3) | 2−log3 |
| (1,2) | (1/3,2/3) | 2−log3 |
| (3,1) | (3/4,1/4) | 3−log4 |

```python
from decimal import Decimal as D, localcontext
with localcontext() as ctx:
    ctx.prec = 80
    def divergence(u, v):
        return sum(a*(a/b).ln()-a+b for a, b in zip(u, v))
    for x in ((D(2), D(1)), (D(1), D(2)), (D(3), D(1))):
        mass = sum(x)
        z = tuple(a/mass for a in x)
        for first in (D(1)/4, D(1)/2, D(3)/4):
            u = (first, 1-first)
            residual = divergence(u, x)-divergence(u, z)-divergence(z, x)
            assert abs(residual) < D('1e-75')
        assert abs(divergence(z, x)-(mass-1-mass.ln())) < D('1e-75')
        print(tuple(x), tuple(z), divergence(z, x))
```

벡터 투영 예제는 위 스칼라 접선 간극의 개념을 확장하며 조절기를 바꾸지 않는다. Decimal로 등식 아홉 사례를 검산한다. 거의 0인 잔차에서 증명을 추측하는 대신, 별도 증명한 항등식을 수치로 확인한다.

## 직접 풀기와 풀이

Quadratic에서 x=3,y=2,z=1이면 D(3,1)=2지만 D(3,2)+D(2,1)=1이다. 삼각부등식의 직접 반례다. Entropy에서 x=2,y=1을 바꾸어 위의 비대칭 간극을 검산하자. x=y에서는 두 divergence가 0이고 어떤 z에도 세 점 항등식이 성립한다.

Mirror descent는 이 생성 함수 기하를 갱신 규칙의 이동 비용으로 사용한다. 생성 함수를 바꾸면 이동 비용과 정의역 보존이 달라지지만, 최소화하려는 목적 자체를 생성 함수로 바꾼 것은 아니다.

더 읽기: [Nemirovski의 Mirror Descent 강의](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf).
