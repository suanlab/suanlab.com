## 실수 손실을 복소 입력으로 표현하기

복소 파라미터 z=x+iy를 쓰는 모델도 학습 손실 L은 보통 실수입니다. L=|z|²=x²+y²은 복소 정칙 함수가 아니지만 R²에서는 매끄럽고 최적화 가능합니다. 정칙성이 없다는 이유로 gradient를 정의할 수 없다고 결론내리면 안 됩니다.

Wirtinger 연산자는 실수 미분의 두 선형 조합입니다.

$$
\partial_z=\frac12(\partial_x-i\partial_y),\qquad
\partial_{\bar z}=\frac12(\partial_x+i\partial_y).
$$

z와 z̄를 형식적으로 독립 변수처럼 다루는 계산 규칙은 이 정의에서 유도됩니다. 원래 두 실수 자유도를 네 실수 자유도로 늘린다는 뜻은 아닙니다.

## 두 미분과 실제 gradient

L=|z−c|²에서

$$
\frac{\partial L}{\partial z}=\overline{z-c},\qquad
\frac{\partial L}{\partial\bar z}=z-c.
$$

실수 gradient를 복소수 g=Lₓ+iLᵧ로 묶으면 g=2∂L/∂z̄입니다. 여기서 정의한 1/2 규약과 유클리드 실수 metric을 유지하면 이 계수 2가 필요합니다. 라이브러리에서 같은 이름의 gradient가 어떤 규약인지 반드시 확인해야 합니다.

## 왜 켤레 쪽이 하강 방향인가

실수 손실의 미소 변화는 복소 표기로

$$
dL=2\operatorname{Re}\left[\left(\frac{\partial L}{\partial\bar z}\right)^*dz\right]
$$

입니다. dz=−2η∂L/∂z̄를 넣으면 일차 변화가 음수 또는 0입니다. ∂L/∂z를 같은 방향으로 그대로 쓰면 일반적으로 실수 gradient 하강과 다릅니다. 예를 들어 순허수 입력에서 켤레를 잘못 선택하면 목표로부터 더 멀어질 수 있습니다.

## 이차 손실의 갱신을 정확히 계산하기

실습은 z_next=z−2η(z−c)를 사용하므로 오차 e=z−c의 갱신은

$$
e_{\text{next}}=(1-2\eta)e,\qquad
L_{\text{next}}=(1-2\eta)^2L.
$$

0<η<1이면 수축하고 η=1이면 부호만 바뀌어 진동하며 η>1이면 비영 오차가 증가합니다. η=0은 갱신하지 않고 η=1/2이면 한 번에 최소점에 도달합니다. 임의의 복소 손실에 같은 보폭 범위를 적용할 수는 없습니다. 이 범위는 이 명시적 이차식의 정확한 결과입니다.

기준 z=1+i,c=0,η=0.25에서는 다음 z=0.5+0.5i,손실은 2에서 0.5로 줄어듭니다. 목표 c=1+i로 바꾸면 원점이 아닌 그 목표에 대한 오차를 사용합니다.

## 그림과 AI 구현

파랑은 실제 갱신을 여섯 번 적용한 경로이고 주황 점은 최소점입니다. 손실 높이를 복소 좌표로 그린 것이 아니라 파라미터의 궤적입니다. 숫자 표의 손실과 함께 읽으세요.

복소 자동 미분을 구현할 때 실수·허수 중앙차분으로 규약을 독립 검산할 수 있습니다. 이 실습은 해석적 미분식을 사용하며 특정 라이브러리의 자동 미분을 실행했다고 주장하지 않습니다. 비매끄러운 복소 절댓값·분기 연산에서는 추가 정의와 검토가 필요합니다.

## 실수 선형층의 두 입력 경로 따라가기

고정 복소수 a,b에 대해 s=az+b·conj(z)로 확장해 봅시다. 이 식은 실수 선형이고 b≠0이면 켤레 경로가 생깁니다. 잔차 r=s−c, 실수 손실 L=|r|²에서 선언한 1/2 규약을 사용해 L=r·conj(r)를 미분하면

$$
\frac{\partial L}{\partial\bar z}=\bar a r+b\bar r,\qquad
g=L_x+iL_y=2(\bar a r+b\bar r).
$$

입니다. 첫 항은 conj(r)를 미분한 기여이고 둘째 항은 r 자체를 미분한 기여입니다. 첫 항만 남기면 층을 정칙 함수처럼 취급하여 켤레 경로를 놓칩니다.

더 일반적으로 g_s가 s에서 실수 하류 손실의 gradient를 복소수로 나타낸 값이면 입력 gradient는

$$
g_z=\bar a g_s+b\bar g_s.
$$

입니다. 방향 dz에 대한 출력 변화 ds=a dz+b·conj(dz)를 Re(conj(g_s)ds)에 대입해 전개하면 같은 실수 미분 Re(conj(g_z)dz)를 얻습니다. 이는 실수 내적 Re(conj(u)v)의 수반이며 보통의 복소 선형 전치 규칙과 구분해야 합니다.

## 한 번의 갱신을 끝까지 계산하기

a=1,b=1/2,c=1+i,z=1+i로 선택하면 실수 좌표에서

$$
L(x,y)=(\tfrac32x-1)^2+(\tfrac12y-1)^2,\qquad
g=\tfrac32-\tfrac12 i.
$$

입니다. η=1/4에서 z−ηg로 갱신하면 다음과 같습니다.

| 값 | 이전 | 이후 |
|---|---|---|
| z | 1+i | 5/8+9i/8 |
| r | (1−i)/2 | −1/16−7i/16 |
| L | 1/2 | 25/128 |

최소점은 z*=2/3+2i입니다. 목표 c는 출력의 값이며 매개변수 최소점 자체가 아닙니다. 켤레 경로를 생략하면 올바른 3/2−i/2 대신 1−i를 얻습니다. (x,y)의 Hessian은 diag(9/2,1/2)이므로 이 갱신이 모든 시작점에서 수렴하려면 0<η<4/9이어야 합니다. 원래 항등 층 예제의 0<η<1과 다릅니다.

## 실수·허수 미분 검산 재현하기

다음 코드는 고정 층 세 개와 매개변수 점 세 개를 독립적인 실수·허수 중앙차분과 대조합니다. 기준 갱신과 최소점도 확인합니다. 위 실습은 기존 |z−c|²을 유지하며 확장한 층은 별도로 실행하는 연습입니다.

```python
def loss(z, a, b, c):
    r = a*z + b*z.conjugate() - c
    return r.real*r.real + r.imag*r.imag

def gradient(z, a, b, c):
    r = a*z + b*z.conjugate() - c
    return 2*(a.conjugate()*r + b*r.conjugate())

cases = [(1+0j, 0.5+0j, 1+1j),
         (1+1j, 0.25-0.5j, 0.5-0.25j),
         (1+0j, 0j, 0j)]
h = 1e-5
for a, b, c in cases:
    for z in [1+1j, -0.5+0.25j, 0j]:
        dx = (loss(z+h, a, b, c) - loss(z-h, a, b, c))/(2*h)
        dy = (loss(z+1j*h, a, b, c) - loss(z-1j*h, a, b, c))/(2*h)
        assert abs(gradient(z, a, b, c) - complex(dx, dy)) < 1e-8

a, b, c = cases[0]
z = 1+1j
assert gradient(z, a, b, c) == 1.5-0.5j
updated = z - 0.25*gradient(z, a, b, c)
assert updated == 5/8 + 9j/8
assert abs(loss(updated, a, b, c) - 25/128) < 1e-14
assert abs(loss(2/3+2j, a, b, c)) < 1e-28
print(updated, loss(updated, a, b, c))
```

프레임워크 연결: [PyTorch 2.15 Gradcheck mechanics](https://docs.pytorch.org/docs/2.15/notes/gradcheck.html)는 실수 손실에서 켤레 Wirtinger 미분의 두 배를 반환하는 규약을 설명합니다. 미분의 이름만 맞추기보다 g=Lₓ+iLᵧ와 대조하세요. 위 코드는 Python 표준 라이브러리와 해석 식을 사용하며 프레임워크 학습을 재현하지 않습니다.

## 확인 문제와 답

1. η=1/2이면? **한 번에 c에 도달합니다.**
2. η=1이면 손실이 줄까요? **일반 비영 오차에서 그대로입니다.**
3. 실수값 비상수 손실을 정칙 함수로 강제해야 하나요? **아니요. R² 또는 Wirtinger 미분을 사용합니다.**

추가 읽기: [MIT Complex Variables with Applications](https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/). 공식 강좌 범위를 확인해 보충 자료로 연결합니다. 본문의 작은 계산과 예제는 직접 구성했습니다.
