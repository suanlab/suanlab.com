## 알고리즘의 관례를 먼저 고정하기

Momentum과 적응형 방법에는 이름이 비슷한 여러 변형이 있다. 실습은 f(x)=(x₁²+κx₂²)/2, 초기 (1,1)에서 heavy-ball과 bias correction을 포함한 Adam을 명시적으로 비교한다. 파란 선은 같은 학습률과 반복 수의 GD, 주황 선은 선택한 방법이다. 목적과 초기값이 같아야 차이를 갱신 규칙에 연결할 수 있다.

이 예는 결정론적 이차함수의 유한 실행이다. 일반화, 확률 잡음, 분리된 weight decay, 비동기 갱신이나 보편적인 수렴 정리를 실험한 것은 아니다.

## 속도를 저장하는 heavy-ball

초기 속도 0에서 구현한 규칙은

$$
v_k=\beta v_{k-1}+g_k,\qquad x_{k+1}=x_k-\eta v_k.
$$

기울기에 1−β를 곱하지 않는 관례다. 이를 곱하는 다른 관례와 학습률 숫자만 같게 놓으면 다른 보폭을 비교하게 된다. β=0으로 두면 정확히 GD가 된다.

곡률 λ인 한 방향에서 속도를 제거하면 특성다항식 r²−(1+β−ηλ)r+β를 얻는다. 0≤β<1에서 두 근이 단위원 내부에 놓이는 조건은 0<ηλ<2(1+β)다. 실습은 이차함수의 두 방향을 검사한다. 이는 선형 점화식의 안정성 조건이며 매 반복 목적값 감소를 뜻하지 않는다. 복소 근과 관성 때문에 안정한 경우에도 진동과 overshoot가 생길 수 있다.

## 두 모멘트를 저장하는 Adam

Adam은 지수 평균 두 개를 사용한다.

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\quad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
\qquad
x_{k+1}=x_k-\eta\frac{m_k/(1-\beta_1^k)}{\sqrt{v_k/(1-\beta_2^k)}+\epsilon}.
$$

제곱·제곱근·나눗셈은 좌표별 연산이다. ε=0.001은 제곱근 밖에 고정한다. 안에 넣으면 다른 알고리즘이다. heavy-ball 모드에서는 beta2가 쓰이지 않는다.

첫 기울기는 (1,κ)다. 0으로 시작한 모멘트의 편향을 보정하면 허용된 beta 값에 관계없이 첫 모멘트는 g, 둘째는 g²다. κ=8, η=0.1이면 새 좌표는 약 (0.900100,0.900012)다. GD의 결과 (0.9,0.2)와 비교하자. Adam의 비슷한 초기 이동량은 정규화 때문이며 두 곡률이 같다는 뜻은 아니다.

## 적응형 배율이 보장하지 않는 것

제곱 기울기 평균은 Hessian이 아니다. 좌표 사이 교차 곡률도 담지 않는다. 좌표를 임의로 회전하면 동일한 알고리즘 궤적을 유지하는 것도 아니다. 유한 그래프가 좋아 보인다고 모든 볼록·비볼록 문제의 수렴이 증명되지는 않는다. 정확한 변형·학습률 일정·확률 가정·정칙 조건을 함께 읽어야 한다.

## Heavy-ball 안정 구간을 증명하고 좌표 변화 검사하기

곡률 λ에서 a=1+β−ηλ로 씁니다. 특성다항식에 r=(1+s)/(1−s)를 넣고 (1−s)²를 곱하면 단위원 내부가 왼쪽 반평면으로 옮겨지고 다음 다항식을 얻습니다.

$$
\big(2(1+\beta)-\eta\lambda\big)s^2
+2(1-\beta)s+\eta\lambda.
$$

0≤β<1에서 이차 Hurwitz 조건은 세 계수의 양수성이므로 정확히 0<ηλ<2(1+β)입니다. 등호에서는 근이 경계에 놓입니다. 고정 이차 방향의 증명이지 손실의 단조성이나 비선형 heavy-ball 수렴률의 보장은 아닙니다.

Adam의 좌표별 정규화에는 다른 한계가 있습니다. 직교 재매개화 u=Rᵀx에서 gradient는 공변 변환되지만 각 성분을 자체 크기로 나누는 연산은 그렇지 않습니다. Python은 회전한 두 기저의 첫 bias-corrected Adam 보폭을 원래 좌표로 되돌려 비교합니다. 같은 계산의 GD는 일치합니다. Epsilon은 실습처럼 제곱근 밖에 둡니다.

| 항목 | 명시한 이차함수의 Heavy-ball | 첫 Adam 갱신 |
|---|---|---|
| 안정성 검사 | 특성근 | 이 근 검사로 보장하지 않음 |
| 매개변수 기저 회전 | Scalar momentum 점화식과 호환 | 좌표 정규화가 일반적으로 보폭을 바꿈 |
| 유리한 유한 손실 | 영원한 단조성 증명 아님 | 보편 수렴 정리 아님 |

## Python으로 계산 재현하기

```python
from math import sqrt,hypot
c=1/sqrt(2); R=[[c,-c],[c,c]]
def mv(A,v): return [sum(row[j]*v[j] for j in range(2)) for row in A]
Rt=list(map(list,zip(*R))); x=[1,1]; g=[1,8]; eta=.1; eps=.001
plain=[x[i]-eta*g[i]/(abs(g[i])+eps) for i in range(2)]
gu=mv(Rt,g); rotated_step=mv(R,[eta*z/(abs(z)+eps) for z in gu])
changed=[x[i]-rotated_step[i] for i in range(2)]
gd=mv(R,mv(Rt,g))
assert hypot(gd[0]-g[0],gd[1]-g[1])<1e-14
assert hypot(plain[0]-changed[0],plain[1]-changed[1])>.1
print('Adam original basis:',plain)
print('Adam rotated basis, mapped back:',changed)
```

## 직접 풀기와 풀이

κ=8,η=0.1,β=0.8의 heavy-ball을 열 번 실행하여 GD와 목적값을 비교하자. 안정성이 매 반복 우위를 뜻하지 않음을 확인한다. β=0으로 바꾸면 두 곡선이 일치한다. 다음 Adam 한 번만 실행하고 보정 모멘트 (1,8), (1,64)를 직접 구하자. 보정을 생략하면 초기 이동량이 왜 달라지는지도 설명하자.

더 읽기: [Kingma와 Ba, Adam](https://arxiv.org/abs/1412.6980), [Bottou, Curtis와 Nocedal](https://arxiv.org/abs/1606.04838).
