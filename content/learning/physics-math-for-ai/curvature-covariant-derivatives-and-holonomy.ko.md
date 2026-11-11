## 1. 움직이는 기저 위의 벡터 미분하기

곡면에서는 접평면이 위치에 따라 바뀝니다. 평면에서도 극좌표 기저는 각도를 따라 회전합니다. 공변 미분은 접속을 사용해 벡터를 비교합니다. 벡터 $V$의 성분은 다음과 같습니다.
$$
(\nabla_i V)^k=\partial_i V^k+\Gamma^k_{ij}V^j.
$$
접속항이 기저의 변화를 보정합니다. 레비치비타 접속은 계량을 보존하고 비틀림이 0인 조건으로 정해집니다. [Tong, Introducing Riemannian Geometry §3.2](https://davidtong.org/teaching/general-relativity/grhtml/S3)

## 2. 회전 대칭인 세 곡면 비교하기

중심을 제외한 곳에서 $f(r)>0$인 계량 $ds^2=dr^2+f(r)^2d\theta^2$를 씁니다. 평면, 반지름 $R$인 구면, 곡률 반지름 $R$인 쌍곡 평면에서는 각각 $f=r$, $R\sin(r/R)$, $R\sinh(r/R)$입니다. 레비치비타 공식에 대입하면 다음을 얻습니다.
$$
\Gamma^r_{\theta\theta}=-ff',\qquad
\Gamma^\theta_{r\theta}=\Gamma^\theta_{\theta r}=f'/f.
$$
평면에서도 이 좌표의 접속 계수는 0이 아닙니다.

## 3. 접속에서 곡률 계산하기

구면의 가우스 곡률이 양수가 되는 부호 관례를 씁니다. 이 계량의 곡률은
$$
K=-f''/f.
$$
예를 들어 반지름과 각도에 대한 성분 $R^r{}_{\theta r\theta}$는 $-ff''$입니다. $(f')^2$에 비례하는 항들은 상쇄됩니다. 평면에서는 접속 계수가 0이 아니어도 $K=0$입니다. 구면은 $K=1/R^2$, 쌍곡 평면은 $K=-1/R^2$입니다. 2차원 스칼라 곡률은 $2K$입니다.

## 4. 원을 따라 벡터 평행이동하기

정규직교 기저 $e_r=\partial_r$, $e_\theta=f^{-1}\partial_\theta$를 씁니다. 반지름을 고정하면 $\nabla_\theta e_r=f'e_\theta$, $\nabla_\theta e_\theta=-f'e_r$입니다. 이 기저에서 각도가 $\alpha$인 평행 벡터는 $d\alpha/d\theta=-f'$를 만족합니다. 양의 방향으로 한 바퀴 돌면 각도 변화는 $-2\pi f'$이고, $2\pi$를 법으로 보면 $2\pi(1-f')$와 같습니다.

평면에서 기저는 한 바퀴 돌고 평행이동한 벡터는 처음 방향으로 돌아옵니다. 구면의 남는 회전은 $2\pi(1-\cos(r/R))$이며 곡률에 원이 둘러싼 구면 마개 면적을 곱한 값입니다. 쌍곡 평면에서는 대응하는 값이 음수입니다. 이 부호 있는 식은 작은 고리에서 0부터 연속적으로 변하는 값을 고른 것입니다. 실제 마지막 방향의 각도는 $2\pi$를 법으로 정해집니다. [Tong, Introducing Riemannian Geometry §3.3](https://davidtong.org/teaching/general-relativity/grhtml/S3)

## 5. 작은 고리의 극한 검산하기

```python
from math import cos, cosh, isclose, pi

R = 2
r = 0.7
sphere_area = 2*pi*R**2*(1-cos(r/R))
hyperbolic_area = 2*pi*R**2*(cosh(r/R)-1)
sphere_turn = 2*pi*(1-cos(r/R))
hyperbolic_turn = 2*pi*(1-cosh(r/R))
assert isclose(sphere_turn, sphere_area/R**2)
assert isclose(hyperbolic_turn, -hyperbolic_area/R**2)
assert sphere_turn > 0 > hyperbolic_turn
small = 0.001
assert isclose(2*(1-cos(small/R))/small**2, 1/R**2, rel_tol=1e-6)
print(sphere_turn, hyperbolic_turn)
```

## 6. 좌표 기호와 기하량 구분하기

- 세 곡면에서 접속 계수와 가우스 곡률을 비교하세요.
- 평면의 회전하는 극좌표 기저에서 벡터를 따라가고 마지막 방향과 처음 방향을 비교하세요.
- 구면에서 고리의 반지름을 늘리고 회전과 면적이 어떻게 변하는지 확인하세요.
- 일반상대론에서는 로런츠 계량에도 같은 접속과 곡률 개념을 적용합니다. 이 실험은 경로와 회전을 보기 쉬운 공간 곡면을 사용합니다.
