## 미분이 없어도 함수 전체를 지지할 기울기는 있습니다

볼록함수 f의 기준점 u에서 subgradient v는 affine 전역 하한을 만듭니다. 단순한 국소 접선 근사가 아닙니다.

$$
v\in\partial f(u)\quad\Longleftrightarrow\quad
f(x)\ge f(u)+v(x-u)\quad\text{for every }x\in\operatorname{dom}f.
$$

미분 가능하면 subdifferential에는 gradient 하나만 있고 cusp에서는 여러 기울기가 허용됩니다. 없는 보통 미분을 추측하는 것이 아니라 별도 정의의 전역 support 인증입니다.

## 실험 함수의 Subdifferential 유도

f(x)=|x|+γx²/2, γ≥0입니다. 0 밖에서는 sign(x)+γx이고 0에서는 이차항의 미분이 0이라 다음과 같습니다.

$$
\partial f(u)=\begin{cases}[-1,1],&u=0,\\
\{\operatorname{sign}(u)+\gamma u\},&u\ne0.
\end{cases}
$$

γ=0,u=0,v=0.5의 직선 0.5x는 |x| 아래입니다. 질의 x=2의 잔차는 1이지만 접점에서 전역 최소는 0입니다. v=1.5면 양수 x에서 잔차가 −0.5x라 −∞로 내려갑니다. 유한 그림 범위가 정의의 전체 영역은 아닙니다.

## Conjugate로 전역 잔차 계산

Fenchel conjugate는 f*(v)=supₓ(vx−f(x))입니다. γ>0에서 양·음 반직선을 각각 최적화하면 다음과 같습니다.

$$
f^*(v)=\frac{(\max\{|v|-1,0\})^2}{2\gamma},\qquad
\inf_x\{f(x)-f(u)-v(x-u)\}=vu-f(u)-f^*(v).
$$

γ=0에서는 |v|≤1일 때 0, 밖에서는 +∞입니다. 유효 subgradient이면 최소 잔차가 0입니다. 잘못된 기울기는 γ>0에서 유한 음수일 수도 있고 γ=0,|v|>1에서는 −∞일 수도 있습니다. 해석적으로 유한한 값이 숫자 표현 범위를 넘으면 숨기며 수학적 무한대로 바꾸지 않습니다.

## Fenchel 등호와 합 규칙의 Qualification

f*의 정의에서 모든 유한 정의역 점 u에 대해 f(u)+f*(v)≥vu입니다. 등호가 성립하는 것은 v가 u에서 f를 전역 지지할 때와 정확히 같습니다. supremum 부등식을 정리하면 subgradient 정의가 돌아오기 때문입니다. 따라서 Fenchel 간극은 그림의 질의점 하나에서 기울기가 다른지만이 아니라 전역 지지 실패를 측정합니다.

유한 차원의 proper 볼록함수 합 규칙에는 다음 유용한 qualification이 있습니다.

$$
\operatorname{ri}(\operatorname{dom}f)\cap\operatorname{ri}(\operatorname{dom}g)\ne\varnothing
\quad\Longrightarrow\quad
\partial(f+g)(u)=\partial f(u)+\partial g(u).
$$

조건이 없어도 지지 부등식을 더하면 오른쪽에서 왼쪽으로의 포함은 얻지만 등호는 실패할 수 있습니다. x≥0에서 f(x)=−√x, 밖에서는 +∞로 두고 g를 {0}의 indicator로 둡니다. 합은 {0}의 indicator이므로 원점 부분미분은 R 전체입니다. 하지만 x=1/n²의 기울기 비가 −n이므로 모든 유한 지지 기울기가 배제되어 ∂f(0)은 공집합입니다. 오른쪽 합도 공집합이고 두 relative interior는 만나지 않습니다.

| n | 질의 x | (f(x)−f(0))/x |
|---:|---:|---:|
| 1 | 1 | -1 |
| 4 | 1/16 | -4 |
| 16 | 1/256 | -16 |

따라서 제약 최적성은 먼저 0∈∂(f+I_C)로 쓰고, 그 뒤 별도의 부분경사와 normal로 나누는 가정을 확인해야 합니다. 미분 계산 규칙에도 정의역 조건이 필요합니다.

## Python으로 계산 재현하기

```python
from fractions import Fraction as F
for n in [1,4,16,64]:
    x=F(1,n*n); value=-F(1,n)
    assert value/x == -n
    print(n,x,value/x)
# Any proposed finite slope fails for a sufficiently large n.
for slope in [-100,-1,0,10]:
    n=max(1,1-slope)
    assert -n < slope
```

## 직접 풀기와 풀이

u=1,γ=1,v=2이면 유일한 subgradient가 2입니다. x=−1에서 함수값은 1.5, 직선은 1.5+2(−2)=−2.5라 잔차는 4입니다. conjugate는 0.5, 전역 최소는 2−1.5−0.5=0입니다. 확인한 뒤 v를 바꾸어 질의 잔차 하나가 양수여도 전역 지지를 보장하지 않음을 설명하세요.

## 최적성과 학습

proper 볼록 목적함수에서 0∈∂f는 전역 최소 조건입니다. 제약은 indicator와 normal cone으로 넣을 수 있으나 합 규칙의 가정도 확인해야 합니다. Subgradient descent와 proximal 알고리즘은 이 구조를 다르게 사용합니다. cusp에서 허용되는 기울기 하나를 고른다고 수렴률이나 학습 성능까지 자동으로 보장되지 않습니다.

추가 읽기: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
