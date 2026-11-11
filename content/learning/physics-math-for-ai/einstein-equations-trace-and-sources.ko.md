## 1. 곡률과 응력·에너지 연결하기

이 단원에서는 시공간 부호 $(-,+,+,+)$를 씁니다. 여기서 채택한 곡률 부호의 장방정식은 다음과 같습니다.
$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=\kappa T_{\mu\nu},\qquad
G_{\mu\nu}=R_{\mu\nu}-\tfrac12R g_{\mu\nu},
\qquad \kappa=\frac{8\pi G}{c^4}.
$$
이 식은 각 시공간 점에서 텐서 사이의 관계를 줍니다. 응력 텐서에는 에너지 밀도, 흐름과 공간 응력이 들어 있습니다. 서로 다른 좌표계의 성분을 그대로 비교하려면 먼저 같은 기저로 변환해야 합니다. [Tong, The Einstein Equations §4.1](https://davidtong.org/teaching/general-relativity/grhtml/S4)

## 2. 4차원에서 대각합 취하기

$g^{\mu\nu}$와 수축하고 $g^{\mu\nu}g_{\mu\nu}=4$를 사용합니다. 아인슈타인 텐서의 대각합은 $-R$이므로 다음을 얻습니다.
$$
R=4\Lambda-\kappa T,\qquad
R_{\mu\nu}=\kappa(T_{\mu\nu}-\tfrac12Tg_{\mu\nu})+\Lambda g_{\mu\nu}.
$$
둘째 식은 대각합을 뒤집어 정리한 형태입니다. 물질 텐서가 간단할 때 편리합니다. 차원을 바꾸면 대각합의 계수도 바뀝니다.

## 3. 완전 유체의 정지계에서 계산하기

한 사건의 정규직교 기저에서 에너지 밀도 $\rho$와 등방 압력 $p$를 같은 단위로 표시합니다. 아래 첨자를 쓴 응력 텐서의 대각 성분은 $(\rho,p,p,p)$이고 $T=-\rho+3p$입니다. 따라서 다음과 같습니다.
$$
R_{\hat0\hat0}=\tfrac{\kappa}{2}(\rho+3p)-\Lambda,\qquad
R_{\hat{i}\hat{i}}=\tfrac{\kappa}{2}(\rho-p)+\Lambda
$$
공간 방향 $i$마다 성립하며 $i$에 대해 합하지 않습니다. 에너지 밀도와 압력이 함께 곡률에 기여합니다.

## 4. 먼지·복사·진공 비교하기

$p=0$, $\Lambda=0$인 먼지 모형의 스칼라 곡률은 $\kappa\rho$입니다. $p=\rho/3$인 복사는 대각합이 0이므로 $\Lambda=0$에서 $R=0$입니다. 이때 리치 텐서의 시간 성분은 $\kappa\rho$입니다. $\Lambda=0$인 진공에서는 리치 텐서 전체가 0입니다. 구대칭 질량의 외부와 같이 리만 곡률은 남을 수 있습니다. 수축에서 사라진 정보는 대각합으로 복원할 수 없습니다.

## 5. 일관성 조건과 예제 검산하기

수축한 비앙키 항등식으로 $\nabla^\mu G_{\mu\nu}=0$입니다. $\Lambda$가 상수이고 계량이 공변 미분에서 보존되면 $\nabla^\mu T_{\mu\nu}=0$도 얻습니다. 이는 물질의 시간 변화를 제약합니다. 시공간 자체를 예측하려면 물질 모형과 적절한 초기 또는 경계 조건도 필요합니다.

```python
from math import isclose

def curvature(rho,p,lam):
    return 4*lam+rho-3*p, (rho+3*p)/2-lam, (rho-p)/2+lam

assert curvature(3,0,0)[0] == 3
assert curvature(3,1,0) == (0,3,1)
for rho,p,lam in [(3,1,0),(2,0,1),(0,0,0.5)]:
    scalar, time, space = curvature(rho,p,lam)
    assert isclose(scalar,-time+3*space)
print(curvature(3,1,0))
```

## 6. 계산으로 결정한 범위 점검하기

- 텐서의 대각합을 구할 때 계량 부호를 적용하세요.
- 밀도를 고정하고 압력을 바꾸어 $R$과 $R_{\hat0\hat0}$를 비교하세요.
- 우주 상수를 더했을 때 두 성분과 스칼라가 얼마나 이동하는지 확인하세요.
- 계산이 스칼라를 정하는지, 한 사건의 리치 텐서를 정하는지, 시공간 해를 구하는지 명시하세요.
