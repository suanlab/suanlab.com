## Martingale은 조건부 평균에 대한 명제다

M_n이 주어진 정보 filtration의 martingale이려면 적분 가능·adapted이어야 하고 E[M_(n+1)|F_n]=M_n입니다. 경로가 평평하거나 예측 불가능해 보인다는 뜻이 아닙니다. 한 실현이 크게 움직여도 조건부 평균 관계는 정확할 수 있습니다.

증분이 확률 p로 +1, 나머지로 -1인 walk의 drift는 d=2p-1입니다. 원래 walk는 d=0일 때 martingale이고 예측 가능한 drift를 빼면 다음입니다.

$$
M_n=S_n-dn,\qquad E[\Delta M_n\mid\mathcal F_n]=0.
$$

Poisson 과정의 N(t)-lambda t도 같은 보상 원리입니다. 여기서는 유한 확률 재귀로 정지 기대값을 정확히 계산할 수 있는 walk를 사용합니다.

## 상태뿐 아니라 시계도 멈춘다

Tau를 ±B의 첫 hit와 N의 최솟값으로 둡니다. 유계 stopping time입니다. 중간 관측 상한 n의 stopped 과정은 M_(min(tau,n))이며 보상도 실제 stopped time을 씁니다.

$$
M_{\tau\wedge n}=S_{\tau\wedge n}-d(\tau\wedge n).
$$

상태가 멈췄는데 계속 dn을 빼면 가짜 drift를 만듭니다. 위 그림은 실제 한 stopped 표본이고 아래 그림은 정확한 기대값입니다. 둘은 같은 곡선일 필요가 없습니다.

## 유계 선택적 정지를 직접 검산한다

모형은 위치와 정지시각의 joint 확률을 추적합니다. 살아 있는 상태는 p와 1-p로 갈라지고 멈춘 상태는 시계를 유지합니다. 보상값의 확률 가중합은 반올림을 제외하면 0입니다. 여기서는 유계성이 필요한 적분 조건을 확보하며 모든 무한 stopping time에 정리를 적용하지 않습니다.

p=0.75, B=2, N=4에서 시간 2에 hit할 확률은 p²+(1-p)²=0.625입니다. 나머지의 정지시각은 4라 E[tau]=2.75, E[S_tau]=0.5×2.75=1.375입니다. E[M_tau]는 0입니다.

## 거의 확실히 유한한 정지시각도 실패할 수 있는 이유

0에서 시작하는 공정 walk를 +1 또는 −B에서 멈추자. 정지시각의 기대값은 B로 유한하고 정지 위치는 유계다. Affine 조화 경계 문제를 풀면 p=Pr(+1 탈출)=B/(B+1)이다. 따라서 E[Sτ]=p−B(1−p)=0이며 선택적 정지가 정당하다. 기대 탈출 시간은 이차 함수의 재귀로 구한다.

이제 B를 늘리자. 공정 walk는 +1에 거의 확실히 도달하지만 그 유계가 아닌 hit 시각의 위치는 반드시 1이므로 E[Sτ]=1이다. 아래쪽 희귀 탈출은 확률 1/(B+1), 크기 B로 기대값 기여가 사라지지 않고 −1로 간다. 이 정지 위치열은 균등 적분 가능하지 않아 극한과 기대값을 교환할 수 없다.

$$
E[|S_{\tau_B}|\,\mathbf1_{\{|S_{\tau_B}|>K\}}]
\ge B/(B+1)\quad(B>\max(K,1)).
$$

| B | 위쪽 탈출 확률 | 아래쪽 탈출의 기대값 기여 |
|---|---:|---:|
| 1 | 1/2 | −1/2 |
| 3 | 3/4 | −3/4 |
| 9 | 9/10 | −9/10 |

```python
from fractions import Fraction as F
for B in (1, 3, 9):
    upper = F(B, B+1)
    lower_contribution = -B*(1-upper)
    assert upper+lower_contribution == 0
    # Harmonic probability and expected time for -B <= x <= 1.
    probability = lambda x: F(x+B, B+1)
    duration = lambda x: (x+B)*(1-x)
    for x in range(-B+1, 1):
        assert probability(x)==(probability(x-1)+probability(x+1))/2
        assert duration(x)==1+F(duration(x-1)+duration(x+1),2)
    assert duration(0)==B
    print(B, upper, lower_contribution, duration(0))
```

유리수 조화 계산이 빠진 꼬리 제어를 드러낸다. 실제 양쪽 경계·유계 horizon 재귀는 계속 유효하다. 관측 성능에 따라 선택한 중단 규칙을 무조건 평균 하나로 정당화할 수 없다. 정보·적분 가능성·정지 조건을 함께 확인해야 한다.

## 직접 풀기: Seed와 정리 구분

같은 편향 예제의 seed를 바꾸면 실제 정지 상태가 달라도 확률 재귀·기대값은 같습니다. B=1이면 첫 step에 반드시 멈춥니다. 관측 시간이 8이어도 보상은 d 한 번입니다. 잘못 8d를 빼면 무엇이 달라지고 왜 martingale 정리의 반례가 아닌지 계산하세요.

더 읽기: [Lawler의 Martingale·정지시각 노트](https://www.math.uchicago.edu/~lawler/finbook.pdf).
