## 1. 시간 방향을 가진 쌍선형 형식

관성 좌표 $X=(ct,x,y,z)$에서 $\eta=\operatorname{diag}(1,-1,-1,-1)$로 둡니다. 두 변위의 민코프스키 곱은 $U^\mathsf T\eta V$입니다. 대칭이고 쌍선형이며 비퇴화합니다. 한 변위를 자기 자신과 곱한 값은 양수, 0 또는 음수일 수 있습니다. 로런츠 변환 $\Lambda$는 $\Lambda^\mathsf T\eta\Lambda=\eta$를 만족해 이 곱을 보존합니다. 전체 부호를 반대로 쓰는 문헌도 있습니다. [Minkowski, Space and Time](https://www.gutenberg.org/files/66944/66944-h/66944-h.htm)

## 2. 간격의 부호 읽기

0이 아닌 변위에서 제곱 간격 $s^2$가 양수이면 시간꼴, 0이면 빛꼴, 음수이면 공간꼴입니다. 예를 들어 $c=1$인 단위에서 $(3,1)$, $(2,2)$, $(1,3)$의 제곱 간격은 $8$, $0$, $-8$입니다. 빛꼴 변위는 0이 아닐 수 있습니다. 제곱 간격은 양의 정부호인 유클리드 거리와 다른 성질을 갖습니다.

## 3. 경로를 따라 흐른 시간 더하기

관성계에서 속력이 $v(t)$인 매끄러운 시간꼴 궤적을 따라 움직이는 이상적인 시계는 다음만큼 진행합니다.
$$
d\tau=\sqrt{dt^2-d\mathbf{x}^2/c^2}
      =dt\sqrt{1-v(t)^2/c^2}.
$$
미래 방향의 직선 구간마다 좌표 시간에 제곱근 인자를 곱합니다. 각 구간의 고유 시간을 더하면 시계의 전체 경과 시간을 얻습니다. 빛꼴 구간의 극한 경로 길이는 0입니다. 공간꼴 구간은 시계가 따르는 궤적이 될 수 없습니다. [Tong, Geodesics in Spacetime §1](https://davidtong.org/teaching/general-relativity/grhtml/S1)

## 4. 같은 두 사건을 잇는 경로 비교하기

$c=1$을 쓰고 출발을 $(0,0)$, 재회를 $(6,0)$으로 고정합니다. 정지한 시계는 6을 기록합니다. $(3,2)$를 지나 직선 두 구간으로 돌아온 여행자의 시계는 $2\sqrt{9-4}=2\sqrt5$를 기록합니다. 두 경로는 끝점 간격을 공유하며 경과 시간은 경로에 따라 달라집니다. 순간적으로 방향을 바꾸는 이상화는 짧고 매끄러운 회전 구간으로 근사할 수 있습니다.

## 5. 직선 경로가 고유 시간을 최대로 만드는 이유

출발과 재회 사이 시간꼴 변위의 정지계를 선택합니다. 그 계에서는 두 끝점의 공간 좌표가 같습니다. 구간별로 매끄러운 모든 시간꼴 경로에서 $\sqrt{1-v^2/c^2}$는 1 이하입니다. 따라서 고유 시간은 그 계의 끝점 좌표 시간차를 넘지 않습니다. 등호는 시간이 흐르는 동안 속력이 0일 때 성립합니다. 미래 방향의 시간꼴 구간에서는 이 관계를 역삼각부등식이라고 부릅니다.

$1+1$차원에서 미래 방향 단위 시간꼴 쌍곡선은 $(\cosh\eta,\sinh\eta)$로 나타낼 수 있습니다. 그 위의 두 벡터의 민코프스키 곱은 $\cosh(\eta_1-\eta_2)\geq 1$입니다. 두 벡터의 급속도 차이가 쌍곡각을 정하므로 쌍선형 곱에서도 역부등식의 부호를 확인할 수 있습니다.

## 6. 수치 검산하기

```python
from math import isclose, sqrt

def proper_time(*events):
    elapsed = 0
    for (ct0,x0), (ct1,x1) in zip(events,events[1:]):
        dt, dx = ct1-ct0, x1-x0
        if dt < abs(dx):
            raise ValueError('The leg is spacelike or past-directed')
        elapsed += sqrt((dt-dx)*(dt+dx))
    return elapsed

A, B = (0,0), (6,0)
assert proper_time(A, B) == 6
assert isclose(proper_time(A,(3,2),B), 2*sqrt(5))
assert proper_time(A,(3,3),B) == 0
for t in range(1,6):
    for x in range(-min(t,6-t),min(t,6-t)+1):
        assert proper_time(A,(t,x),B) <= proper_time(A,B)
print(proper_time(A,(3,2),B))
```

## 7. 실험 점검하기

- 출발과 재회 사건을 고정하고 방향을 바꾸는 사건을 옮기세요. 두 경로의 경과 시간을 비교합니다.
- 방향을 바꾸는 시각을 3에 두고 위치를 0에서 3까지 옮기세요. 두 구간이 빛의 원뿔에 접근할 때 어떻게 되는지 확인합니다.
- 방향 전환 사건을 허용 영역 밖으로 옮기세요. 어느 구간이 공간꼴이 되는지 확인한 뒤 시계 읽기를 해석합니다.
