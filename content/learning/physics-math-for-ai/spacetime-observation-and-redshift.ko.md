## 1. 무엇을 측정하는지 정하기

시계는 자기 고유 시간을 기록합니다. 수신기는 그 고유 시간의 단위마다 도착한 빛의 마루를 셉니다. 미래 방향 광자 파동벡터 $k$와 관측자의 사원속도 $u$에 대해 부호 $(-,+,+,+)$에서 측정 각주파수는 $\omega=-g(u,k)$입니다. 이 수축값은 스칼라입니다. 같은 관측자의 읽기는 좌표를 바꿔도 같고, 관측자 자체가 바뀌면 달라질 수 있습니다.

## 2. 구대칭 질량 바깥의 정지 시계 비교하기

$G=c=1$에서 중심 질량을 $M$, 슈바르츠실트 외부 $r>2M$의 함수를 $f(r)=1-2M/r$라 둡니다. 정지 시계는 $d\tau=\sqrt{f(r)}\,dt$를 따릅니다. 시간 병진 대칭으로 광선을 따라 $-k_t$가 보존되므로 정지 관측자는 $\omega=-k_t/\sqrt{f(r)}$를 측정합니다. $r_e$에서 정지 방출하고 $r_o$에서 정지 수신할 때
$$
\frac{\omega_o}{\omega_e}=\sqrt{\frac{f(r_e)}{f(r_o)}}.
$$
$r_o>r_e$이면 비율은 1보다 작습니다. 두 관측자는 각자의 반지름을 고정합니다. [Philipp 외, Redshift and frequency comparison in Schwarzschild spacetime](https://arxiv.org/abs/1711.01237)

## 3. 방출기의 국소 운동 구분하기

바깥으로 향하는 방사 방향 광선을 택합니다. $r_e$에 있는 정지 관측자가 재는 방출기의 국소 속도를 같은 바깥 방향에 대해 $\beta$라 둡니다. 방출기 정지계에서의 주파수는 국소 정지 관측자의 주파수에 $\gamma(1-\beta)$를 곱한 값입니다. 따라서 수신 주파수와 방출 주파수의 비는
$$
\frac{\omega_o}{\omega_e}=
\sqrt{\frac{f(r_e)}{f(r_o)}}\sqrt{\frac{1+\beta}{1-\beta}} .
$$
첫 인자는 두 반지름에 정지한 관측자의 비교에서 오고, 둘째 인자는 방출 지점의 국소 도플러 인자입니다. 방출기가 충분히 빠르게 바깥으로 움직이면 중력 적색편이가 상쇄될 수 있습니다. $z=\omega_e/\omega_o-1$로 정의하면 적색편이에서 양수입니다.

## 4. 상쇄되는 예제 계산하기

$r_e=4M$, $r_o=12M$으로 두면 정지 주파수비는 $\sqrt{3/5}$입니다. $\beta=1/4$에서는 도플러 인자의 제곱이 $5/3$이므로 전체 비율은 1입니다. 양의 속도를 더 키우면 비율은 1을 넘습니다. 정지한 수신기로 가는 방사 방향 광선과 방출 순간에 지정한 방출기 속도를 가정한 계산입니다.

```python
from math import isclose, sqrt

def observed_ratio(emitter, receiver, beta):
    f = lambda r: 1 - 2/r
    return sqrt(f(emitter)/f(receiver) * (1+beta)/(1-beta))

assert observed_ratio(4,12,0) < 1
assert isclose(observed_ratio(4,12,.25), 1)
assert observed_ratio(4,12,.5) > 1
assert observed_ratio(4,12,-.25) < observed_ratio(4,12,0)
print(observed_ratio(4,12,0), observed_ratio(4,12,.25))
```

## 5. 주파수 비교 점검하기

- 방출기를 정지시킨 채 수신 반지름을 늘려 비율과 $z$의 부호를 확인하세요.
- 같은 두 반지름에서 두 효과가 상쇄되도록 방출 속도를 바꾸세요.
- 방출기가 안쪽으로 움직이고 광선은 바깥으로 나갈 때 도플러 보정의 부호를 확인하세요.
- 적색편이 정의를 적용하기 전에 두 주파수를 각각 누가 측정하는지 명시하세요.
