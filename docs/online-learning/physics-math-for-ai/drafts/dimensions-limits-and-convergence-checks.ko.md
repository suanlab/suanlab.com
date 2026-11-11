## 1. 정확히 풀 수 있는 기준 문제 고르기

단위 질량 조화진동자는 $\ddot x=-\omega^2x$를 따릅니다. 길이 단위를 정해 $x(0)=1$, $\dot x(0)=0$으로 두면 정확한 해는 $x(t)=\cos(\omega t)$, $v(t)=-\omega\sin(\omega t)$입니다. 시뮬레이터를 검사할 궤적, 보존 에너지, 시간 척도를 모두 알 수 있습니다. [Hairer의 기하 수치 적분 연구](https://access.archive-ouverte.unige.ch/access/metadata/b8e62cf8-dac5-4a29-880a-7e34961ecffb/download)

## 2. 수치를 비교하기 전에 차원 확인하기

$\tau=\omega t$, $p=v/\omega$로 두면 $x$와 $p$의 단위가 같습니다. 식은 $dx/d\tau=p$, $dp/d\tau=-x$가 됩니다. 물리 시간 간격 $\Delta t$는 무차원 간격 $h=\omega\Delta t$로 바뀝니다. 시간 단위를 바꾸면 주파수와 시간 간격의 수치도 함께 바꿔 곱을 유지합니다.

## 3. 수치 에너지의 변화 유도하기

명시적 오일러 방법은 두 변수를 이전 상태에서 함께 갱신합니다.

$$
x_{n+1}=x_n+hp_n,\qquad p_{n+1}=p_n-hx_n.
$$

$E_n=(x_n^2+p_n^2)/2$로 두고 제곱을 전개하면 교차항이 없어져 $E_{n+1}=(1+h^2)E_n$입니다. 0이 아닌 간격에서는 매번 에너지가 증가합니다. $N$단계 뒤의 비율은 $(1+h^2)^N$입니다.

## 4. 같은 물리 시간에서 수렴 확인하기

최종 무차원 시간을 $T=4$, 간격을 $h=T/N$으로 둡니다. 그러면 $N\to\infty$에서 $E_N/E_0=(1+T^2/N^2)^N\to1$입니다. $N$을 고정하고 $h$만 바꾸면 최종 시간도 달라집니다. 에너지와 궤적 오차를 같은 시간에서 비교합니다.

## 5. 독립적인 식으로 검산하기

```python
from math import cos, hypot, isclose, sin

T = 4.
previous_error = None
for steps in (20, 40, 80, 160):
    h = T/steps
    x, p = 1., 0.
    for _ in range(steps):
        x, p = x+h*p, p-h*x
    energy_ratio = x*x+p*p
    assert isclose(energy_ratio, (1+h*h)**steps, rel_tol=1e-12)
    error = hypot(x-cos(T), p+sin(T))
    if previous_error is not None:
        assert error < previous_error
    previous_error = error
    print(steps, energy_ratio, error)
```

## 6. 검증 조건표 만들기

초기 조건, 단위, 보존량, 정확히 풀 수 있는 극한, 간격 세분화 계획을 기록합니다. 예상하지 못한 드리프트가 선택한 방법의 성질인지 단위나 갱신 순서의 오류인지 구분합니다. 학습한 시뮬레이터에서는 같은 물리 조건에서 장시간 보존량, 상태 오차, 학습 범위 밖 시간 간격을 비교합니다.
