## Cell Average는 보존되는 내용을 나타낸다

[0,1]에서 양의 속도 c인 u_t+c u_x=0을 생각합시다. 유한체적 상태는 점 표본도 cell 질량 자체도 아닌 cell average입니다. 폭 dx를 곱해야 내용량이 됩니다. 초기 사각 pulse [1/4,1/2]를 cell과의 정확한 겹침 길이로 초기화해 경계가 정렬되지 않아도 질량이 A/4입니다.

파랑·주황 stem은 같은 물리 시간의 계산 average와 정확히 이동한 pulse의 average입니다. 확률질량이 아닙니다. 정확한 profile을 각 cell에서 적분하여, 분해하지 못한 점값과 수치 오차를 섞지 않습니다.

## 공유 Flux에서 갱신을 유도한다

c>0인 upwind flux는 계면 왼쪽 상태를 씁니다. 이웃 cell은 같은 계면 flux를 공유합니다.

$$
u_i^{n+1}=u_i^n-\frac{\Delta t}{\Delta x}
(F_{i+1/2}-F_{i-1/2}),\qquad
F_{i+1/2}=c u_i^n.
$$

Nu=CFL=c dt/dx이면 u_new=(1-nu)u_i+nu u_left입니다. 전체 내용량을 더하면 내부 계면 기여가 상쇄됩니다. 이 telescoping이 이산 보존의 원리이며 개별 cell 값이 일정해야 한다는 뜻은 아닙니다.

## 경계가 전체 질량 변화를 정한다

Periodic에서는 첫 cell의 왼쪽 이웃을 마지막 cell로 두어 순 경계 flux가 0입니다. Inflow에서는 왼쪽에 상수 g를 주고 오른쪽 outflow는 마지막 내부 값을 사용합니다.

$$
M^{n+1}-M^n=\Delta t(cg-cu_{N-1}^n).
$$

실험은 매 step의 실제 경계 유입을 누적해 M-M0와 비교합니다. 최종 질량만 보면 올바른 열린 경계 계산을 보존 위반으로 오해할 수 있습니다. 초기값·inflow 데이터·outflow 관례는 서로 다른 명세입니다.

## 보존과 안정성은 다르다

CFL이 0부터 1이면 비음수 이웃 값의 convex combination이라 positivity와 maximum bound를 유지합니다. 1을 넘으면 합은 여전히 상쇄되어도 진동·음수·진폭 증가가 생길 수 있습니다. 작은 질량 잔차만으로 정확도나 안정성을 주장하면 안 됩니다.

CFL=1의 periodic 갱신은 배열을 정확히 한 cell 옮깁니다. 비교 기준도 cell average이므로 보통 산술 오차를 제외하면 일치합니다. 더 작은 CFL의 일차 해법은 날카로운 경계를 퍼뜨려, 질량이 맞아도 수치 확산 오차가 남습니다.

## L1 수축과 수치 확산의 무작위 보행 표현

0≤ν≤1인 periodic upwind 두 해의 차이는 eᵢⁿ⁺¹=(1−ν)eᵢⁿ+νeᵢ₋₁ⁿ를 만족한다. 삼각부등식과 index 이동으로 ΔxΣ|eᵢⁿ⁺¹|≤ΔxΣ|eᵢⁿ|를 얻는다. 같은 inflow 데이터를 주면 outflow 손실과 함께 같은 논증이 성립한다. 따라서 질량 보존뿐 아니라 L1 수축이다. CFL이 1을 넘으면 비음수 가중치 증명은 실패하지만 flux 상쇄는 유지된다.

무한 격자의 단위 cell pulse를 반복 갱신하면 이항 가중치가 나온다. 매 step에서 1−ν의 가중치로 제자리에 남고 ν로 오른쪽 한 cell 이동한다. 정규화된 비음수 가중치의 표현이며 임의 cell average를 확률로 재해석하는 것은 아니다. n회 뒤 평균 물리 이동은 nνΔx=ct, 분산은 nν(1−ν)Δx²다. 분산 2Dt와 비교하면

$$
D_{\mathrm{num}}=\frac{c\Delta x}{2}(1-\nu).
$$

시간 이산화의 uₜₜ 항까지 포함한 Taylor 전개도 같은 계수를 준다. ν=1에서는 계수가 0이고 정확한 cell 이동이 된다. 물리 시간과 CFL을 고정해 격자를 줄이면 선도 수치 확산이 줄어든다. 불연속 근처의 점별 이차 오차 정리라는 뜻은 아니다.

![무한 격자에서 CFL 이분의 일로 계산한 세 단계 수의 이항 upwind pulse 가중치와 퍼짐.](/assets/learning/differential-binomial-ko.svg)

| n, CFL=1/2 | 평균 cell 이동 | cell 단위 분산 |
|---|---:|---:|
| 4 | 2 | 1 |
| 8 | 4 | 2 |
| 16 | 8 | 4 |

```python
from fractions import Fraction as F
from math import comb
nu = F(1, 2)
for n in (4, 8, 16):
    weights = [F(comb(n, j))*nu**j*(1-nu)**(n-j) for j in range(n+1)]
    mass = sum(weights)
    mean = sum(j*w for j, w in enumerate(weights))
    variance = sum((j-mean)**2*w for j, w in enumerate(weights))
    assert mass == 1 and mean == n*nu and variance == n*nu*(1-nu)
    print(n, mass, mean, variance)
```

유리수로 질량·평균·분산을 실제 유한 영역 solver와 독립 검산한다. 그림의 가로축은 평균으로부터의 cell 거리이고 step이 많을수록 퍼진다. Periodic 감김과 inflow 경계에서는 해석을 따로 해야 한다. 공유 flux가 수지를 인증하는 것과 안정성 증명이 별개이므로 CFL>1의 보존적 계산도 실패할 수 있다.

## 직접 풀기: 물리 시간을 맞춰 비교하기

dt=CFL dx/c이므로 n회 뒤 시간은 n CFL/(Nc)입니다. 고정 update 수에서 N을 바꾸면 다른 시간을 보게 됩니다. N=12, c=1, CFL=0.8, n=8이면 t=8/15입니다. Periodic 질량 1/4을 확인하고 inflow로 바꾸어 질량 변화가 누적 경계 flux와 맞는지 검산하세요. CFL=1.2에서는 overshoot나 음수 cell이 생겨도 질량 수지가 맞을 수 있음을 설명하세요.

더 읽기: [Clawpack 보존 법칙·Riemann 체계](https://www.clawpack.org/riemann_book/html/Burgers.html).
