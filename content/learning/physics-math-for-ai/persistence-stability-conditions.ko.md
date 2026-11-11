## 안정성은 가정을 갖춘 정량적 주장입니다

점이 조금 움직여 보인다는 말에는 거리·대응·filtration을 먼저 정해야 합니다. 기준 정사각형 P에서 꼭짓점 2만 scale·(δ,−δ)만큼 옮겨 Q를 만듭니다. 최대 이름별 변위는 η=√2·scale·|δ|입니다.

쌍별 거리 변화는 두 끝점의 이동 합 이하입니다. 실제 최대 거리 왜곡 D를 정의하면 다음과 같습니다.

$$
D=\max_{i,j}|d_P(i,j)-d_Q(i,j)|\le2\eta.
$$

2η는 편리하지만 느슨할 수 있습니다. 수치 표는 이 상계와 실제 왜곡을 구별합니다.

## 공통 이름이 Filtration 포함을 만듭니다

두 Rips filtration은 같은 추상 꼭짓점 집합을 씁니다. simplex의 값은 최대 간선 거리이므로 역시 D 이내로 변합니다. 꼭짓점의 항등 대응으로 양방향 포함을 얻습니다.

$$
\mathrm{VR}_\varepsilon(P)\subseteq\mathrm{VR}_{\varepsilon+D}(Q),\qquad
\mathrm{VR}_\varepsilon(Q)\subseteq\mathrm{VR}_{\varepsilon+D}(P).
$$

체 위의 이 유한 filtration에는 persistence 안정성으로 bottleneck 거리가 D 이하임을 얻습니다. 공통 복합체의 명시적 대응 논증이며, 서로 무관한 점 집합에 함수 sup-norm 정리를 가정 없이 적용한 것이 아닙니다.

## 대각선을 포함해 실제 Matching 계산하기

Persistence diagram은 중복을 허용한 birth/death 쌍입니다. bottleneck matching은 최대 L∞ 비용을 최소화하며 feature를 대각선에 보낼 수도 있습니다.

$$
\|(b,d)-(b',d')\|_\infty
=\max(|b-b'|,|d-d'|),\qquad
\operatorname{dist}_\infty((b,d),\Delta)=\frac{d-b}{2}.
$$

구현은 대각선 복사본까지 포함한 유한 assignment 상태로 정확한 matching을 풉니다. 무한 H₀는 양쪽 모두 birth 0이라 비용 0으로 맞고 유한 H₀·H₁ 거리는 따로 계산합니다. δ=0.25, scale=1에서는 H₁ 점끼리 직접 맞추는 것보다 둘 다 대각선으로 보내는 편이 낫습니다. H₁ 거리는 (√2−1)/2≈0.2071입니다.

## 가까운 Diagram도 Threshold Betti는 다를 수 있습니다

δ=0.1, ε=1.05에서 기준 정사각형에는 이미 cycle이 있지만 변형 사각형의 마지막 변은 아직 들어오지 않았습니다. β₁은 1과 0이며 H₁ bottleneck 거리는 약 0.1045입니다. 생성·소멸 위치의 안정성이 birth 근처에서 읽는 indicator의 연속성을 뜻하지는 않습니다.

## Bottleneck 보증이 실제로 보존하는 것

![정사각형의 세 변형에서 H1 구간은 가깝게 유지되지만 임계값 epsilon 1.05에서는 생존 판정이 달라집니다.](/assets/learning/geometry-persistence-ko.svg)

각 simplex에서 최대 간선 길이의 차이는 최대 쌍별 왜곡 D 이하입니다. 양쪽 포함 사상은 filtration을 D만큼 옮기고, 합성하면 2D만큼 옮기는 원래 구조 사상입니다. Homology를 적용하면 D-interleaving이 됩니다. 이 유한 persistence module의 대수적 안정성 정리를 적용해 비용 D 이하인 diagram matching을 얻습니다. 마지막 단계는 module에 대한 정리이며 그림의 점들이 가까워 보인다는 사실만으로 나오지 않습니다. [Chazal·de Silva·Glisse·Oudot](https://arxiv.org/abs/1207.3674)를 참고하세요.

수명이 2D보다 긴 bar는 대각선까지 거리가 수명 절반이므로 그 비용 안에서 대각선에 대응될 수 없습니다. 실제 bar와 짝지어져야 합니다. 끝점 이동이 D 이하인 쌍에서 ε가 기준 bar의 양 끝점으로부터 D보다 멀면 해당 feature의 생존 여부는 같습니다. 이는 짝지은 feature별 명제이며 짧은 미대응 feature가 전체 Betti 수를 바꿀 수는 있습니다.

두 diagram에 유한 H₁ bar가 하나씩이면 정확 bottleneck 거리는 직접 끝점 matching과 두 bar 모두 대각선으로 보내는 비용 중 작은 값입니다. 코드는 persistence 라이브러리나 assignment solver 없이 변형 사각형의 이 공식을 독립 계산합니다.

| 변형 δ | 기준 H₁ | 변형 H₁ |
|---:|---|---|
| 0 | [1,√2) | [1,√2) |
| 0.1 | [1,√2) | [√1.22,√2) |
| 0.25 | [1,√2) | [√1.625,√2) |

## Python으로 계산 재현하기

```python
from math import hypot, sqrt
for delta in [0, .1, .25]:
    P = [(0,0),(1,0),(1,1),(0,1)]
    Q = [(0,0),(1,0),(1+delta,1-delta),(0,1)]
    birth = max(hypot(Q[(i+1)%4][0]-Q[i][0],Q[(i+1)%4][1]-Q[i][1]) for i in range(4))
    death = min(hypot(Q[2][0]-Q[0][0],Q[2][1]-Q[0][1]), sqrt(2))
    distortion = max(abs(hypot(P[i][0]-P[j][0],P[i][1]-P[j][1])-hypot(Q[i][0]-Q[j][0],Q[i][1]-Q[j][1])) for i in range(4) for j in range(4))
    direct = max(abs(birth-1), abs(death-sqrt(2)))
    diagonal = max((sqrt(2)-1)/2, max(0,death-birth)/2)
    bottleneck = min(direct,diagonal)
    assert bottleneck <= distortion+1e-12
    print(delta, round(birth,9), round(death,9), round(bottleneck,9))
```

## 직접 풀기와 적용 한계

δ=0이면 변위·왜곡·두 diagram 거리가 모두 0입니다. 이어 δ를 고정하고 scale과 ε를 두 배로 하세요. 거리 상계와 유한 persistence 좌표는 두 배이고 재조정한 Betti 패턴은 같습니다. 수명이 bottleneck 오차 두 배보다 긴 feature가 왜 그 오차 안에서 대각선으로 사라질 수 없는지도 설명하세요.

실제 데이터에서는 metric·filtration 관례·계수 체·잡음 모형·대응 또는 표본화 정리를 기록해야 합니다. diagram 안정성은 선택한 구성의 perturbation을 제어하며 미지 생성 시스템의 위상이나 인과적 의미를 자동으로 증명하지 않습니다.

추가 읽기: [Cohen-Steiner, Edelsbrunner and Harer, Stability of Persistence Diagrams](https://scholars.duke.edu/publication/770168).
