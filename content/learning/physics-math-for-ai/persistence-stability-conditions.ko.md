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

## 직접 풀기와 적용 한계

δ=0이면 변위·왜곡·두 diagram 거리가 모두 0입니다. 이어 δ를 고정하고 scale과 ε를 두 배로 하세요. 거리 상계와 유한 persistence 좌표는 두 배이고 재조정한 Betti 패턴은 같습니다. 수명이 bottleneck 오차 두 배보다 긴 feature가 왜 그 오차 안에서 대각선으로 사라질 수 없는지도 설명하세요.

실제 데이터에서는 metric·filtration 관례·계수 체·잡음 모형·대응 또는 표본화 정리를 기록해야 합니다. diagram 안정성은 선택한 구성의 perturbation을 제어하며 미지 생성 시스템의 위상이나 인과적 의미를 자동으로 증명하지 않습니다.

추가 읽기: [Cohen-Steiner, Edelsbrunner and Harer, Stability of Persistence Diagrams](https://scholars.duke.edu/publication/770168).
