## 평균 보장과 모든 입력의 보장은 다르다

무작위 알고리즘은 어떤 확률 공간에서 평균을 취하는지, 근사 알고리즘은 어떤 최적값에 대해 비율을 보장하는지 명시해야 합니다. 현재 여섯 labeled node의 triangle, star, cycle, empty 그래프를 사용합니다. Cut 문제의 무작위 bit 할당과 vertex cover 문제의 maximal matching은 서로 다른 최적화 과제입니다. 하나의 그래프에서 두 종류의 논증을 비교하는 것이 목적입니다.

파랑 PMF는 64개 독립 균일 bit 할당을 정확하게 전수 계산한 분포입니다. 주황은 한 seeded 표본의 점수에 해당하는 PMF 항을 강조합니다. 표본 확률을 하나로 두거나 이 한 표본으로 분포를 추정한 것이 아닙니다. 위 그림은 따로 선택한 cut mask이며 표본 mask와 같을 필요가 없습니다.

## Random Cut의 기대 성능

각 node를 독립적인 공정한 bit로 양쪽에 배치합니다. Edge 양 끝 bit가 다를 확률은 1/2입니다. Edge 사건들이 모두 독립일 필요 없이 기대값의 선형성을 사용할 수 있습니다.

$$
C=\sum_{e=1}^m I_e,\qquad E[C]=m/2\ge \mathrm{OPT}_{cut}/2.
$$

따라서 기대값에서 1/2 근사를 얻지만 모든 표본이 최적값의 절반 이상이라는 말은 아닙니다. 예를 들어 triangle에서는 모든 bit가 같은 경우 cut 0이 나옵니다. 높은 확률 보장이나 derandomization에는 별도의 논증이 필요합니다. 현재 seed는 재현용 결정론적 설정이며 허용 seed 집합의 분포를 이론적 균일 bit 분포로 가정하지 않습니다.

## Matching으로 Vertex Cover 만들기

끝점을 공유하지 않는 edge를 greedy로 더해 더 이상 추가할 수 없는 maximal matching M을 만듭니다. Maximum matching과는 다른 개념입니다. M의 모든 끝점을 고른 집합은 cover입니다. 만약 덮이지 않은 edge가 있으면 양 끝이 모두 unmatched여서 M에 추가할 수 있으므로 maximality와 모순입니다.

$$
|M|\le\mathrm{OPT}_{cover},\qquad
|cover|=2|M|\le2\mathrm{OPT}_{cover}.
$$

어떤 cover도 서로 끝점이 분리된 matching edge마다 최소 한 node를 가져야 하므로 첫 부등식을 얻습니다. 실제 matching endpoint pair와 cover bitmask를 readout으로 확인할 수 있습니다. 빈 그래프에서는 최적값도 0이므로 비율은 정의하지 않습니다.

## 손계산과 직접 풀기

Triangle의 cut은 0 또는 2입니다. 세 활성 node의 bit가 모두 같은 두 경우가 8개 할당 중 2개이므로 P(C=0)=1/4, P(C=2)=3/4입니다. 나머지 세 isolated node bit는 비율에 영향을 주지 않습니다. 평균은 1.5이며 edge 수 3의 절반입니다.

직접 풀기: 중앙 하나와 leaf 다섯인 star에서 greedy cover의 비율을 구하세요. 풀이: matching은 edge 하나만 가질 수 있어 양 끝 cover는 2개입니다. 최적 cover는 중앙 1개이므로 비율은 2로 보장 경계에 도달합니다. Random cut의 기대값 2.5와 cover 크기 2는 다른 목적값이므로 직접 비교할 수 없습니다.

## AI 휴리스틱의 평가

학습한 조합 해법에서는 동일한 목적함수·feasibility·baseline·최적값 또는 bound를 사용하세요. 평균 표본 성능, 최악 근사 보장, 실패 확률은 다른 보고 항목입니다. 작은 전수 최적값을 이용한 검산은 중요하지만 그 자체가 큰 입력의 성능 보장은 아닙니다.

참고: [MIT 계산복잡도와 무작위성](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
