## 합을 보존하는 것과 모두 같아지는 것

여섯 node가 초기값 [x₀,1,2,3,4,5]를 나누어 갖습니다. 모두가 전체 데이터를 모으지 않고 이웃과만 통신해 평균을 구하려면 그래프 연결성, 갱신 방식과 메시지 비용을 명시해야 합니다. 값의 총합이 보존돼도 서로 분리된 성분은 다른 값에 머물 수 있습니다. 이 실험은 ring, path, 두 성분, complete, empty 망과 고정된 첫 edge 제거를 비교합니다.

위 그림은 현재 node 값과 실제 남은 edge입니다. 파랑은 전체 max-minus-min, 주황은 현재 전역 평균과 초기 평균의 차이입니다. 주황이 0이라는 사실만으로 파랑도 0이라고 결론낼 수 없습니다. 표의 componentwise limit는 고정 망·현재 갱신 규칙 아래의 극한이며 현재 유한 step 결과와 다릅니다.

## Synchronous Metropolis 평균

각 active edge의 weight와 diagonal을 다음처럼 정합니다.

$$
W_{ij}=\frac1{1+\max(d_i,d_j)},\qquad
W_{ii}=1-\sum_{j\ne i}W_{ij},\qquad x^{k+1}=Wx^k.
$$

대칭 W는 행 합과 열 합이 모두 1이므로 전체 합을 보존합니다. Diagonal이 양수라 주기성 문제도 피합니다. 고정된 연결 그래프에서는 반복 평균이 전역 초기 평균에 수렴합니다. 분리된 망에서는 block마다 자기 초기 합을 보존하므로 각 component의 평균으로 갑니다. 이 수렴 결론은 임의의 시간 가변 실패나 악의적인 메시지에 대한 보장이 아닙니다.

## Pair Gossip의 실제 교환

다른 모드에서는 active edge를 순환 순서로 하나씩 골라 그 두 끝점만 평균냅니다. 반드시 이전 두 값을 함께 읽고 동시에 바꿉니다.

$$
x_i^{new}=x_j^{new}=\tfrac12(x_i^{old}+x_j^{old}),\qquad
V_{new}=V_{old}-\tfrac12(x_i-x_j)^2.
$$

V는 고정 평균에 대한 제곱 오차 합입니다. 각 pair는 합을 보존하고 disagreement를 줄입니다. 고정된 연결 망에서 모든 edge가 반복적으로 선택되는 fairness가 필요합니다. 현재 순환 규칙은 이 유한 edge 집합을 빠뜨리지 않습니다. 아직 몇 번만 실행했다면 정확한 평균에 도달했다고 주장하지 않습니다.

## 올바른 불변 평균이 있어야 Spectral 수축을 쓴다

고정 대칭 stochastic W에서 x를 평균 Jx와 disagreement e=(I−J)x로 나누자. J=11ᵀ/N이다. 연결 망과 양의 diagonal에서는 상수 mode만 고유값 1이고 나머지 절댓값은 1보다 작다. 그 최댓값이 ρ이면

$$
\|x^k-Jx^0\|_2\le\rho^k\|x^0-Jx^0\|_2.
$$

통신 round당 오차 bound다. Ring과 complete는 round당 메시지 수가 다르다. 여섯 node ring의 고유값은 1/3+(2/3)cos(2πj/6)이므로 ρ=2/3이다. 두 triangle은 각 component가 평균에 도달해도 전체 component 차이의 고유값 1을 남긴다.

![고정된 세 6-node 망의 정확 유리수 평균 갱신에서 directed 메시지 수에 따른 disagreement를 비교하는 그림.](/assets/learning/algorithm-gossip-ko.svg)

| 망 | 전체 Disagreement 배율 ρ | Round당 Directed Message |
|---|---:|---:|
| Ring | 2/3 | 12 |
| Complete | 0 | 30 |
| 두 Triangle | 1 | 12 |

```python
from fractions import Fraction as F
initial=list(map(F,range(6)));mean=sum(initial)/6
ring=[[F(int(i==j or (i-j)%6 in (1,5)),3) for j in range(6)] for i in range(6)]
x=initial[:];energy=lambda v:sum((a-mean)**2 for a in v)
start=energy(x)
for k in range(1,11):
    x=[sum(a*b for a,b in zip(row,x)) for row in ring]
    assert sum(x)==sum(initial) and energy(x)<=F(4,9)**k*start
print('ring after 120 messages:',float(energy(x)))
W=[[F(9,10),F(1,10)],[F(2,5),F(3,5)]]
z=[F(0),F(1)];weights=[F(4,5),F(1,5)]
for _ in range(20):
    z=[sum(a*b for a,b in zip(row,z)) for row in W]
    assert sum(a*b for a,b in zip(weights,z))==F(1,5)
print('directed values:',list(map(float,z)))
```

행 확률성만으로는 상수 벡터를 보존하지만 산술평균은 아니다. 방향 두-state 행렬 [[0.9,0.1],[0.4,0.6]]의 정상 left weight는 (0.8,0.2)다. (0,1)에서 consensus는 0.5 대신 0.2로 간다. 보조 유리수 예제는 이 가중 불변량을 검산하고 live 망은 무방향으로 유지한다. 시간 가변·비동기·Byzantine에는 다른 가정이 필요하며 고정 대칭 spectrum으로 인증할 수 없다.

## 메시지 회계와 손계산

Synchronous round는 active edge마다 양방향 scalar 하나씩, pair step은 선택된 edge에서 양방향 하나씩 교환한다고 가정합니다. Payload는 double 8 byte로 고정하고 header·동기화·망 지연은 제외합니다. Round 수가 같다고 통신량이 같은 것이 아닙니다.

Complete 망에서는 degree 5, 모든 weight 1/6이므로 [0,1,2,3,4,5]가 한 round에 모두 2.5가 됩니다. Edge 15개에서 directed message 30개, payload 240 byte입니다. 분리된 두 triangle에서는 극한이 [1,1,1,4,4,4]이고 평균은 여전히 2.5이지만 전체 spread는 3입니다.

직접 풀기: ring에서 30회 synchronous round와 30회 pair step의 메시지 수를 구하세요. 풀이: ring edge가 6개이므로 전자는 30·12=360개, 후자는 30·2=60개입니다. 같은 수의 step에서 나온 spread를 동일한 통신 예산의 비교로 해석하면 안 됩니다.

## 분산 AI 연구의 보고 기준

Distributed training과 decentralized learning에서는 local 계산량, 참여 graph, payload, round 정의, 실패·지연 가정을 함께 기록하세요. 현재 모형은 고정된 무방향 망의 정확한 scalar averaging입니다. Byzantine consensus나 일반 asynchronous protocol의 안전성을 구현했다고 부르지 않습니다. 합 보존·연결성·수렴 조건·유한 관측 오차·통신 예산을 각각 확인하는 출발점입니다.

참고: [MIT 병렬·분산 성능 공학](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
