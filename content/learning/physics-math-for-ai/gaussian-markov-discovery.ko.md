## 독립성에서 그래프 집합으로

공분산은 통계적 관계를 나타낸다. 인과 방향을 찾으려면 이 관계를 구조 그래프와 연결하는 가정이 필요하다. 인과 Markov 조건은 그래프의 d-separation이 분포의 조건부 독립성을 뜻한다고 한다. Faithfulness는 역으로, 경로 계수의 상쇄 같은 이유로 추가 독립성이 생기지 않는다고 요구한다. 예측 적합도가 좋다는 사실만으로 이 가정들이 성립하지는 않는다.

실습은 관측 변수 A,B,C, 비순환 선형 Gaussian 구조식, 독립 단위 분산 innovation, 숨은 공통 원인과 선택의 부재를 가정한다. 세 노드의 DAG 25개를 모두 비교한다. 표본 검정 오차까지 보정한 유한 자료 발견 알고리즘은 아니다.

## 네 생성 구조

Chain A→B→C, fork A←B→C, collider A→B←C, 그리고 A→C를 더한 triangle을 선택한다. a,b는 양수다. Triangle은

$$
A=E_A,\quad B=aA+E_B,\quad C=cA+bB+E_C.
$$

각 모형은 독립 innovation의 선형 변환 T이므로 Sigma=T Tᵀ다. 그림은 선언한 생성 그래프다. 특수한 계수에서 비활성화될 수 있는 허용 경로도 포함한다. 관측 공분산으로 이미 입증한 구조를 그린 것이 아니다.

## Gaussian 조건부 독립

양의 정부호 Gaussian 모형에서는 상관 0이 주변 독립, partial correlation 0이 해당 조건부 독립을 뜻한다. B를 조건으로 주면

$$
\Sigma_{AC\mid B}=\Sigma_{AC}-\Sigma_{AB}\Sigma_{BC}/\Sigma_{BB}.
$$

두 조건부 분산의 곱의 제곱근으로 나누어 partial correlation을 얻는다. 모든 변수 쌍에 대해 무조건·남은 변수 조건을 검사하므로 질의는 여섯 개다. 정확한 모집단 계산에서 수치 허용오차 1e-9로 0을 표시한다. 이는 유의수준이 아니며 실제 자료의 작은 상관이 인과 부재를 증명한다는 뜻도 아니다.

## 후보 두 종류 구분

각 DAG의 모든 d-separation을 열거한다. Markov-CI 양립 후보는 그래프가 함의하는 독립성이 공분산에서도 나타나야 한다. Faithful-CI 양립 후보는 독립성 패턴이 정확히 같아야 한다. 완전 DAG는 독립성을 함의하지 않아 추가 독립성이 있는 자료에도 Markov 후보로 남을 수 있다.

Chain과 같은 skeleton의 두 비-collider 방향은 A와 C가 B 조건에서 독립이라는 패턴을 공유한다. 충실한 후보가 세 개 남는 Markov 동등성이다. 방향을 바꾸어 동일 공분산을 표현할 때 계수와 잡음 분산도 다시 정할 수 있다. 모든 방향에서 같은 단위 잡음 계수를 그대로 쓴 공분산이 항상 같다는 뜻은 아니다.

## 개입을 비교하기 전에 관측 공분산을 맞추기

Markov 동등한 방향도 계수·잡음 분산을 바꾸면 같은 관측 Gaussian 법칙을 표현할 수 있다. 독립 단위 분산 잡음으로 chain A=E_A,B=A+E_B,C=B+E_C를 만들면 공분산은 [[1,1,1],[1,2,2],[1,2,3]]이다.

Fork에서는 B를 분산 2인 독립 root로 두고 A=B/2+N_A, Var(N_A)=1/2, C=B+N_C, Var(N_C)=1로 둔다. 독립 innovation의 공분산을 계산하면 정확히 같아 중심 Gaussian 관측 법칙 전체가 일치한다. 조건부 regression E[C|A=a]=a도 같다.

그러나 do(A=a)에서는 chain만 B를 통해 a를 C로 전달한다. Fork는 root B를 바꾸지 않는다. 개입이 관측상 동등한 두 모형을 구분하는 예다.

| 고정 A | Chain의 do(A) 아래 C 평균 | Fork의 do(A) 아래 C 평균 |
|---|---:|---:|
| −1 | −1 | 0 |
| 0 | 0 | 0 |
| 1 | 1 | 0 |

```python
from fractions import Fraction as F
chain=[[F(1),0,0],[F(1),1,0],[F(1),1,1]]
fork=[[F(1,2),1,0],[F(1),0,0],[F(1),0,1]]
def covariance(T,variances):return [[sum(T[i][k]*T[j][k]*variances[k] for k in range(3)) for j in range(3)] for i in range(3)]
a=covariance(chain,[F(1)]*3);b=covariance(fork,[F(2),F(1,2),F(1)])
assert a==b==[[1,1,1],[1,2,2],[1,2,3]]
for query in (-1,0,1):
    conditional=a[2][0]/a[0][0]*query
    chain_do=F(query);fork_do=F(0)
    assert conditional==chain_do
    print(query,chain_do,fork_do)
```

같은 CI 패턴의 그래프 두 개를 나열한 데 그치지 않고 모집단 법칙을 직접 일치시켰다. 방향마다 innovation 분산을 모두 1로 고정하면 추가 식별 제약을 도입할 수 있다는 설명이기도 하다. 화면은 선언한 1e-9 수치 zero 규칙 아래 25개 DAG의 CI 패턴을 비교하며 다른 잡음 분산 적합이나 실제 개입 실험을 수행하지 않는다.

실제 개입은 대상·배정·유지 기제가 정당화될 때만 방향 정보를 더한다. Faithfulness 실패·숨은 원인·선택·피드백은 CI 후보의 해석을 무효화할 수 있다. 유한 표본의 거의 0인 상관에는 정확한 공분산 비교 외에 별도 추론 문제도 남는다.

## 확인 문제와 풀이

Collider에서 a=b=1로 두고 A,C의 주변 상관과 B 조건 상관을 비교하라.

**풀이.** A,C는 독립 innovation이므로 주변 상관은 0이다. B=A+C+E_B를 조건으로 삼으면 공통 결과를 통해 연관된다. 조건부 공분산은 -1/3, 두 조건부 분산은 각각 2/3이어서 partial correlation은 -1/2다. 선언한 가정 아래 이 패턴의 충실한 세 노드 후보는 collider 하나다. 인과 충분성·Markov·faithfulness 조건이 빠지면 같은 결론을 보장하지 않는다.

## 상쇄가 역추론을 깨뜨리는 예

Triangle에서 a=b=1,c=-1로 두면 C=E_B+E_C다. 선언한 직접·간접 경로가 있어도 A와 C는 주변 독립이다. B 조건 상관은 -1/sqrt(3)이다. 독립성 패턴은 collider와 닮지만 실제 선언한 triangle은 비충실하다. Faithfulness를 무조건 강제하면 여기서는 잘못된 생성 구조를 고른다.

## 연구와 연결하기

독립성 기반 발견은 단일 그래프보다 동등류를 줄 수 있다. 개입, 비-Gaussian 모형, 함수 제약은 추가 정보를 제공하지만 별도의 정당화가 필요하다. 유한 표본의 검정 오차·다중 비교·상쇄 근처 불안정성, 숨은 원인·피드백·선택도 검토하자. 발견한 화살표를 확립된 기제라고 제시하기 전에 가정과 후보 집합을 보고해야 한다.

## 원전

Peters·Janzing·Schölkopf, [Elements of Causal Inference](https://web.math.ku.dk/~peters/elements.html), 2017. 공분산 예제와 후보 전수 비교는 직접 구현했다.
