## 위상은 어떤 구별을 열린집합으로 볼지 정합니다

Euclidean 공간에서는 거리로 연속성을 정의하는 데 익숙합니다. 위상은 거리 없이도 이웃의 개념을 유지합니다. X의 위상은 열린집합이라 부르는 부분집합들의 모음이며 공집합과 X를 포함하고, 임의 합집합과 유한 교집합에 닫혀 있어야 합니다. 유한 공간에서는 이 정의 전체를 직접 열거할 수 있습니다.

여기서는 X={0,1,2}, Y={0,1}입니다. bit mask로 부분집합을 표시합니다. 원공간 mask 5는 {0,2}, 목표공간 mask 1은 {0}입니다. 그림의 점 위치는 원소 이름을 배치한 것일 뿐 거리나 이웃 관계가 아닙니다.

## 모든 열린 역상을 확인하기

연속 함수의 정의는 다음과 같습니다.

$$
f\text{ continuous}\quad\Longleftrightarrow\quad
f^{-1}(V)=\{x\in X:f(x)\in V\}\text{ is open for every open }V\subseteq Y.
$$

indiscrete 원공간의 열린 mask는 0과 7뿐입니다. 함수 mask 1은 f(0)=1, f(1)=f(2)=0을 뜻합니다. 목표가 discrete이면 두 singleton이 모두 열려 있고 역상은 mask 1과 6입니다. 둘 다 원공간에서 열리지 않으므로 연속이 아닙니다. stem 그림이 매끄러워 보이는지와 무관한 정확한 판정입니다.

상수 함수의 역상은 공집합 또는 전체이므로 어떤 원공간에서도 연속입니다. 반대로 discrete 원공간은 모든 부분집합이 열려 있어 모든 함수가 연속입니다. indiscrete 목표공간으로 가는 함수도 모두 연속입니다. 양쪽 위상을 먼저 정해야 하는 이유입니다.

## 연결성은 분리가 가능한지 묻습니다

연결 공간은 두 개의 서로소인 비어 있지 않은 열린 부분으로 나눌 수 없습니다. 동등하게 열린 동시에 닫힌 clopen 집합은 공집합과 전체뿐입니다. 이 유한 예제에서는 clopen 집합을 모두 열거하고 어떤 clopen 집합으로도 분리되지 않는 원소들을 같은 성분으로 묶습니다.

$$
X=A\sqcup B,\quad A,B\ne\varnothing,\quad A,B\text{ open}
\quad\Longrightarrow\quad X\text{ disconnected}.
$$

indiscrete 원공간의 연결 성분은 1개, 세 점 discrete 공간은 3개입니다. 열린 mask가 {0,4,6,7}인 chain 위상은 indiscrete보다 열린집합이 많아도 연결입니다. 열린집합 개수만으로 연결 성분 수를 정할 수 없습니다.

## 유한 연결 성분 알고리즘이 맞는 이유

유한 공간에서는 x를 포함하는 열린 이웃을 모두 교차해 최소 열린 이웃 U(x)를 얻습니다. y가 U(x)에 속하거나 그 반대이면 무방향 간선을 그립니다. 열린 동시에 닫힌 집합은 각 간선 양 끝을 함께 포함해야 합니다. 열림은 소속을 이웃으로 전파하고, 여집합의 열림은 경계를 넘는 전파를 막기 때문입니다. 역으로 각 graph 성분과 그 여집합은 최소 이웃들의 합집합이므로 열립니다. 따라서 graph 성분이 정확히 연결 성분입니다. 거리를 가정하지 않고 유한 알고리즘을 증명한 것입니다.

유한성은 중요합니다. 일반 공간에서 clopen 집합으로 분리되지 않는 점들의 모임은 quasi-component이며 연결 성분과 다를 수 있습니다. 연결성이 경로 연결성을 항상 뜻하지도 않습니다. 교실의 유한 모형을 모든 모집단 공간에 그대로 옮기면 안 됩니다.

아래 전수 검사는 두 label의 fiber가 모두 열리는지 독립적으로 확인해 연속 이진 사상을 셉니다. 목표가 discrete이면 두 fiber는 서로 여집합인 열린집합이어야 합니다. 세 원공간의 결과는 2, 8, 2입니다. 목표를 indiscrete로 바꾸면 여덟 함수가 모두 연속인 이유도 설명하세요.

| 원공간 위상 | 열린 mask | Discrete Y로 가는 연속 사상 |
|---|---|---:|
| Indiscrete | 0, 7 | 2 |
| Discrete | 0부터 7 | 8 |
| Chain | 0, 4, 6, 7 | 2 |

## Python으로 계산 재현하기

```python
topologies = [{0, 7}, set(range(8)), {0, 4, 6, 7}]
for opens in topologies:
    continuous = [m for m in range(8) if m in opens and (7 ^ m) in opens]
    print(sorted(opens), len(continuous), continuous)
assert [sum(m in o and (7 ^ m) in o for m in range(8)) for o in topologies] == [2, 8, 2]
```

## 직접 풀기와 풀이

indiscrete 원공간에서 discrete 목표의 두 값을 모두 찍는 연속 함수가 있을까요? 없습니다. 연결 공간의 연속상은 연결이어야 합니다. 상이 두 열린 부분으로 분리된다면 그 역상들이 X도 분리하기 때문입니다. 따라서 이 경우 연속 함수는 상수입니다. 함수 mask 8개로 유한 예제를 확인한 뒤 일반적인 역상 논증도 설명하세요.

## 데이터 분석과의 연결

연결 입력 공간을 discrete label로 분류하는 함수는 연속이 아닐 수 있습니다. 결정 경계의 불연속이 자동으로 코드 오류가 되는 것은 아닙니다. 반대로 표본 점의 이웃 graph가 연결이라고 미지의 모집단 공간까지 연결임을 증명한 것도 아닙니다. graph 이웃·거리 이웃·추상 위상을 어떤 규칙으로 구성했는지부터 명시하세요.

추가 읽기: [Allen Hatcher, Notes on Introductory Point-Set Topology](https://pi.math.cornell.edu/~hatcher/Top/TopNotes.pdf).
