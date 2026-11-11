## 유한 문제와 계산 가능성의 범위

결정 문제는 어떤 입력 하나의 답이 아니라 yes인 입력 encoding들의 언어입니다. 현재 실험은 세 변수·세 clause의 작은 SAT 공식을 다룹니다. 여덟 진리 할당과 아홉 occurrence vertex의 모든 선택을 전수 검사할 수 있습니다. 이것은 크기가 고정된 유한 문제의 검증이며 모든 프로그램의 정지를 판별하는 일반 절차와 다릅니다.

진리 할당 bit i는 변수 xᵢ₊₁의 참·거짓입니다. Vertex bit는 clause 순서대로 나열한 literal occurrence를 선택합니다. 같은 literal이 반복돼도 occurrence는 서로 다른 vertex입니다. 그래프의 주황 node는 query mask의 선택이며, 색칠했다는 것만으로 유효한 증거가 되지는 않습니다.

## 구성적 환원과 양방향 증거

Clause마다 세 occurrence를 clique로 연결하고, 서로 모순인 x와 not x의 occurrence도 연결합니다. Clause 수 m=3에 대해 다음이 성립합니다.

$$
\varphi\text{ satisfiable}\iff\alpha(G_\varphi)\ge m.
$$

참 할당이 있으면 clause마다 참 literal 하나를 고릅니다. 한 clause에서 하나만 골랐고 같은 할당에서 x와 not x가 동시에 참일 수 없으므로 선택은 독립집합입니다. 반대로 크기 3의 독립집합은 clause clique마다 정확히 하나를 선택해야 합니다. 모순 literal은 함께 선택할 수 없어 일관된 변수 값을 정할 수 있고 미선택 변수는 임의로 확장해도 모든 clause가 만족됩니다.

일반 입력에서도 vertex는 3m개, edge는 최대 O(m²)이므로 이 구성은 다항 크기입니다. 한 작은 예시의 양방향 검산과 일반 크기 bound의 증명은 구분하세요. 환원 A≤pB는 B의 효율적인 해법으로 A를 풀 수 있다는 방향이지 그 반대가 아닙니다.

## Halting의 대각선 논증

프로그램과 입력을 문자열로 encoding할 수 있다고 합시다. 모든 P,x에 대해 P(x)의 정지 여부를 항상 맞게 끝내는 계산 가능한 H가 있다고 가정합니다. 다음 프로그램 D를 구성합니다.

$$
D(P)=\begin{cases}\text{loop forever},&H(P,P)=\text{halts},\\
\text{halt},&H(P,P)=\text{does not halt}.\end{cases}
$$

D 자신의 encoding을 넣으면 H가 정지라고 답할 때 D는 멈추지 않고, 멈추지 않는다고 답할 때 D는 멈춥니다. 어느 답도 맞을 수 없어 그런 total H는 없습니다. 이 일반 모순은 SAT 화면의 유한 탐색 결과나 timeout에서 추론한 것이 아닙니다. SAT의 유한 변수 전수 결정과 halting 불가능성을 혼동하지 마세요.

## Decision Oracle로 증거 복원하기

임의 공식의 satisfiability를 결정하는 oracle을 가정하자. 처음 yes를 받은 뒤 실제 변수를 하나씩 고정한다. 그 변수를 false로 둔 현재 공식이 여전히 satisfiable한지 묻고,yes이면 false,아니면 true를 유지한다. 적어도 하나의 만족 completion이 남는다는 불변식이다. n개를 모두 고정하면 할당 자체가 증거이며 처음 질의를 포함해 최대 n+1회의 결정 질의를 쓴다.

변수 이름은 서로 다른 occurrence 이름을 세어 재번호화할 수 있다. n은 가장 큰 이진 identifier가 아니라 실제 변수 수다. 고정값은 unit clause 또는 literal을 세 번 반복한 3-CNF clause로 추가할 수 있다. 공식 증가와 원래 환원은 encoding 길이에 다항이다. 결정 oracle이 다항이면 이 증거 탐색도 다항이지만 여기서 그 oracle을 제공한 것은 아니다.

| 입력 | 이 절차의 결정 질의 |
|---|---:|
| 세 변수 Satisfiable 예제 | 4 |
| Unsatisfiable 예제 | 1 |
| 일반 n변수 Satisfiable 입력 | 최대 n+1 |

```python
from itertools import product
def recover(clauses,n):
    calls=0
    def oracle(partial):
        nonlocal calls
        calls+=1
        return any(all(bits[i-1]==b for i,b in partial.items()) and all(any(bits[abs(l)-1]==(l>0) for l in c) for c in clauses) for bits in product((False,True),repeat=n))
    partial={}
    if not oracle(partial):return None,calls
    for variable in range(1,n+1):
        partial[variable]=False
        if not oracle(partial):partial[variable]=True
    assert all(any(partial[abs(l)]==(l>0) for l in c) for c in clauses)
    return partial,calls
sat=[[1,2,3],[-1,2,3],[1,-2,-3]]
unsat=[[1,1,1],[-1,-1,-1],[2,2,2]]
assert recover(sat,3)[1]==4 and recover(unsat,3)==(None,1)
print(recover(sat,3),recover(unsat,3))
```

위의 작은 oracle은 할당을 전수 열거하므로 실행 예제가 다항 SAT 해법은 아니다. 유한 증거 검사·oracle 탐색-결정 환원·정지 불가능성은 별도 주장이다. 실제 정지한 실행은 실행으로 인식할 수 있지만 임의 timeout이 비정지나 unsatisfiability의 인증은 아니다.

## 손계산과 직접 풀기

첫 공식에서 모든 변수를 참으로 두면 각 clause의 첫 참 occurrence를 고른 forward mask는 81입니다. Query mask 273은 다른 세 occurrence를 골라서 변수 할당 3으로 복원할 수 있습니다. 원래 할당 7과 달라도 둘 다 같은 공식을 만족할 수 있으므로 환원이 원래 증거를 유일하게 복원해야 하는 것은 아닙니다.

직접 풀기: (x∨x∨x)와 (not x∨not x∨not x), (y∨y∨y)를 동시에 만족할 수 있나요? 풀이: 앞 두 clause가 모순이므로 불가능합니다. 대응 그래프의 최대 독립집합은 2이며, 각 clause에서 하나씩 고른 세 node는 첫 두 clause 사이의 contradiction edge 때문에 독립일 수 없습니다.

## AI 추론과 검증 가능한 증거

LLM이 SAT 답이나 프로그램의 성질을 주장할 때 후보 증거의 유한 검증과 일반 알고리즘 존재 주장을 분리하세요. Satisfying assignment나 독립집합은 직접 확인할 수 있지만 timeout은 unsatisfiable 증명서가 아닙니다. NP의 빠른 증거 검증과 빠른 증거 발견도 같은 성질이 아닙니다. 이 단원은 임의 프로그램을 실행하지 않고 명시한 유한 공식만 검산합니다.

참고: [MIT 계산 가능성·환원 강의](https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/).
