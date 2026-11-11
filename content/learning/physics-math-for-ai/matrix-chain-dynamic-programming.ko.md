## 순서는 같아도 계산량은 달라진다

행렬 곱은 정확한 수학에서 결합법칙을 만족하지만 순서를 바꿀 수 있는 교환법칙은 없습니다. A₁A₂A₃에서 괄호를 바꾸는 것만으로 중간 행렬 크기와 곱셈 비용이 달라집니다. 이 단원은 실제 행렬 원소를 곱하지 않고, 고전적 dense 곱셈의 scalar multiplication 수를 최적화합니다. 유한 정밀도의 결합 순서가 값에 주는 영향이나 sparse 구조는 별도 문제입니다.

차원 배열 d₀,...,dₙ에서 Aᵢ의 크기는 dᵢ₋₁×dᵢ입니다. Uniform, 길고 얇은 alternating, 증가하는 차원 가족을 선택할 수 있습니다. 파랑은 prefix별 최적 비용, 주황은 항상 왼쪽부터 괄호를 붙인 비용입니다.

## 최적 부분구조의 증명

구간 i부터 j의 마지막 곱셈이 k에서 나뉜다고 생각합니다. 왼쪽과 오른쪽의 차원이 정해지므로 마지막 곱셈 비용은 고정입니다. 한쪽이 최적이 아니라면 그 부분만 더 싼 방법으로 교체해 전체를 개선할 수 있으므로 모순입니다.

$$
M_{i,j}=\min_{i\le k<j}\{M_{i,k}+M_{k+1,j}+d_i d_{k+1}d_{j+1}\},\qquad M_{i,i}=0.
$$

식은 0-based index입니다. 마지막 곱의 차원은 dᵢ×dₖ₊₁와 dₖ₊₁×dⱼ₊₁입니다. 비용을 더하는 것이 현재의 작업 모형이고 행렬을 실제로 재배열하는 명령이 아닙니다.

## 분할정복과 동적계획

가능한 split을 재귀로 모두 시도하면 같은 구간을 여러 번 다시 계산합니다. Memoization은 방문한 결과를 저장하고, tabulation은 짧은 구간부터 표를 채웁니다. 현재 실험은 bottom-up DP와 저장하지 않는 재귀를 둘 다 실제 실행합니다. 재귀 호출 수와 DP split 평가 수는 서로 다른 계수입니다.

$$
\#\mathrm{DP\ splits}=\frac{n(n^2-1)}6,\qquad
\mathrm{space}=O(n^2),\quad\mathrm{time}=O(n^3).
$$

Split 표에서 최적 괄호를 복원할 수 있으므로 비용 값뿐 아니라 증거도 남깁니다. 일반적인 분할정복의 예인 mergesort는 겹치지 않는 두 절반과 선형 merge로 T(n)=2T(n/2)+O(n)을 가집니다. 그 재귀와 중복 구간을 재사용하는 현재 DP를 같은 것으로 부르지 마세요.

## 메모리 제약이 DP의 충분 상태를 바꾼다

Scalar 작업량만으로 feasible한 contraction 순서를 고르지 못할 수 있다. 차원 (1,10,100,5)를 생각하자. 원래 입력 저장은 제외하고 새 중간·최종 출력을 세며 kernel scratch는 0으로 가정한다. 부모 출력을 할당할 때 두 child 결과를 유지하고 곱셈 후 child 중간값을 해제한다.

왼쪽 먼저 평가할 때 child peak pL,pR,유지 중간 크기 sL,sR,출력 s의 peak는 max(pL,sL+pR,sL+sR+s)다. 오른쪽 우선은 L,R을 바꾼다. Leaf 추가 저장은 0이다. 각 평가 순서를 비교하면 서로 지배하지 않는 두 plan을 얻는다. 왼쪽 결합은 작업 1500·peak105 scalar,오른쪽은 작업5050·peak55다.

| 임시·출력 용량 | Feasible한 최소 작업 Plan | 작업량 |
|---|---|---:|
| 50 | 없음 | 미정 |
| 60 | A₁(A₂A₃) | 5050 |
| 110 | (A₁A₂)A₃ | 1500 |

```python
d=(1,10,100,5)
def plans(i,j):
    if i==j:return [(0,0,0,f'A{i+1}')]
    out=[]
    for k in range(i,j):
        for wl,pl,sl,tl in plans(i,k):
            for wr,pr,sr,tr in plans(k+1,j):
                size=d[i]*d[j+1]
                work=wl+wr+d[i]*d[k+1]*d[j+1]
                left=max(pl,sl+pr,sl+sr+size)
                right=max(pr,sr+pl,sl+sr+size)
                out.append((work,min(left,right),size,f'({tl}{tr})'))
    return out
candidates=plans(0,2)
assert {(w,p) for w,p,_,_ in candidates}=={(1500,105),(5050,55)}
for capacity in (50,60,110):
    feasible=[p for p in candidates if p[1]<=capacity]
    best=min(feasible) if feasible else None
    print(capacity,best)
```

확장 planner는 구간마다 (work,peak) Pareto frontier를 유지할 수 있다. 같은 경계 shape를 보존하며 두 좌표 모두 더 나쁘지 않은 상태가 있어야 기존 상태를 버릴 수 있다. 최저 작업 subplan 하나만 보관하면 나중 메모리 제한을 만족하는 유일한 plan을 잃을 수 있다. Frontier 크기와 열거 비용은 별도 분석이 필요하며 원래 scalar O(n³) bound가 자동으로 유지되지 않는다.

아래는 custom 차원의 보조 전수 planner다. 실제 차원 가족을 바꾸거나 GPU allocator·입력 residency·곱셈 workspace를 추정하지 않는다.

## 손계산과 직접 풀기

차원이 1,2,3,4이면 (A₁A₂)A₃는 1·2·3+1·3·4=18입니다. A₁(A₂A₃)는 2·3·4+1·2·4=32입니다. 수학적으로 같은 곱이지만 비용이 다릅니다. 차원을 모두 2배 하면 각 곱셈 비용이 8배가 됩니다.

직접 풀기: 차원 5,30,5,30의 두 괄호 비용을 구하세요. 풀이: 왼쪽은 5·30·5+5·5·30=1500, 오른쪽은 30·5·30+5·30·30=9000입니다. 현재 n=3 표의 split과 비교하세요. n=6에서는 DP 후보가 35개, 현재 저장 없는 재귀 호출은 243개이며 둘을 같은 연산이라고 직접 나눠 속도비를 주장할 수는 없습니다.

## AI 계산 그래프와 최적화

Tensor contraction 계획, query optimizer, sequence decoding은 부분문제와 중간 표현의 비용에 민감합니다. 최적 부분구조가 성립하는지, 경계 상태가 충분한지, 연산 모형이 실제 kernel 비용과 맞는지 먼저 확인하세요. 최소 scalar 곱셈 수가 메모리 traffic이나 GPU 실행 시간까지 최소화한다고 단정할 수 없습니다.

참고: [MIT 알고리즘과 동적계획 자료](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/).
