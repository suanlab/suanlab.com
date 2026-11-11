## 연산을 줄이는 것과 동시에 실행하는 것

희소성은 0과 곱하는 작업을 없애고 병렬성은 독립 작업을 동시에 실행합니다. 둘은 같은 최적화가 아닙니다. 이 단원은 대각 2, 선택한 off-diagonal band -1인 행렬에 대해 실제 dense 곱, CSR 곱, balanced reduction task 그래프를 모두 계산합니다. 파랑·주황은 각각 CSR와 dense 출력이고 표에는 실제 스케줄로 얻은 출력도 남깁니다.

CSR는 row pointer, column index, value로 nonzero를 저장합니다. 각 row의 범위는 rowptr[i]부터 rowptr[i+1] 직전까지입니다. 교육 모형은 비교를 위해 dense 행렬과 trace도 생성하므로 화면 전체 준비 비용을 O(nnz)라고 부르지 않습니다. 아래 core multiplication의 비용과 setup을 분리합니다.

## 실제 작업 DAG 만들기

각 nonzero product를 한 unit task로 만들고 같은 row의 항들을 이진 트리로 더합니다. 두 child가 끝나야 parent addition이 준비됩니다. 홀수 개 항에서는 마지막 항을 다음 layer로 넘깁니다. 모든 row에는 diagonal nonzero가 있어 빈 reduction은 없습니다.

$$
W=\sum_i\{r_i+(r_i-1)\}=2\,\mathrm{nnz}-N,\qquad
S=\max_i\{1+\lceil\log_2r_i\rceil\}.
$$

rᵢ는 row의 nonzero 수입니다. W는 product와 addition의 전체 task 수, S는 product부터 output까지 가장 긴 의존 경로입니다. Task마다 같은 시간, 무료 데이터 접근, 독립 processor라는 이상화가 들어갑니다. Sparse index 접근이나 실제 GPU kernel 비용까지 포함한 cycle 예측은 아닙니다.

## 유한 Processor에서 스케줄하기

$$
T_P\ge\max\{\lceil W/P\rceil,S\}.
$$

아무리 잘 스케줄해도 전체 W개 작업을 P개보다 많이 동시에 실행할 수 없고 한 의존 경로를 건너뛸 수도 없습니다. 현재 구현은 매 tick의 시작에 모든 dependency가 끝난 task 중 index 순으로 최대 P개를 골라 실제 실행합니다. 이 greedy schedule이 항상 최적이라는 주장은 하지 않습니다. 각 task를 한 번만 실행했는지, child가 이전 tick에 끝났는지, 결과가 같은지를 독립 검사합니다.

## 작업을 비워 두지 않는 스케줄에는 상한도 있다

Unit-task DAG의 greedy list scheduler는 준비 작업이 P개 이상이면 P개를 실행한다. Tick을 full과 underfull로 나누자. Underfull에서는 준비 작업을 모두 실행해 남은 최장 경로를 최소 1 줄이므로 그런 tick은 최대 S개다. Full마다 P개,underfull마다 최소 하나를 세면

$$
\max(W/P,S)\le T_P\le W/P+(1-1/P)S.
$$

이상적 작업 모형의 부등식이며 ready 작업을 찾는 실제 CPU 시간이 아니다. Live 구현은 dependency 기록을 훑어 스케줄을 만들고 그 진단 비용을 unit DAG 작업으로 세지 않는다. 유용 병렬성 W/S는 메모리 비용이 없어도 processor를 더 늘려 효과가 사라지는 이유를 설명한다.

![Processor 수별 실제 unit-task list 스케줄 tick과 work/span 상계를 비교하며 실측 시간이 아님을 구분하는 그림.](/assets/learning/algorithm-work-span-ko.svg)

| N=8,bandwidth1의 P | 정수 하한 | 실제 List Tick |
|---|---:|---:|
| 1 | 36 | 36 |
| 4 | 9 | 10 |
| 16 | 3 | 4 |

```python
from fractions import Fraction as F
tasks=[]
for count in (2,3,3,3,3,3,3,2):
    layer=[]
    for _ in range(count):layer.append(len(tasks));tasks.append([])
    while len(layer)>1:
        nxt=[]
        for k in range(0,len(layer),2):
            if k+1==len(layer):nxt.append(layer[k])
            else:nxt.append(len(tasks));tasks.append([layer[k],layer[k+1]])
        layer=nxt
depth=[]
for deps in tasks:depth.append(1+max([depth[d] for d in deps],default=0))
W,S=len(tasks),max(depth)
for P in (1,4,16):
    done=set();ticks=0
    while len(done)<W:
        ready=[i for i,deps in enumerate(tasks) if i not in done and all(d in done for d in deps)][:P]
        assert ready;done.update(ready);ticks+=1
    assert max(F(W,P),S)<=ticks<=F(W,P)+(1-F(1,P))*S
    print(P,W,S,ticks)
data=[1e16,1.,-1e16,1.];serial=0.
for x in data:serial+=x
balanced=(data[0]+data[1])+(data[2]+data[3])
assert serial==1 and balanced==0 and sum(map(F,data))==2
```

독립 DAG 예제로 양쪽 bound를 확인한다. 산술 정확도도 별도 제약이다. 저장 수 (10¹⁶,1,−10¹⁶,1)의 순차 double 합은 1,지정한 균형 합은 0인데 정확 합은 2다. 트리가 오차 상계를 개선해도 모든 입력에서 더 정확한 것은 아니다. 작업 감소와 병렬성 증가가 재결합한 수치 계산의 검증을 대신하지 않는다.

## 손계산과 직접 풀기

N=8, band 폭 1이면 양 끝 row는 각각 2개, 내부 여섯 row는 각각 3개 nonzero입니다. nnz=22, W=44-8=36, S=3입니다. P=1이면 36 tick이고 P=4의 하한은 9 tick입니다. 현재 greedy 실행은 10 tick으로 하한보다 한 tick 큽니다. 하한과 달라도 유효한 스케줄일 수 있습니다.

직접 풀기: band 폭 0, N=P=8이면 W,S,T를 구하세요. 풀이: row마다 diagonal product 하나뿐이므로 W=8, S=1이며 8개 task를 같은 tick에 실행해 T=1입니다. Processor를 16개로 늘려도 span 1보다 작아지지 않습니다.

## AI의 희소·병렬 계산 평가

Sparse attention, graph aggregation과 sparse linear layer는 nonzero 수만으로 실제 가속을 보장하지 않습니다. Index 저장, 불규칙 접근, synchronization과 occupancy가 영향을 줍니다. 또한 균형 reduction은 덧셈 순서를 바꾸므로 유한 정밀도에서 serial 합과 bitwise 동일할 필요가 없습니다. 정확도 tolerance와 데이터 이동 비용도 함께 기록하세요. 현재 모형은 작업·의존성의 검증 가능한 기준이며 실제 장치 timing은 별도입니다.

참고: [MIT 성능 공학 강의](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/).
